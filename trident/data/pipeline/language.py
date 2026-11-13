"""JSONL, tokenizers and loss-safe language collation for the new data pipeline."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .dataset import Dataset


SPECIAL_TOKENS = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
                  "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


class ByteTokenizer:
    """Dependency-free UTF-8 tokenizer for offline tests and small examples.

    Production training should normally use train_bpe_tokenizer or an existing
    fast tokenizer. This fixed vocabulary is intentionally not a learned BPE.
    """
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    pad_token, bos_token, eos_token = SPECIAL_TOKENS[:3]
    vocab_size = 272

    def __len__(self):
        return self.vocab_size

    def encode_with_offsets(self, text):
        ids, offsets = [], []
        cursor = 0
        pattern = re.compile("(" + "|".join(re.escape(item) for item in SPECIAL_TOKENS) + ")")
        for match in pattern.finditer(text):
            for index in range(cursor, match.start()):
                encoded = text[index].encode("utf-8")
                ids.extend(byte + 16 for byte in encoded)
                offsets.extend([(index, index + 1)] * len(encoded))
            ids.append(SPECIAL_TOKENS.index(match.group()))
            offsets.append(match.span())
            cursor = match.end()
        for index in range(cursor, len(text)):
            encoded = text[index].encode("utf-8")
            ids.extend(byte + 16 for byte in encoded)
            offsets.extend([(index, index + 1)] * len(encoded))
        return ids, offsets

    def encode(self, text, add_special_tokens=False):
        ids, _ = self.encode_with_offsets(text)
        return ([self.bos_token_id] + ids + [self.eos_token_id]) if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=False):
        output, buffer = [], bytearray()
        def flush():
            if buffer:
                output.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()
        for token in ids:
            token = int(token)
            if 16 <= token < 272:
                buffer.append(token - 16)
            else:
                flush()
                if not skip_special_tokens and 0 <= token < len(SPECIAL_TOKENS):
                    output.append(SPECIAL_TOKENS[token])
        flush()
        return "".join(output)

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "trident_tokenizer.json").write_text(json.dumps({"type": "byte", "version": 1}), encoding="utf-8")


class BPETokenizer(ByteTokenizer):
    def __init__(self, backend):
        self.backend = backend
        self.vocab_size = backend.get_vocab_size()
        self.pad_token_id = backend.token_to_id(self.pad_token)
        self.bos_token_id = backend.token_to_id(self.bos_token)
        self.eos_token_id = backend.token_to_id(self.eos_token)

    def encode_with_offsets(self, text):
        encoded = self.backend.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def decode(self, ids, skip_special_tokens=False):
        return self.backend.decode([int(token) for token in ids], skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path / "tokenizer.json"))
        (path / "trident_tokenizer.json").write_text(json.dumps({"type": "bpe", "version": 1}), encoding="utf-8")


def train_bpe_tokenizer(texts, vocab_size=6400, min_frequency=2):
    """Train from an iterable of strings; never downloads a corpus."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must fit the byte alphabet and special tokens")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return BPETokenizer(tokenizer)


def load_tokenizer(directory):
    path = Path(directory)
    metadata = path / "trident_tokenizer.json"
    if metadata.exists():
        config = json.loads(metadata.read_text(encoding="utf-8"))
        if config["type"] == "byte":
            return ByteTokenizer()
        if config["type"] == "bpe":
            from tokenizers import Tokenizer
            return BPETokenizer(Tokenizer.from_file(str(path / "tokenizer.json")))
        raise ValueError("unknown Trident tokenizer format")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


class JSONLSource:
    """Indexed JSONL reader with a content fingerprint for resume validation."""
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.offsets = []
        digest = hashlib.sha256()
        with open(self.path, "rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if line.strip():
                    self.offsets.append(offset)
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with open(self.path, "rb") as stream:
            stream.seek(self.offsets[index])
            try:
                sample = json.loads(stream.readline())
            except (ValueError, UnicodeError) as error:
                raise ValueError("invalid JSONL record at byte {} in {}".format(self.offsets[index], self.path)) from error
        if not isinstance(sample, dict):
            raise ValueError("JSONL records must be objects")
        if "conversations" in sample and "messages" not in sample:
            sample["messages"] = sample.pop("conversations")
        return sample

    def state_dict(self):
        return {"fingerprint": self.fingerprint}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("JSONL content changed since checkpoint")


def jsonl_dataset(path):
    return Dataset(JSONLSource(path), map_style=True)


class ChatTemplate:
    """Explicit role template with character spans, never delimiter scanning.

    Assistant spans include their end-of-turn marker. Tool observations and
    role headers are context only. `thinking` controls the generation prefix.
    """
    def render(self, messages, tools=None, add_generation_prompt=False, thinking=False):
        parts, spans, length = [], [], 0
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a nonempty list")
        for index, message in enumerate(messages):
            role = message.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError("unsupported chat role: " + str(role))
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("text-only chat content must be a string")
            if index == 0 and tools:
                prefix = "<|im_start|>system\nTools: " + json.dumps(tools, ensure_ascii=False) + "<|im_end|>\n"
                parts.append(prefix)
                length += len(prefix)
            header = "<|im_start|>" + role + "\n"
            if role == "assistant" and message.get("reasoning_content"):
                content = "<think>" + message["reasoning_content"] + "</think>\n" + content
            if role == "assistant" and message.get("tool_calls"):
                calls = message["tool_calls"]
                if isinstance(calls, str):
                    calls = json.loads(calls)
                content += "<tool_call>" + json.dumps(calls, ensure_ascii=False) + "</tool_call>"
            text = header + content + "<|im_end|>\n"
            if role == "assistant":
                spans.append((length + len(header), length + len(text) - 1))
            parts.append(text)
            length += len(text)
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n" + ("<think>" if thinking else ""))
        return "".join(parts), spans


def encode_offsets(tokenizer, text):
    if hasattr(tokenizer, "encode_with_offsets"):
        return tokenizer.encode_with_offsets(text)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("assistant masking requires a fast tokenizer with offset mappings")
    output = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(output["input_ids"]), list(output["offset_mapping"])


def encode_chat(tokenizer, template, messages, max_length, tools=None):
    text, spans = template.render(messages, tools=tools)
    tokens, offsets = encode_offsets(tokenizer, text)
    tokens, offsets = tokens[:max_length], offsets[:max_length]
    labels = [token if end > start and any(start >= left and end <= right for left, right in spans) else -100
              for token, (start, end) in zip(tokens, offsets)]
    return tokens, labels


def _pad_rows(rows, pad_token_id, pad_to_multiple_of=None):
    if not rows:
        raise ValueError("cannot collate an empty batch")
    length = max(len(tokens) for tokens, _ in rows)
    if length < 2:
        raise ValueError("language sequences need at least two tokens")
    if pad_to_multiple_of:
        length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    inputs = np.full((len(rows), length), pad_token_id, dtype=np.int64)
    labels = np.full_like(inputs, -100)
    mask = np.zeros_like(inputs)
    for index, (tokens, targets) in enumerate(rows):
        inputs[index, :len(tokens)] = tokens
        labels[index, :len(tokens)] = targets
        mask[index, :len(tokens)] = 1
    return {"input_ids": inputs, "attention_mask": mask, "labels": labels}


class CausalLMCollator:
    accepts_schema = True

    def __init__(self, tokenizer, max_length=512, text_field="text", pad_to_multiple_of=None):
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define PAD and EOS IDs")
        self.tokenizer, self.max_length = tokenizer, max_length
        self.text_field, self.pad_to_multiple_of = text_field, pad_to_multiple_of

    def rows(self, samples):
        result = []
        for sample in samples:
            prefix = [] if self.tokenizer.bos_token_id is None else [self.tokenizer.bos_token_id]
            tokens = prefix + list(self.tokenizer.encode(str(sample[self.text_field]), add_special_tokens=False))[:self.max_length - len(prefix) - 1]
            tokens.append(self.tokenizer.eos_token_id)
            result.append((tokens, list(tokens)))
        return result

    def __call__(self, samples, schema=None):
        return _pad_rows(self.rows(samples), self.tokenizer.pad_token_id, self.pad_to_multiple_of)


class SFTCollator(CausalLMCollator):
    def __init__(self, tokenizer, max_length=1024, template=None, **kwargs):
        super().__init__(tokenizer, max_length, **kwargs)
        self.template = template or ChatTemplate()

    def rows(self, samples):
        return [encode_chat(self.tokenizer, self.template,
                            sample.get("messages", sample.get("conversations")),
                            self.max_length, sample.get("tools")) for sample in samples]


class PreferenceCollator(SFTCollator):
    """Pair format: prompt messages + chosen/rejected strings or full chats."""
    def __call__(self, samples, schema=None):
        chosen, rejected = [], []
        for sample in samples:
            prompt = sample.get("prompt", [])
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            pair = []
            for field in ("chosen", "rejected"):
                value = sample[field]
                messages = prompt + [{"role": "assistant", "content": value}] if isinstance(value, str) else value
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError("preference branches must end with an assistant answer")
                pair.append(messages)
            if pair[0][:-1] != pair[1][:-1]:
                raise ValueError("chosen and rejected must have identical prompt messages")
            # Only the final answer is an action in a preference pair.
            for target, messages in zip((chosen, rejected), pair):
                text, spans = self.template.render(messages, tools=sample.get("tools"))
                tokens, offsets = encode_offsets(self.tokenizer, text)
                left, right = spans[-1]
                labels = [token if end > start and start >= left and end <= right else -100
                          for token, (start, end) in zip(tokens, offsets)]
                tokens, labels = tokens[:self.max_length], labels[:self.max_length]
                if not any(label != -100 for label in labels[1:]):
                    raise ValueError("preference answer was entirely truncated")
                target.append((tokens, labels))
        output = {}
        for name, rows in (("chosen", chosen), ("rejected", rejected)):
            output.update({name + "_" + key: value for key, value in
                           _pad_rows(rows, self.tokenizer.pad_token_id, self.pad_to_multiple_of).items()})
        return output


class PackedCollator:
    """Pack whole samples; segment IDs prevent cross-document attention."""
    accepts_schema = True

    def __init__(self, collator, max_length):
        if max_length < collator.max_length:
            raise ValueError("packing length must fit each collated sample")
        self.collator, self.max_length = collator, max_length

    def __call__(self, samples, schema=None):
        packed, segments, positions = [], [], []
        tokens, labels, segment, position = [], [], [], []
        for index, (item, target) in enumerate(self.collator.rows(samples)):
            if len(tokens) + len(item) > self.max_length:
                packed.append((tokens, labels))
                segments.append(segment)
                positions.append(position)
                tokens, labels, segment, position = [], [], [], []
            # The preceding document must not predict the next document's BOS.
            tokens.extend(item)
            labels.extend([-100] + target[1:])
            segment.extend([index] * len(item))
            position.extend(range(len(item)))
        if tokens:
            packed.append((tokens, labels))
            segments.append(segment)
            positions.append(position)
        output = _pad_rows(packed, self.collator.tokenizer.pad_token_id)
        output["segment_ids"] = np.full_like(output["input_ids"], -1)
        output["position_ids"] = np.zeros_like(output["input_ids"])
        for index, (segment, position) in enumerate(zip(segments, positions)):
            output["segment_ids"][index, :len(segment)] = segment
            output["position_ids"][index, :len(position)] = position
        return output
