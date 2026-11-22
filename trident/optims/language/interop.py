"""Optional Transformers interop. Training never depends on HF trainers."""
import json
from pathlib import Path

from trident.models.pytorch_causal_lm import CausalLM


CHAT_TEMPLATE = """{% if tools %}{{ '<|im_start|>system\nTools: ' }}{{ tools | tojson }}{{ '<|im_end|>\n' }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% if message['reasoning_content'] is defined and message['reasoning_content'] %}{{ '<think>' + message['reasoning_content'] + '</think>\n' }}{% endif %}{{ message['content'] or '' }}{% if message['tool_calls'] is defined and message['tool_calls'] %}{{ '<tool_call>' }}{{ message['tool_calls'] | tojson }}{{ '</tool_call>' }}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if open_thinking is defined and open_thinking %}{{ '<think>' }}{% endif %}{% endif %}"""


def to_hf_tokenizer(tokenizer):
    from transformers import PreTrainedTokenizerFast
    from trident.data.pipeline.language import BPETokenizer, ByteTokenizer, SPECIAL_TOKENS
    if isinstance(tokenizer, BPETokenizer):
        backend = tokenizer.backend
    elif isinstance(tokenizer, ByteTokenizer):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        vocab.update({"<|reserved_{}|>".format(index): index for index in range(len(vocab), 16)})
        vocab.update({character: byte + 16 for byte, character in bytes_to_unicode().items()})
        backend = Tokenizer(models.BPE(vocab, []))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.add_special_tokens(SPECIAL_TOKENS)
    else:
        return tokenizer
    output = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token=tokenizer.bos_token,
                                    eos_token=tokenizer.eos_token, pad_token=tokenizer.pad_token,
                                    additional_special_tokens=SPECIAL_TOKENS[3:],
                                    clean_up_tokenization_spaces=False)
    output.chat_template = CHAT_TEMPLATE
    return output


def to_huggingface(model):
    if not isinstance(model, CausalLM):
        if hasattr(model, "save_pretrained"):
            return model
        raise TypeError("HF export supports native CausalLM or a HF PreTrainedModel")
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    values = model.config.to_dict()
    values.update(head_dim=model.config.head_dim, hidden_act="silu", attention_bias=False,
                  use_sliding_window=False)
    if model.config.num_experts:
        values["moe_intermediate_size"] = model.config.intermediate_size
        values["norm_topk_prob"] = True
        config = Qwen3MoeConfig(**values)
        target = Qwen3MoeForCausalLM(config)
    else:
        config = Qwen3Config(**values)
        target = Qwen3ForCausalLM(config)
    from .quantization import NF4Linear
    state = model.state_dict()
    quantized = set()
    for name, module in model.named_modules():
        if isinstance(module, NF4Linear):
            state.pop(name + ".packed_weight")
            state.pop(name + ".absmax")
            quantized.add(name + ".weight")
            with torch.no_grad():
                target.get_submodule(name).weight.copy_(module.unpack(target.dtype))
    missing, unexpected = target.load_state_dict(state, strict=False)
    if set(missing) != quantized or unexpected:
        raise ValueError("incompatible native weights for Hugging Face export")
    return target.eval()


def export_model(model, tokenizer, directory, format="native", *, max_shard_size=None):
    directory = Path(directory)
    if format not in ("native", "huggingface"):
        raise ValueError("format must be native or huggingface")
    directory.mkdir(parents=True, exist_ok=True)
    if format == "huggingface":
        exported = to_huggingface(model)
        options = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
        exported.save_pretrained(str(directory), safe_serialization=True, **options)
        if tokenizer is not None:
            to_hf_tokenizer(tokenizer).save_pretrained(str(directory))
    else:
        if not isinstance(model, CausalLM):
            raise TypeError("external models must be exported as huggingface")
        model.save_pretrained(directory, max_shard_size=max_shard_size)
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)
    (directory / "trident_export.json").write_text(json.dumps({"format": format, "version": 1}), encoding="utf-8")
    return directory
