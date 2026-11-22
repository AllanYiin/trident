"""Weights-only native model archives, including indexed, memory-mapped shards."""
import json
import os
import re
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from .checkpoint import atomic_torch_save
from .quantization import NF4Linear


_INDEX = "model.pt.index.json"


def _write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _byte_limit(value):
    if type(value) is int and value > 0:
        return value
    match = re.fullmatch(r"([1-9][0-9]*)\s*(B|KiB|MiB|GiB|KB|MB|GB)", str(value), re.I)
    if not match:
        raise ValueError("max_shard_size must be positive bytes or a size such as '1GiB'")
    unit = match[2].upper()
    exponent = {"B": 0, "K": 1, "M": 2, "G": 3}[unit[0]]
    return int(match[1]) * (1024 if "I" in unit else 1000) ** exponent


def save_pretrained(model, directory, max_shard_size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = model.config.to_dict()
    state = model.state_dict()
    if any(key.endswith((".lora_A", ".lora_B")) for key in state):
        raise ValueError("use save_adapter or export a merged adapter before saving pretrained weights")
    if any(t.is_meta for t in state.values()):
        raise ValueError("reload offloaded weights on one device before exporting")
    if max_shard_size is None and not (directory / _INDEX).exists():
        atomic_torch_save(state, directory / "model.pt")
    else:
        limit = _byte_limit(max_shard_size if max_shard_size is not None else "1GiB")
        generation = uuid.uuid4().hex
        weight_map, shard, size = {}, {}, 0
        def flush():
            name = f"model-{generation}-{len(weight_map):06d}.bin"
            atomic_torch_save(shard, directory / name)
            weight_map.update(dict.fromkeys(shard, name))
            shard.clear()
        for key, value in state.items():
            if config["tie_word_embeddings"] and key == "lm_head.weight":
                continue
            nbytes = value.numel() * value.element_size()
            if shard and size + nbytes > limit:
                flush()
                size = 0
            shard[key] = value
            if config["tie_word_embeddings"] and key == "model.embed_tokens.weight":
                shard["lm_head.weight"] = value
            size += nbytes
        if shard:
            flush()
        # Immutable shard names keep an in-flight reader on one complete generation.
        _write_json(directory / _INDEX, {"weight_map": weight_map, "config": config})
    _write_json(directory / "trident_config.json", config)


def _mapped(path):
    try:
        return torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        return torch.load(path, weights_only=True, map_location="cpu")


class _Archive:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / _INDEX
        self.single = None
        self.config = None
        if self.path.is_file():
            index = json.loads(self.path.read_text(encoding="utf-8"))
            self.weights = index["weight_map"]
            self.config = index.get("config")
            for name in set(self.weights.values()):
                if (Path(name).name != name or ":" in name or "\\" in name or
                        (self.directory / name).resolve().parent != self.directory.resolve()):
                    raise ValueError("invalid checkpoint shard path")
        else:
            self.path = self.directory / "model.pt"
            self.single = _mapped(self.path)
            self.weights = dict.fromkeys(self.single, self.path.name)

    def tensors(self, keys):
        groups = defaultdict(list)
        for key in keys:
            groups[self.weights[key]].append(key)
        for name, names in groups.items():
            state = self.single if self.single is not None else _mapped(self.directory / name)
            for key in names:
                yield key, state[key]
            del state

    def dispatch(self, model, **options):
        from accelerate import load_checkpoint_and_dispatch
        if self.single is not None:
            return load_checkpoint_and_dispatch(model, str(self.path), **options)
        # Accelerate reopens the index. Keep the generation we already validated
        # even if a concurrent exporter publishes a new one in this directory.
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / _INDEX
            paths = {key: str((self.directory / name).absolute()) for key, name in self.weights.items()}
            _write_json(index, {"weight_map": paths})
            return load_checkpoint_and_dispatch(model, str(index), **options)


def load_pretrained_weights(target, path, source_prefix="", *, strict=True):
    """Copy the selected subtree into an existing module. Report target-relative keys."""
    archive = _Archive(path)
    prefix = source_prefix.rstrip(".")
    prefix = prefix + "." if prefix else ""
    selected = {key[len(prefix):]: key for key in archive.weights if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"no checkpoint weights match {source_prefix!r}")
    current = target.state_dict()
    missing = sorted(current.keys() - selected.keys())
    unexpected = sorted(selected.keys() - current.keys())
    matched = [selected[key] for key in current if key in selected]
    loaded, mismatch = [], []
    # Validate every selected tensor before changing the existing target.
    for key, tensor in archive.tensors(matched):
        local = key[len(prefix):]
        if current[local].is_meta:
            raise ValueError("partial loading needs a materialized target; use from_pretrained for a meta model")
        if tensor.is_meta or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"unsupported tensor storage for {key}")
        if tensor.shape == current[local].shape:
            loaded.append(local)
        else:
            mismatch.append(local)
    report = dict(loaded=loaded, missing=missing, unexpected=unexpected, mismatched=mismatch)
    if strict and (missing or unexpected or mismatch):
        raise ValueError(f"incompatible selected weights: {report}")
    with torch.no_grad():
        for key, tensor in archive.tensors([selected[key] for key in loaded]):
            current[key[len(prefix):]].copy_(tensor)
    return report


def load_pretrained(cls, directory, device, dtype, device_map, max_memory, offload_folder, quantization_config):
    from trident.models.pytorch_causal_lm import CausalLMConfig, RotaryEmbedding
    archive = _Archive(directory)
    config = CausalLMConfig(**(archive.config or json.loads(
        (Path(directory) / "trident_config.json").read_text(encoding="utf-8"))))
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("dtype must be a floating-point torch dtype")
    if quantization_config not in (None, {"format": "nf4"}):
        raise ValueError("use quantization_config={'format': 'nf4'} for native NF4 weights")
    if device_map is None and (max_memory is not None or offload_folder is not None):
        raise ValueError("set device_map when supplying an inference memory budget or offload folder")
    with torch.device("meta"):
        model = cls(config)
    # Rotary frequencies are non-persistent and have no entry in the checkpoint.
    for layer in model.model.layers:
        layer.self_attn.rotary = RotaryEmbedding(config)

    packed = {key.rsplit(".", 1)[0] for key in archive.weights if key.endswith(".packed_weight")}
    quantize = {}
    with torch.device("meta"):
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name != "lm_head" and (name in packed or quantization_config):
                replacement = NF4Linear(module)
                parent, _, child = name.rpartition(".")
                setattr(model.get_submodule(parent), child, replacement)
                if name not in packed:
                    quantize[name + ".weight"] = (replacement, module.weight.shape)
    expected = model.state_dict()
    required = set(expected)
    for key in quantize:
        prefix = key[:-len("weight")]
        required.difference_update((prefix + "packed_weight", prefix + "absmax"))
        required.add(key)
    if required != archive.weights.keys():
        raise ValueError(f"incompatible checkpoint keys: missing={sorted(required - archive.weights.keys())}, "
                         f"unexpected={sorted(archive.weights.keys() - required)}")
    if device_map is not None:
        if quantize:
            raise ValueError("save a native NF4 checkpoint on one device before dispatching packed weights")
        for key, tensor in archive.tensors(required):
            if expected[key].shape != tensor.shape:
                raise ValueError(f"incompatible checkpoint shape for {key}")
            if key.endswith(".packed_weight") and tensor.dtype != torch.uint8:
                raise ValueError(f"expected packed NF4 bytes for {key}")
        # Packed weights live in buffers. Disk offload needs both these buffers
        # and their scales, while RotaryEmbedding can be rebuilt from config.
        # Two active models must not overwrite each other's disk-backed tensors.
        offload_dir = (str(Path(offload_folder) / uuid.uuid4().hex) if offload_folder is not None else None)
        model = archive.dispatch(
            model, device_map=device_map, max_memory=max_memory,
            offload_folder=offload_dir, no_split_module_classes=["DecoderBlock"],
            dtype=dtype, offload_buffers=bool(packed), skip_keys="past_key_values", strict=True)
        if packed:
            model.requires_grad_(False)
        return model.eval()

    for key, tensor in archive.tensors(archive.weights):
        if key in quantize:
            module, shape = quantize[key]
            if tensor.shape != shape:
                raise ValueError(f"incompatible checkpoint shape for {key}")
            module.pack(tensor, device)
            continue
        if tensor.shape != expected[key].shape:
            raise ValueError(f"incompatible checkpoint shape for {key}")
        if key.endswith(".packed_weight") and tensor.dtype != torch.uint8:
            raise ValueError(f"expected packed NF4 bytes for {key}")
        if config.tie_word_embeddings and key == "lm_head.weight":
            continue
        name, _, field = key.rpartition(".")
        module = model.get_submodule(name)
        target_dtype = expected[key].dtype
        if dtype is not None and tensor.is_floating_point() and not isinstance(module, NF4Linear):
            target_dtype = dtype
        value = tensor.to(device=device, dtype=target_dtype)
        if field in module._parameters:
            value = nn.Parameter(value, requires_grad=module._parameters[field].requires_grad)
        setattr(module, field, value)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    for layer in model.model.layers:
        layer.self_attn.rotary.to(device)
    if packed or quantize:
        model.requires_grad_(False)
    return model
