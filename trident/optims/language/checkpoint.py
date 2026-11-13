"""Atomic, weights-only-loadable training state (including RNG and data state)."""
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_rng():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "gaussian": numpy_state[3], "cached": numpy_state[4]},
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    value = state["numpy"]
    np.random.set_state((value["name"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["gaussian"], value["cached"]))
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("exact resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def loader_config(loader):
    dataset = getattr(loader, "dataset", None)
    source = getattr(dataset, "source", None)
    try:
        length = len(dataset) if dataset is not None else len(loader)
    except TypeError:
        length = None
    return {"type": type(loader).__module__ + "." + type(loader).__name__,
            "length": length,
            "batch_size": getattr(loader, "batch_size", None),
            "seed": getattr(loader, "seed", None),
            "shuffle": getattr(loader, "shuffle", None),
            "drop_last": getattr(loader, "drop_last", None),
            "collator": collator_config(getattr(loader, "collate", getattr(loader, "collate_fn", None))),
            "fingerprint": getattr(source, "fingerprint", None)}


def tokenizer_config(tokenizer):
    if tokenizer is None:
        return None
    import hashlib
    import json
    getter = getattr(tokenizer, "get_vocab", None)
    if getter is None and hasattr(tokenizer, "backend"):
        getter = tokenizer.backend.get_vocab
    vocab = getter() if callable(getter) else {"vocab_size": getattr(tokenizer, "vocab_size", None)}
    return {"type": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
            "vocab_hash": hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest(),
            "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id}


def collator_config(collator):
    if collator is None:
        return None
    output = {"type": type(collator).__module__ + "." + type(collator).__name__}
    for key in ("max_length", "text_field", "pad_to_multiple_of"):
        if hasattr(collator, key):
            output[key] = getattr(collator, key)
    if hasattr(collator, "tokenizer"):
        output["tokenizer"] = tokenizer_config(collator.tokenizer)
    if hasattr(collator, "collator"):
        output["collator"] = collator_config(collator.collator)
    return output


def data_state(loader, epoch, offset):
    state_method = getattr(loader, "state_dict", None)
    return {"config": loader_config(loader), "epoch": epoch, "offset": offset,
            "iterator": state_method() if callable(state_method) else None}


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "trident.language" or state.get("version") != 1:
        raise ValueError("unsupported language training checkpoint")
    return state
