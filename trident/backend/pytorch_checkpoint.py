"""Portable checkpoints for the legacy PyTorch trainer."""
import pickle
import shutil
import warnings
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from torch import nn
from torch.torch_version import TorchVersion

from trident.backend.checkpoint import staged_checkpoints
from trident.backend.common import OrderedDict, Signature, TensorShape
from trident.backend.tensorspec import ObjectType, TensorSpec


_LEGACY_METADATA = (OrderedDict, Signature, TensorShape, TensorSpec, ObjectType, TorchVersion)
_METADATA_LOAD_LOCK = RLock()


def _load_weights_checkpoint(path):
    serialization = torch.serialization
    if not hasattr(serialization, 'safe_globals'):
        return torch.load(path, map_location='cpu', weights_only=True)

    # Old Trident state-dict archives embed the input/output signature as objects.
    # Scope the allowlist to this load and preserve the caller's registrations.
    with _METADATA_LOAD_LOCK:
        registered = set(serialization.get_safe_globals())
        metadata = [kind for kind in _LEGACY_METADATA if kind not in registered]
        with serialization.safe_globals(metadata):
            return torch.load(path, map_location='cpu', weights_only=True)


def _plain(value):
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    raise TypeError("Unsupported checkpoint metadata type: {}".format(type(value).__name__))


def _encode_spec(spec):
    if spec is None:
        return None
    dtype = spec.dtype
    if isinstance(dtype, torch.dtype):
        dtype = str(dtype)
    elif isinstance(dtype, type) and dtype.__module__ == 'builtins':
        dtype = 'builtins.' + dtype.__name__
    elif dtype is not None:
        dtype = 'numpy.' + np.dtype(dtype).name
    return dict(name=spec.name, shape=None if spec.shape is None else list(spec.shape.dims),
                dtype=dtype, object_type=None if spec.object_type is None else spec.object_type.value,
                ndim=spec.ndim, min_ndim=spec.min_ndim, max_ndim=spec.max_ndim,
                axes=spec.axes, optional=spec.optional, default=spec.default)


def _decode_spec(values):
    if values is None:
        return None
    values = dict(values)
    dtype = values['dtype']
    if dtype is not None:
        family, name = dtype.split('.', 1)
        if family == 'torch':
            dtype = getattr(torch, name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("Unsupported torch dtype " + name)
        elif family == 'builtins':
            types = {t.__name__: t for t in (bool, int, float, str, bytes, tuple, list, dict, type(None))}
            dtype = types[name]
        elif family == 'numpy':
            dtype = np.dtype(name)
        else:
            raise ValueError("Unknown dtype family " + family)
    values['dtype'] = dtype
    if values['shape'] is not None:
        values['shape'] = TensorShape(values['shape'])
    if values['object_type'] is not None:
        values['object_type'] = ObjectType(values['object_type'])
    return TensorSpec(**values)


def _signature_fields(signature):
    if signature is None:
        return None
    return _plain(dict(name=signature.name,
                       inputs={k: _encode_spec(v) for k, v in signature.inputs.items()},
                       outputs={k: _encode_spec(v) for k, v in signature.outputs.items()}))


def write_model_checkpoint(model, path, metadata, save_full_model=False):
    """Prepare both legacy filenames before replacing either existing file."""
    archive = Path(path)
    weights = archive.with_suffix('')  # name.pth.tar -> name.pth
    archive.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(_plain(metadata), state_dict=model.state_dict(),
                   signature=_signature_fields(getattr(model, 'signature', None)))
    with staged_checkpoints(weights, archive) as (new_weights, new_archive):
        with new_archive.open('wb') as stream:
            torch.save(payload, stream)
        if save_full_model:
            import dill
            # Legacy wrappers attach bound methods to native nn.Module instances.
            # dill preserves those methods when exporting an explicitly requested whole model.
            with new_weights.open('wb') as stream:
                torch.save(model, stream, pickle_module=dill, _use_new_zipfile_serialization=False)
        else:
            shutil.copyfile(new_archive, new_weights)


def restore_model_checkpoint(model, path, *, strict=False, trusted_legacy=False):
    """Return the key comparison and optional metadata after loading a state dict."""
    try:
        if trusted_legacy:
            archive = torch.load(path, map_location='cpu', weights_only=False)
        else:
            archive = _load_weights_checkpoint(path)
    except pickle.UnpicklingError as exc:
        if trusted_legacy:
            raise
        raise ValueError("Weights-only load failed. For a trusted legacy whole-model archive, "
                         "pass trusted_legacy=True.") from exc
    if isinstance(archive, nn.Module) and trusted_legacy:
        from trident.backend.pytorch_backend import Layer, fix_layer
        if isinstance(archive, Layer):
            archive = fix_layer(archive)
        archive = {'state_dict': archive.state_dict(), 'signature': getattr(archive, 'signature', None)}
    if not isinstance(archive, dict):
        raise ValueError("Expected a PyTorch state dict or model checkpoint")
    if 'state_dict' not in archive:
        archive = {'state_dict': archive}
    if archive.get('backend', 'pytorch') != 'pytorch':
        raise ValueError("Checkpoint backend is not PyTorch")
    signature = archive.get('signature')
    if isinstance(signature, dict):
        signature = Signature(name=signature['name'], **{
            side: OrderedDict((k, _decode_spec(v)) for k, v in signature[side].items())
            for side in ('inputs', 'outputs')})
    weights = archive['state_dict']
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint state_dict must be a mapping of parameter names to values")
    current = model.state_dict()
    common = current.keys() & weights.keys()
    missing = sorted(current.keys() - weights.keys())
    unexpected = sorted(weights.keys() - current.keys())
    if (not common and (current or weights)) or (strict and (missing or unexpected)):
        raise RuntimeError(f"Incompatible checkpoint: missing keys {missing}, unexpected keys {unexpected}")
    # Native load_state_dict can copy earlier tensors before it discovers a mismatch.
    for key in common:
        target, source = current[key], weights[key]
        if torch.is_tensor(target) and not torch.nn.parameter.is_lazy(target):
            if not torch.is_tensor(source) or target.shape != source.shape:
                raise RuntimeError(f"Incompatible checkpoint tensor for {key}")
    result = model.load_state_dict(weights, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        warnings.warn(f"Partial checkpoint load: {result}", UserWarning, stacklevel=3)
    if signature is not None:
        model._signature = signature
        if 'signature' in vars(model):
            model.signature = signature
    model.eval()
    return result, archive
