import pytest
import torch

from trident.layers.pytorch_layers import Dense
from trident.optims.pytorch_trainer import Model


def test_nonfinite_weights_preserve_existing_files(tmp_path):
    item = Model(input_shape=(1,), output=Dense(1))
    path = tmp_path / "weights.pth.tar"
    item.save_model(path)
    previous = {p: p.read_bytes() for p in tmp_path.iterdir()}
    with torch.no_grad():
        item.model.weight.fill_(float("nan"))
    assert item.save_model(path) is False
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == previous


def test_serialization_failure_preserves_existing_files(tmp_path, monkeypatch):
    item = Model(input_shape=(1,), output=Dense(1))
    path = tmp_path / "weights.pth.tar"
    item.save_model(path)
    previous = {p: p.read_bytes() for p in tmp_path.iterdir()}

    def fail(obj, stream):
        stream.write(b"partial")
        raise OSError("write failed")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="write failed"):
        item.save_model(path)
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == previous


def test_failure_replacing_archive_restores_its_companion(tmp_path, monkeypatch):
    from trident.backend import checkpoint

    item = Model(input_shape=(1,), output=Dense(1))
    path = tmp_path / "weights.pth.tar"
    item.save_model(path)
    previous = {p: p.read_bytes() for p in tmp_path.glob("weights.pth*")}
    with torch.no_grad():
        item.model.weight.add_(2)
    replace = checkpoint.os.replace

    def fail_last(source, target):
        if target == path:
            raise PermissionError("destination locked")
        replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", fail_last)
    with pytest.raises(PermissionError, match="destination locked"):
        item.save_model(path)
    assert {p: p.read_bytes() for p in tmp_path.glob("weights.pth*")} == previous
