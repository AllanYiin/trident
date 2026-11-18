import numpy as np
import pytest
import torch

from trident.data.pipeline import DataProvider, Dataset, DatasetSchema, FieldSpec
from trident.layers.pytorch_layers import Dense
from trident.optims.pytorch_trainer import Model
from trident.optims.trainers import TrainingPlan


def make_plan(tmp_path, rows=7, batch_size=2, accumulation=1):
    schema = DatasetSchema([FieldSpec("x", metadata={"role": "input"}),
                            FieldSpec("target", kind="label", metadata={"role": "target"})])
    samples = [{"x": np.array([i], np.float32), "target": np.array([2 * i], np.float32)}
               for i in range(1, rows + 1)]
    data = DataProvider(train=Dataset(samples, schema=schema)).for_trident_trainer("train", batch_size=batch_size)
    item = Model(input_shape=(1,), output=Dense(1, use_bias=False))
    item.with_optimizer(torch.optim.SGD, lr=0.001).with_loss("MSELoss").with_accumulate_grads(accumulation)
    with torch.no_grad():
        item.model.weight.fill_(1)
    item.training_context["save_path"] = str(tmp_path / "latest.pth.tar")
    return TrainingPlan().add_training_item(item).with_data_loader(data).with_batch_size(batch_size), item


@pytest.mark.parametrize("bad_stage", ["forward", "backward", "step", "interrupt"])
def test_failure_reaches_caller_and_preserves_checkpoint(tmp_path, monkeypatch, bad_stage):
    plan, item = make_plan(tmp_path, accumulation=2)
    item.save_model()
    files = {p: p.read_bytes() for p in tmp_path.glob("latest.pth*")}
    assert len(files) == 2
    error = KeyboardInterrupt if bad_stage == "interrupt" else RuntimeError

    def fail(*args, **kwargs):
        raise error("injected training failure")

    if bad_stage in ("forward", "interrupt"):
        monkeypatch.setattr(item.model, "forward", fail)
    elif bad_stage == "backward":
        item.model.weight.register_hook(fail)
    else:
        monkeypatch.setattr(item.optimizer, "step", fail)
    plan.save_model_scheduling(1, unit="epoch")
    with pytest.raises(error, match="injected training failure"):
        plan.start_now()
    assert all(p.read_bytes() == original for p, original in files.items())
    assert item.model.weight.grad is None


def test_epochs_consume_short_final_batch_without_unscheduled_saves(tmp_path, monkeypatch):
    plan, item = make_plan(tmp_path)
    seen = []
    original = Model.train_model

    def record(self, train_data, *args, **kwargs):
        seen.append(train_data["x"].ravel().tolist())
        return original(self, train_data, *args, **kwargs)

    monkeypatch.setattr(Model, "train_model", record)
    plan.repeat_epochs(2).start_now()
    assert seen == [[1, 2], [3, 4], [5, 6], [7]] * 2
    assert plan.steps == 8
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rows,large,small,window", [(4, 4, 2, 2), (5, 5, 2, 4), (7, 4, 2, 2)])
def test_cpu_accumulation_matches_large_batch_including_short_tail(tmp_path, rows, large, small, window):
    direct, reference = make_plan(tmp_path, rows, large)
    split, accumulated = make_plan(tmp_path, rows, small, window)
    direct.repeat_epochs(2).start_now()
    split.repeat_epochs(2).start_now()
    torch.testing.assert_close(accumulated.model.weight, reference.model.weight, rtol=1e-6, atol=1e-7)


def test_mixed_sum_and_mean_losses_match_a_single_batch(tmp_path):
    def squared_sum(output, target):
        return (output - target).square().sum()

    squared_sum.reduction = "sum"
    direct, reference = make_plan(tmp_path, 5, 5)
    split, accumulated = make_plan(tmp_path, 5, 2, 4)
    for item in (reference, accumulated):
        item.with_loss(squared_sum, loss_weight=0.25)
    direct.start_now()
    split.start_now()
    torch.testing.assert_close(accumulated.model.weight, reference.model.weight)


def test_restore_native_module_and_report_missing_keys(tmp_path):
    layer = torch.nn.Linear(1, 1)
    item = Model(input_shape=(1,), output=layer)
    path = tmp_path / "weights.pth.tar"
    torch.save({"state_dict": {"weight": torch.tensor([[9.0]]), "unused": torch.zeros(1)}}, path)
    before = layer.weight.detach().clone()
    with pytest.raises(RuntimeError, match="missing keys"):
        item.load_model(path, strict=True)
    torch.testing.assert_close(layer.weight, before)
    with pytest.warns(UserWarning, match="Partial"):
        result = item.load_model(path)
    assert result.missing_keys == ["bias"]
    assert result.unexpected_keys == ["unused"]
    torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, 9))


def test_portable_roundtrip_and_opt_in_legacy_load(tmp_path):
    _, item = make_plan(tmp_path)
    sample = torch.ones(2, 1, device=item.model.weight.device)
    expected = item.model(sample).detach().clone()
    item.save_model()
    for suffix in (".pth", ".pth.tar"):
        path = tmp_path / ("latest" + suffix)
        state = torch.load(path, weights_only=True)
        assert isinstance(state["signature"], dict)
        with torch.no_grad():
            item.model.weight.zero_()
        item.load_model(path, strict=True)
        torch.testing.assert_close(item.model(sample), expected)
    legacy = torch.nn.Linear(1, 1, bias=False)
    path = tmp_path / "legacy.pth"
    torch.save(legacy, path)
    native = Model(input_shape=(1,), output=torch.nn.Linear(1, 1, bias=False))
    with pytest.raises(ValueError, match="trusted_legacy"):
        native.load_model(path)
    native.load_model(path, trusted_legacy=True, strict=True)
    torch.testing.assert_close(native.model.weight.cpu(), legacy.weight)


@pytest.mark.parametrize("native", [False, True])
def test_explicit_full_model_export_roundtrips(tmp_path, native):
    layer = torch.nn.Linear(1, 1, bias=False) if native else Dense(1, use_bias=False)
    item = Model(input_shape=(1,), output=layer)
    path = tmp_path / "old-consumer.pth"
    with torch.no_grad():
        layer.weight.fill_(3)
    item.save_model(path, save_full_model=True)
    with torch.no_grad():
        layer.weight.zero_()
    item.load_model(path, trusted_legacy=True, strict=True)
    assert layer.weight.item() == 3


def test_step_limit_flushes_partial_window(tmp_path):
    reference, direct = make_plan(tmp_path, rows=6, batch_size=4)
    limited, accumulated = make_plan(tmp_path, rows=6, batch_size=2, accumulation=4)
    assert reference.only_steps(1) is True
    assert limited.only_steps(2) is True
    assert limited.steps == 2
    torch.testing.assert_close(accumulated.model.weight, direct.model.weight)
    assert list(tmp_path.iterdir()) == []
    limited.only_steps(2)
    assert limited.steps == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA mixed precision")
def test_cuda_amp_accumulates_actual_number_of_examples(tmp_path):
    direct, reference = make_plan(tmp_path, rows=5, batch_size=5)
    split, accumulated = make_plan(tmp_path, rows=5, batch_size=2, accumulation=4)
    for item in (reference, accumulated):
        item.with_automatic_mixed_precision_training()
        assert item.is_autocast_enabled
        item.gradscaler = torch.amp.GradScaler("cuda", init_scale=8)
    direct.start_now()
    split.start_now()
    assert reference.model.weight.item() != 1
    torch.testing.assert_close(accumulated.model.weight, reference.model.weight, atol=1e-4, rtol=1e-4)
