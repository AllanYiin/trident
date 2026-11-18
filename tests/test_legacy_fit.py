import os

import numpy as np
import pytest

if os.environ.get("TRIDENT_BACKEND", "pytorch") == "tensorflow":
    from trident.backend.tensorflow_ops import reduce_mean
    from trident.layers.tensorflow_layers import Dense
    from trident.optims.tensorflow_trainer import Model
else:
    from trident.backend.pytorch_ops import reduce_mean
    from trident.layers.pytorch_layers import Dense
    from trident.optims.pytorch_trainer import Model


def ids(value):
    if os.environ.get("TRIDENT_BACKEND", "pytorch") == "pytorch" and hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value).reshape(-1).tolist()


def mean_label(output, target):
    return reduce_mean(target)


@pytest.mark.parametrize("validation", ["split", "explicit", "none"])
def test_fit_trains_on_requested_ids_and_validates_only_held_out_ids(tmp_path, monkeypatch, validation):
    monkeypatch.chdir(tmp_path)
    x = np.arange(12, dtype=np.float32).reshape(-1, 1)
    item = Model(input_shape=(1,), output=Dense(1, use_bias=False))
    item.with_optimizer("Adam", lr=0.01).with_loss("MSELoss").with_metric(mean_label)
    original_weight = ids(item.model.weight)
    item.training_context["save_path"] = str(tmp_path / "fit.pth.tar")
    seen_train, seen_validation = [], []
    original = Model.do_calculate_forward

    def record(self, is_training=True):
        batch = self.training_context["train_data" if is_training else "test_data"]
        (seen_train if is_training else seen_validation).append(ids(batch["input"]))
        return original(self, is_training=is_training)

    monkeypatch.setattr(Model, "do_calculate_forward", record)
    options = {}
    if validation == "split":
        options["validation_split"] = 0.25
    elif validation == "explicit":
        external = x[:3] + 100
        options.update(validation_data=(external, external), validation_split=0.25,
                       validation_freq=2, validation_steps=1)

    assert item.fit(x, x, epochs=2, batch_size=4, validation_batch_size=2,
                    shuffle=False, **options) is item
    expected_epoch = [list(range(4)), list(range(4, 8))]
    expected_epoch += [[8]] if validation == "split" else [list(range(8, 12))]
    assert seen_train == expected_epoch * 2
    expected_validation = {"split": [[9, 10], [11]] * 2, "explicit": [[100, 101]], "none": []}
    assert seen_validation == expected_validation[validation]
    assert ids(item.model.weight) != original_weight
    _, metric_values = item.training_context["out_sample_metrics"].get_series("mean_label")
    assert metric_values == {"split": [10.0, 10.0], "explicit": [100.5], "none": []}[validation]


def test_failed_fit_keeps_last_good_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    item = Model(input_shape=(1,), output=Dense(1, use_bias=False))
    item.with_optimizer("Adam", lr=0.01).with_loss("MSELoss")
    item.save_model(str(tmp_path / "fit.pth.tar"))
    previous = {p: p.read_bytes() for p in tmp_path.glob("*.pth*")}
    assert len(previous) == 2

    def fail(self, is_training=True):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(Model, "do_calculate_forward", fail)
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    with pytest.raises(RuntimeError, match="forward failed"):
        item.fit(x, x, batch_size=2, validation_split=0.25)
    assert {p: p.read_bytes() for p in tmp_path.glob("*.pth*")} == previous
