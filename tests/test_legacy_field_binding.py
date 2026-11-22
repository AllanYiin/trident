import os

import numpy as np
import pytest

from trident import TensorShape, TensorSpec, loss, metric
from trident.data.pipeline import DataProvider, Dataset, DatasetSchema, FieldSpec
from trident.optims.losses import Loss
from trident.optims.metrics import Metric
from trident.optims.trainers import TrainingPlan

if os.environ.get("TRIDENT_BACKEND", "pytorch") == "tensorflow":
    import tensorflow as tf
    from trident.backend.tensorflow_backend import Layer
    from trident.layers.tensorflow_layers import Dense
    from trident.optims.tensorflow_trainer import Model

    def tensor(array):
        return tf.constant(array, dtype=tf.float32)

    def accuracy(pred, truth):
        return tf.reduce_mean(tf.cast((pred > 0) == (truth > 0), tf.float32))

    def squared(x):
        return tf.reduce_mean(tf.square(x))

    def set_weight(weight):
        weight.assign([[1.]])

    def scalar(value):
        return float(value.numpy().item())

else:
    import torch
    from trident.backend.pytorch_backend import Layer
    from trident.backend.pytorch_backend import get_device
    from trident.layers.pytorch_layers import Dense
    from trident.optims.pytorch_trainer import Model

    def tensor(array):
        return torch.tensor(array, dtype=torch.float32, device=get_device())

    def accuracy(pred, truth):
        return ((pred > 0) == (truth > 0)).float().mean()

    def squared(x):
        return x.square().mean()

    def set_weight(weight):
        with torch.no_grad():
            weight.fill_(1.)

    def scalar(value):
        return value.item()


@pytest.fixture(autouse=True)
def isolated_training_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class Heads(Layer):
    def __init__(self, reverse=False):
        super().__init__()
        self.project = Dense(1, use_bias=False)
        self.reverse = reverse

    def forward(self, front, side):
        heads = {"color_logits": self.project(front), "size_logits": self.project(side)}
        return dict(reversed(heads.items())) if self.reverse else heads


def setup_heads(reverse=False):
    example = tensor([[1.], [2.]])
    model = Model(inputs=(example, example), output=Heads(reverse))
    model.data_feed = {"front": "front_image", "side": "side_image"}
    model.with_optimizer("Adam", lr=.001)
    set_weight(model.model.project.weight)
    schema = DatasetSchema([
        FieldSpec("front_image"), FieldSpec("side_image"),
        FieldSpec("color_id", kind="label"), FieldSpec("size_id", kind="label"),
    ])
    rows = [{k: np.array([v], np.float32) for k, v in
             dict(front_image=i, side_image=-i, color_id=-i, size_id=-i).items()} for i in (1, 2)]
    provider = DataProvider(train=Dataset(rows, schema=schema)).for_trident_trainer("train", batch_size=2)
    plan = TrainingPlan().add_training_item(model).with_data_loader(provider).with_batch_size(2)
    return model, plan


@pytest.mark.parametrize("reverse, weight", [(False, .5), (True, 0.)])
def test_multihead_bindings_and_single_detached_loss(reverse, weight, monkeypatch):
    model, plan = setup_heads(reverse)
    calls = []

    @loss
    def joint(color_logits, color_id, *, size_logits, size_id):
        calls.append(1)
        return squared(color_logits - color_id), squared(size_logits - size_id)

    @metric(input_names={"pred": "size_logits", "truth": "size_id"})
    def score(pred, *, truth, scale=1):
        return scale * accuracy(pred, truth)

    assert isinstance(joint, Loss) and isinstance(score, Metric)
    assert scalar(score(tensor([1.]), truth=tensor([1.]))) == 1.
    model.with_loss(joint, as_metric=True, loss_weight=weight)
    colors = {"pred": "color_logits", "truth": "color_id"}
    for name in (("color", "size") if reverse else ("size", "color")):
        model.with_metric(score, name=name, input_names=colors if name == "color" else {}, scale=2)
    assert 'joint' not in model.metrics
    history = model.training_context['tmp_metrics']
    observed = {}
    original = history.collect

    def record(key, step, value):
        if key == 'joint':
            assert isinstance(value, (np.ndarray, np.number, float))
        observed[key] = float(value) if not hasattr(value, 'shape') else float(np.asarray(
            value.detach().cpu() if hasattr(value, 'detach') else value))
        return original(key, step, value)

    monkeypatch.setattr(history, 'collect', record)
    plan.start_now()
    assert calls == [1]
    assert observed['color'] == 0 and observed['size'] == 2
    assert observed['joint'] == pytest.approx(10.)
    assert (scalar(model.model.project.weight) < 1.) == bool(weight)


def test_ambiguous_output_reports_both_candidates_and_fix():
    model, plan = setup_heads()

    @loss(input_names={"target": "size_id"})
    def objective(output, target):
        return squared(output - target)

    model.with_loss(objective)
    with pytest.raises(ValueError, match=r"objective.output.*color_logits.*size_logits.*input_names"):
        plan.start_now()
    assert scalar(model.model.project.weight) == 1.


def test_bad_explicit_field_and_shape():
    model, _ = setup_heads()
    data = dict(front_image=tensor([[1.]]), side_image=tensor([[1.]]))
    model.data_feed = {"front": "absent", "side": "side_image"}
    with pytest.raises(ValueError, match=r"model.front.*absent"):
        model._prepare_model_fields(data)
    model.data_feed = {"front": "front_image", "side": "side_image"}
    with pytest.raises(ValueError, match=r"model.side.*side_image.*expected shape"):
        model._prepare_model_fields(dict(data, side_image=tensor([[1., 2.]])))
    with pytest.raises(ValueError, match=r"batch dimension"):
        model._prepare_model_fields(dict(data, side_image=tensor([[1.], [2.]])))


def test_metric_ambiguity_stops_before_optimizer_update():
    model, plan = setup_heads()

    @loss
    def distance(color_logits, color_id):
        return squared(color_logits - color_id)

    model.with_loss(distance).with_metric("accuracy", input_names={"target": "size_id"})
    with pytest.raises(ValueError, match=r"metric accuracy.output.*color_logits.*size_logits.*input_names"):
        plan.start_now()
    assert scalar(model.model.project.weight) == 1.


def test_validation_uses_local_bindings_and_declared_shape_filters_outputs():
    model, _ = setup_heads()
    one_column = TensorSpec(shape=TensorShape([None, 1]))

    @metric
    def measured(output: one_column, /, *, target):
        return squared(output - target)

    model.with_metric(measured, input_names={"target": "truth"})
    model._provider_fields = (("front_image", "side_image"), ("truth",))
    data = dict(front_image=tensor([[1.]]), side_image=tensor([[2.]]),
                size_logits=tensor([[4.]]), color_logits=tensor([[8., 9.]]), truth=tensor([[1.]]))
    fn = model.metrics["measured"]
    assert scalar(model._call_objective("metric", "measured", fn, data)) == 9.
    model.with_metric(measured, name="wrong", input_names={"output": "color_logits", "target": "truth"})
    with pytest.raises(ValueError, match="expected shape"):
        model._call_objective("metric", "wrong", model.metrics["wrong"], data)


def test_fit_held_out_uses_decorator_bindings_and_fixed_options():

    @metric(input_names={"observed": "target"})
    def label_value(*, observed, factor=1):
        return squared(observed) * factor

    item = Model(input_shape=(1,), output=Dense(1, use_bias=False))
    item.with_optimizer("Adam").with_loss("MSELoss").with_metric(label_value, factor=2)
    data = np.ones((3, 1), np.float32)
    item.fit(data, data, validation_data=(data[:2], 3 * data[:2]), batch_size=2, epochs=1)
    _, values = item.training_context["out_sample_metrics"].get_series("label_value")
    assert values == [18.]

@pytest.mark.parametrize("dense", [True, False])
def test_accuracy_accepts_classification_shapes_and_rejects_unrelated_axis(dense):
    model, _ = setup_heads()
    model.with_metric("accuracy", input_names={"output": "color_logits", "target": "color_id"})
    scores = tensor([[.9, .1, 0.], [.1, .1, .8]])
    if dense:
        labels = tensor([[1., 0., 0.], [0., 0., 1.]])
    elif os.environ.get("TRIDENT_BACKEND", "pytorch") == "tensorflow":
        labels = tf.constant([0, 2], dtype=tf.int64)
    else:
        labels = torch.tensor([0, 2], dtype=torch.int64, device=get_device())
    data = dict(color_logits=scores, color_id=labels)
    fn = model.metrics["accuracy"]
    assert scalar(model._call_objective("metric", "accuracy", fn, data)) == 1.
    with pytest.raises(ValueError, match="classification target shape"):
        model._call_objective("metric", "accuracy", fn, dict(data, color_id=tensor([[0., 1.], [1., 0.]])))


def test_output_field_cannot_overwrite_batch_field():
    model, _ = setup_heads()
    data = {"color_logits": tensor([1.])}
    with pytest.raises(ValueError, match="collides.*color_logits"):
        model._store_forward_outputs(data, {"color_logits": tensor([2.])})
    assert scalar(data["color_logits"]) == 1.

def test_existing_loss_constructor_bindings_and_scalar_snapshot():
    model, _ = setup_heads()
    if os.environ.get("TRIDENT_BACKEND", "pytorch") == "tensorflow":
        from trident.optims.tensorflow_losses import MSELoss
    else:
        from trident.optims.pytorch_losses import MSELoss
    criterion = MSELoss(input_names={"output": "size_logits", "target": "size_id"})
    model.with_loss(criterion, as_metric=True, name="distance", loss_weight=0.)
    model.training_context.update(current_epoch=0, current_loss=tensor(0.),
                                  train_data={"size_logits": tensor([[2.]]), "size_id": tensor([[4.]])})
    model.do_calculate_losses()
    assert scalar(model.training_context["current_loss"]) == 0.
    assert model.training_context["tmp_metrics"].get_last("distance")[1] == 4.
