import os
import numpy as np
import pytest

if os.environ.get('TRIDENT_BACKEND') != 'tensorflow':
    pytest.skip('run with TRIDENT_BACKEND=tensorflow', allow_module_level=True)
tf = pytest.importorskip('tensorflow')

from trident.callbacks import ConstantWarmup, CosineWarmup, ExponentialWarmup, LinearWarmup
from trident.data.pipeline import Dataset, DataProvider, DatasetSchema, FieldSpec
from trident.layers.tensorflow_layers import Dense
from trident.layers.tensorflow_rnn import standard_lstm
from trident.optims.tensorflow_trainer import Model
from trident.optims.tensorflow_optimizers import Adam, AdamW, LARS, PlainRAdam, RAdam, SGD
from trident.optims.trainers import TrainingPlan


def test_dynamic_classification_uses_valid_weight_on_each_batch():
    from trident.optims.tensorflow_losses import CrossEntropyLoss
    criterion = CrossEntropyLoss(from_logits=True, sample_weight=[1., 3., .5])

    @tf.function(input_signature=[tf.TensorSpec((None, None, 3), tf.float32),
                                  tf.TensorSpec((None, None), tf.int64)])
    def compute(scores, labels):
        return criterion(scores, labels), criterion.gradient_denominator

    for labels, denominator in [([[0, 1]], 4.), ([[2], [-100]], .5), ([[-100]], 0.)]:
        indices = tf.constant(labels, tf.int64)
        value, weight = compute(tf.zeros((*indices.shape, 3)), indices)
        assert float(weight) == pytest.approx(denominator)
        assert float(value) == pytest.approx(np.log(3.) if denominator else 0.)


def collect_gradients(batch_size, accumulate, threshold=None, rates=None, warmup=None):
    schema = DatasetSchema([FieldSpec("x", metadata={"role": "input"}),
                            FieldSpec("target", kind="label", metadata={"role": "target"})])
    samples = [{"x": np.ones(1, np.float32), "target": np.zeros(1, np.float32)} for _ in range(5)]
    provider = DataProvider(train=Dataset(samples, schema=schema)).for_trident_trainer(
        "train", batch_size=batch_size)
    model = Model(input_shape=(1,), output=Dense(1, use_bias=False))
    model.with_optimizer('Adam').with_loss('MSELoss').with_accumulate_grads(accumulate)
    model.model.weight.assign([[1.]])
    if threshold is not None:
        model.with_grad_clipping(threshold)
    gradients = []
    if warmup is not None:
        model.with_learning_rate_scheduler(warmup=warmup)
    original_step = model.optimizer.step

    def record(pairs):
        gradients.extend(float(g.numpy()[0, 0]) for g, _ in pairs)
        if rates is not None:
            rates.append(model.optimizer.lr)
            assert model.training_context['current_lr'] == model.optimizer.lr
            return original_step(pairs)

    model.optimizer.step = record
    TrainingPlan().add_training_item(model).with_data_loader(provider).with_batch_size(batch_size).start_now()
    return gradients


def test_accumulation_flushes_uneven_window_and_clips():
    assert collect_gradients(5, 1) == pytest.approx([2.])
    assert collect_gradients(2, 4) == pytest.approx([2.])
    assert collect_gradients(2, 4, .01) == pytest.approx([.01])


@pytest.mark.parametrize('warmup, expected', [
    (LinearWarmup(2, .2), [.0002, .001, .001]),
    (ConstantWarmup(2, .2), [.0002, .0002, .001]),
    (ExponentialWarmup(3, .01), [.00001, .0001, .001]),
    ("cosine", [.001 * CosineWarmup(360).factor_at(i) for i in range(3)]),
])
def test_tf_warmup_only_counts_actual_updates(warmup, expected):
    rates = []
    collect_gradients(1, 2, rates=rates, warmup=warmup)
    assert rates == pytest.approx(expected)


@pytest.mark.parametrize('configure_optimizer_first', [True, False])
def test_center_loss_parameters_are_registered_and_updated(configure_optimizer_first):
    from trident.optims.tensorflow_losses import CenterLoss

    schema = DatasetSchema([FieldSpec("x", metadata={"role": "input"}),
                            FieldSpec("target", kind="label", metadata={"role": "target"})])
    samples = [dict(x=np.ones(1, np.float32), target=np.int64(1)) for _ in range(2)]
    provider = DataProvider(train=Dataset(samples, schema=schema)).for_trident_trainer("train", batch_size=2)
    item = Model(input_shape=(1,), output=Dense(2, use_bias=False))
    criterion = CenterLoss(num_classes=2, feat_dim=2)
    if configure_optimizer_first:
        item.with_optimizer('Adam')
    item.with_loss(criterion)
    if not configure_optimizer_first:
        item.with_optimizer('Adam')
    before = criterion.centers.numpy().copy()
    assert any(p is criterion.centers for g in item.optimizer.param_groups for p in g['params'])
    TrainingPlan().add_training_item(item).with_data_loader(provider).with_batch_size(2).start_now()
    np.testing.assert_array_equal(criterion.centers.numpy()[0], before[0])
    assert not np.array_equal(criterion.centers.numpy()[1], before[1])


@pytest.mark.parametrize('optimizer', [Adam, AdamW, RAdam, PlainRAdam, SGD, LARS])
def test_optimizer_retains_momentum_with_a_zero_gradient(optimizer):
    parameter = tf.Variable([1.])
    kwargs = {'momentum': .9} if optimizer in (SGD, LARS) else {}
    opt = optimizer([parameter], lr=.1, **kwargs)
    opt.step([(tf.ones_like(parameter), parameter)])
    first = float(parameter[0])
    opt.step([(tf.zeros_like(parameter), parameter)])
    assert float(parameter[0]) < first < 1
    before = parameter.numpy().copy()
    with pytest.raises(tf.errors.InvalidArgumentError, match='Nonfinite'):
        opt.step([(tf.fill((1,), float('nan')), parameter)])
    np.testing.assert_array_equal(parameter.numpy(), before)


def test_lstm_loss_on_last_step_has_gradient_to_first_input():
    sequence = tf.Variable([[[.2], [.4], [.7]]])
    initial = tf.zeros((1, 1, 1))
    kernel = [tf.ones((4, 1))]
    with tf.GradientTape() as tape:
        output, _, _ = standard_lstm(sequence, initial, initial, kernel, kernel,
                                    False, None, True, False, None, False)
        last = tf.reduce_sum(output[:, -1])
    gradient = tape.gradient(last, sequence)
    assert abs(float(gradient[0, 0, 0])) > 1e-5
