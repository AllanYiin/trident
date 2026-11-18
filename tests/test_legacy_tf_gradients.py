import os
import numpy as np
import pytest

if os.environ.get('TRIDENT_BACKEND') != 'tensorflow':
    pytest.skip('run with TRIDENT_BACKEND=tensorflow', allow_module_level=True)
tf = pytest.importorskip('tensorflow')

from trident.data.pipeline import Dataset, DataProvider, DatasetSchema, FieldSpec
from trident.layers.tensorflow_layers import Dense
from trident.layers.tensorflow_rnn import standard_lstm
from trident.optims.tensorflow_trainer import Model
from trident.optims.tensorflow_optimizers import Adam, AdamW, LARS, PlainRAdam, RAdam, SGD
from trident.optims.trainers import TrainingPlan


def collect_gradients(batch_size, accumulate, threshold=None):
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
    model.optimizer.step = lambda pairs: gradients.extend(float(g.numpy()[0, 0]) for g, _ in pairs)
    TrainingPlan().add_training_item(model).with_data_loader(provider).with_batch_size(batch_size).start_now()
    return gradients


def test_accumulation_flushes_uneven_window_and_clips():
    assert collect_gradients(5, 1) == pytest.approx([2.])
    assert collect_gradients(2, 4) == pytest.approx([2.])
    assert collect_gradients(2, 4, .01) == pytest.approx([.01])


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
