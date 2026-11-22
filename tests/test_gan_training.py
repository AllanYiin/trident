"""Exercise GanTrainingPlan through independent backend processes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('backend,package', [('pytorch', 'torch'), ('tensorflow', 'tensorflow'), ('jax', 'jax')])
def test_gan_training_across_backends(backend, package, tmp_path):
    if importlib.util.find_spec(package) is None:
        pytest.skip(f'{package} is optional')
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, TRIDENT_BACKEND=backend, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg',
               TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONPATH=str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    result = subprocess.run([sys.executable, __file__, str(tmp_path)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def check_training(directory):
    from trident.callbacks.callback_base import CallbackBase
    from trident.data.pipeline import DataProvider, Dataset, DatasetSchema, FieldSpec
    from trident.optims import gan
    from trident.optims.trainers import GanTrainingPlan, JaxGanModel

    backend = os.environ['TRIDENT_BACKEND']
    lr = 0.05
    calls = []
    if backend == 'pytorch':
        import torch
        from trident.backend.pytorch_backend import Layer, Parameter, get_device
        from trident.optims.pytorch_trainer import Model

        class Linear(Layer):
            def __init__(self, weight, tag):
                super().__init__()
                self.weight = Parameter(torch.tensor(weight, device=get_device()))
                self.tag = tag

            def forward(self, values):
                calls.append((self.tag, self.training))
                if self.tag == 'quadratic_critic':
                    values = values.square() * 0.25
                return self.weight * values

        def tensor(data):
            return torch.tensor(data, device=get_device(), dtype=torch.float32)

        class ConditionalLinear(Linear):
            def forward(self, values, condition):
                return super().forward(values + condition)

        def player(weight, tag, conditioned=False, default_optimizer=False):
            example = tensor([[1.]])
            model = (Model(inputs=(example, example), output=ConditionalLinear(weight, tag)) if conditioned
                     else Model(inputs=example, output=Linear(weight, tag)))
            if not default_optimizer:
                model.with_optimizer('SGD', lr=lr)
            return model.with_model_save_path(str(directory / f'{tag}.pth'))

        def weight_of(item):
            return item.model.weight.item()

    elif backend == 'tensorflow':
        import tensorflow as tf
        from trident.backend.tensorflow_backend import Layer
        from trident.optims.tensorflow_trainer import Model

        class Linear(Layer):
            def __init__(self, weight, tag):
                super().__init__()
                self.weight = tf.Variable(weight, trainable=True, name='weight')
                self.tag = tag

            def forward(self, values):
                calls.append((self.tag, self.training))
                if self.tag == 'quadratic_critic':
                    values = tf.square(values) * 0.25
                return self.weight * values

        def tensor(data):
            return tf.constant(data, dtype=tf.float32)

        class ConditionalLinear(Linear):
            def forward(self, values, condition):
                return super().forward(values + condition)

        def player(weight, tag, conditioned=False, default_optimizer=False):
            example = tensor([[1.]])
            model = (Model(inputs=(example, example), output=ConditionalLinear(weight, tag)) if conditioned
                     else Model(inputs=example, output=Linear(weight, tag)))
            if not default_optimizer:
                model.with_optimizer('SGD', lr=lr)
            return model.with_model_save_path(str(directory / f'{tag}.pth'))

        def weight_of(item):
            return float(item.model.weight.numpy())

    else:
        import jax
        import jax.numpy as jnp

        class SGD:
            def init(self, params):
                return ()

            def update(self, gradients, state, params):
                return jax.tree_util.tree_map(lambda g: -lr * g, gradients), state

        def tensor(data):
            return jnp.asarray(data, dtype=jnp.float32)

        def player(weight, tag, conditioned=False, default_optimizer=False):
            optimizer = None if default_optimizer else SGD()
            if conditioned:
                return JaxGanModel(lambda p, x, c: p['w'] * (x + c), {'w': tensor(weight)}, optimizer)
            if tag == 'quadratic_critic':
                return JaxGanModel(lambda p, x: p['w'] * x ** 2 / 4, {'w': tensor(weight)}, optimizer)
            return JaxGanModel(lambda p, x: p['w'] * x, {'w': tensor(weight)}, optimizer)

        def weight_of(item):
            return float(item.params['w'])

    z = np.array([0.5, 1.], dtype=np.float32)
    real = np.array([1.5, 2.], dtype=np.float32)
    schema = DatasetSchema([FieldSpec('examples'), FieldSpec('latent', metadata={'role': 'unpaired'})])
    rows = [{'examples': np.array([x]), 'latent': np.array([noise])} for x, noise in zip(real, z)]

    def plan_for(kind, generator_first=False, conditioned=False):
        generator = player(0.6, 'G')
        critic_tag = 'quadratic_critic' if kind == 'gan-r1' else 'D'
        discriminator = player(0.4, critic_tag, conditioned)
        plan = GanTrainingPlan(seed=123).with_gan_type(kind)
        if conditioned:
            plan.with_batch_fields('examples', 'latent', condition='latent')
        if generator_first:
            plan.with_generator(generator).with_discriminator(discriminator)
        else:
            plan.with_discriminator(discriminator).with_generator(generator)
        provider = DataProvider(train=Dataset(rows, schema=schema)).for_trident_trainer('train', batch_size=2)
        plan.with_data_loader(provider).with_batch_size(2).print_progress_frequency = 10000
        return plan, generator, discriminator, provider

    for kind in ('lsgan', 'wgan-gp'):
        for generator_first in (False, True):
            plan, generator, discriminator, provider = plan_for(kind, generator_first)
            calls.clear()
            expected_g, expected_d = 0.6, 0.4
            for step in range(2):
                for role in ('GD' if generator_first else 'DG'):
                    fake = expected_g * z
                    if role == 'G':
                        gradient = (-expected_d * z.mean() if kind == 'wgan-gp' else
                                    np.mean(2 * (expected_d * fake - 1) * expected_d * z))
                        expected_g -= lr * gradient
                    else:
                        gradient = (fake.mean() - real.mean() + 20 * (expected_d - 1) if kind == 'wgan-gp' else
                                    np.mean((expected_d * real - 1) * real + expected_d * fake ** 2))
                        gradient = np.clip(gradient, -10, 10)
                        expected_d -= lr * gradient
                assert plan.only_steps(1, keep_gradient_history=True)
                assert plan.steps == generator.steps == discriminator.steps == step + 1
                assert provider.traindata.unpair.symbol == 'latent'
                np.testing.assert_allclose([weight_of(generator), weight_of(discriminator)],
                                           [expected_g, expected_d], rtol=3e-5, atol=1e-6)
            if backend != 'jax':
                assert calls.count(('G', True)) == 2
                assert calls.count(('G', False)) == 2
                assert calls.count(('D', False)) == 4

    for kind in ('gan', 'dcgan', 'lsgan1', 'rasgan', 'wgan', 'wgan-div', 'began', 'ebgan'):
        plan, generator, discriminator, _ = plan_for(kind)
        assert plan.only_steps(2)
        assert np.isfinite([weight_of(generator), weight_of(discriminator)]).all()
        assert weight_of(generator) != pytest.approx(0.6)
        assert weight_of(discriminator) != pytest.approx(0.4)
        if kind == 'wgan':
            assert abs(weight_of(discriminator)) <= 0.0100001

    def reference_loss(kind, role, weights, gamma):
        gw, dw = weights
        real_score = dw * real.astype(np.float64)
        fake_score = dw * gw * z.astype(np.float64)
        softplus = lambda x: np.logaddexp(0., x).mean()
        if kind == 'hinge':
            return (-fake_score.mean() if role == 0 else
                    np.maximum(0, 1 - real_score).mean() + np.maximum(0, 1 + fake_score).mean())
        if kind == 'rsgan':
            return softplus((1 - 2 * role) * (real_score - fake_score))
        if kind == 'gan-r1':
            real_score = dw * np.square(real.astype(np.float64)) / 4
            fake_score = dw * np.square(gw * z.astype(np.float64)) / 4
            r1 = gamma * np.square(dw * real.astype(np.float64) / 2).mean() / 2
            return softplus(-fake_score) if role == 0 else softplus(-real_score) + softplus(fake_score) + r1
        sign = 2 * role - 1
        real_margin = sign * (real_score - fake_score.mean())
        fake_margin = sign * (real_score.mean() - fake_score)
        if kind == 'ralsgan':
            return np.square(real_margin - 1).mean() + np.square(fake_margin - 1).mean()
        return np.maximum(0, 1 - real_margin).mean() + np.maximum(0, 1 - fake_margin).mean()

    for kind in ('hinge', 'rsgan', 'ralsgan', 'rahinge', 'gan-r1'):
        for generator_first in (False, True):
            plan, generator, discriminator, _ = plan_for(kind, generator_first)
            gamma = 3. if generator_first else 10.
            if kind == 'gan-r1' and generator_first:
                plan.with_gan_type(kind, r1_gamma=gamma)
            calls.clear()
            expected = np.array([.6, .4])
            losses = np.empty(2)
            roles = (0, 1) if generator_first else (1, 0)
            for role in roles:
                delta = np.zeros(2)
                delta[role] = 1e-5
                losses[role] = reference_loss(kind, role, expected, gamma)
                gradient = (reference_loss(kind, role, expected + delta, gamma) -
                            reference_loss(kind, role, expected - delta, gamma)) / (2 * delta[role])
                expected[role] -= lr * np.clip(gradient, -10, 10)
            assert plan.only_steps(1)
            np.testing.assert_allclose([weight_of(generator), weight_of(discriminator)], expected, rtol=3e-5,
                                       atol=1e-6, err_msg=kind)
            actual_losses = [model.training_context['losses']['total_losses'][-1][1]
                             for model in (generator, discriminator)]
            np.testing.assert_allclose(actual_losses, losses, rtol=3e-5, atol=1e-6, err_msg=kind)
            if kind == 'gan-r1' and backend != 'jax':
                assert calls.count(('quadratic_critic', True)) == 2

    with pytest.raises(ValueError, match='finite and nonnegative'):
        GanTrainingPlan().with_gan_type('gan-r1', r1_gamma=float('inf'))
    with pytest.raises(ValueError, match='only supported for gan-r1'):
        GanTrainingPlan().with_gan_type('hinge', r1_gamma=10.)

    class Abort(CallbackBase):
        def on_overall_batch_end(self, context):
            raise RuntimeError('test interruption')

    plan, generator, discriminator, provider = plan_for('gan')
    provider.mode = 'tuple'
    plan.with_callbacks(Abort(is_shared=True))
    with pytest.raises(RuntimeError, match='test interruption'):
        plan.only_steps(1)
    assert provider.mode == 'tuple'
    if backend != 'jax':
        assert generator.output_fn is None and discriminator.output_fn is None

    for kind in ('gan', 'rasgan'):
        plan, generator, discriminator, _ = plan_for(kind)
        plan.with_label_smoothing(one_side=False).with_label_flipping(0.25).with_noised_real_images()
        assert plan.only_steps(5)
        assert np.isfinite(list(generator.training_context['losses']['total_losses'])).all()

    plan, generator, discriminator, provider = plan_for('wgan-gp', conditioned=True)
    assert plan.only_steps(3)
    assert plan.is_condition_gan
    assert weight_of(generator) != pytest.approx(0.6)
    assert weight_of(discriminator) != pytest.approx(0.4)

    plan, generator, discriminator, _ = plan_for('lsgan', generator_first=True)

    def singular_gradient(img_fake):
        return gan.mean((img_fake - gan.stop_gradient(img_fake)) ** 0.5)

    generator.with_loss(singular_gradient)
    with pytest.raises(Exception, match='Nonfinite gradients|Nonfinite GAN loss or gradients'):
        plan.only_steps(1)
    assert weight_of(generator) == pytest.approx(0.6)
    assert weight_of(discriminator) == pytest.approx(0.4)

    generator = player(.6, 'G', default_optimizer=True)
    discriminator = player(.4, 'D', default_optimizer=True)
    plan = GanTrainingPlan(generator, discriminator).with_gan_type('wgan')
    plan.with_data_loader(provider).with_batch_size(2).repeat_epochs(4)
    assert plan.start_now()
    assert generator.steps == discriminator.steps == 4
    assert weight_of(generator) > .6
    assert abs(weight_of(discriminator)) <= 0.0100001

    # The original GAN example uses the legacy provider and RandomNoiseDataset.
    from trident.data.data_provider import DataProvider as LegacyProvider
    from trident.data.dataset import Iterator, NumpyDataset, RandomNoiseDataset

    if backend == 'jax':
        generator = JaxGanModel(lambda p, x: x @ p, tensor(np.full((2, 4), 0.1)))
        discriminator = JaxGanModel(lambda p, x: x @ p, tensor(np.full((4, 1), 0.1)))
    else:
        if backend == 'pytorch':
            from trident.layers.pytorch_layers import Dense
        else:
            from trident.layers.tensorflow_layers import Dense
        generator = Model(input_shape=(2,), output=Dense(4, use_bias=False))
        discriminator = Model(input_shape=(4,), output=Dense(1, use_bias=False))
    samples = NumpyDataset(np.arange(12, dtype=np.float32).reshape(3, 4) / 10, symbol='sample')
    noise = RandomNoiseDataset((2,), symbol='custom_noise')
    legacy_data = Iterator(data=samples, unpair=noise, batch_size=2, is_shuffle=False)
    legacy_provider = LegacyProvider('gan_legacy', traindata=legacy_data, batch_size=2)
    plan = GanTrainingPlan(generator, discriminator, seed=13).with_gan_type('wgan-gp')
    plan.with_data_loader(legacy_provider).with_batch_size(2)
    assert plan.start_now() and plan.steps == 2
    assert noise.symbol == 'custom_noise' and legacy_provider.mode == 'tuple'
    for item in (generator, discriminator):
        assert np.isfinite(item.training_context['losses']['total_losses']).all()

    assert float(gan.binary_loss(tensor([1000., -1000.]), tensor([0., 1.]))) == pytest.approx(1000.)
    assert float(gan.pull_away(tensor([[0., 0.]]))) == 0.


if __name__ == '__main__':
    check_training(Path(sys.argv[1]))
