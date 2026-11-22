"""Differentiable GAN primitives and the functional JAX training item."""

from contextlib import contextmanager, nullcontext
import inspect
import math
from pathlib import Path

import numpy as np

from trident.backend.common import OrderedDict, get_backend
from trident.callbacks.callback_base import CallbackBase
from trident.loggers.history import HistoryBase
from trident.optims.field_binding import FunctionObjective, prepare_objective, resolve_fields


backend = get_backend()
if backend == 'pytorch':
    import torch
    from torch.nn import functional as F
elif backend == 'tensorflow':
    import tensorflow as tf
elif backend == 'jax':
    import jax
    import jax.numpy as jnp


def as_like(value, reference):
    if backend == 'pytorch':
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if backend == 'tensorflow':
        with tf.device(reference.device):
            return tf.convert_to_tensor(value, dtype=reference.dtype)
    return jnp.asarray(value, dtype=reference.dtype)


def float_loss(value):
    """Keep reductions and norm epsilons out of half precision."""
    if backend == 'pytorch':
        return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    if backend == 'tensorflow':
        return tf.cast(value, tf.float32) if value.dtype in (tf.float16, tf.bfloat16) else value
    return value.astype(jnp.float32) if value.dtype in (jnp.float16, jnp.bfloat16) else value


def stop_gradient(value):
    if backend == 'pytorch':
        return value.detach()
    if backend == 'tensorflow':
        return tf.stop_gradient(value)
    return jax.lax.stop_gradient(value)


def mean(value, axis=None):
    value = float_loss(value)
    if backend == 'tensorflow':
        return tf.reduce_mean(value, axis=axis)
    return value.mean(axis=axis)


def finite(value):
    if backend == 'pytorch':
        return bool(torch.isfinite(value).all())
    if backend == 'tensorflow':
        return bool(tf.reduce_all(tf.math.is_finite(value)))
    return bool(jnp.all(jnp.isfinite(value)))


def binary_loss(score, label, from_logits=True):
    score = float_loss(score)
    label = float_loss(label)
    if backend == 'pytorch':
        if from_logits:
            return F.binary_cross_entropy_with_logits(score, label)
        return F.binary_cross_entropy(score.clamp(1e-7, 1 - 1e-7), label)
    if backend == 'tensorflow':
        if from_logits:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=score))
        score = tf.clip_by_value(score, 1e-7, 1 - 1e-7)
        return -tf.reduce_mean(label * tf.math.log(score) + (1 - label) * tf.math.log1p(-score))
    if from_logits:
        return -jnp.mean(label * jax.nn.log_sigmoid(score) + (1 - label) * jax.nn.log_sigmoid(-score))
    score = jnp.clip(score, 1e-7, 1 - 1e-7)
    return -jnp.mean(label * jnp.log(score) + (1 - label) * jnp.log1p(-score))


def hinge_loss(score, positive_label):
    score = float_loss(score)
    return mean(positive_label * relu(1 - score) + (1 - positive_label) * relu(1 + score))


def scores_and_input_gradient_norm_squared(critic, inputs):
    """Evaluate the critic once and retain the input norm's parameter derivative."""
    inputs = stop_gradient(inputs)
    axes = tuple(range(1, len(inputs.shape)))
    if backend == 'pytorch':
        inputs = inputs.requires_grad_(True)
        scores = critic(inputs)
        # A constant critic has zero slope. The zero term also handles unused inputs.
        gradient, = torch.autograd.grad(scores.sum() + inputs.sum() * 0, inputs, create_graph=True)
        squared = float_loss(gradient).square()
        return scores, squared.sum(dim=axes) if axes else squared
    if backend == 'tensorflow':
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            scores = critic(inputs)
        gradient = tape.gradient(scores, inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return scores, tf.reduce_sum(tf.square(float_loss(gradient)), axis=axes)

    def summed_scores(value):
        scores = critic(value)
        return jnp.sum(scores), scores

    (_, scores), gradient = jax.value_and_grad(summed_scores, has_aux=True)(inputs)
    return scores, jnp.sum(jnp.square(float_loss(gradient)), axis=axes)


def gradient_penalty(critic, real, fake, rng, divergence=False):
    if real.shape != fake.shape:
        raise ValueError(f'Gradient penalty requires matching real and generated shapes, got {real.shape} and {fake.shape}')
    if divergence:
        _, real_norm = scores_and_input_gradient_norm_squared(critic, real)
        _, fake_norm = scores_and_input_gradient_norm_squared(critic, fake)
        return mean((real_norm ** 3 + fake_norm ** 3) * 0.5)
    alpha = as_like(rng.random((len(real),) + (1,) * (len(real.shape) - 1), dtype=np.float32), real)
    _, norm_squared = scores_and_input_gradient_norm_squared(critic, alpha * real + (1 - alpha) * fake)
    return mean(((norm_squared + 1e-12) ** 0.5 - 1) ** 2)


def pull_away(features):
    features = float_loss(features)
    n = len(features)
    if n < 2:
        return mean(features * 0)
    if backend == 'tensorflow':
        flat = tf.reshape(features, (n, -1))
        normal = flat * tf.math.rsqrt(tf.reduce_sum(flat ** 2, axis=1, keepdims=True) + 1e-12)
        similarity = tf.matmul(normal, normal, transpose_b=True)
        return (tf.reduce_sum(similarity) - tf.linalg.trace(similarity)) / (n * (n - 1))
    flat = features.reshape(n, -1)
    normal = flat / ((flat ** 2).sum(axis=1, keepdims=True) + 1e-12) ** 0.5
    similarity = normal @ normal.T
    return (similarity.sum() - similarity.trace()) / (n * (n - 1))


def total_variation(images):
    images = float_loss(images)
    if len(images.shape) != 4:
        raise ValueError('GAN total variation requires a rank-4 image batch')
    if backend == 'pytorch':
        images = images.permute(0, 2, 3, 1)
    result = mean(images) * 0
    if images.shape[1] > 1:
        result = result + mean(abs(images[:, 1:] - images[:, :-1]))
    if images.shape[2] > 1:
        result = result + mean(abs(images[:, :, 1:] - images[:, :, :-1]))
    return result


def relu(value):
    if backend == 'pytorch':
        return torch.relu(value)
    if backend == 'tensorflow':
        return tf.nn.relu(value)
    return jax.nn.relu(value)


@contextmanager
def frozen_model(model, differentiate_input):
    """Use inference state for the inactive player, even during the peer's backward pass."""
    if backend == 'tensorflow' and isinstance(model, tf.keras.Model):
        yield lambda *inputs: model(*inputs, training=False)
        return
    modes = [(layer, layer.training) for layer in model.modules()]
    parameters = []
    try:
        model.eval()
        for layer, _ in modes:
            layer.training = False
        if backend == 'pytorch':
            parameters = [p for p in model.parameters() if p.requires_grad]
            for parameter in parameters:
                parameter.requires_grad_(False)
            def forward(*inputs):
                with torch.set_grad_enabled(torch.is_grad_enabled() and differentiate_input):
                    return model(*inputs)

            yield forward
        else:
            yield model
    finally:
        for layer, mode in modes:
            layer.training = mode
        for parameter in parameters:
            parameter.requires_grad_(True)


def autocast(enabled):
    if backend == 'pytorch' and enabled and torch.cuda.is_available():
        return torch.amp.autocast('cuda')
    return nullcontext()


def has_sigmoid_head(item):
    if backend == 'jax':
        return False
    model = item.model
    while True:
        if model.__class__.__name__.lower() == 'sigmoid':
            return True
        activation = getattr(model, 'activation', None)
        if (getattr(activation, '__name__', '').lower() == 'sigmoid' or
                activation.__class__.__name__.lower() == 'sigmoid'):
            return True
        children = list(model.layers) if backend == 'tensorflow' and isinstance(model, tf.keras.Model) else list(model.children())
        if not children:
            return False
        model = children[-1]


def shared_parameters(generator, discriminator):
    def identities(model):
        if backend == 'pytorch':
            return {id(p) for p in model.parameters() if p.requires_grad}
        return {id(p if isinstance(p, tf.Variable) else p.value) for p in model.trainable_variables}

    return generator is discriminator or bool(identities(generator) & identities(discriminator))


def feature_layer(model, layer):
    if layer is not None:
        if not hasattr(layer, 'keep_output'):
            raise ValueError('A GAN feature layer must support keep_output')
        return layer
    layers = list(model.modules()) if hasattr(model, 'modules') else ()
    return next((child for child in reversed(layers)
                 if hasattr(child, 'keep_output') and ('conv' in type(child).__name__.lower() or
                                                       getattr(child, 'name', None) == 'embeddings')), None)


class CriticWeightClip(CallbackBase):
    """Classic WGAN constraint, applied only on an actual critic update."""

    def on_optimization_step_end(self, training_context):
        model = training_context['current_model']
        if backend == 'pytorch':
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.clamp_(-0.01, 0.01)
        else:
            for variable in model.trainable_variables:
                variable.assign(tf.clip_by_value(variable, -0.01, 0.01))


class JaxGanModel:
    """GAN player backed by a pure ``apply_fn(params, inputs[, condition])`` and a parameter PyTree.

    The critic may return ``(scores, features)`` for feature matching and pull-away loss.
    Pass an Optax-compatible transformation to replace the default JAX Adam optimizer.
    """

    def __init__(self, apply_fn, params, optimizer=None, lr=2e-4):
        if backend != 'jax':
            raise RuntimeError("JaxGanModel requires TRIDENT_BACKEND=jax")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be positive and finite')
        self.apply_fn = apply_fn
        try:
            inspect.signature(apply_fn).bind(None, None, None)
        except (TypeError, ValueError):
            self.accepts_condition = False
        else:
            self.accepts_condition = True
        self.params = jax.tree_util.tree_map(jnp.asarray, params)
        leaves = jax.tree_util.tree_leaves(self.params)
        if not leaves or any(not jnp.issubdtype(p.dtype, jnp.floating) for p in leaves):
            raise ValueError('JaxGanModel needs a nonempty PyTree of floating-point parameters')
        self.name = None
        self.lr = lr
        self.optimizer = optimizer
        if optimizer is None:
            from jax.example_libraries.optimizers import adam

            init, self._adam_update, self._adam_params = adam(lambda _: self.lr, b1=0.5, b2=0.999)
            # Adam's moments require float32 even when a model computes in half precision.
            self._optimizer_state = init(jax.tree_util.tree_map(float_loss, self.params))
        else:
            self._optimizer_state = optimizer.init(self.params)
        self.updates = 0
        self.steps = 0
        self.grad_clipping_threshold = 10.
        self.callbacks = []
        self._objectives = {'loss': {}, 'metric': {}}
        self.training_context = dict(callbacks=self.callbacks, steps=0, current_epoch=0,
                                     losses=HistoryBase(name='losses'), metrics=HistoryBase(name='metrics'))
        self.gradients_history = []
        self.weights_history = []

    def __call__(self, *inputs):
        return self.apply_fn(self.params, *inputs)

    def with_callbacks(self, *callbacks):
        self.callbacks.extend(callbacks)
        return self

    def with_model_save_path(self, save_path):
        self.training_context['save_path'] = save_path
        return self

    def save_model(self, save_path):
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save numerical leaves in PyTree order. The caller supplies the architecture.
        with path.open('wb') as stream:
            np.savez(stream, *jax.tree_util.tree_leaves(self.params))

    def do_on_excution_exception(self):
        for callback in self.callbacks:
            callback.on_excution_exception(self.training_context)

    def with_grad_clipping(self, clipping_threshold=10.):
        if not math.isfinite(clipping_threshold) or clipping_threshold <= 0:
            raise ValueError('clipping_threshold must be positive and finite')
        self.grad_clipping_threshold = clipping_threshold
        return self

    def _register(self, kind, fn, name, input_names, options, weight=1., start_epoch=0):
        registry = self._objectives[kind]
        alias, objective, bindings = prepare_objective(fn, name, input_names, options, registry, None, FunctionObjective)
        registry[alias] = objective, bindings, weight, start_epoch
        return self

    def with_loss(self, loss, loss_weight=1., start_epoch=0, name='', input_names=None, **options):
        return self._register('loss', loss, name, input_names, options, loss_weight, start_epoch)

    def with_metric(self, metric, name='', input_names=None, **options):
        return self._register('metric', metric, name, input_names, options)

    def _evaluate(self, kind, fields, epoch):
        values = {}
        for name, (fn, bindings, weight, start_epoch) in self._objectives[kind].items():
            if epoch >= start_epoch:
                mapping = resolve_fields(fn.signature, fields, {}, owner=f'{kind} {name}', local=bindings,
                                         fixed=getattr(fn, 'options', ()))
                args = {arg: fields[field] for arg, field in mapping.items()}
                value = fn.call_fields(args) if isinstance(fn, FunctionObjective) else fn(**args)
                values[name] = weight * mean(jnp.asarray(value))
        return values

    def _step(self, make_fields, epoch, waiting, collect, log_gradients, log_weights, weight_clip):
        context = self.training_context

        def objective(params):
            fields = make_fields(lambda *inputs: self.apply_fn(params, *inputs))
            losses = self._evaluate('loss', fields, epoch)
            adversarial = fields['gan_adversarial_loss']
            return sum(losses.values(), adversarial), (fields, dict(losses, gan_loss=adversarial))

        if waiting:
            total, (fields, losses) = objective(self.params)
        else:
            (total, (fields, losses)), gradients = jax.value_and_grad(objective, has_aux=True)(self.params)
            if not bool(jnp.isfinite(total)) or not self._finite(gradients):
                raise FloatingPointError('Nonfinite GAN loss or gradients; optimizer was not updated')
            leaves = jax.tree_util.tree_leaves(gradients)
            peak = jnp.max(jnp.stack([jnp.max(jnp.abs(float_loss(g)), initial=0) for g in leaves]))
            scale = jnp.maximum(peak, 1.)
            norm = jnp.sqrt(sum(jnp.sum((float_loss(g) / scale) ** 2) for g in leaves))
            factor = jnp.minimum(1., (self.grad_clipping_threshold / scale) / jnp.maximum(norm, 1e-12))
            gradients = jax.tree_util.tree_map(lambda g: (float_loss(g) * factor).astype(g.dtype), gradients)
            if self.optimizer is None:
                state = self._adam_update(self.updates, jax.tree_util.tree_map(float_loss, gradients), self._optimizer_state)
                params = jax.tree_util.tree_map(lambda p, old: p.astype(old.dtype), self._adam_params(state), self.params)
            else:
                changes, state = self.optimizer.update(gradients, self._optimizer_state, self.params)
                params = jax.tree_util.tree_map(lambda p, change: p + change, self.params, changes)
            if not self._finite(params) or not self._finite(state):
                raise FloatingPointError('GAN optimizer produced nonfinite parameters or state')
            if weight_clip is not None:
                params = jax.tree_util.tree_map(lambda p: jnp.clip(p, -weight_clip, weight_clip), params)
                if self.optimizer is None:
                    # Keep clipped weights as the next Adam iterate while retaining both moments.
                    state = state._replace(packed_state=[(float_loss(p), *entry[1:]) for p, entry in
                                                         zip(jax.tree_util.tree_leaves(params), state.packed_state)])
            self.params, self._optimizer_state = params, state
            self.updates += 1
            if log_gradients and collect:
                self.gradients_history.append(gradients)
        if not bool(jnp.isfinite(total)):
            raise FloatingPointError('Nonfinite GAN loss')
        context.update(train_data=fields, current_loss=total)
        if collect:
            for name, value in dict(losses, total_losses=total).items():
                context['losses'].collect(name, self.steps, float(value))
            metrics = self._evaluate('metric', fields, epoch)
            metrics['d_fake'] = mean(fields['d_fake'])
            if context['gan_role'] == 'discriminator':
                metrics['d_real'] = mean(fields['d_real'])
            for name, value in metrics.items():
                context['metrics'].collect(name, self.steps, float(value))
            if log_weights:
                self.weights_history.append(self.params)

    @staticmethod
    def _finite(tree):
        checks = [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]
        return not checks or bool(jnp.stack(checks).all())
