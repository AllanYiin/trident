"""Differentiable loss kernels for Trident's active tensor backend.

Class axes are moved to the last dimension for the calculation. All other
axes retain their order, including for sparse segmentation targets.
"""
import math
import numpy as np

from trident.backend.common import get_backend
from trident.optims.losses import Loss

backend = get_backend()
if backend == 'tensorflow':
    import tensorflow as tf
    import tensorflow.experimental.numpy as xp
    stop_gradient = tf.stop_gradient
    log_softmax = tf.nn.log_softmax
    softplus = tf.nn.softplus
    shape = tf.shape
elif backend == 'jax':
    import jax
    import jax.numpy as xp
    stop_gradient = jax.lax.stop_gradient
    log_softmax = jax.nn.log_softmax
    softplus = jax.nn.softplus
    shape = xp.shape
else:
    import torch as xp
    import torch.nn.functional as F
    stop_gradient = xp.Tensor.detach
    log_softmax = F.log_softmax
    softplus = F.softplus

    def shape(x):
        return x.shape


def as_like(value, reference):
    if backend == 'pytorch':
        return xp.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return xp.asarray(value, dtype=reference.dtype)


def moveaxis(x, source, destination):
    return xp.moveaxis(x, source, destination)


def magnitude(x):
    # JAX chooses +1 at zero for abs. PyTorch chooses the zero subgradient.
    return xp.where(x == 0, xp.zeros_like(x), xp.abs(x))


def flat_batch(x):
    return xp.reshape(x, (shape(x)[0], -1)) if len(x.shape) else x


def hard_examples(loss, fraction, weight=None):
    flat = xp.reshape(loss, (-1,))
    ranking = flat if weight is None else xp.where(xp.reshape(weight, (-1,)) != 0, flat, -math.inf)
    if backend == 'tensorflow':
        count = tf.minimum(tf.size(flat), tf.maximum(1, tf.cast(tf.cast(tf.size(flat), tf.float64) * fraction, tf.int32)))
        indices = tf.math.top_k(ranking, k=count).indices
        return tf.gather(flat, indices), None if weight is None else tf.gather(tf.reshape(weight, (-1,)), indices)
    count = max(1, int(flat.shape[0] * fraction))
    indices = xp.argsort(-ranking)[:count]
    return flat[indices], None if weight is None else xp.reshape(weight, (-1,))[indices]


class PairwiseLoss(Loss):
    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=.7, input_names=None,
                 output_names=None, name=None, axis=None):
        super().__init__(reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, axis=axis, name=name or type(self).__name__)

    def preprocess(self, output, target, **kwargs):
        if self.flatten_pair:
            target = dense_pair_target(output, target, self.axis)
        return output, target

    flatten_pair = False

    def forward(self, output, target, **kwargs):
        if not self.built:
            self.build(output, target)
        output, target = self.preprocess(output, target, **kwargs)
        loss = self.calculate_loss(output, stop_gradient(as_like(target, output)))
        if self.enable_ohem and len(loss.shape):
            loss, _ = hard_examples(loss, self.ohem_thresh)
        return self._get_reduction(self._handel_abnormal(loss))


def dense_pair_target(output, target, axis):
    """Preserve Trident's scalar and sparse-class regression target conventions."""
    if np.isscalar(target) or len(target.shape) == 0:
        return xp.ones_like(output) * as_like(target, output)
    if len(target.shape) + 1 == len(output.shape) and 'int' in str(target.dtype):
        axis = (-1 if backend != 'pytorch' else 1) if axis is None else axis
        classes = shape(output)[axis]
        if backend == 'tensorflow':
            return tf.one_hot(target, classes, axis=axis % len(output.shape), dtype=output.dtype)
        if backend == 'jax':
            return jax.nn.one_hot(target, classes, axis=axis, dtype=output.dtype)
        return moveaxis(F.one_hot(target.long(), classes).to(output), -1, axis)
    return target


class L1Loss(PairwiseLoss):
    flatten_pair = True

    def calculate_loss(self, output, target):
        return magnitude(flat_batch(output) - flat_batch(target))


class L2Loss(PairwiseLoss):
    flatten_pair = True

    def calculate_loss(self, output, target):
        return .5 * (flat_batch(output) - flat_batch(target)) ** 2


class MSELoss(PairwiseLoss):
    flatten_pair = True

    def calculate_loss(self, output, target):
        return (flat_batch(output) - flat_batch(target)) ** 2


class SmoothL1Loss(PairwiseLoss):
    flatten_pair = True

    def calculate_loss(self, output, target):
        error = magnitude(flat_batch(output) - flat_batch(target))
        return xp.where(error < 1, .5 * error ** 2, error - .5)


class HuberLoss(PairwiseLoss):
    """PyTorch Huber loss with a configurable positive transition point."""
    def __init__(self, reduction='mean', delta=1., **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        if not delta > 0:
            raise ValueError("delta must be positive")
        self.delta = delta

    def calculate_loss(self, output, target):
        error = magnitude(output - target)
        return xp.where(error < self.delta, .5 * error ** 2, self.delta * (error - .5 * self.delta))


class BCEWithLogitsLoss(PairwiseLoss):
    """Elementwise binary cross entropy. weight and pos_weight follow broadcasting."""
    def __init__(self, weight=None, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.pos_weight = pos_weight

    def calculate_loss(self, output, target):
        if backend == 'tensorflow':
            tf.debugging.assert_equal(tf.shape(output), tf.shape(target))
        elif output.shape != target.shape:
            raise ValueError("BCEWithLogitsLoss requires matching output and target shapes")
        positive = target if self.pos_weight is None else target * as_like(self.pos_weight, output)
        loss = positive * softplus(-output) + (1 - target) * softplus(output)
        return loss if self.weight is None else loss * as_like(self.weight, output)


class WingLoss(PairwiseLoss):
    def __init__(self, omega=10., epsilon=2., **kwargs):
        super().__init__(**kwargs)
        self.omega, self.epsilon = omega, epsilon

    def calculate_loss(self, output, target):
        error = magnitude(output - target)
        if len(output.shape) == 2:
            error = xp.reshape(error, (shape(output)[0], -1, 2))
        c = self.omega * (1 - math.log1p(self.omega / self.epsilon))
        return xp.where(error < self.omega, self.omega * xp.log1p(error / self.epsilon), error - c)


class AdaptiveWingLoss(PairwiseLoss):
    def __init__(self, omega=14., theta=.5, epsilon=1., alpha=2.1, **kwargs):
        super().__init__(**kwargs)
        self.omega, self.theta, self.epsilon, self.alpha = omega, theta, epsilon, alpha

    def calculate_loss(self, output, target):
        error = magnitude(output - target)
        power = self.alpha - target
        t = self.theta / self.epsilon
        slope = self.omega * power * t ** (power - 1) / (self.epsilon * (1 + t ** power))
        return xp.where(error < self.theta, self.omega * xp.log1p((error / self.epsilon) ** power),
                        slope * (error - self.theta) + self.omega * xp.log1p(t ** power))


class ExponentialLoss(PairwiseLoss):
    def calculate_loss(self, output, target):
        return -xp.expm1(-((output - target) ** 2) / math.pi)


class ItakuraSaitoLoss(PairwiseLoss):
    def calculate_loss(self, output, target):
        low_x, low_y = xp.min(output), xp.min(target)
        shift = as_like((low_x >= -1) & (low_x < 0) & (low_y >= -1) & (low_y < 0), output)
        x, y = output + shift + 1e-8, target + shift
        return y / x - xp.log((y + 1e-8) / x) - 1


class CosineSimilarityLoss(PairwiseLoss):
    def calculate_loss(self, output, target):
        axis = self.axis if self.axis is not None else 1
        def normalize(x):
            squared = xp.sum(x * x, axis=axis, keepdims=True)
            return x / xp.sqrt(xp.maximum(squared, as_like(1e-16, x)))
        return 1 - xp.sum(normalize(output) * normalize(target), axis=axis)


class KLDivergenceLoss(PairwiseLoss):
    """KL(target || prediction), with predictions in log space."""
    def calculate_loss(self, output, target):
        safe = xp.where(target == 0, xp.ones_like(target), target)
        return target * (xp.log(safe) - output)


class EdgeLoss(PairwiseLoss):
    def calculate_loss(self, output, target):
        if len(output.shape) == 3:
            output, target = output[..., None], target[..., None]
        elif self.axis is not None:
            output, target = moveaxis(output, self.axis, -1), moveaxis(target, self.axis, -1)
        def edges(x):
            corner = x[:, :-1, :-1]
            return magnitude(corner - x[:, 1:, :-1]), magnitude(corner - x[:, :-1, 1:])
        dx, dy = edges(output)
        tx, ty = edges(target)
        return xp.mean((dx - tx) ** 2) + xp.mean((dy - ty) ** 2)


class TripletLoss(Loss):
    def __init__(self, margin=1., p=2., eps=1e-6, swap=False, reduction='mean'):
        super().__init__(reduction=reduction, name='TripletLoss')
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.margin, self.p, self.eps, self.swap = margin, p, eps, swap

    def __call__(self, anchor, positive, negative):
        return self.forward(anchor, positive, negative)

    def forward(self, anchor, positive, negative):
        def distance(x, y):
            if backend == 'tensorflow':
                return tf.norm(x - y + self.eps, ord=self.p, axis=-1)
            return xp.linalg.norm(x - y + self.eps, ord=self.p, axis=-1)
        negative_distance = distance(anchor, negative)
        if self.swap:
            negative_distance = xp.minimum(negative_distance, distance(positive, negative))
        loss = self.margin + distance(anchor, positive) - negative_distance
        return self._get_reduction(xp.maximum(loss, xp.zeros_like(loss)))


TripletMarginLoss = TripletLoss

def _tiny(x):
    dtype = x.dtype.as_numpy_dtype if backend == 'tensorflow' else x.dtype
    return float(xp.finfo(dtype).tiny) if backend != 'tensorflow' else float(np.finfo(dtype).tiny)


def _log_probs(x, from_logits):
    if from_logits is True:
        return log_softmax(x, -1)
    log_p = xp.log(xp.maximum(x, as_like(_tiny(x), x)))
    if from_logits is False:
        return log_p
    is_probability = (xp.min(x) >= 0) & (xp.max(x) <= 1) & (xp.mean(xp.abs(xp.sum(x, -1) - 1)) < .2)
    # Keep exp finite on the unselected logits branch.
    is_log_probability = (xp.max(x) <= 0) & (xp.mean(xp.abs(xp.sum(xp.exp(xp.minimum(x, xp.zeros_like(x))), -1) - 1)) < .2)
    return xp.where(is_log_probability, x, xp.where(is_probability, log_p, log_softmax(x, -1)))


def _smoothing(value):
    amount = .1 if value is True else float(value or 0.)
    if not 0 <= amount <= 1:
        raise ValueError("label_smooth must be in [0, 1]")
    return amount


def classification_preprocess(obj, output, target, **kwargs):
    """Prepare custom calculate_loss implementations without changing from_logits."""
    sparse = len(target.shape) + 1 == len(output.shape)
    smooth = _smoothing(obj.label_smooth)
    if sparse and (obj.need_target_onehot or smooth):
        target = dense_pair_target(output, target, obj.axis)
        sparse = False
    obj.is_target_onehot = not sparse
    if smooth:
        classes = as_like(shape(output)[obj.axis], output)
        target = as_like(target, output) * (1 - smooth) + smooth / classes
    return output, stop_gradient(target)


def _class_inputs(obj, output, target):
    axis = obj.axis % len(output.shape)
    x = moveaxis(output, axis, -1)
    if 'float16' in str(x.dtype):
        x = tf.cast(x, tf.float32) if backend == 'tensorflow' else (x.float() if backend == 'pytorch' else x.astype(xp.float32))
    classes = shape(x)[-1]
    if obj.auto_balance and not obj.built:
        obj.build(output, target)
    weights = as_like(obj.sample_weight, x) if obj.sample_weight is not None else as_like(xp.ones((classes,)), x)
    if backend == 'tensorflow':
        tf.debugging.assert_equal(tf.shape(weights), tf.reshape(classes, (1,)), message="class weight shape mismatch")
    elif weights.shape != (classes,):
        raise ValueError("class weight shape mismatch")
    if backend == 'tensorflow':
        indices = tf.range(classes)
    elif backend == 'pytorch':
        indices = xp.arange(classes, device=x.device)
    else:
        indices = xp.arange(classes)
    for ignored in obj.ignore_index:
        if ignored is not None:
            weights = xp.where(indices == ignored, xp.zeros_like(weights), weights)
    sparse = len(target.shape) != len(output.shape)
    if sparse:
        if len(target.shape) + 1 != len(output.shape):
            raise ValueError("expected class indices or a distribution with the same rank as output")
        labels = target
        valid = xp.ones_like(labels) != 0
        for ignored in obj.ignore_index:
            if ignored is not None:
                valid = valid & (labels != ignored)
        if backend == 'tensorflow':
            tf.debugging.assert_equal(tf.shape(labels), tf.shape(x)[:-1])
            tf.debugging.assert_equal(tf.reduce_all(~valid | ((labels >= 0) & (labels < tf.cast(classes, labels.dtype)))), True)
        else:
            if labels.shape != x.shape[:-1]:
                raise ValueError("sparse target shape must match non-class dimensions")
            invalid = valid & ((labels < 0) | (labels >= classes))
            if backend == 'pytorch' or not isinstance(invalid, jax.core.Tracer):
                if bool(xp.any(invalid)):
                    raise ValueError("class index out of range")
        if backend == 'tensorflow':
            indices = tf.cast(indices, labels.dtype)
        distribution = as_like(labels[..., None] == indices, x)
        if backend == 'jax':
            distribution = xp.where(invalid[..., None], xp.nan, distribution)
    else:
        distribution = moveaxis(as_like(target, x), axis, -1)
        if backend == 'tensorflow':
            tf.debugging.assert_equal(tf.shape(distribution), tf.shape(x))
        elif distribution.shape != x.shape:
            raise ValueError("soft target shape must match output")
        valid = xp.sum(distribution * as_like(weights != 0, x), -1) != 0
    obj.num_classes = x.shape[-1]
    obj._built = True
    return x, stop_gradient(distribution), stop_gradient(weights), valid, sparse


def _reduce(obj, loss, weights=None):
    loss = obj._handel_abnormal(loss)
    if backend != 'jax':
        obj.gradient_denominator = None
    if obj.enable_ohem:
        loss, weights = hard_examples(loss, obj.ohem_thresh, weights)
    if weights is not None and obj.reduction in ('mean', 'batch_mean'):
        denominator = xp.sum(weights)
        if backend != 'jax':
            obj.gradient_denominator = stop_gradient(denominator)
        return xp.sum(loss) / xp.where(denominator > 0, denominator, xp.ones_like(denominator))
    return obj._get_reduction(loss)


class ClassificationLoss(Loss):
    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False, ohem_thresh=.7,
                 input_names=None, output_names=None, name=None):
        super().__init__(axis=axis, reduction=reduction, sample_weight=sample_weight, enable_ohem=enable_ohem,
                         ohem_thresh=ohem_thresh, input_names=input_names, output_names=output_names,
                         name=name or type(self).__name__)
        self.from_logits = from_logits
        self.auto_balance = auto_balance
        self.ignore_index = ignore_index if isinstance(ignore_index, (tuple, list)) else [ignore_index]
        self.cutoff, self.label_smooth = cutoff, label_smooth

    def build(self, output, target, **kwargs):
        if self.auto_balance:
            from trident.optims.loss_weights import dataset_class_weights
            self.sample_weight = dataset_class_weights(self, target, output.shape[self.axis])
        super().build(output, target)


    def calculate_loss(self, output, target, **kwargs):
        if type(self).loss_terms is ClassificationLoss.loss_terms:
            raise NotImplementedError
        return self.loss_terms(output, target)[0]

    def loss_terms(self, output, target):
        return self.calculate_loss(output, target), None

    def forward(self, output, target, **kwargs):
        loss, weight = self.loss_terms(output, target)
        return _reduce(self, loss, weight)


def categorical_loss(obj, output, target, *, log_input=False):
    x, distribution, weights, valid, sparse = _class_inputs(obj, output, target)
    logp = x if log_input else _log_probs(x, obj.from_logits)
    denominator = xp.sum(distribution * weights, -1) if sparse else as_like(valid, x)
    smoothing = _smoothing(obj.label_smooth)
    distribution = distribution * (1 - smoothing) + smoothing / as_like(shape(x)[-1], x)
    weighted = distribution * weights
    raw = -xp.sum(xp.where(weighted != 0, logp, xp.zeros_like(logp)) * weighted, -1)
    return xp.where(valid, raw, xp.zeros_like(raw)), denominator


class CrossEntropyLoss(ClassificationLoss):
    def loss_terms(self, output, target):
        return categorical_loss(self, output, target)


class NLLLoss(ClassificationLoss):
    def loss_terms(self, output, target):
        return categorical_loss(self, output, target, log_input=True)


class FocalLoss(ClassificationLoss):
    def __init__(self, alpha=.25, gamma=2., normalized=False, threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.gamma, self.normalized, self.threshold = alpha, gamma, normalized, threshold

    def loss_terms(self, output, target):
        x, dist, weights, valid, sparse = _class_inputs(self, output, target)
        if not sparse:
            indices = (tf.range(shape(x)[-1]) if backend == 'tensorflow' else
                       xp.arange(x.shape[-1], device=x.device) if backend == 'pytorch' else xp.arange(x.shape[-1]))
            chosen = xp.argmax(dist, axis=-1)
            if backend == 'tensorflow':
                indices = tf.cast(indices, chosen.dtype)
            dist = as_like(chosen[..., None] == indices, x)
        logp = _log_probs(x, self.from_logits)
        smooth = _smoothing(self.label_smooth)
        distribution = dist * (1 - smooth) + smooth / as_like(shape(x)[-1], x)
        ce = -xp.sum(xp.where(distribution != 0, logp, xp.zeros_like(logp)) * distribution, -1)
        pt = xp.exp(-ce)
        modulation = (1 - pt) ** self.gamma
        if self.threshold is not None:
            modulation = xp.where(pt < self.threshold, xp.ones_like(pt), ((1 - pt) / self.threshold) ** self.gamma)
        alpha = self.alpha * dist[..., 0] + (1 - self.alpha) * (1 - dist[..., 0])
        weight = xp.sum(dist * weights, -1)
        valid = valid & (weight != 0)
        raw = xp.where(valid, alpha * modulation * ce * weight, xp.zeros_like(ce))
        return raw, xp.where(valid, modulation if self.normalized else xp.ones_like(pt), xp.zeros_like(pt))


def _binary_probs(x, from_logits):
    if from_logits is True:
        return xp.exp(-softplus(-x))
    if from_logits is False:
        return x
    return xp.where((xp.min(x) >= 0) & (xp.max(x) <= 1), x, xp.exp(-softplus(-x)))


def _probability_bce_value(p, target):
    # torch clamps each logarithm to -100, independently of its backward epsilon.
    lp = xp.maximum(xp.log(p), as_like(-100., p))
    lq = xp.maximum(xp.log1p(-p), as_like(-100., p))
    return -(target * lp + (1 - target) * lq)


if backend == 'tensorflow':
    @tf.custom_gradient
    def probability_bce(p, target):
        def grad(upstream):
            divisor = tf.maximum(p * (1 - p), tf.cast(1e-12, p.dtype))
            return upstream * (p - target) / divisor, upstream * (tf.math.log1p(-p) - tf.math.log(p))
        return _probability_bce_value(p, target), grad
elif backend == 'jax':
    probability_bce = jax.custom_jvp(_probability_bce_value)

    @probability_bce.defjvp
    def _bce_jvp(inputs, tangents):
        p, target = inputs
        dp, dt = tangents
        divisor = xp.maximum(p * (1 - p), 1e-12)
        dy = dp * (p - target) / divisor
        # A constant target must not introduce 0 * inf at an endpoint.
        log_odds = xp.maximum(xp.log1p(-p), -100.) - xp.maximum(xp.log(p), -100.)
        return probability_bce(p, target), dy + dt * log_odds
else:
    def probability_bce(p, target):
        return F.binary_cross_entropy(p, target, reduction='none')


def binary_cross_entropy(output, target, from_logits=False):
    target = xp.broadcast_to(as_like(target, output), shape(output))
    if from_logits:
        return target * softplus(-output) + (1 - target) * softplus(output)
    if 'float16' in str(output.dtype):
        output = tf.cast(output, tf.float32) if backend == 'tensorflow' else (output.float() if backend == 'pytorch' else output.astype(xp.float32))
        target = as_like(target, output)
    if backend == 'tensorflow':
        tf.debugging.assert_greater_equal(output, tf.zeros_like(output))
        tf.debugging.assert_less_equal(output, tf.ones_like(output))
    return probability_bce(output, target)


class BCELoss(ClassificationLoss):
    def loss_terms(self, output, target):
        if len(output.shape) == 1:
            output, target = output[:, None], target[:, None]
        x, dist, weights, valid, sparse = _class_inputs(self, output, target)
        smooth = _smoothing(self.label_smooth)
        dist = dist * (1 - smooth) + .5 * smooth
        if self.from_logits is None:
            is_prob = (xp.min(x) >= 0) & (xp.max(x) <= 1)
            is_log_prob = (xp.max(x) <= 0) & (xp.mean(xp.abs(xp.sum(xp.exp(xp.minimum(x, xp.zeros_like(x))), -1) - 1)) < .2)
            prob = xp.where(is_log_prob, xp.exp(xp.minimum(x, xp.zeros_like(x))), xp.clip(x, 0., 1.))
            raw = xp.where(is_prob | is_log_prob, probability_bce(prob, dist), binary_cross_entropy(x, dist, True))
        else:
            raw = binary_cross_entropy(x, dist, self.from_logits)
        loss = xp.sum(raw * weights, -1)
        if sparse:
            loss = xp.where(valid, loss, xp.zeros_like(loss))
        return loss, None


def overlap_loss(obj, output, target, kind):
    x, dist, weights, valid, sparse = _class_inputs(obj, output, target)
    smooth = _smoothing(obj.label_smooth)
    dist = dist * (1 - smooth) + smooth / as_like(shape(x)[-1], x)
    if kind == 'soft_iou':
        prob = xp.exp(log_softmax(x, -1))
    elif kind in ('iou', 'f1'):
        prob = xp.exp(_log_probs(x, obj.from_logits))
    else:
        prob = _binary_probs(x, obj.from_logits)
    if sparse:
        prob = xp.where(valid[..., None], prob, xp.zeros_like(prob))
        dist = xp.where(valid[..., None], dist, xp.zeros_like(dist))
    if kind == 'f1':
        tp = xp.sum(dist * prob * weights, -1)
        divisor = obj.beta ** 2 * xp.sum(dist, -1) + xp.sum(prob, -1)
        return 1 - (1 + obj.beta ** 2) * tp / xp.where(divisor != 0, divisor, xp.ones_like(divisor)), None
    batch, classes = shape(x)[0], shape(x)[-1]
    prob = xp.reshape(prob, (batch, -1, classes))
    dist = xp.reshape(dist, (batch, -1, classes))
    intersection = xp.sum(prob * dist, 1)
    total = xp.sum(prob + dist, 1)
    if kind == 'soft_iou':
        return -xp.mean(intersection / (total - intersection + 1e-16)), None
    inter = xp.sum(intersection * weights, -1)
    mass = xp.sum(total * weights, -1)
    if kind == 'iou':
        union = mass - inter
        return xp.where(union > 0, 1 - inter / xp.where(union > 0, union, xp.ones_like(union)), xp.zeros_like(union)), None
    return 1 - (2 * inter + obj.smooth) / (mass + obj.smooth), None


class F1ScoreLoss(ClassificationLoss):
    def __init__(self, beta=1., **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def loss_terms(self, output, target):
        return overlap_loss(self, output, target, 'f1')


class DiceLoss(ClassificationLoss):
    def __init__(self, smooth=1., **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth

    def loss_terms(self, output, target):
        return overlap_loss(self, output, target, 'dice')


class GeneralizedDiceLoss(DiceLoss):
    """Trident's class-weighted Dice."""


class IoULoss(ClassificationLoss):
    def __init__(self, *args, ignore_index=0, **kwargs):
        super().__init__(*args, ignore_index=ignore_index, **kwargs)

    def loss_terms(self, output, target):
        return overlap_loss(self, output, target, 'iou')


class SoftIoULoss(IoULoss):
    def loss_terms(self, output, target):
        return overlap_loss(self, output, target, 'soft_iou')


class ActiveContourLoss(ClassificationLoss):
    def __init__(self, lambdaP=1., mu=1., **kwargs):
        super().__init__(**kwargs)
        self.lambdaP, self.mu = lambdaP, mu

    def loss_terms(self, output, target):
        x, dist, weights, valid, sparse = _class_inputs(self, output, target)
        if len(x.shape) != 4:
            raise ValueError("ActiveContourLoss requires a 2-D image batch")
        prob = _binary_probs(x, self.from_logits)
        dx = (prob[:, 1:] - prob[:, :-1])[:, 1:, :-2]
        dy = (prob[:, :, 1:] - prob[:, :, :-1])[:, :-2, 1:]
        length = xp.mean(xp.sqrt(dx ** 2 + dy ** 2 + 1e-8))
        inside = xp.abs(xp.mean(prob[..., 0] * (dist[..., 0] - 1) ** 2))
        outside = xp.abs(xp.mean((1 - prob[..., 0]) * dist[..., 0] ** 2))
        return length + self.lambdaP * (self.mu * inside + outside), None


def _gather(x, indices, axis):
    if backend == 'tensorflow':
        return tf.transpose(tf.gather(tf.transpose(x), tf.transpose(indices), batch_dims=1))
    return xp.take_along_axis(x, indices, axis=axis) if backend == 'jax' else xp.gather(x, axis, indices)


class LovaszSoftmax(ClassificationLoss):
    """Multiclass Lovasz extension of the Jaccard loss, averaged over classes."""
    def __init__(self, *args, ignore_index=0, **kwargs):
        super().__init__(*args, ignore_index=ignore_index, **kwargs)

    def loss_terms(self, output, target):
        x, dist, weights, valid, sparse = _class_inputs(self, output, target)
        classes = shape(x)[-1]
        prob = xp.exp(_log_probs(x, self.from_logits))
        truth = xp.reshape(dist, (-1, classes))
        active = xp.reshape(valid, (-1, 1))
        error = xp.where(active, magnitude(xp.reshape(prob, (-1, classes)) - truth), 0.)
        order = xp.argsort(-error, axis=0)
        foreground = _gather(truth * as_like(active, truth), order, 0)
        sorted_error = _gather(error, order, 0)
        sorted_active = _gather(as_like(active, truth) * xp.ones_like(truth), order, 0)
        total = xp.sum(foreground, 0)
        intersection = total - xp.cumsum(foreground, 0)
        union = total + xp.cumsum(sorted_active - foreground, 0)
        jaccard = 1 - intersection / xp.where(union > 0, union, xp.ones_like(union))
        delta = xp.concatenate((jaccard[:1], jaccard[1:] - jaccard[:-1]), axis=0)
        return xp.sum(sorted_error * delta, 0) * weights, weights


class StyleLoss(PairwiseLoss):
    def calculate_loss(self, output, target):
        def gram(x):
            b, h, w, c = (shape(x)[i] for i in range(4))
            features = xp.reshape(moveaxis(x, -1, 1), (b * c, h * w))
            return features @ xp.transpose(features)
        size = shape(output)
        return xp.mean((gram(output) - gram(target)) ** 2) / as_like(size[1] * size[2] * size[3], output)


class MS_SSIMLoss(PairwiseLoss):
    """Separable Gaussian MS-SSIM for channels-last images and volumes."""
    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False,
                 weights=None, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        if window_size < 1 or window_size % 2 != 1 or window_sigma <= 0:
            raise ValueError("expected a positive odd window size and positive sigma")
        coords = np.arange(window_size) - window_size // 2
        kernel = np.exp(-coords ** 2 / (2 * window_sigma ** 2))
        self.kernel = kernel / kernel.sum()
        weights = np.asarray([.0448, .2856, .3001, .2363, .1333] if weights is None else weights)
        self.weights = weights / weights.sum()
        self.data_range = data_range

    def _blur(self, x):
        for axis in range(1, len(x.shape) - 1):
            def convolve(value):
                length = shape(value)[axis] - len(self.kernel) + 1
                slices = [slice(None)] * len(value.shape)
                total = 0.
                for i, weight in enumerate(self.kernel):
                    slices[axis] = slice(i, i + length)
                    total = total + float(weight) * value[tuple(slices)]
                return total
            if x.shape[axis] is None:
                x = tf.cond(tf.shape(x)[axis] >= len(self.kernel), lambda: convolve(x), lambda: x)
            elif x.shape[axis] >= len(self.kernel):
                x = convolve(x)
        return x

    def _pool(self, x):
        spatial = len(x.shape) - 2
        # Match torch avg_pool's symmetric zero-padding on odd spatial dimensions.
        pads = [(0, 0)] + [(shape(x)[i] % 2,) * 2 for i in range(1, spatial + 1)] + [(0, 0)]
        if backend == 'tensorflow':
            return tf.nn.pool(tf.pad(x, pads), [2] * spatial, 'AVG', strides=[2] * spatial, padding='VALID')
        window = (1,) + (2,) * spatial + (1,)
        return jax.lax.reduce_window(x, 0., jax.lax.add, window, window, pads) / (2 ** spatial)

    def calculate_loss(self, output, target):
        x, y = output, target
        axes = tuple(range(1, len(x.shape) - 1))
        data_range = 1. if self.data_range == 2 else self.data_range
        c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
        value = 1.
        for i, weight in enumerate(self.weights):
            mx, my = self._blur(x), self._blur(y)
            vx, vy = self._blur(x * x) - mx ** 2, self._blur(y * y) - my ** 2
            cs = (2 * (self._blur(x * y) - mx * my) + c2) / (vx + vy + c2)
            if i + 1 == len(self.weights):
                cs = cs * (2 * mx * my + c1) / (mx * mx + my * my + c1)
            else:
                x, y = self._pool(x), self._pool(y)
            score = xp.mean(cs, axis=axes)
            value = value * xp.where(score > 0, score, xp.zeros_like(score)) ** float(weight)
        return 1 - value


class CenterLoss(Loss):
    """Distance to learnable class centers. with_centers accepts explicit JAX parameters."""
    def __init__(self, num_classes=751, feat_dim=2048, reduction='mean', reduced=False, *, key=None):
        super().__init__(reduction=reduction, name='CenterLoss')
        self.num_classes, self.feat_dim, self.reduced = num_classes, feat_dim, reduced
        bound = 1 / math.sqrt(feat_dim)
        if backend == 'tensorflow':
            self.centers = tf.Variable(tf.random.uniform((num_classes, feat_dim), -bound, bound), name='centers')
        else:
            key = jax.random.PRNGKey(0) if key is None else key
            self.centers = jax.random.uniform(key, (num_classes, feat_dim), minval=-bound, maxval=bound)

    @property
    def trainable_variables(self):
        return (self.centers,)

    def forward(self, output, target):
        return self.with_centers(output, target, self.centers)

    def with_centers(self, features, target, centers):
        if backend == 'tensorflow':
            tf.debugging.assert_non_negative(target)
            tf.debugging.assert_less(target, tf.cast(self.num_classes, target.dtype))
            # Dense center gradients also work with Trident's dense-only Adam.
            selected = tf.matmul(tf.one_hot(target, self.num_classes, dtype=centers.dtype), centers)
        else:
            selected = centers[target]
        distance = xp.sum((features - as_like(selected, features)) ** 2, -1)
        return self._get_reduction(xp.clip(distance, 1e-8, 1e5))


class PerceptionLoss(PairwiseLoss):
    """Mean MSE over frozen VGG features or features returned by a callable."""
    feature_names = ('block1_relu2', 'block2_relu2', 'block3_relu3', 'block4_relu3')

    def __init__(self, net, reduction='mean', input_names=None, output_names=None, name='PerceptionLoss'):
        super().__init__(reduction=reduction, input_names=input_names, output_names=output_names, name=name)
        self.ref_model = net.model if hasattr(net, 'training_context') else net
        self.feature_layers = []
        if backend == 'tensorflow' and isinstance(self.ref_model, tf.keras.Model):
            names = {layer.name for layer in self.ref_model.layers}
            chosen = [name if name in names else name.replace('relu', 'conv') for name in self.feature_names]
            if all(name in names for name in chosen):
                self.ref_model = tf.keras.Model(self.ref_model.inputs,
                                                [self.ref_model.get_layer(name).output for name in chosen])
            self.ref_model.trainable = False
        elif hasattr(self.ref_model, 'named_modules'):
            modules = dict(self.ref_model.named_modules())
            for name in self.feature_names:
                if name in modules:
                    modules[name].keep_output = True
                    self.feature_layers.append(modules[name])
            if hasattr(self.ref_model, 'eval'):
                self.ref_model.eval()
            if backend == 'pytorch':
                self.ref_model.requires_grad_(False)
            elif hasattr(self.ref_model, 'trainable'):
                self.ref_model.trainable = False

    def _features(self, image):
        if backend == 'pytorch':
            image = moveaxis(image, 1, -1)
        x = (image + 1) * .5 * as_like([.485, .456, .406], image) + as_like([.229, .224, .225], image)
        if backend == 'tensorflow' and isinstance(self.ref_model, tf.keras.Model):
            features = self.ref_model(x, training=False)
        else:
            features = self.ref_model(moveaxis(x, -1, 1) if backend == 'pytorch' else x)
        if self.feature_layers:
            return [layer.output for layer in self.feature_layers]
        if isinstance(features, dict):
            return [features[k] for k in self.feature_names] if all(k in features for k in self.feature_names) else list(features.values())
        return list(features) if isinstance(features, (tuple, list)) else [features]

    def calculate_loss(self, output, target):
        prediction, reference = self._features(output), self._features(target)
        return sum(xp.mean((a - stop_gradient(b)) ** 2) for a, b in zip(prediction, reference)) / len(prediction)


class TransformInvariantLoss(Loss):
    """Compare embedded predictions with a bilinearly rotated and scaled target."""
    def __init__(self, loss, embedded_func):
        super().__init__(name='TransformInvariantLoss')
        self.loss = loss if loss is not None else MSELoss()
        self.embedded_func = embedded_func
        self.rotation_range = 20.
        self.zoom_range = .1

    def forward(self, output, target, *, angle=None, scale=None, key=None):
        image = moveaxis(target, 1, -1) if backend == 'pytorch' else target
        batch, h, w = (shape(image)[i] for i in range(3))
        if angle is None or scale is None:
            if backend == 'jax':
                if key is None:
                    raise ValueError("provide a JAX PRNG key or explicit angle and scale")
                noise = jax.random.uniform(key, (2, batch), minval=-1., maxval=1.)
            elif backend == 'tensorflow':
                noise = tf.random.uniform((2, batch), -1., 1., dtype=image.dtype)
            else:
                noise = xp.rand((2, batch), device=image.device, dtype=image.dtype) * 2 - 1
            angle = noise[0] * self.rotation_range if angle is None else angle
            scale = 1 + noise[1] * self.zoom_range if scale is None else scale
        radians = xp.reshape(as_like(angle, image), (-1, 1, 1)) * (math.pi / 180)
        zoom = xp.reshape(as_like(scale, image), (-1, 1, 1))
        def indices(n):
            if backend == 'tensorflow':
                return tf.range(n)
            return xp.arange(n, device=image.device) if backend == 'pytorch' else xp.arange(n)
        cx, cy = as_like(w - 1, image) * .5, as_like(h - 1, image) * .5
        xx = as_like(indices(w), image)[None, None, :] - cx
        yy = as_like(indices(h), image)[None, :, None] - cy
        u = (xp.cos(radians) * xx - xp.sin(radians) * yy) / zoom + cx
        v = (xp.sin(radians) * xx + xp.cos(radians) * yy) / zoom + cy
        x0, y0 = xp.floor(u), xp.floor(v)
        warped = xp.zeros_like(image)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x, y = x0 + dx, y0 + dy
            active = (x >= 0) & (x < as_like(w, x)) & (y >= 0) & (y < as_like(h, y))
            ix = xp.clip(x, 0, as_like(w - 1, x))
            iy = xp.clip(y, 0, as_like(h - 1, y))
            b = indices(batch)[:, None, None]
            if backend == 'tensorflow':
                coords = tf.stack([tf.broadcast_to(b, (batch, h, w)), tf.broadcast_to(tf.cast(iy, b.dtype), (batch, h, w)),
                                   tf.broadcast_to(tf.cast(ix, b.dtype), (batch, h, w))], axis=-1)
                pixels = tf.gather_nd(image, coords)
            elif backend == 'jax':
                pixels = image[b, iy.astype(int), ix.astype(int)]
            else:
                pixels = image[b, iy.long(), ix.long()]
            weight = (1 - xp.abs(u - x)) * (1 - xp.abs(v - y)) * as_like(active, image)
            warped = warped + pixels * weight[..., None]
        target = moveaxis(warped, -1, 1) if backend == 'pytorch' else warped
        return self.loss(self.embedded_func(output), self.embedded_func(target))


class TverskyLoss(ClassificationLoss):
    """1 - weighted Tversky similarity, with separate FP and FN penalties."""
    def __init__(self, alpha=.5, beta=.5, smooth=1., **kwargs):
        super().__init__(**kwargs)
        if alpha < 0 or beta < 0 or alpha + beta <= 0 or smooth <= 0:
            raise ValueError("expected nonnegative FP/FN penalties, a positive total penalty and smooth > 0")
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def loss_terms(self, output, target):
        x, dist, weights, valid, sparse = _class_inputs(self, output, target)
        p = _binary_probs(x, self.from_logits)
        mask = as_like(valid[..., None], x) if sparse else 1.
        axes = tuple(range(1, len(x.shape)))
        tp = xp.sum(p * dist * mask * weights, axis=axes)
        fp = xp.sum(p * (1 - dist) * mask * weights, axis=axes)
        fn = xp.sum((1 - p) * dist * mask * weights, axis=axes)
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth), None


class HardTripletLoss(Loss):
    def __init__(self, margin=.1, hardest=False, squared=False):
        super().__init__(name='HardTripletLoss')
        self.margin, self.hardest, self.squared = margin, hardest, squared

    def forward(self, embeddings, labels):
        differences = embeddings[:, None, :] - embeddings[None, :, :]
        distances = xp.sum(differences ** 2, axis=-1)
        if not self.squared:
            positive = distances > 0
            distances = xp.where(positive, xp.sqrt(xp.where(positive, distances, xp.ones_like(distances))), 0.)
        same = labels[:, None] == labels[None, :]
        if backend == 'pytorch':
            diagonal = xp.eye(len(labels), dtype=xp.bool, device=labels.device)
        elif backend == 'tensorflow':
            diagonal = tf.eye(tf.shape(labels)[0], dtype=tf.bool)
        else:
            diagonal = xp.eye(shape(labels)[0], dtype=bool)
        positive = same & ~diagonal
        if self.hardest:
            farthest = xp.amax(xp.where(positive, distances, xp.zeros_like(distances)), axis=1)
            nearest = xp.amin(xp.where(~same, distances, as_like(math.inf, distances)), axis=1)
            valid = xp.any(positive, axis=1) & xp.any(~same, axis=1)
            raw = xp.where(valid, farthest - nearest + self.margin, xp.zeros_like(farthest))
            count = xp.sum(as_like(valid, embeddings))
        else:
            valid = positive[:, :, None] & ~same[:, None, :]
            diff = distances[:, :, None] - distances[:, None, :] + self.margin
            raw = xp.where(valid, diff, xp.zeros_like(diff))
            count = xp.sum(as_like(raw > 1e-16, embeddings))
        return xp.sum(xp.where(raw > 0, raw, xp.zeros_like(raw))) / xp.maximum(count, as_like(1., embeddings))


class GPLoss(Loss):
    """WGAN-GP on interpolated samples. alpha or a JAX RNG key controls sampling."""
    def __init__(self, discriminator, l=10):
        super().__init__(name='GPLoss')
        self.discriminator, self.l = discriminator, l

    def forward(self, real_data, fake_data, *, alpha=None, key=None):
        batch = shape(real_data)[0]
        dims = (batch,) + (1,) * (len(real_data.shape) - 1)
        if alpha is None:
            if backend == 'jax':
                if key is None:
                    raise ValueError("provide a JAX PRNG key or explicit alpha")
                alpha = jax.random.uniform(key, dims)
            else:
                alpha = tf.random.uniform(dims, dtype=real_data.dtype)
        samples = stop_gradient(real_data * alpha + fake_data * (1 - alpha))
        if backend == 'jax':
            gradient = jax.grad(lambda x: xp.sum(self.discriminator(x)))(samples)
        else:
            with tf.GradientTape() as tape:
                tape.watch(samples)
                score = tf.reduce_sum(self.discriminator(samples))
            gradient = tape.gradient(score, samples)
        norm = xp.sqrt(xp.sum(flat_batch(gradient) ** 2, axis=1) + 1e-12)
        return self.l * xp.mean((norm - 1) ** 2)
