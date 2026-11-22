"""Run in a fresh process since Trident selects a backend at import time."""
import importlib
import os
import sys

import numpy as np

import trident
from trident.backend.common import get_backend

backend = get_backend()
losses = importlib.import_module('trident.optims.' + backend + '_losses')
if backend == 'pytorch':
    import torch

    def run(loss, x, y):
        variable = torch.tensor(x, requires_grad=True)
        output = loss(variable, torch.tensor(y))
        gradient, = torch.autograd.grad(output.sum(), variable)
        return output.detach().cpu().numpy(), gradient.detach().cpu().numpy()
elif backend == 'tensorflow':
    import tensorflow as tf

    def run(loss, x, y):
        @tf.function(autograph=False)
        def compute(variable, target):
            with tf.GradientTape() as tape:
                tape.watch(variable)
                output = loss(variable, target)
                scalar = tf.reduce_sum(output)
            return output, tape.gradient(scalar, variable)
        return compute(tf.constant(x), tf.constant(y))
else:
    import jax
    import jax.numpy as jnp

    def run(loss, x, y):
        def compute(variable, target):
            output = loss(variable, target)
            return jnp.sum(output), output
        (_, output), gradient = jax.jit(jax.value_and_grad(compute, has_aux=True))(jnp.asarray(x), jnp.asarray(y))
        return output, gradient


def main(destination):
    values = {}
    covered = set()
    for name in losses.__all__:
        if name != 'get_loss':
            assert trident.get_loss(name) is getattr(losses, name)
    x = np.array([[[.15, -.25], [.8, .6], [-.42, .2]],
                  [[.12, .24], [-.18, .74], [.45, -.1]]], np.float32)
    y = np.array([[[.5, .3], [.3, .5], [.2, .2]], [[.1, .2], [.45, .5], [.45, .3]]], np.float32)

    def check(name, pred=x, target=y, tag='', **options):
        fn = losses.get_loss(name)(**options)
        covered.add(name)
        output, grad = run(fn, pred, target)
        key = name + tag
        values[key], values[key + '_grad'] = np.asarray(output), np.asarray(grad)

    for reduction in ('none', 'mean', 'sum', 'batch_sum', 'batch_mean'):
        for name in ('MSELoss', 'L1Loss', 'L2Loss', 'SmoothL1Loss'):
            check(name, reduction=reduction, tag=reduction)
    check('MSELoss', np.array(.5, np.float32), np.array(.2, np.float32), tag='scalar', reduction='none')
    check('L1Loss', x, np.array(.3, np.float32), tag='scalar_target')
    check('SmoothL1Loss', x[:1, :1, :1], y[:1, :1, :1], tag='single_ohem', enable_ohem=True)
    check('HuberLoss', delta=.2, reduction='none')
    check('BCEWithLogitsLoss', pos_weight=[.7, 2.], reduction='none')
    check('BCEWithLogitsLoss', np.array([-80., 0., 80.], np.float32), np.array([1., .25, 0.], np.float32), tag='extreme')
    check('WingLoss')
    check('AdaptiveWingLoss')
    check('CosineSimilarityLoss')
    check('KLDivergenceLoss', np.log(y), y)
    check('KLDivergenceLoss', np.log(y), y * .8, reduction='batchmean', tag='batchmean')
    check('ItakuraSaitoLoss', x + 1, y)
    check('ExponentialLoss')
    sparse = np.array([[1, -100], [0, 2]], np.int32)
    logp = x - np.log(np.exp(x).sum(1, keepdims=True))
    for tag, prediction in [('log_probs', logp), ('probabilities', np.exp(logp)), ('raw_logits', x)]:
        check('CrossEntropyLoss', prediction, sparse, axis=1, tag='auto_' + tag)
    for reduction in ('none', 'mean', 'sum', 'batch_sum', 'batch_mean'):
        for labels, tag in ((sparse, 'sparse'), (y, 'soft')):
            check('CrossEntropyLoss', x, labels, axis=1, sample_weight=[.25, 2., 0.], label_smooth=.1,
                  from_logits=True, reduction=reduction, tag=tag+reduction)
        check('NLLLoss', logp, sparse, axis=1, sample_weight=[.25, 2., 0.], reduction=reduction, tag=reduction)
    check('CrossEntropyLoss', x, np.full_like(sparse, -100), axis=1, from_logits=True, tag='ignored')
    check('CrossEntropyLoss', x[0, :, 0], np.array(1, np.int64), from_logits=True, tag='unbatched')
    check('CrossEntropyLoss', np.moveaxis(x, 1, -1).copy(), sparse, axis=-1, from_logits=True, tag='lastaxis')
    for name in ('BCELoss', 'FocalLoss', 'F1ScoreLoss', 'DiceLoss', 'GeneralizedDiceLoss', 'IoULoss', 'SoftIoULoss', 'TverskyLoss'):
        check(name, x, y, axis=1, from_logits=True, reduction='none')
    check('CrossEntropyLoss', x, sparse, tag='ohem', axis=1, from_logits=True, enable_ohem=True, ohem_thresh=.5)
    rng = np.random.default_rng(19)
    images = rng.uniform(.1, .9, (2, 19, 21, 3)).astype(np.float32)
    reference = images * .7 + .1
    def channels_last(v):
        return np.moveaxis(v, 1, -1) if backend == 'pytorch' else v
    def image_case(name, fn, target=reference):
        covered.add(name)
        pred = np.moveaxis(images, -1, 1).copy() if backend == 'pytorch' else images
        label = np.moveaxis(target, -1, 1).copy() if backend == 'pytorch' and target.ndim == 4 else target
        value, grad = run(fn, pred, label)
        values[name] = np.asarray(value)
        values[name + '_grad'] = np.asarray(channels_last(grad))
    for name in ('EdgeLoss', 'StyleLoss'):
        image_case(name, losses.get_loss(name)())
    image_case('MS_SSIMLoss', losses.MS_SSIMLoss(window_size=3, weights=[.4, .6], data_range=1.))
    labels = np.argmax(reference, axis=-1).astype(np.int64)
    for name in ('ActiveContourLoss', 'LovaszSoftmax'):
        image_case(name, losses.get_loss(name)(from_logits=True, ignore_index=-100), labels)
    def extractor(image):
        return [image, image * image]
    image_case('PerceptionLoss', losses.PerceptionLoss(extractor))
    transform = losses.TransformInvariantLoss(None, lambda image: image)
    image_case('TransformInvariantLoss', lambda a, b: transform.forward(a, b, angle=13., scale=.85))
    features = x[:, :, 0].copy()
    for name in ('TripletLoss', 'TripletMarginLoss'):
        covered.add(name)
        triplet = losses.get_loss(name)(margin=.4, swap=True)
        check_fn = lambda a, b: triplet(a, b, b + .3)
        value, grad = run(check_fn, features, y[:, :, 0].copy())
        values[name], values[name + '_grad'] = np.asarray(value), np.asarray(grad)
    center = losses.CenterLoss(num_classes=2, feat_dim=3)
    centers = np.array([[.1, .2, .3], [.6, .1, .8]], np.float32)
    if backend == 'pytorch':
        with torch.no_grad():
            center.centers.copy_(torch.tensor(centers))
    elif backend == 'tensorflow':
        center.centers.assign(centers)
    else:
        center.centers = jnp.asarray(centers)
    value, grad = run(center, features, np.array([1, 0], dtype=np.int64))
    values['CenterLoss'], values['CenterLoss_grad'] = np.asarray(value), np.asarray(grad)
    if backend == 'pytorch':
        center(torch.tensor(features), torch.tensor([1, 0])).backward()
        center_grad = center.centers.grad.detach().cpu()
    elif backend == 'tensorflow':
        with tf.GradientTape() as tape:
            center_value = center(tf.constant(features), tf.constant([1, 0]))
        center_grad = tape.gradient(center_value, center.centers)
    else:
        center_grad = jax.jit(jax.grad(lambda c: center.with_centers(jnp.asarray(features), jnp.array([1, 0]), c)))(center.centers)
    values['CenterLoss_centers_grad'] = np.asarray(center_grad)
    covered.add('CenterLoss')
    vectors = rng.normal(size=(5, 3)).astype(np.float32)
    for hardest in (False, True):
        check('HardTripletLoss', vectors, np.array([0, 1, 0, 1, 0]), margin=.6, hardest=hardest, tag=str(hardest))
    # The gradient-penalty derivative is with respect to the critic, not the data.
    def penalty(coefficient, data):
        def critic(z):
            return coefficient * z
        alpha = data * 0 + .5
        return losses.GPLoss(critic).forward(data, data + 1, alpha=alpha)
    value, grad = run(penalty, np.array(.3, np.float32), vectors)
    values['GPLoss'], values['GPLoss_grad'] = np.asarray(value), np.asarray(grad)
    covered.add('GPLoss')
    check('BCELoss', np.array([[0., 1.], [.4, .7]], np.float32),
          np.array([[1., 0.], [.2, 1.]], np.float32), from_logits=False, tag='endpoints')
    assert covered == set(losses.__all__) - {'get_loss', '_ClassificationLoss'}
    ops = importlib.import_module('trident.backend.' + backend + '_ops')
    for name in ('binary_cross_entropy', 'cross_entropy', 'binary_hinge'):
        fn = getattr(ops, name)
        value, grad = run(fn, y, y * .5)
        values[name], values[name + '_grad'] = np.asarray(value), np.asarray(grad)
    np.savez(destination, **values)


if __name__ == '__main__':
    main(sys.argv[1])
