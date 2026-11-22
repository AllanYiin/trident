"""Run each Trident backend in its own process and check against NumPy references."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import erf, expit


@pytest.mark.parametrize("backend,package", [
    ("pytorch", "torch"), ("tensorflow", "tensorflow"), ("jax", "jax"),
])
def test_llm_activations(backend, package):
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is optional")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, TRIDENT_BACKEND=backend, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, __file__, backend], cwd=root, env=env, check=True)


def check_backend(backend):
    acts = importlib.import_module(f"trident.layers.{backend}_activations")
    public = importlib.import_module("trident.backend")
    if backend == "pytorch":
        import torch

        def tensor(x):
            return torch.tensor(x, requires_grad=True)

        def evaluate(fn, x):
            output = fn(x)
            gradient, = torch.autograd.grad(output.sum(), x)
            return output.detach().cpu().numpy(), gradient.detach().cpu().numpy()
    elif backend == "tensorflow":
        import tensorflow as tf
        tensor = tf.constant

        def evaluate(fn, x):
            @tf.function(autograph=False)
            def compute(x):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    output = fn(x)
                    total = tf.reduce_sum(output)
                return output, tape.gradient(total, x)
            return compute(x)
    else:
        import jax
        import jax.numpy as jnp
        tensor = jnp.asarray

        def evaluate(fn, x):
            def compute(x):
                output = fn(x)
                return jnp.sum(output), output
            (_, output), gradient = jax.jit(jax.value_and_grad(compute, has_aux=True))(x)
            return output, gradient

    def silu(x):
        return x * expit(x)

    def gelu(x):
        return x * 0.5 * (1 + erf(x / np.sqrt(2)))

    def tanh_gelu(x):
        return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def gated(x, activation):
        gate, up = np.split(x, 2, axis=-1)
        return activation(gate) * up

    def gpt_oss(x):
        gate = np.minimum(x[..., ::2], 7)
        up = np.clip(x[..., 1::2], -7, 7)
        return gate * expit(1.702 * gate) * (up + 1)

    x = np.array([[[-10, 9, 8, -11], [-1.5, 0.3, 2.7, -0.6]]], dtype=np.float64)
    variable = tensor(x.astype(np.float32))
    cases = [
        ("swiglu", acts.SwiGLU, lambda x: gated(x, silu)),
        ("geglu", acts.GeGLU, lambda x: gated(x, gelu)),
        ("gpt_oss_swiglu", acts.GptOssSwiGLU, gpt_oss),
        ("silu", acts.Silu, silu), ("gelu", acts.Gelu, gelu),
        ("gelu_new", acts.GptGelu, tanh_gelu),
        ("gelu_pytorch_tanh", acts.GptGelu, tanh_gelu),
        ("relu2", acts.SquaredRelu, lambda x: np.maximum(x, 0)**2),
    ]
    for name, layer_type, reference in cases:
        fn = acts.get_activation(name)
        layer = acts.get_activation(name, only_layer=True)
        assert isinstance(layer, layer_type), name
        assert isinstance(acts.get_activation(layer_type.__name__), layer_type)
        expected = reference(x)
        gradient = np.empty_like(x)
        for index in np.ndindex(x.shape):
            delta = np.zeros_like(x)
            delta[index] = 1e-5
            gradient[index] = (reference(x + delta).sum() - reference(x - delta).sum()) / 2e-5
        for activation in (fn, layer):
            output, grad = evaluate(activation, variable)
            assert output.shape == expected.shape, name
            np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6, err_msg=name)
            np.testing.assert_allclose(grad, gradient, rtol=2e-5, atol=3e-6, err_msg=name)

    output, _ = evaluate(acts.GeGLU(approximate=True), variable)
    np.testing.assert_allclose(output, gated(x, tanh_gelu), rtol=2e-5, atol=2e-6)

    # At a clipping endpoint the reference implementation passes the gradient through.
    edge = tensor(np.array([[7, -7, 7, 7]], np.float32))
    _, grad = evaluate(acts.GptOssSwiGLU(), edge)
    sigmoid = expit(1.702 * 7)
    derivative = sigmoid * (1 + 7 * 1.702 * (1 - sigmoid))
    np.testing.assert_allclose(grad, [[-6 * derivative, 7 * sigmoid, 8 * derivative, 7 * sigmoid]], rtol=2e-5)

    for name in ("swiglu", "geglu", "gpt_oss_swiglu"):
        fn = acts.get_activation(name)
        layer = acts.get_activation(fn, only_layer=True)
        assert isinstance(layer, type(acts.get_activation(name, only_layer=True)))
        assert getattr(public, type(layer).__name__) is type(layer)
        half = tensor(x.astype(np.float16))
        assert layer(half).dtype == half.dtype
        if backend == "tensorflow":
            fn = tf.function(fn, input_signature=[tf.TensorSpec([None, None], tf.float32)])
            fn(tf.zeros((1, 4)))
        with pytest.raises((ValueError, IndexError, RuntimeError) if backend != "tensorflow"
                           else (ValueError, tf.errors.InvalidArgumentError)):
            fn(tensor(np.zeros((1, 3), np.float32)))


if __name__ == "__main__":
    check_backend(sys.argv[1])
