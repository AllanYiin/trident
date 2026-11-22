"""Compare independent backend processes, including compiled TF/JAX gradients."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def probe(backend, directory):
    root = Path(__file__).resolve().parents[1]
    path = directory / (backend + '.npz')
    env = dict(os.environ, TRIDENT_BACKEND=backend, CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    subprocess.run([sys.executable, str(root / 'tests/loss_backend_probe.py'), str(path)], env=env, check=True)
    with np.load(path) as values:
        return dict(values)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    return probe('pytorch', tmp_path_factory.mktemp('loss-reference'))


@pytest.mark.parametrize('backend', ['tensorflow', 'jax'])
def test_loss_values_and_gradients_match_pytorch(backend, reference, tmp_path):
    if importlib.util.find_spec(backend) is None:
        pytest.skip(backend + ' is optional')
    actual = probe(backend, tmp_path)
    assert actual.keys() == reference.keys()
    for key, expected in reference.items():
        assert np.isfinite(expected).all(), key
        assert actual[key].shape == expected.shape, key
        np.testing.assert_allclose(actual[key], expected, rtol=3e-5, atol=2e-6, err_msg=key)
