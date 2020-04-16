from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
import importlib
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import cast_to_floatx
from .common import image_data_format
from .common import set_image_data_format
from .common import normalize_data_format
from .common import symbolic, eager

# Set Keras base dir path given TRIDENT_HOME env variable, if applicable.
# Otherwise either ~/.trident or /tmp.
if 'TRIDENT_HOME' in os.environ:
    _trident_dir = os.environ.get('TRIDENT_HOME')
else:
    _trident_base_dir = os.path.expanduser('~')
    if not os.access(_trident_base_dir, os.W_OK):
        _trident_base_dir = '/tmp'
    _trident_dir = os.path.join(_trident_base_dir, '.trident')

# Default backend: Pytorch.
_BACKEND = 'pytorch'

# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_trident_dir, 'trident.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _floatx = _config.get('floatx', floatx())
    assert _floatx in {'float16', 'float32', 'float64'}
    _epsilon = _config.get('epsilon', epsilon())
    assert isinstance(_epsilon, float)
    _backend = _config.get('backend', _BACKEND)

    # _image_data_format = _config.get('image_data_format',image_data_format())
    # assert _image_data_format in {'channels_last', 'channels_first'}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        'floatx': floatx(),
        'epsilon': epsilon(),
        'backend': _BACKEND
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on TRIDENT_BACKEND flag, if applicable.
if 'TRIDENT_BACKEND' in os.environ:
    _backend = os.environ['TRIDENT_BACKEND']
    if _backend:
        _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'cntk':
    sys.stderr.write('Using CNTK backend\n')
    from .cntk_backend import *
elif _BACKEND == 'theano':
    sys.stderr.write('Using Theano backend.\n')
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend.\n')
    from .tensorflow_backend import *
else:
    # Try and load external backend.
    try:
        backend_module = importlib.import_module(_BACKEND)
        entries = backend_module.__dict__
        # Check if valid backend.
        # Module is a valid backend if it has the required entries.
        required_entries = ['placeholder', 'variable', 'function']
        for e in required_entries:
            if e not in entries:
                raise ValueError('Invalid backend. Missing required entry : ' + e)
        namespace = globals()
        for k, v in entries.items():
            # Make sure we don't override any entries from common, such as epsilon.
            if k not in namespace:
                namespace[k] = v
        sys.stderr.write('Using ' + _BACKEND + ' backend.\n')
    except ImportError:
        raise ValueError('Unable to import backend : ' + str(_BACKEND))


def backend():
    """Returns the name of the current backend (e.g. "tensorflow").

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> trident.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND