from __future__ import absolute_import
from __future__ import print_function
import threading
import json
from sys import stderr
import os
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import set_image_channel_order
from .common import set_image_data_format

_SESSION = threading.local()

def get_session():
    return _SESSION


# Set Keras base dir path given TRIDENT_HOME env variable, if applicable.
# Otherwise either ~/.trident or /tmp.




if 'TRIDENT_HOME' in os.environ:
    _trident_dir = os.environ.get('TRIDENT_HOME')
else:
    _trident_base_dir = os.path.expanduser('~')
    if not os.access(_trident_base_dir, os.W_OK):
        _trident_base_dir = '/tmp'
    _trident_dir = os.path.join(_trident_base_dir, '.trident')

_SESSION.trident_dir=_trident_dir


           


# Default backend: Pytorch.
_BACKEND = 'pytorch'
_IMAGE_BACKEND='pillow'

def write_config(_config_path):
    _config = {
        'floatx': floatx(),
        'epsilon': epsilon(),
        'backend': _BACKEND,
        'image_backend':_IMAGE_BACKEND
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass


# Attempt to read Trident config file.
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
    _image_backend = _config.get('image_backend', _IMAGE_BACKEND)

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    _BACKEND = _backend
    _IMAGE_BACKEND=_image_backend
    
    _SESSION.floatx=_floatx
    _SESSION.epsilon=_epsilon
    _SESSION.backend =  _backend
    _SESSION.image_backend =  _image_backend

# Save config file, if possible.
if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


# Set backend based on TRIDENT_BACKEND flag, if applicable.
if 'TRIDENT_BACKEND' in os.environ:
    _backend = os.environ['TRIDENT_BACKEND']
    if _backend!=_BACKEND:
        _BACKEND = _backend
        write_config(_config_path)
        _SESSION.backend = _backend


if _BACKEND == 'cntk':
    stderr.write('Using CNTK backend\n')
    stderr.write('Image Data Format: channels_first.\n')
    stderr.write('Image Channel Order: bgr.\n')
    set_image_data_format('channels_first')
    set_image_channel_order('bgr')
    from .cntk_backend import *
    from layers.cntk_normalizations import *
    from layers.cntk_activations import *
elif _BACKEND == 'pytorch':
    stderr.write('Using Pytorch backend.\n')
    stderr.write('Image Data Format: channels_first.\n')
    stderr.write('Image Channel Order: bgr.\n')
    set_image_data_format('channels_first')
    set_image_channel_order('bgr')
    from .pytorch_backend import *
    from data.pytorch_datasets import *
elif _BACKEND == 'tensorflow':
    stderr.write('Using TensorFlow backend.\n')
    stderr.write('Image Data Format: channels_last.\n')
    stderr.write('Image Channel Order: rgb.\n')
    set_image_data_format('channels_last')
    set_image_channel_order('rgb')
    from .tensorflow_backend import *
    from data.tensorflow_datasets import *
    from layers.tensorflow_normalizations import *
    from layers.tensorflow_activations import *



if 'TRIDENT_IMG_BACKEND' in os.environ:
    _image_backend = os.environ['TRIDENT_IMG_BACKEND']
    if _image_backend:
        _IMAGE_BACKEND = _image_backend
        _SESSION.image_backend = _image_backend

if _IMAGE_BACKEND == 'opencv':
    stderr.write('Using opencv image backend\n')
    from .opencv_backend import *
elif _IMAGE_BACKEND == 'pillow':
    stderr.write('Using pillow image backend.\n')
    from .pillow_backend import *


if not os.path.exists(_config_path):
    write_config(_config_path)


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

def image_backend():
    """Returns the name of the current backend (e.g. "tensorflow").

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> trident.backend.image_backend()
        'tensorflow'
    ```
    """
    return _IMAGE_BACKEND


def get_trident_dir():
    return _trident_dir

