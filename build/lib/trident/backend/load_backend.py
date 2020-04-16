from __future__ import absolute_import
from __future__ import print_function
import threading
import json
from sys import stderr
import os
from .common import *




_session=get_session()
_trident_dir=get_trident_dir()





# Default backend: Pytorch.
_BACKEND = 'pytorch'
_IMAGE_BACKEND='pillow'

def write_config(_config_path):
    _config = {
        'floatx': _session.floatx,
        'epsilon': _session.epsilon,
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
    _floatx = _config.get('floatx', _session.floatx)
    assert _floatx in {'float16', 'float32', 'float64'}
    _epsilon = _config.get('epsilon', _session.epsilon)
    assert isinstance(_epsilon, float)
    _BACKEND = _config.get('backend', _BACKEND)
    _IMAGE_BACKEND = _config.get('image_backend', _IMAGE_BACKEND)


    
    _session.floatx=_floatx
    _session.epsilon=_epsilon
    _session.backend =  _BACKEND
    _session.image_backend =  _IMAGE_BACKEND

# Save config file, if possible.
if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


def get_backend():
    return _BACKEND

def get_image_backend():
    return _IMAGE_BACKEND




# Set backend based on TRIDENT_BACKEND flag, if applicable.
if 'TRIDENT_BACKEND' in os.environ:
    _backend = os.environ['TRIDENT_BACKEND']
    if _backend!=_BACKEND:
        _BACKEND = _backend
        _session.backend = _backend
        write_config(_config_path)



if _BACKEND == 'cntk':
    stderr.write('Using CNTK backend\n')
    stderr.write('Image Data Format: channels_first.\n')
    stderr.write('Image Channel Order: bgr.\n')
    _session.backend = 'pytorch'
    _session.image_data_format = 'channels_first'
    _session.image_channel_order = 'bgr'
    from .cntk_backend import *

elif _BACKEND == 'pytorch':
    stderr.write('Using Pytorch backend.\n')
    stderr.write('Image Data Format: channels_first.\n')
    stderr.write('Image Channel Order: bgr.\n')
    _session.backend='pytorch'
    _session.image_data_format='channels_first'
    _session.image_channel_order='bgr'
    from .pytorch_backend import *



elif _BACKEND == 'tensorflow':
    stderr.write('Using TensorFlow backend.\n')
    stderr.write('Image Data Format: channels_last.\n')
    stderr.write('Image Channel Order: rgb.\n')
    _session.backend = 'tensorflow'
    _session.image_data_format = 'channels_last'
    _session.image_channel_order = 'rgb'
    from .tensorflow_backend import *


if 'TRIDENT_IMG_BACKEND' in os.environ:
    _image_backend = os.environ['TRIDENT_IMG_BACKEND']
    if _image_backend:
        _IMAGE_BACKEND = _image_backend
        _session.image_backend = _image_backend




if _IMAGE_BACKEND == 'opencv':
    stderr.write('Using opencv image backend\n')
    #from .opencv_backend import *
elif _IMAGE_BACKEND == 'pillow':
    stderr.write('Using pillow image backend.\n')
    #from .pillow_backend import *
#image_common會載入image backend
# from .image_common import  *
# from data.datasets_common import *

if not os.path.exists(_config_path):
    write_config(_config_path)


