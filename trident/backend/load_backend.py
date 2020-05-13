from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
from sys import stderr
from trident.backend.common import *
from trident.backend.model import *
from trident.backend.optimizer import *


_session=get_session()
_trident_dir=get_trident_dir()
_config_path = os.path.expanduser(os.path.join(_trident_dir, 'trident.json'))


def write_config(_config_path):
    # _config = {
    #     'floatx': _session.floatx,
    #     'epsilon': _session.epsilon,
    #     'backend': _session.backend ,
    #     'image_backend':_session.image_backend
    # }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_session.__dict__, indent=4))
    except IOError:
        # Except permission denied.
        pass


# Save config file, if possible.
if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


def get_backend():
    return _session.backend

def get_image_backend():
    return _session.image_backend


# Set backend based on TRIDENT_BACKEND flag, if applicable.
if 'TRIDENT_BACKEND' in os.environ:
    if _session.backend!=os.environ['TRIDENT_BACKEND']:
        _session.backend = os.environ['TRIDENT_BACKEND']
        write_config(_config_path)


if _session.backend == 'pytorch':
    stderr.write('Using Pytorch backend.\n')
    stderr.write('Image Data Format: channels_first.\n')
    stderr.write('Image Channel Order: rgb.\n')
    _session.backend='pytorch'
    _session.image_data_format='channels_first'
    _session.image_channel_order='rgb'



    #module = importlib.import_module(mName)
    #layers=importlib.import_module('layers.pytorch_layers')


elif _session.backend == 'tensorflow':
    stderr.write('Using TensorFlow backend.\n')
    stderr.write('Image Data Format: channels_last.\n')
    stderr.write('Image Channel Order: rgb.\n')
    _session.backend = 'tensorflow'
    _session.image_data_format = 'channels_last'
    _session.image_channel_order = 'rgb'



if 'TRIDENT_IMG_BACKEND' in os.environ:
    _image_backend = os.environ['TRIDENT_IMG_BACKEND']
    if _image_backend and _session.image_backend!=_image_backend:
        _session.image_backend = _image_backend





if _session.image_backend == 'opencv':
    stderr.write('Using opencv image backend\n')
elif _session.image_backend== 'pillow':
    stderr.write('Using pillow image backend.\n')




if not os.path.exists(_config_path):
    write_config(_config_path)

