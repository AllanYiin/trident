import json
import os
from sys import stderr
from .common import *

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

#
# # Attempt to read Trident config file.
# _config_path = os.path.expanduser(os.path.join(_trident_dir, 'trident.json'))
# if os.path.exists(_config_path):
#     try:
#         with open(_config_path) as f:
#             _config = json.load(f)
#     except ValueError:
#         _config = {}
#     for k,v in _config.items():
#         if k=='floatx':
#             assert v in {'float16', 'float32', 'float64'}
#         _session[k]=v



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



if _session.backend == 'cntk':
    stderr.write('Using CNTK backend\n')
    stderr.write('Image Data Format: channels_first.\n')
    stderr.write('Image Channel Order: rgb.\n')
    _session.backend = 'cntk'
    _session.image_data_format = 'channels_first'
    _session.image_channel_order = 'rgb'
    from .cntk_backend import *
    from .cntk_ops import *

elif _session.backend == 'pytorch':
    stderr.write('Using Pytorch backend.\n')
    stderr.write('Image Data Format: channels_first.\n')
    stderr.write('Image Channel Order: rgb.\n')
    _session.backend='pytorch'
    _session.image_data_format='channels_first'
    _session.image_channel_order='rgb'
    from .pytorch_ops import *
    from .pytorch_backend import *

    #module = importlib.import_module(mName)
    #layers=importlib.import_module('layers.pytorch_layers')


elif _session.backend == 'tensorflow':
    stderr.write('Using TensorFlow backend.\n')
    stderr.write('Image Data Format: channels_last.\n')
    stderr.write('Image Channel Order: rgb.\n')
    _session.backend = 'tensorflow'
    _session.image_data_format = 'channels_last'
    _session.image_channel_order = 'rgb'
    from .tensorflow_ops import *
    from .tensorflow_backend import  *


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

