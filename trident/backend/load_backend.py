from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import random
from sys import stderr,stdout
from trident.backend.common import *


__all__ = []

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
            session=_session.__dict__.copy()
            session.pop('_thread_local_info')
            session.pop('_context_handle')
            session.pop('_module_dict')
            session.pop('print')
            f.write(json.dumps(session, indent=4))
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


# Set backend based on TRIDENT_BACKEND flag, if applicable.
if 'TRIDENT_BACKEND' in os.environ:
    if _session.backend!=os.environ['TRIDENT_BACKEND']:
        _session.backend = os.environ['TRIDENT_BACKEND']
    if 'TRIDENT_WORKING_DIR' in os.environ:
        _session.working_directory = os.environ['TRIDENT_WORKING_DIR']
        os.chdir(os.environ['TRIDENT_WORKING_DIR'])
    write_config(_config_path)



if _session.backend== 'pytorch':
    stdout.write('Using Pytorch backend.\n')
    stdout.write('Image Data Format: channels_first.\n')
    stdout.write('Image Channel Order: rgb.\n')
    _session.image_data_format='channels_first'
    _session.image_channel_order='rgb'
    import torch
    from trident.backend.pytorch_ops import *

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        set_session('device','cuda')
    elif is_tpu_available():
        import torch_xla.core.xla_model as xm

        # os.environ['XLA_USE_BF16'] = '1'
        # os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'
        set_session('device', 'tpu')
        set_session('print', xm.master_print)




    from trident.backend.pytorch_backend import *

elif _session.backend == 'tensorflow':
    stdout.write('Using TensorFlow backend.\n')
    stdout.write('Image Data Format: channels_last.\n')
    stdout.write('Image Channel Order: rgb.\n')
    _session.image_data_format = 'channels_last'
    _session.image_channel_order = 'rgb'


    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], False)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            set_session('device', '/gpu:0')
            tf.config.set_visible_devices(gpus[0], 'GPU')

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    else:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        set_session('device', '/cpu:0')

    from trident.backend.tensorflow_ops import *
    from trident.backend.tensorflow_backend import *

elif _session.backend == 'jax':
    stdout.write('Using Jax backend.\n')
    stdout.write('Image Data Format: channels_last.\n')
    stdout.write('Image Channel Order: rgb.\n')
    _session.image_data_format = 'channels_last'
    _session.image_channel_order = 'rgb'

    import jax
    from trident.backend.jax_ops import *

    if is_gpu_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        set_session('device','cuda')
    elif is_tpu_available():
        import torch_xla.core.xla_model as xm
        set_session('device', 'tpu')
        set_session('print', xm.master_print)
    else:
        set_session('device', 'cpu')
    #from trident.backend.jax_backend import *


elif _session.backend == 'onnx':
    stdout.write('Using ONNX backend.\n')
    stdout.write('Image Data Format: channels_first.\n')
    stdout.write('Image Channel Order: rgb.\n')
    _session.image_data_format = 'channels_first'
    _session.image_channel_order = 'rgb'



from trident.backend.opencv_backend import *









if not os.path.exists(_config_path):
    write_config(_config_path)

