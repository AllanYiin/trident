import jax.numpy as jnp
from jax.lib import xla_bridge
from trident.backend import common
from trident.backend.common import to_list, addindent, camel2snake, unpack_singleton, enforce_singleton, OrderedDict, get_session, set_session, get_session_value, \
    PrintException, Signature, TensorShape,  get_args_spec,is_instance
from trident.backend.tensorspec import *
from trident.backend import jax_ops as jops
from trident.backend import dtype
from trident import context
from trident.context import split_path, make_dir_if_need, sanitize_path


ctx = context._context()
_backend = ctx.get_backend()


DTYPE_MAPPING = {
    jnp.bool: dtype.bool,
    jnp.int8: dtype.int8,
    jnp.int16: dtype.int16,
    jnp.int32: dtype.int32,
    jnp.int64: dtype.int64,
    jnp.uint8: dtype.uint8,
    jnp.float16: dtype.float16,
    jnp.float32: dtype.float32,
    jnp.float64: dtype.float64,
    jnp.complex64: dtype.complex64,
    jnp.complex128: dtype.complex128,

}


def get_device():
    """get current device

    Returns: device string ('cpu', 'cuda)

    """
    if ctx.device is None:
        set_device("cuda" if torch.cuda.is_available() else 'xpu' if is_tpu_available() else  "cpu")

    return get_session().device


def set_device(device=None):

    if device is None:
        if xla_bridge.get_backend().platform=='gpu':
            device='cuda'
        elif xla_bridge.get_backend().platform=='tpu':
            device='xpu'
        else:
            device='cpu'
    device = device.lower().replace('gpu', 'cuda')
    if device == 'cuda' and xla_bridge.get_backend().platform!='gpu':
        raise ValueError('Gpu is not available...')
    if device == 'xpu' and xla_bridge.get_backend().platform!='tpu':
        raise ValueError('Tpu is not available...')
    # try:
    #     device_=device
    # #     if device=='xpu':
    # #         import torch_xla.core.xla_model as xm
    # #         device_ = xm.xla_device()
    # #     set_session('device', device_)
    # #
    # #     gcitems = gc.get_objects()
    # #     for i in range(len(gcitems)):
    # #         obj = gcitems[i]
    # #         try:
    # #             if torch.is_tensor(obj) :
    # #                 obj.to(device_)
    # #             elif isinstance(obj, nn.Module):
    # #                 obj.to(device_)
    # #         except Exception:
    # #             pass
    # # except Exception as e:
    # #     print(e)

