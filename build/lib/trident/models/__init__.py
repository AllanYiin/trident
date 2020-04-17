
from ..backend.load_backend import get_backend

__all__ = ['vgg','resnet','densenet','efficientnet','mobilenet','gan','deeplab','arcfacenet','mtcnn','rfbnet']

if get_backend()=='pytorch':
    from . import pytorch_vgg as vgg
    from . import pytorch_resnet as resnet
    from . import pytorch_densenet as densenet
    from . import pytorch_efficientnet as efficientnet
    from . import pytorch_mobilenet as mobilenet
    from . import pytorch_gan as gan
    from . import pytorch_deeplab as deeplab
    from . import pytorch_arcfacenet as arcfacenet
    from . import pytorch_mtcnn as mtcnn
    from . import pytorch_rfbnet as rfbnet
elif get_backend()=='tensorflow':
    from . import tensorflow_resnet as resnet
    from ..backend.tensorflow_backend import  to_numpy,to_tensor
elif get_backend()=='cntk':
    from ..backend.cntk_backend import  to_numpy,to_tensor

