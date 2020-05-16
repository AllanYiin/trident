from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trident.backend.load_backend import get_backend

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
    from . import pytorch_ssd as ssd
    from . import pytorch_yolo as yolo
elif get_backend()=='tensorflow':
    from . import tensorflow_vgg as vgg
    from . import tensorflow_resnet as resnet
    from . import tensorflow_efficientnet as efficientnet
    from . import tensorflow_densenet as densenet
    from . import tensorflow_mobilenet as mobilenet

#__all__ = ['vgg','resnet','densenet','efficientnet','mobilenet','gan','deeplab','arcfacenet','mtcnn','rfbnet','ssd','yolo']


