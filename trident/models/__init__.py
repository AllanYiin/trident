
from trident.backend.load_backend import get_backend

__all__ = ['vgg','resnet','densenet','efficientnet','mobilenet','gan','deeplab','arcfacenet','mtcnn','rfbnet','yolo']

if get_backend()=='pytorch':
    from trident.models import pytorch_vgg as vgg
    from trident.models import pytorch_resnet as resnet
    from trident.models import pytorch_densenet as densenet
    from trident.models import pytorch_efficientnet as efficientnet
    from trident.models import pytorch_mobilenet as mobilenet
    from trident.models import pytorch_gan as gan
    from trident.models import pytorch_deeplab as deeplab
    from trident.models import pytorch_arcfacenet as arcfacenet
    from trident.models import pytorch_mtcnn as mtcnn
    from trident.models import pytorch_rfbnet as rfbnet
    from trident.models import pytorch_ssd as ssd
    from trident.models import pytorch_yolo as yolo
elif get_backend()=='tensorflow':
    from trident.models import tensorflow_resnet as resnet
    from trident.models import tensorflow_efficientnet as efficientnet



