from math import exp

import cntk as C
import numpy as np
from cntk.ops import *

from ..backend.cntk_backend import *
from ..backend.cntk_ops import *
from ..backend.common import get_session, get_function, snake2camel, camel2snake

_device = "cuda" if 'GPU' in str(C.all_devices()[0]) else "cpu"
_session=get_session()

__all__ = ['MSELoss','CrossEntropyLoss','get_loss' ]



def MSELoss(reduction='mean',name=''):
    def loss(output,target):
        if reduction=='mean':
            return C.reduce_mean(C.squared_error(output,target))
        else:
            return C.reduce_sum(C.squared_error(output, target))
    return loss


def CrossEntropyLoss(reduction='mean',name=''):
    def loss(output, target):
        if reduction == 'mean':
            return C.reduce_mean(C.cross_entropy_with_softmax(output, target,axis=-1))
        else:
            return C.reduce_sum(C.cross_entropy_with_softmax(output, target,axis=-1))
    return loss




def CosineRankingLoss(margin=0.1,reduction='mean',name=''):
    def loss(output,target):
        q = output[:, 0]
        correct = output[:, 1]
        incorrect = output[:, 2]
        mr_loss =C.clip(margin - C.cosine_distance(q,correct)+C.cosine_distance(q,incorrect),0,5)- target [0 ] *0
        if reduction=='mean':
            return C.reduce_mean(mr_loss)
        else:
            return C.reduce_sum(mr_loss)
    return loss

def MarginRankingLoss(margin=0.1,reduction='mean',name=''):
    def loss(output,target):
        pos = output[:, 0]
        neg = output[:, 1]
        mr_loss = -C.sigmoid(pos - neg)  # use loss = K.maximum(1.0 + neg - pos, 0.0) if you want to
        if reduction=='mean':
            return C.reduce_mean(mr_loss)+ 0 * target
        else:
            return C.reduce_sum(mr_loss)+ 0 * target
    return loss

def ContrastiveLoss(margin=1,reduction='mean',name=''):
    """Contrastive loss from Hadsell-et-al.'06
     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

     y = 0 if image is similar
     y = 1 if image is different
     according to Tokukawa: https://github.com/fchollet/keras/issues/4980 it has to be:
     return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
     instead of:
     return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

     """
    def loss(output,target):
        c_loss = (1 - target) * C.square(output) + target * C.square(C.clip(margin - output, 0,np.inf))
        if reduction=='mean':
            return C.reduce_mean(c_loss)
        else:
            return C.reduce_sum(c_loss)
    return loss






def get_loss(loss_name):
    if loss_name is None:
        return None
    loss_modules = ['trident.optims.cntk_losses']
    if loss_name in __all__:
        loss_fn = get_function(loss_name, loss_modules)
    else:
        try:
            loss_fn = get_function(camel2snake(loss_name), loss_modules)
        except Exception :
            loss_fn = get_function(loss_name,loss_modules)
    return loss_fn
