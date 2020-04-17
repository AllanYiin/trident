import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops.losses import util as tf_losses_utils

from ..backend.common import get_session, get_function, camel2snake, get_class
from ..backend.tensorflow_ops import *

# def cosine_similarity(y_true, y_pred):
#     assert y_true.ndim == 2
#     assert y_pred.ndim == 2
#     y_true = l2_normalize(y_true, axis=1)
#     y_pred = l2_normalize(y_pred, axis=1)
#     return T.sum(y_true * y_pred, axis=1, keepdims=False)
# def cosine_ranking_loss(y_true, y_pred):
#     q = y_pred[: ,:args.hidden_size]
#     a_correct = y_pred[: ,args.hidden_size: 2 *args.hidden_size]
#     a_incorrect = y_pred[: , 2 *args.hidden_size: 3 *args.hidden_size]
#
#     return mean \
#         (T.maximum(0., args.margin - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect)) - y_true
#             [0 ] *0, axis=-1)



__all__ = ['get_loss','CrossEntropyLoss','MSELoss','EdgeLoss','NllLoss']


def CrossEntropyLoss(output,target):
    return  tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(output,target),axis=-1))

# class CrossEntropyLoss(LossFunctionWrapper):
#     def __init__(self, reduction='mean', name='CrossEntropyLoss'):
#         func = tf.nn.weighted_cross_entropy_with_logits(.losses.categorical_crossentropy
#         super(CrossEntropyLoss, self).__init__(func, name=name, reduction=losses_utils.ReductionV2.AUTO)

def MSELoss(output,target):
    return  tf.reduce_mean((square(output-target)))





def NllLoss(output,target):
    return  tf.reduce_mean(-tf.math.log(output+1e-8)*target)


class EdgeLoss(object):
    def __init__(self ,name='EdgeLoss'):
        self.name=name
        super(EdgeLoss, self).__init__()
    def first_order(self, x, axis=2):
        h, w = x.shape[1:3]
        if axis == 1:
            return tf.math.abs((x[:,  :h - 1, :w - 1,:] - x[:,  1:, :w - 1,:]))
        elif axis == 2:
            return tf.math.abs(x[:, :h - 1, :w - 1,:] - x[:, :h - 1, 1:,:])
        else:
            return None
    def call(self, y_true, y_pred):
        loss1=tf.reduce_mean(tf.math.square(self.first_order(y_pred, 1) - self.first_order(y_true, 1)) )
        loss2=tf.reduce_mean(tf.math.square(self.first_order(y_pred, 2) - self.first_order(y_true, 2)))
        return loss1+loss2


def get_loss(loss_name):
    if loss_name is None:
        return None
    loss_modules = ['trident.optims.tensorflow_losses']
    if loss_name in __all__:
        loss_fn = get_class(loss_name, loss_modules)
    else:
        try:
            loss_fn = get_class(camel2snake(loss_name), loss_modules)
        except Exception :
            loss_fn = get_class(loss_name,loss_modules)
    return loss_fn

