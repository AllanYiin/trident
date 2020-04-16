from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import collections
import itertools
import cntk as C
import cntk.ops
from ..backend.common import get_session,addindent,get_time_suffix,get_function,get_class,format_time,get_terminal_size,snake2camel,camel2snake


__all__ = ['accuracy','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metrics']


def accuracy(output, target, topk=1,axis=-1):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    if target.shape!=output.shape:
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))

    if topk==1:
        return C.reduce_mean(C.equal(C.argmax(target,-1),C.argmax(output,-1)))
    else:
        return C.reduce_mean(C.classification_error(output, target, axis,topk))


def psnr(output, target):
    if target.shape!=output.shape :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    diff =C.reduce_mean(C.square(output - target),[1,2,3])
    return C.reduce_mean( (10 * C.log(255**2 / diff)/C.log(10)))


def mean_absolute_error(output, target):
    if target.shape!=output.shape :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return C.reduce_mean(C.abs(output -  target))
mae=mean_absolute_error


def mean_squared_error(output, target):
    if target.shape!=output.shape :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return C.reduce_mean(C.square(output -  target))
mse=mean_squared_error




def root_mean_squared_error(output, target):
    if target.shape!=output.shape :
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return C.sqrt(C.reduce_mean(C.square(output -  target)))
rmse=root_mean_squared_error



def mean_squared_logarithmic_error(output, target):
    if target.shape!=output.shape :
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return C.reduce_mean(C.square(C.log(1 +output)- C.log(1 + target)))
msle=mean_squared_logarithmic_error



def get_metrics(metrics_name):
    if metrics_name is None:
        return None
    metrics_modules = ['trident.optimizers.cntk_metrics']
    try:
        metrics_fn = get_function(camel2snake(metrics_name), metrics_modules)
    except Exception :
        metrics_fn = get_function(metrics_name, metrics_modules)
    return metrics_fn

