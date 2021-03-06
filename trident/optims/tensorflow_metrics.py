from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import itertools
import copy
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from trident.backend.common import get_session,addindent,get_time_suffix,get_function,get_class,format_time,get_terminal_size,snake2camel,camel2snake
from trident.backend.tensorflow_ops import *

__all__ = ['accuracy','recall','pixel_accuracy','alpha_pixel_accuracy','iou','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metric']


def flatten_check(output, target):
    "Check that `out` and `targ` have the same number of elements and flatten them."
    if ndim(output) > 2 and ndim(output) == ndim(target) + 1:
        shp = int_shape(output)
        output = output.reshape((shp[0], -1, shp[-1]))
        target = cast(target.reshape((shp[0], -1)), 'int64')
        return output, target
    elif ndim(output) > 2 and ndim(output) == ndim(target):
        shp = int_shape(output)
        output = output.reshape((shp[0], -1, shp[-1]))
        if ndim(target) > 2:
            target = target.reshape((shp[0], -1, shp[-1]))
        return output, target
    elif ndim(output) == 2 and ndim(output) == ndim(target):
        return output, target
    else:
        raise ValueError('output and target have diffent elements.')


def accuracy(output, target, topk=1, axis=-1,ignore_index=-100, exclude_mask=False):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """

    input_tensor = output.copy().detach()
    target_tensor = target.copy().detach()
    num_classes = int_shape(output)[axis]
    _dtype=input_tensor.dtype

    is_logsoftmax = None
    from_logits = None
    output_exp = exp(input_tensor)
    if (ndim(input_tensor) >= 1 and 'float' in str(input_tensor.dtype) and input_tensor.min() >= 0 and input_tensor.max() <= 1):
        is_logsoftmax = False
        from_logits = True
        input_tensor = clip(input_tensor, min=1e-8, max=1 - 1e-8)

    elif (ndim(output_exp) >= 1 and 'float' in str(output_exp.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1):
        is_logsoftmax = True
        from_logits = True
        input_tensor = clip(output_exp, min=1e-8, max=1 - 1e-8)
    else:
        is_logsoftmax = False
        from_logits = False

    if input_tensor.dtype != tf.int64 and topk == 1:
        if len(int_shape(input_tensor)) == 1:  # binary
            input_tensor =greater_equal(input_tensor,0.5)
        else:
            input_tensor = argmax(input_tensor, axis).squeeze()
    if target_tensor.dtype != tf.int64:
        target_tensor = argmax(target_tensor, axis).squeeze()
    if input_tensor.shape != target_tensor.shape and topk == 1:
        raise ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape, target_tensor.shape))
    input_tensor_numpy=to_numpy(input_tensor)
    input_mask=np.ones(int_shape(input_tensor))
    if isinstance(ignore_index, int) and 0 <= ignore_index < num_classes:
        input_mask[input_tensor_numpy==int(ignore_index)] = 0.
    elif isinstance(ignore_index, (list, tuple)):
        for idx in ignore_index:
            if isinstance(idx, int) and 0 <= idx < int_shape(output)[axis]:
                input_mask[input_tensor_numpy ==int( idx)] = 0.
    input_mask=to_tensor(input_mask,dtype=_dtype)

    batch_size = int_shape(target_tensor)[0]
    if topk == 1:
        return reduce_sum(equal(input_tensor,target_tensor,dtype=_dtype)*input_mask)/clip(reduce_sum(input_mask),min=1)
    else:
        _,pred = input_tensor.topk(topk)
        pred = cast(tf.transpose(pred),'float32')
        target_tensor= cast(repeat_elements(expand_dims(target_tensor,0),topk,axis=0),'float32')
        correct = equal(pred,target_tensor).sum()
        return correct/batch_size


def recall(output, target, axis=-1,ignore_index=-100):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """

    input_tensor = output.copy().detach()
    target_tensor = target.copy().detach()
    num_classes = int_shape(output)[axis]
    _dtype = input_tensor.dtype

    is_logsoftmax = None
    from_logits = None
    output_exp = exp(input_tensor)
    if (ndim(input_tensor) >= 1 and 'float' in str(input_tensor.dtype) and input_tensor.min() >= 0 and input_tensor.max() <= 1):
        is_logsoftmax = False
        from_logits = True
        input_tensor = clip(input_tensor, min=1e-8, max=1 - 1e-8)

    elif (ndim(output_exp) >= 1 and 'float' in str(output_exp.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1):
        is_logsoftmax = True
        from_logits = True
        input_tensor = clip(output_exp, min=1e-8, max=1 - 1e-8)
    else:
        is_logsoftmax = False
        from_logits = False

    if input_tensor.dtype != tf.int64 :
        if len(int_shape(input_tensor)) == 1:  # binary
            input_tensor =greater_equal(input_tensor,0.5)
        else:
            input_tensor = argmax(input_tensor, axis).squeeze()
    if target_tensor.dtype != tf.int64:
        target_tensor = argmax(target_tensor, axis).squeeze()
    if input_tensor.shape != target_tensor.shape :
        raise ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape, target_tensor.shape))

    target_tensor_numpy = to_numpy(target_tensor)
    target_mask = np.ones(int_shape(target_tensor))
    if isinstance(ignore_index, int) and 0 <= ignore_index < num_classes:
        target_mask[target_tensor_numpy == int(ignore_index)] = 0.
    elif isinstance(ignore_index, (list, tuple)):
        for idx in ignore_index:
            if isinstance(idx, int) and 0 <= idx < int_shape(output)[axis]:
                target_mask[target_tensor_numpy == int(idx)] = 0.
    target_mask = to_tensor(target_mask, dtype=_dtype)

    batch_size = int_shape(target_tensor)[0]

    return reduce_sum(equal(input_tensor,target_tensor,dtype=_dtype)*target_mask)/clip(reduce_sum(target_mask),min=1)


def pixel_accuracy(output, target):
    output, target=flatten_check(output, target)
    input_tensor = output.copy().detach()
    target_tensor = target.copy().detach()
    if input_tensor.dtype!=tf.int64 :
        input_tensor=argmax(input_tensor,axis=-1).squeeze()
    return equal(cast(input_tensor,'float32'),cast(target_tensor,'float32')).mean()

def alpha_pixel_accuracy(output,alpha):
    output, alpha = flatten_check(output, alpha)
    output_tensor = to_numpy(output)
    alpha_tensor =  to_numpy(alpha)

    trimap=alpha_tensor.copy()
    trimap[(0<trimap)*(trimap<1)==True]=0.5
    if len(output_tensor.shape)==len(alpha_tensor.shape)+1 and output_tensor.shape[1]==2:
        # trimap_out=mask2trimap()(np.argmax(output_tensor,1))
        # trimap_out1=trimap_out.copy()
        # trimap_out1[trimap_out1==0.5]=0
        # trimap_out2=trimap_out.copy()
        # trimap_out2[trimap_out2 ==1] = 0
        output_tensor=output_tensor[:,1,:,:]*np.argmax(output_tensor,1)#trimap_out*trimap_out1+output_tensor[:,1,:,:]*trimap_out2
        output_tensor[output_tensor>0.95]=1
    pixel_labeled = (output_tensor > 0).sum()
    pixel_correct = ((output_tensor == alpha_tensor)*(trimap == 1)).sum()+ (np.less(np.abs(output_tensor - alpha_tensor),0.1).astype(np.float32)*(trimap == 0.5)).sum()
    return pixel_correct/max(pixel_labeled,1)



def iou(output, target,axis=-1):
    output, target = flatten_check(output, target)
    input_tensor = output.copy().detach()
    target_tensor =target.copy().detach()

    num_classes = int_shape(input_tensor)[axis]
    sample_weight=np.ones(num_classes)
    sample_weight[0]=0
    sample_weight = to_tensor(sample_weight)
    reshape_shape = [1] * ndim(input_tensor)
    reshape_shape[-1] = num_classes
    sample_weight=reshape(sample_weight,reshape_shape)

    is_logsoftmax = None
    from_logits = None
    is_target_onehot=None
    output_exp = exp(input_tensor)
    if (ndim(input_tensor) >= 1 and 'float' in str(input_tensor.dtype) and input_tensor.min() >= 0 and input_tensor.max() <= 1):
        is_logsoftmax = False
        from_logits = True
        input_tensor = clip(input_tensor, min=1e-8, max=1 - 1e-8)

    elif (ndim(output_exp) >= 1 and 'float' in str(output_exp.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1):
        is_logsoftmax = True
        from_logits = True
        input_tensor = clip(exp(input_tensor), min=1e-8, max=1 - 1e-8)
    else:
        is_logsoftmax = False
        from_logits = False

    if target_tensor.dtype == str2dtype('long'):
        is_target_onehot = False
    elif target_tensor.dtype != str2dtype('long') and (target_tensor.min() >= 0 and target_tensor.max() <= 1 and abs(target_tensor.sum(-1).mean() - 1) < 1e-4):
        target_tensor = clip(target_tensor, min=1e-8, max=1 - 1e-8)
        is_target_onehot = True

    if target_tensor.dtype == tf.int64 and is_target_onehot == False:
        target_tensor = cast(make_onehot(target_tensor, num_classes=num_classes, axis=axis),output.dtype)

    intersection = reduce_sum(input_tensor * target_tensor *greater_equal(input_tensor,0.5,dtype=output.dtype)* sample_weight)
    union = reduce_sum((input_tensor *greater_equal(input_tensor,0.5,dtype=output.dtype)+ target_tensor) * sample_weight) - intersection
    return intersection/maximum(union,1)

def psnr(output, target):
    if int_shape(target)!=int_shape(output) :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    rmse = tf.math.sqrt(tf.math.reduce_mean((output - target) ** 2))
    return 20.0 * (tf.math.log(tf.math.divide_no_nan(255.0 , rmse))-tf.math.log(10.0))


def mean_absolute_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.reduce_mean(tf.math.abs( output -  target))
mae=mean_absolute_error


def mean_squared_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.reduce_mean(tf.math.square( output -  target))
mse=mean_squared_error




def root_mean_squared_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square( output -  target)))
rmse=root_mean_squared_error



def mean_squared_logarithmic_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.reduce_mean(tf.math.square(tf.math.log(1 + output)- tf.math.log(1 + target)))
msle=mean_squared_logarithmic_error


def get_metric(metric_name):
    if metric_name is None:
        return None
    metric_modules = ['trident.optims.tensorflow_metrics']
    try:
        metric_fn = get_function(camel2snake(metric_name), metric_modules)
    except Exception :
        metric_fn = get_function(metric_name, metric_modules)
    return metric_fn

