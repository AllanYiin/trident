from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from collections import Sized, Iterable
from functools import partial
from typing import Tuple,List, Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  trident.backend.common import *

__all__ = ['is_tensor','to_numpy','to_tensor','ndim','int_shape','is_sparse','is_nan','is_inf','is_abnormal_number','any_nan','any_inf','any_abnormal_number','less','equal','greater','greater_equal','not_equal','less_equal','argmax','argmin','argsort','maximum','minimum','floor','ceil','round','dot','sqrt','square','abs','pow','log','exp','clip','add','subtract','true_divide','pi','matmul','sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','element_times','element_max','element_min','element_divide','element_cosine_distance','where','reduce_mean','reduce_sum','reduce_max','reduce_min','mean','sum','max','min','reduce_logsumexp','reduce_prod','depth_to_space','space_to_depth','identity','sigmoid','relu','relu6','leaky_relu','leaky_relu6','smooth_relu','p_relu','swish','elu','hard_sigmoid','hard_swish','selu','lecun_tanh','soft_sign','soft_plus','hard_tanh','logit','log_log','mish','softmax','log_softmax','bert_gelu','gpt_gelu','ones','ones_like','zeros','zeros_like','meshgrid','reshape','permute','transpose','squeeze','expand_dims','concate','stack','gram_matrix','shuffle','random_choice','get_rotation_matrix2d','warp_affine']



def _get_device():
    return get_session().device


def is_tensor(x):
    return isinstance(x,torch.Tensor)

def to_numpy(x) -> np.ndarray:

    """
    Convert whatever to numpy array
    :param x: List, tuple, PyTorch tensor or numpy array
    :return: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.clone().cpu().detach_().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, tuple):
        return np.array(list(x))
    elif  'int' in str(type(x)) or  'float' in str(type(x)):
        return np.array([x])
    elif x is None:
        return None
    else:
        raise ValueError("Unsupported type")

def to_tensor(x, dtype=torch.float32,requires_grad=None) -> torch.Tensor:
    ''''
     Convert input  to a tensor as possible
    Args:
        x (int,float,list,tuple,ndarray,tensor):
        dtype :
        requires_grad (bool): wheather need grade

    Returns: output tensor
    Examples:
        >>> to_tensor(2)
        tensor(2, dtype=torch.int32)
        >>> to_tensor([1.0,2.0,3.0],requires_grad=True)
        tensor([1., 2., 3.], requires_grad=True)
        >>> to_tensor([1.0,2.0,3.0],requires_grad=False)
        tensor([1., 2., 3.])
        >>> to_tensor([1.0,2.0,3.0])
        tensor([1., 2., 3.])
        >>> to_tensor((1.0,2.0,3.0))
        tensor([1., 2., 3.])
        >>> to_tensor(np.arange(0,5))
        tensor([0, 1, 2, 3, 4])

    '''
    if isinstance(x,  torch.Tensor):
        x = x.clone().detach()
        x = x.to(_get_device())
        if dtype is not None:
            x = x.type(dtype)
        if requires_grad ==False:
            x.requires_grad =False
        elif requires_grad ==True:
            x.requires_grad=True

        return x
    elif isinstance(x, int):
        return torch.tensor(x).int().to(_get_device()) if requires_grad is None else torch.tensor(x, requires_grad=requires_grad).int().to(_get_device())
    elif isinstance(x, float):
        return torch.tensor(x).float().to(_get_device()) if requires_grad is None else torch.tensor(x, requires_grad=requires_grad).float().to(_get_device())
    elif isinstance(x, (list, tuple)):
        if isinstance(x[0],int):
            x =torch.tensor(x).int() if requires_grad is None else torch.tensor(x,requires_grad=requires_grad).int()
        else:
            x=torch.tensor(x).float() if requires_grad is None else torch.tensor(x,requires_grad=requires_grad).float()
        x = x.to(_get_device())
        return x
    elif isinstance(x, np.ndarray):
        npdtype=x.dtype
        x = torch.tensor(x)
        if 'int' in str(npdtype):
            x = x.type(torch.int64)
        else:
            x = x.type(dtype)
        x = x.to(_get_device())
        if requires_grad == False:
            x.requires_grad = False
        elif requires_grad == True:
            x.requires_grad = True
        return x
    else:
        raise ValueError("Unsupported input type" + str(type(x)))

############################
## tensor attribute
###########################
def ndim(x:torch.Tensor):
    return x.ndim

def int_shape(x:torch.Tensor):
    '''

    Args:
        x : input tensor

    Returns: tuple of integer as shape representation

    Examples:
    >>> int_shape(ones((3,3,7)))
    (3, 3, 7)

    '''
    return tuple(list(x.size()))

def is_sparse(x):
    return 'sparse' in str(type(x))

def cast(x, dtype):
    if isinstance(dtype,torch.dtype):
       if dtype==torch.float64 or dtype==torch.double:
           return x.double()
       elif dtype==torch.float16 or dtype==torch.half:
           return x.float()
       elif dtype==torch.float32:
           return x.float()
       elif dtype==torch.int64:
           return x.long()
       elif dtype==torch.int32:
           return x.int()
       elif dtype==torch.int16:
           return x.short()
       elif dtype==torch.int8:
           return x.char()
       elif dtype==torch.uint8:
           return x.byte()
       elif dtype==torch.bool:
           return x.bool()
    elif isinstance(dtype,str):
        if 'float64' in  dtype.lower() or 'double' in  dtype.lower():
            return x.type
        elif 'float16' in  dtype.lower() or 'half' in  dtype.lower():
            return x.half()
        elif 'float' in  dtype.lower():
            return x.float()
        elif 'int64' in dtype.lower() or 'long' in dtype.lower():
            return x.long()
        elif 'int16' in dtype.lower() or 'short' in dtype.lower():
            return x.short()
        elif 'uint8' in dtype.lower() or 'byte' in dtype.lower():
            return x.byte()
        elif 'int8' in dtype.lower() or 'char' in dtype.lower():
            return x.char()
        elif 'int32' in dtype.lower() or 'int' in dtype.lower():
            return x.int()
        elif 'bool' in dtype.lower() :
            return x.bool()



############################
## check operation
###########################

def is_nan(x):
    if isinstance(x,torch.Tensor):
        return torch.isnan(x)
    elif isinstance(x,nn.Module):
        return [torch.isnan(para) for para in x.parameters()]
    elif isinstance(x, np.ndarray):
        return np.isnan(x)
    else:
        raise NotImplementedError

def is_inf(x):
    if isinstance(x,torch.Tensor):
        return torch.isinf(x)
    elif isinstance(x,nn.Module):
        return [torch.isinf(para) for para in x.parameters()]
    elif isinstance(x, np.ndarray):
        return np.isinf(x)
    else:
        raise NotImplementedError

def is_abnormal_number(x):
    return is_nan(x) or is_inf(x)

def any_nan(x):
    if isinstance(x,torch.Tensor):
        return torch.isnan(x).any()
    elif isinstance(x,nn.Module):
        for para in x.parameters():
            if torch.isnan(para).any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isnan(x).any()
    else:
        raise NotImplementedError

def any_inf(x):
    if isinstance(x,torch.Tensor):
        return torch.isinf(x).any()
    elif isinstance(x,nn.Module):
        for para in x.parameters():
            if torch.isinf(para).any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isinf(x).any()
    else:
        raise NotImplementedError

def any_abnormal_number(x):
    return any_nan(x) or any_inf(x)


############################
## compare operation
###########################


def less(left:torch.Tensor, right:torch.Tensor):
    '''
    Elementwise 'less' comparison of two tensors. Result is 1 if left < right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        Result is 1 if left < right else 0.

    Example:
   >>> less(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
   tensor([1., 0., 0.])
   >>> less(to_tensor([-1,0,1]), 0)
   tensor([1., 0., 0.])
    '''

    return left.lt(right).float()

def equal(left:torch.Tensor, right:torch.Tensor):
    '''
    Elementwise 'equal' comparison of two tensors. Result is 1 if values are equal 0 otherwise.
    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if values are equal 0 otherwise

    Example:
    >>> equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    tensor([0., 1., 0.])
    >>> equal(to_tensor([-1,0,1]), 1)
    tensor([0., 0., 1.])
    '''
    return left.eq(right).float()

def greater(left:torch.Tensor, right:torch.Tensor):
    '''
    Elementwise 'greater' comparison of two tensors. Result is 1 if left > right else 0.
    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left > right else 0.

    Example:
    >>> greater(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    tensor([0., 0., 1.])
    >>> greater(to_tensor([-1,0,1]), 0)
    tensor([0., 0., 1.])
    '''
    return left.gt(right).float()

def greater_equal(left:torch.Tensor, right:torch.Tensor):
    '''
    Elementwise 'greater equal' comparison of two tensors. Result is 1 if left >= right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left >= right else 0

    Example:
    >>> greater_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    tensor([0., 1., 1.])
    >>> greater_equal(to_tensor([-1,0,1]), 0)
    tensor([0., 1., 1.])
    '''
    return left.ge(right).float()

def not_equal(left:torch.Tensor, right:torch.Tensor):
    '''
    Elementwise 'not equal' comparison of two tensors. Result is 1 if left != right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left != right else 0.

    Example:
    >>> not_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    tensor([1., 0., 1.])
    >>> not_equal(to_tensor([-1,0,1]), 0)
    tensor([1., 0., 1.])
    '''
    return 1-(left.eq(right).float())

def less_equal(left:torch.Tensor, right:torch.Tensor):
    '''
    Elementwise 'less equal' comparison of two tensors. Result is 1 if left <= right else 0.

    Args:
        left: left side tensor
        right: right side tensor

    Returns:
        :Result is 1 if left <= right else 0.
    Example:
    >>> less_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    tensor([1., 1., 0.])
    >>> less_equal(to_tensor([-1,0,1]), 0)
    tensor([1., 1., 0.])

    '''
    return left.le(right).float()


def argmax(x:torch.Tensor,axis=1)-> torch.Tensor:
    if len(x.shape)>axis:
         _, idx = x.max(dim=axis)
    else:
        _, idx = x.max()
    return idx
def argmin(x:torch.Tensor,axis=1)-> torch.Tensor:
    if len(x.shape)>axis:
         _, idx = x.min(dim=axis)
    else:
        _, idx = x.min()
    return idx
def argsort(x:torch.Tensor,axis=1,descending=True)-> torch.Tensor:
    return torch.argsort(x,dim=axis,descending=descending)

def maximum(x:torch.Tensor,other:(torch.Tensor,int,float))-> torch.Tensor:
    if isinstance(other,torch.Tensor):
        return torch.max(x,other)
    elif isinstance(other,(int,float)):
        return x.clamp(min=other)
def minimum(x:torch.Tensor,other:(torch.Tensor,int,float))-> torch.Tensor:
    if isinstance(other,torch.Tensor):
        return torch.min(x,other)
    elif isinstance(other,(int,float)):
        return x.clamp(max=other)





############################
## basic math operation
###########################
def add(x, y):
    return torch.add(x,y)
def subtract(x, y):
    return torch.sub(x,y)
def dot(x, y):
    return torch.dot(x,y)
def true_divide(x, y):
    return torch.true_divide(x,y)
def pi():
    return to_tensor(np.pi)

def matmul(a,b,transpose_a=False,transpose_b=False):
    if transpose_a:
        a=a.T
    if transpose_b:
        b=b.T
    return torch.matmul(a,b)


def prod(x):
    return torch.prod(x)

def floor(x:(torch.Tensor,float)):
    if not is_tensor(x):
        x=to_tensor(x)
    return x.floor()

def ceil(x:(torch.Tensor,float)):
    if not is_tensor(x):
        x=to_tensor(x)
    return x.ceil()

def round(x:(torch.Tensor,float),digit:int=0):
    '''

    Args:
        x ():
        digit ():

    Returns:
    Examples;
    >>> round(to_tensor([[1,2,3,4,5]])/3,0)
    tensor([[0., 1., 1., 1., 2.]])
    >>> round(to_tensor([[1,2,3,4,5]])/3,2)
    tensor([[0.3300, 0.6700, 1.0000, 1.3300, 1.6700]])
     >>> round(to_tensor([[11.6,24.3,35.2,14.4,23.5]])/3,-1)
     tensor([[ 0., 10., 10.,  0., 10.]])

    '''
    if not is_tensor(x):
        x=to_tensor(x,dtype=torch.float32)
    if digit!=0:
        factor=to_tensor(float(math.pow(10,-1*digit)))
        return (x/factor).round()*factor
    else:
        return torch.round(x)

def sqrt(x:torch.Tensor):
    if not is_tensor(x):
        x=to_tensor(x,dtype=torch.float32)
    return x.sqrt()

def square(x:torch.Tensor):
    if not is_tensor(x):
        x=to_tensor(x,dtype=torch.float32)
    return x**2

def abs(x:torch.Tensor):
    if not is_tensor(x):
        x=to_tensor(x,dtype=torch.float32)
    return x.abs()

def pow(x:torch.Tensor,y):
    if not is_tensor(x):
        x=to_tensor(x,dtype=torch.float32)
    return x.pow(y)

def log(x:torch.Tensor):
    if not is_tensor(x):
        x=to_tensor(x,dtype=torch.float32)
    return x.log()

def exp(x:torch.Tensor):
    if not is_tensor(x):
        x=to_tensor(x,dtype=torch.float32)
    return x.exp()


def clip(x:torch.Tensor,min=-np.inf,max=np.inf):
    return x.clamp(min,max)


def sin(x:torch.Tensor):
    '''
    Computes the element-wise sine
    Args:
        x (tensor):input tensor

    Returns: element-wise sine

    Examples
    >>> sin(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[ 0.8415,  0.4794],
            [-0.2474, -0.6816]])
    '''
    return torch.sin(x.float())
def cos(x:torch.Tensor):
    '''
    Computes the element-wise cosine
    Args:
        x (tensor):input tensor

    Returns: element-wise cosine

    Examples
    >>> cos(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[0.5403, 0.8776],
            [0.9689, 0.7317]])
    '''
    return torch.cos(x.float())
def tan(x:torch.Tensor):
    '''
    Computes the element-wise tan
    Args:
        x (tensor):input tensor

    Returns: element-wise tan

    Examples
    >>> tan(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[ 1.5574,  0.5463],
            [-0.2553, -0.9316]])
    '''
    return torch.tan(x.float())


def asin(x:torch.Tensor):
    '''
    Computes the element-wise arcsin (inverse sine)
    Args:
        x (tensor):input tensor

    Returns: element-wise arcsin

    Examples
    >>> asin(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[ 1.5708,  0.5236],
            [-0.2527, -0.8481]])
    '''
    return torch.asin(x.float())
def acos(x:torch.Tensor):
    '''
    Computes the element-wise arccos (inverse cosine)
    Args:
        x (tensor):input tensor

    Returns: element-wise arccos

    Examples
    >>> acos(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[0.0000, 1.0472],
            [1.8235, 2.4189]])
    '''
    return torch.acos(x.float())
def atan(x:torch.Tensor):
    '''
    Computes the element-wise arctan (inverse tan)
    Args:
        x (tensor):input tensor

    Returns: element-wise arccos

    Examples
    >>> atan(to_tensor([-1, 0, 1])).cpu()
    tensor([-0.7854,  0.0000,  0.7854])
    '''
    return torch.atan(x.float())


def sinh(x:torch.Tensor):
    '''
    Computes the element-wise sinh
    Args:
        x (tensor):input tensor

    Returns: element-wise sinh

    Examples
    >>> sinh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[ 1.1752,  0.5211],
            [-0.2526, -0.8223]])
    '''
    return torch.sinh(x.float())
def cosh(x:torch.Tensor):
    '''
    Computes the element-wise cosh
    Args:
        x (tensor):input tensor

    Returns: element-wise cosh

    Examples
    >>> cosh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[1.5431, 1.1276],
            [1.0314, 1.2947]])
    '''
    return torch.cosh(x.float())
def tanh(x:torch.Tensor):
    '''
    Computes the element-wise tanh
    Args:
        x (tensor):input tensor

    Returns: element-wise tanh

    Examples
    >>> tanh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[ 0.7616,  0.4621],
            [-0.2449, -0.6351]])
    '''
    return torch.tanh(x.float())










############################
## elementwise operation
###########################

def element_times(left, right):
    '''
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Example:
    >>> element_times(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    tensor([0.5000, 0.2500, 0.1250, 0.0000])
    >>> element_times(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    tensor([10., 20., 30., 60.])
    >>> element_times(to_tensor([[5., 10.], [15., 30.]]), to_tensor([[1., 2.], [3.,1.]]))
    tensor([[ 5., 20.],
            [45., 30.]])
    '''
    return left*right

def element_max(left, right):
    '''
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Example:
    >>> element_max(to_tensor([1., 1., 0., -1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    tensor([1.0000, 1.0000, 0.1250, 0.0000])
    >>> element_max(to_tensor([5., 10., 15., 30.]),to_tensor([20.]))
    tensor([20., 20., 20., 30.])
    >>> element_max(to_tensor([5., 10., 15., 30.]), to_tensor([10., 2., 8., 2.]))
    tensor([10., 10., 15., 30.])
    '''
    return torch.max(left,right)

def element_min (left, right):
    '''
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Example:
    >>> element_min(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    tensor([0.5000, 0.2500, 0.1250, 0.0000])
    >>> element_min(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    tensor([2., 2., 2., 2.])
    >>> element_min(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
    tensor([1., 2., 1., 2.])
    '''
    return torch.min(left, right)

def element_divide (left, right):
    '''
    The output of this operation is the element-wise divide of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise divide of the two  input

    Example:
    >>> element_divide(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    tensor([2., 4., 8., inf])
    >>> element_divide(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    tensor([ 2.5000,  5.0000,  7.5000, 15.0000])
    >>> element_divide(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
    tensor([ 5.,  5., 15., 15.])
    '''
    return torch.true_divide(left, right)



def element_cosine_distance(v1, v2, axis=-1):
    reduce_dim = -1
    cos = (v1 * v2).sum(dim=reduce_dim,keepdims=False) /((v1 * v1).sum(dim=reduce_dim, keepdims=False).sqrt()*(v2 * v2).sum(dim=reduce_dim, keepdims=False).sqrt())
    return cos

def where(flag, value_if_true, value_if_false):
    '''
    return either ``value_if_true`` or ``value_if_false`` based on the value of ``flag``.
    If ``flag`` != 0 ``value_if_true`` is returned, otherwise ``value_if_false``.
    Behaves analogously to numpy.where(...).

    Args:
        flag: condition tensor
        value_if_true: true branch tensor
        value_if_false: false branch tensor
    Returns:
        :conditional selection

    Example:
    >>> x=to_tensor([0.1, 0.9, 0.8, 0.4, 0.5])
    >>> where(x>0.5, x, zeros_like(x))
    tensor([0.0000, 0.9000, 0.8000, 0.0000, 0.0000])
    '''
    return torch.where(flag, value_if_true, value_if_false)


############################
## reduce operation
###########################

def reduce_mean(x:torch.Tensor,axis=None,keepdims=False,**kwargs):
    '''
    Computes the mean of the input tensor's elements across a specified axis or a list of specified axes.

    Args:
        x (torch.Tensor):input tensor
        axis (int,list):  axis along which the reduction will be performed
        keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
        **kwargs ():

    Returns:


    Exsample:
    >>> data = to_tensor(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
    >>> print(reduce_mean(data, 0).cpu())
    tensor([[30.,  1.],
            [40.,  2.]])
    >>> print(reduce_mean(data, axis=0).cpu())
    tensor([[30.,  1.],
            [40.,  2.]])
    >>> print(reduce_mean(data, axis=[0,2]).cpu())
    tensor([15.5000, 21.0000])





    '''
    axis=kwargs.get('dim',axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if axis is None:
        return x.mean()
    elif isinstance(axis,int):
        return x.mean(dim=axis,keepdim=keepdims)
    elif isinstance(axis, list)  :
        axis=sorted(axis)
        axis.reverse()
        for a in axis:
            x=x.mean(dim=a,keepdim=keepdims)
        return x

def reduce_sum(x:torch.Tensor,axis=None,keepdims=False,**kwargs):
    '''
        Computes the sum of the input tensor's elements across a specified axis or a list of specified axes.

    Args:
        x (torch.Tensor):input tensor
        axis (int,list):  axis along which the reduction will be performed
        keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
        **kwargs ():

    Returns:


    Exsample:
    >>> data = to_tensor(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
    >>> print(reduce_sum(data).cpu())
    tensor(219.)
    >>> print(reduce_sum(data, 0).cpu())
    tensor([[ 90.,   3.],
            [120.,   6.]])
    >>> print(reduce_sum(data, axis=0).cpu())
    tensor([[ 90.,   3.],
            [120.,   6.]])
    >>> print(reduce_sum(data, axis=[0,2]).cpu())
    tensor([ 93., 126.])
        '''
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if axis is None:
        return x.sum()
    elif isinstance(axis, int):
        return x.sum(dim=axis, keepdim=keepdims)
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            x = x.sum(dim=a, keepdim=keepdims)
        return x

def reduce_max(x:torch.Tensor,axis=None,keepdims=False,**kwargs):
    '''
        Computes the maximum of the input tensor's elements across a specified axis or a list of specified axes.

        Args:
            x (torch.Tensor):input tensor
            axis (int,list):  axis along which the reduction will be performed
            keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
            **kwargs ():

        Returns:


        Exsample:
        >>> data = to_tensor(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
        >>> print(reduce_max(data, 0).cpu())
        tensor([[55.,  1.],
                [60.,  2.]])
        >>> print(reduce_max(data, axis=0).cpu())
        tensor([[55.,  1.],
                [60.,  2.]])
        >>> print(reduce_max(data, axis=[0,2]).cpu())
        tensor([55., 60.])


        '''
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if axis is None:
        return x.max()
    elif isinstance(axis, int):
        arr, idx = x.max(dim=axis, keepdim=keepdims)
        return arr
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = x.max(dim=a, keepdim=keepdims)
            x = arr
        return x

def reduce_min(x:torch.Tensor,axis=None,keepdims=False,**kwargs):
    '''
    Computes the minimum of the input tensor's elements across a specified axis or a list of specified axes.

    Args:
        x (torch.Tensor):input tensor
        axis (int,list):  axis along which the reduction will be performed
        keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
        **kwargs ():

    Returns:


    Exsample:
    >>> data = to_tensor(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
    >>> print(reduce_min(data, 0).cpu())
    tensor([[ 5.,  1.],
            [20.,  2.]])
    >>> print(reduce_min(data, axis=0).cpu())
    tensor([[ 5.,  1.],
            [20.,  2.]])
    >>> print(reduce_min(data, axis=[0,2]).cpu())
    tensor([1., 2.])

        '''
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if axis is None:
        return x.min()
    elif isinstance(axis, int):
        arr, idx = x.min(dim=axis, keepdim=keepdims)
        return arr
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = x.min(dim=a, keepdim=keepdims)
            x = arr
        return x


def reduce_logsumexp(x:torch.Tensor,axis=None,keepdims=False,**kwargs):
    '''

     Args:
         x ():
         axis ():
         keepdims ():

     Returns:
     Examples
     >>> x = to_tensor([[0., 0., 0.], [0., 0., 0.]])
     >>> reduce_logsumexp(x)
     tensor(1.7918)
     >>> reduce_logsumexp(x, 0)
     tensor([0.6931, 0.6931, 0.6931])
     >>> reduce_logsumexp(x, [0, 1])
     tensor(1.7918)
     '''
    if axis is None:
        return log(reduce_sum(exp(x)))
    else:
        return log(reduce_sum(exp(x),axis=axis,keepdims=keepdims))

def reduce_prod(x:torch.Tensor,axis=None,keepdims=False,**kwargs):
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if isinstance(axis, int):
        arr, idx = x.prod(dim=axis, keepdim=keepdims)
        return arr
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = x.prod(dim=a, keepdim=keepdims)
            x = arr
        return x



#reduce_log_sum_exp
#reduce_prod
#reduce_l1
#reduce_l2
#reduce_sum_square

mean=reduce_mean
sum=reduce_sum
max=reduce_max
min=reduce_min




############################
## activationoperation
###########################


def identity(x):
    return x



def relu(x):
    '''relu activation function
    '''
    return torch.relu(x)

def relu6(x):
    '''relu6 activation function
    '''
    return F.relu6(x)


def leaky_relu(x,slope=0.2):
    '''leaky_relu activation function
    '''
    return F.leaky_relu(x, negative_slope=slope)

def leaky_relu6(x,slope=0.2):
    '''leaky_relu6 activation function
    '''
    return torch.clamp(F.leaky_relu(x, negative_slope=slope), -6, 6)

def smooth_relu(x):
    '''smooth_relu activation function
    '''
    return torch.log(1 + torch.exp(x))

'''p_relu activation function 
'''



def p_relu(x,weight):
    '''

    Args:
        x ():
        weight ():

    Returns:

    '''
    return torch.prelu(x,weight=weight)


def sigmoid(x):
    '''softmax activation function
    '''
    return torch.sigmoid(x)



'''swish activation function 
'''


def swish(x):
    return x * sigmoid(x)


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)

def hard_tanh(x):
    return torch.clamp(x, -1, 1)

def selu(x):
    '''
    selu activation function
    Scaled exponential linear unit operation. Computes the element-wise exponential linear
    of ``x``: ``scale * x`` for ``x >= 0`` and ``x``: ``scale * alpha * (exp(x)-1)`` otherwise.
    scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717

    paper: https://arxiv.org/abs/1706.02515
    Self-Normalizing Neural Networks
    Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter

    Args:
        x (tensor): input tensor

    Returns:The output tensor has the same shape as ``x``
    Example:
        >>> selu(to_tensor([[-1, -0.5, 0, 1, 2]]))
        tensor([[-1.1113, -0.6918,  0.0000,  1.0507,  2.1014]])
    '''
    return torch.selu(x)


def elu(x):
    return F.elu(x)

def lecun_tanh(x):
    return 1.7159 * torch.tanh(2 / 3 * x)


def soft_sign(x):
    return x.exp().add(1).log()


def soft_plus(x):
    return F.softplus(x)


def logit(x):
    return (x / (1 - x)).log()

def log_log(x):
    return 1 - torch.exp(-torch.exp(x))


def mish(x):
    '''
        mish activation function
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1
    Args:
        x ():

    Returns:

    '''
    return x * (torch.tanh(F.softplus(x)))


def softmax(x,axis=1):
    '''
    Computes the gradient of :math:`f(z)=\log\sum_i\exp(z_i)` at ``z = x``. Concretely,
    :math:`\mathrm{softmax}(x)=\left[\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\ldots\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\right]`
    with the understanding that the implementation can use equivalent formulas
    for efficiency and numerical stability.
    The output is a vector of non-negative numbers that sum to 1 and can
    therefore be interpreted as probabilities for mutually exclusive outcomes
    as in the case of multiclass classification.
    If ``axis`` is given as integer, then the softmax will be computed along that axis.
    If the provided ``axis`` is -1, it will be computed along the last axis. Otherwise,
    softmax will be applied to all axes.

    Args:
        x: input tensor
        axis (int) : axis along which the softmax operation will be performed

    Returns:
        :output tensor

    Example:
    >>> softmax(to_tensor([[1, 1, 2, 3]]))
    tensor([[0.0826, 0.0826, 0.2245, 0.6103]])
    >>> softmax(to_tensor([1., 1.]))
    tensor([0.5000, 0.5000])
    >>> softmax(to_tensor([[[1, 1], [3, 5]]]), axis=-1)
    tensor([[[0.5000, 0.5000],
             [0.1192, 0.8808]]])
    >>> softmax(to_tensor([[[1, 1], [3, 5]]]), axis=1)
    tensor([[[0.1192, 0.0180],
             [0.8808, 0.9820]]])

    '''
    if x.ndim==1:
        return  x.exp().true_divide(x.exp().sum().clamp(min=epsilon()))
    return torch.softmax(x.float(), dim=axis)


def log_softmax(x,axis=1):
    '''
    Computes the logsoftmax normalized values of x. That is, y = x - log(reduce_sum(exp(x), axis))
    (the implementation uses an equivalent formula for numerical stability).
    It is also possible to use `x - reduce_log_sum_exp(x, axis)` instead of log_softmax:
    this can be faster (one reduce pass instead of two), but can behave slightly differently numerically.
    Args:
        x: a tensor
        axis (int): axis along which the logsoftmax operation will be performed (the default is the last axis)

    Returns:
        :output tensor
    '''
    return x -reduce_logsumexp(x)





def bert_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gpt_gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



############################
## tensor shape operation
###########################
def reshape(x,shape=None)-> torch.Tensor:
    if shape is None:
        return x
    elif isinstance(shape,list):
        return torch.reshape(x,to_list(shape))
    elif isinstance(shape,tuple):
        shape=to_list(shape)
        return torch.reshape(x,shape)
    else:
        return x


def transpose(x,pattern=None)-> torch.Tensor:
    return x.transpose(pattern) if x.is_contiguous() else x.transpose(pattern).contiguous()

def permute(x,pattern=None)-> torch.Tensor:
    return x.permute(pattern) if x.is_contiguous() else x.permute(pattern).contiguous()




def squeeze(t:torch.Tensor,axis=0):
    return t.squeeze(axis)

def expand_dims(t:torch.Tensor,axis=0):
    return t.unsqueeze(axis)

def depth_to_space(x:torch.Tensor,block_size=2):
    '''
    Rearranges elements in the input tensor from the depth dimension into spatial blocks.
    The equivalent to Pixel-Shuffle

    Args:
        x (tensor): Input tensor, with dimensions CHW or NCHW
        block_size (int):

    Returns: resized tensor

    Examples
    >>> x = to_tensor(np.tile(np.array(np.reshape(range(8), (8, 1, 1)), dtype=np.float32), (1, 2, 3)))
    >>> x
    tensor([[[0., 0., 0.],
             [0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [1., 1., 1.]],
    <BLANKLINE>
            [[2., 2., 2.],
             [2., 2., 2.]],
    <BLANKLINE>
            [[3., 3., 3.],
             [3., 3., 3.]],
    <BLANKLINE>
            [[4., 4., 4.],
             [4., 4., 4.]],
    <BLANKLINE>
            [[5., 5., 5.],
             [5., 5., 5.]],
    <BLANKLINE>
            [[6., 6., 6.],
             [6., 6., 6.]],
    <BLANKLINE>
            [[7., 7., 7.],
             [7., 7., 7.]]])
    >>> arr=depth_to_space(x,block_size=2)
    >>> print(arr.shape)
    torch.Size([2, 4, 6])
    >>> arr
    tensor([[[0., 1., 0., 1., 0., 1.],
             [2., 3., 2., 3., 2., 3.],
             [0., 1., 0., 1., 0., 1.],
             [2., 3., 2., 3., 2., 3.]],
    <BLANKLINE>
            [[4., 5., 4., 5., 4., 5.],
             [6., 7., 6., 7., 6., 7.],
             [4., 5., 4., 5., 4., 5.],
             [6., 7., 6., 7., 6., 7.]]])

    '''
    if ndim(x)  not in (3,4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-3]%(block_size*block_size)!=0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim=ndim(x)
        if orig_ndim ==3:
            x=expand_dims(x,0)
        x=F.pixel_shuffle(x, block_size)
        if orig_ndim == 3:
            return x[0]
        return x

def space_to_depth(x:torch.Tensor,block_size=2):
    '''
    Rearranges elements in the input tensor from the spatial dimensions to the depth dimension.

    This is the reverse transformation of depth_to_space. This operation is useful for implementing and testing sub-pixel convolution that is part of models for image super-resolution .
    It rearranges elements of an input tensor of shape (N, C, H, W) to a tensor of shape (N, C*b*b, H/b, W/b), where b is the block_size,
    by rearranging non-overlapping spatial blocks of size block_size x block_size into the depth/channel dimension at each location.

    Args:
        x (tensor): Input tensor, with dimensions CHW or NCHW
        block_size (int):

    Returns: resized tensor
    Examples
    >>> arr=space_to_depth(to_tensor([[[0., 1., 0., 1., 0., 1.],[2., 3., 2., 3., 2., 3.],[0., 1., 0., 1., 0., 1.],[2., 3., 2., 3., 2., 3.]],[[4., 5., 4., 5., 4., 5.],[6., 7., 6., 7., 6., 7.], [4., 5., 4., 5., 4., 5.],[6., 7., 6., 7., 6., 7.]]]),block_size=2)
    >>> arr
    tensor([[[0., 0., 0.],
             [0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [1., 1., 1.]],
    <BLANKLINE>
            [[2., 2., 2.],
             [2., 2., 2.]],
    <BLANKLINE>
            [[3., 3., 3.],
             [3., 3., 3.]],
    <BLANKLINE>
            [[4., 4., 4.],
             [4., 4., 4.]],
    <BLANKLINE>
            [[5., 5., 5.],
             [5., 5., 5.]],
    <BLANKLINE>
            [[6., 6., 6.],
             [6., 6., 6.]],
    <BLANKLINE>
            [[7., 7., 7.],
             [7., 7., 7.]]])
    >>> print(arr.shape)
    torch.Size([8, 2, 3])
    '''
    if ndim(x)  not in (3,4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-2]%block_size!=0 or  x.shape[-1]%block_size!=0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim=ndim(x)
        if orig_ndim ==3:
            x=expand_dims(x,0)
        orig_shape=list(int_shape(x))
        x=reshape(x,(orig_shape[0],orig_shape[1],orig_shape[2]//block_size,block_size,orig_shape[3]//block_size,block_size))
        x=permute(x,[0,1,3,5,2,4])
        x=reshape(x,(orig_shape[0],orig_shape[1]*block_size*block_size,orig_shape[2]//block_size,orig_shape[3]//block_size))
        if orig_ndim == 3:
            return x[0]
        return x








############################
## tensor generation
###########################

def ones(shape,dtype=torch.float32,requires_grad=False):
    return torch.ones(shape,dtype=dtype,requires_grad=requires_grad).to(_get_device())

def ones_like(a,dtype=torch.float32,requires_grad=False):
    return torch.ones(a.shape,dtype=dtype,requires_grad=requires_grad).to(_get_device())

def zeros(shape,dtype=torch.float32,requires_grad=False):
    return torch.zeros(shape,dtype=dtype,requires_grad=requires_grad).to(_get_device())

def zeros_like(a,dtype=torch.float32,requires_grad=False):
    return torch.zeros(a.shape,dtype=dtype,requires_grad=requires_grad).to(_get_device())

def eye_like(a,dtype=torch.float32,requires_grad=False):
    '''
    Creates a matrix with diagonal set to 1s and of the same shape and the same dynamic axes as ``x``. To be a
    matrix,
     ``x`` must have exactly two axes (counting both dynamic and static axes).

    Args:
        a: numpy array or  that outputs a tensor of rank 2
        requires_grad ():
        dtype ():

    Returns:
        :class:`~cntk.ops.functions.Function`

    Example:
    >>> eye_like(torch.Tensor(3,4))
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]])

    '''
    if a.ndim==2:
        return torch.eye(a.shape[0],a.shape[1],dtype=dtype,requires_grad=requires_grad).to(_get_device())
    else:
        raise ValueError('input tensor must have exactly two axe.')

def one_hot(a, num_classes, axis=-1):
    '''
    Create one hot tensor based on the input tensor
    Args:
        a: input tensor, the value must be positive integer and less than num_class
        num_classes: the number of class in one hot tensor
        axis: The axis to fill (default: -1, a new inner-most axis).
    Returns:
        :onehot tensor
    Example:
    >>> one_hot(to_tensor([[1, 2],[1, 3]]).long(), 4, axis=-1)
    tensor([[[0., 1., 1., 0.],
             [0., 1., 0., 1.]],
    <BLANKLINE>
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.]]])

    '''

    one_hot_shape = list(a.size())
    flattend_a=a.view(-1)
    one_hot_result = zeros((len(flattend_a),num_classes)).float()
    target = one_hot_result.scatter_(1,a,1)
    target=target.view(*one_hot_shape,num_classes).contiguous()
    return target

def meshgrid(x, y, normalized_coordinates=False,requires_grad=False):
    '''Return meshgrid in range x & y.

    Args:
      requires_grad ():
      normalized_coordinates ():
      x: (int) first dim range.
      y: (int) second dim range.

    Returns:
      (tensor) meshgrid, sized [y,x,2]

    Example:
    >>> grid=meshgrid(3,2)
    >>> grid.cpu()
    tensor([[[0., 0.],
             [0., 1.]],
    <BLANKLINE>
            [[1., 0.],
             [1., 1.]],
    <BLANKLINE>
            [[2., 0.],
             [2., 1.]]])
    >>> print(grid[0,0,:].cpu())
     tensor([0., 0.])
    >>> print(grid[:,0,0].cpu())
    tensor([0., 1., 2.])
    >>> print(grid.shape)
    torch.Size([3, 2, 2])


    >>> grid1=meshgrid(3,2,normalized_coordinates=True)
    >>> grid1.cpu()
    tensor([[[0.0000, 0.0000],
             [0.0000, 1.0000]],
    <BLANKLINE>
            [[0.5000, 0.0000],
             [0.5000, 1.0000]],
    <BLANKLINE>
            [[1.0000, 0.0000],
             [1.0000, 1.0000]]])
    >>> grid1.shape
    torch.Size([3, 2, 2])
    '''
    xs = torch.linspace(0, int(x - 1), int(x), device=_get_device(), dtype=torch.float, requires_grad=requires_grad)
    ys = torch.linspace(0, int(y - 1), int(y), device=_get_device(), dtype=torch.float, requires_grad=requires_grad)
    if normalized_coordinates:
        xs = torch.linspace(0, 1, int(x), device=_get_device(), dtype=torch.float, requires_grad=requires_grad)
        ys = torch.linspace(0, 1, int(y), device=_get_device(), dtype=torch.float, requires_grad=requires_grad)

    return torch.stack(torch.meshgrid([ys, xs]),-1).to(_get_device())




############################
## tensor manipulation
###########################

def concate(x:List[torch.Tensor],axis=1):
    return torch.cat(x,dim=axis)

def stack(x:List[torch.Tensor],axis=1):
    return torch.stack(x,dim=axis)


def gram_matrix(x:torch.Tensor):
    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL
    features=features-features.mean(-1)
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G#.div(a * b * c * d)






############################
## random
###########################


def shuffle(t:torch.Tensor):
    order = np.random.shuffle(np.array(range(t.size(0))))
    t[np.array(range(t.size(0)))] = t[order]
    return t

def random_choice(t:torch.Tensor):
    idx = np.random.choice(np.array(range(t.size(0))))
    return t[idx]


############################
## loss
###########################

def binary_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = output.sigmoid_()
    output = output.clamp_( epsilon(), 1.0 - epsilon())
    output = -target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)
    return output





def torch_rot90_(x: torch.Tensor):
    return x.transpose_(2, 3).flip(2)


def torch_rot90(x: torch.Tensor):
    return x.transpose(2, 3).flip(2)


def torch_rot180(x: torch.Tensor):
    return x.flip(2).flip(3)


def torch_rot270(x: torch.Tensor):
    return x.transpose(2, 3).flip(3)


def torch_flipud(x: torch.Tensor):
    """
    Flip image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2)


def torch_fliplr(x: torch.Tensor):
    """
    Flip image tensor horizontally
    :param x:
    :return:
    """
    return x.flip(3)






def pad_image_tensor(image_tensor: torch.Tensor, pad_size: int = 32):
    """Pad input tensor to make it's height and width dividable by @pad_size

    :param image_tensor: Input tensor of shape NCHW
    :param pad_size: Pad size
    :return: Tuple of output tensor and pad params. Second argument can be used to reverse pad operation of model output
    """
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    if (
        isinstance(pad_size, Sized)
        and isinstance(pad_size, Iterable)
        and len(pad_size) == 2
    ):
        pad_height, pad_width = [int(val) for val in pad_size]
    elif isinstance(pad_size, int):
        pad_height = pad_width = pad_size
    else:
        raise ValueError(
            "Unsupported pad_size: {pad_size}, must be either tuple(pad_rows,pad_cols) or single int scalar."
        )

    if rows > pad_height:
        pad_rows = rows % pad_height
        pad_rows = pad_height - pad_rows if pad_rows > 0 else 0
    else:
        pad_rows = pad_height - rows

    if cols > pad_width:
        pad_cols = cols % pad_width
        pad_cols = pad_width - pad_cols if pad_cols > 0 else 0
    else:
        pad_cols = pad_width - cols

    if pad_rows == 0 and pad_cols == 0:
        return image_tensor, (0, 0, 0, 0)

    pad_top = pad_rows // 2
    pad_btm = pad_rows - pad_top

    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    pad = [pad_left, pad_right, pad_top, pad_btm]
    image_tensor = torch.nn.functional.pad(image_tensor, pad)
    return image_tensor, pad


def unpad_image_tensor(image_tensor, pad):
    pad_left, pad_right, pad_top, pad_btm = pad
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    return image_tensor[..., pad_top : rows - pad_btm, pad_left : cols - pad_right]


def unpad_xyxy_bboxes(bboxes_tensor: torch.Tensor, pad, dim=-1):
    pad_left, pad_right, pad_top, pad_btm = pad
    pad = torch.tensor(
        [pad_left, pad_top, pad_left, pad_top], dtype=bboxes_tensor.dtype
    ).to(bboxes_tensor.device)

    if dim == -1:
        dim = len(bboxes_tensor.size()) - 1

    expand_dims = list(set(range(len(bboxes_tensor.size()))) - {dim})
    for i, dim in enumerate(expand_dims):
        pad = pad.unsqueeze(dim)

    return bboxes_tensor - pad







def angle_to_rotation_matrix(angle) -> torch.Tensor:
    """
    Creates a rotation matrix out of angles in degrees
    Args:
        angle: (torch.Tensor): tensor of angles in degrees, any shape.

    Returns:
        torch.Tensor: tensor of *x2x2 rotation matrices.

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*, 2, 2)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad =angle*np.pi/180
    cos_a= torch.cos(ang_rad)
    sin_a= torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(center: torch.Tensor,angle,scale) -> torch.Tensor:
    r"""Calculates an affine matrix of 2D rotation.

    The function calculates the following matrix:

    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}

    where

    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})

    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.

    Args:
        center (Tensor,tuple): center of the rotation in the source image.
        angle (Tensor,float): rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor,float): isotropic scale factor.

    Returns:
        Tensor: the affine matrix of 2D rotation.

    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> angle = 45. * torch.ones(1)
        >>> get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    center=to_tensor(center)
    angle = to_tensor(angle)
    scale = to_tensor(scale)

    if len(center)==2 and ndim(center)==1:
        center=center.unsqueeze(0)
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}"
                         .format(center.shape))

    # convert angle and apply scale
    scaled_rotation = angle_to_rotation_matrix(angle) * scale.view(-1, 1, 1)
    alpha= scaled_rotation[:, 0, 0]
    beta= scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x = center[..., 0]
    y= center[..., 1]

    # create output tensor
    batch_size= center.shape[0]
    M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (torch.tensor(1.) - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (torch.tensor(1.) - alpha) * y
    return M







def _compute_rotation_matrix(angle: torch.Tensor,
                             center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale_tensor = torch.ones_like(angle)
    matrix_tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix_tensor


def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for translation."""
    matrix_tensor = torch.eye( 3, device=translation.device, dtype=translation.dtype)
    matrix = matrix_tensor.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_scaling_matrix(scale: torch.Tensor,
                            center: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for scaling."""
    angle_tensor= torch.zeros_like(scale)
    matrix_tensor = get_rotation_matrix2d(center, angle_tensor, scale)
    return matrix_tensor


def _compute_shear_matrix(shear: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for shearing."""
    matrix_tensor = torch.eye(3, device=shear.device, dtype=shear.dtype)
    matrix = matrix_tensor.repeat(shear.shape[0], 1, 1)

    shx, shy = torch.chunk(shear, chunks=2, dim=-1)
    matrix[..., 0, 1:2] += shx
    matrix[..., 1, 0:1] += shy
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166

def normal_transform_pixel(height, width):
    tr_mat = torch.Tensor([[1.0, 0.0, -1.0],[0.0, 1.0, -1.0],[0.0, 0.0, 1.0]])  # 1x3x3
    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)

    tr_mat = tr_mat.unsqueeze(0)

    return tr_mat


def dst_norm_to_dst_norm(dst_pix_trans_src_pix, dsize_src, dsize_dst):
    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    # the devices and types
    device = dst_pix_trans_src_pix.device
    dtype = dst_pix_trans_src_pix.dtype
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(
        src_h, src_w).to(device).to(dtype)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(
        dst_h, dst_w).to(device).to(dtype)
    # compute chain transformations
    dst_norm_trans_src_norm = torch.matmul(
        dst_norm_trans_dst_pix, torch.matmul(
            dst_pix_trans_src_pix, src_pix_trans_src_norm))
    return dst_norm_trans_src_norm

def transform_points(trans_01: torch.Tensor,points_1: torch.Tensor) -> torch.Tensor:

    r"""Function that applies transformations to a set of points.
    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.
    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """
    if not torch.is_tensor(trans_01) or not torch.is_tensor(points_1):
        raise TypeError("Input type is not a torch.Tensor")
    if not trans_01.device == points_1.device:
        raise TypeError("Tensor must be in the same device")
    if not trans_01.shape[0] == points_1.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differe by one unit")
    # to homogeneous
    points_1_h = torch.nn.functional.pad(points_1, [0, 1], "constant", 1.0)
    # transform coordinates
    points_0_h = torch.matmul(
        trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    z_vec_tensor = points_0_h[..., -1:]
    mask_tensor = torch.abs(z_vec_tensor) >  1e-8
    scale_tensor= torch.ones_like(z_vec_tensor).masked_scatter_(mask_tensor, torch.tensor(1.0) / z_vec_tensor[mask_tensor])

    return scale_tensor * points_0_h[..., :-1]

def warp_grid(dst_homo_src: torch.Tensor,dsize) -> torch.Tensor:
    r"""Computes the grid to warp the coordinates grid by an homography.

    Args:
        dst_homo_src (torch.Tensor): Homography or homographies (stacked) to
                          transform all points in the grid. Shape of the
                          homography has to be :math:`(N, 3, 3)`.

    Returns:
        torch.Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
    """
    height, width = dsize
    grid = meshgrid(height, width, normalized_coordinates=True)

    batch_size= dst_homo_src.shape[0]
    device= dst_homo_src.device
    dtype= dst_homo_src.dtype
    # expand grid to match the input batch size
    grid_tensor = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
    if len(dst_homo_src.shape) == 3:  # local homography case
        dst_homo_src = dst_homo_src.view(batch_size, 1, 3, 3)  # NxHxWx3x3
    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow_tensor = transform_points(dst_homo_src, grid_tensor.to(device).to(dtype))  # NxHxWx2
    return flow_tensor.view(batch_size, height, width, 2)  # NxHxWx2

def warp_affine(src: torch.Tensor,
                M: torch.Tensor,
                dsize: Tuple[int, int],
                mode: Optional[str] = 'bilinear',
                padding_mode: Optional[str] = 'zeros') -> torch.Tensor:
    r"""Applies an affine transformation to a tensor.

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )

    Args:
        src (torch.Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (torch.Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (Optional[str]): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (Optional[str]): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.

    Returns:
        torch.Tensor: the warped tensor.

    Shape:
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://github.com/arraiyopensource/
       kornia/blob/master/docs/source/warp_affine.ipynb>`__.
    """
    if not torch.is_tensor(src):
        raise TypeError("Input src type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(M):
        raise TypeError("Input M type is not a torch.Tensor. Got {}"
                        .format(type(M)))
    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))
    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}"
                         .format(src.shape))
    try:
        # we generate a 3x3 transformation matrix from 2x3 affine
        M_3x3_tensor= F.pad(M, [0, 0, 0, 1, 0, 0], mode="constant", value=0)
        M_3x3_tensor[:, 2, 2] += 1.0

        dst_norm_trans_dst_norm =dst_norm_to_dst_norm(M_3x3_tensor, (src.shape[-2:]), dsize)
        # launches the warper
        return F.grid_sample(src, warp_grid(torch.inverse(dst_norm_trans_dst_norm),dsize=dsize), mode= 'bilinear', padding_mode= 'zeros')
    except Exception:
        PrintException()
        return None

def affine(tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    r"""Apply an affine transformation to the image.

    Args:
        tensor (torch.Tensor): The image tensor to be warped.
        matrix (torch.Tensor): The 2x3 affine transformation matrix.

    Returns:
        torch.Tensor: The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    warped_tensor = warp_affine(tensor, matrix, (height, width))

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped_tensor, dim=0)

    return warped_tensor


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185

def rotate(tensor: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    r"""Rotate the image anti-clockwise about the centre.

    See :class:`~kornia.Rotate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))

    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))


    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = torch.tensor([(tensor.size(4)-1)/2,(tensor.size(3)-1)/2]).expand(tensor.shape[0], -1).to(tensor.device)
    rotation_matrix = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3])


def translate(tensor: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    r"""Translate the tensor in pixel units.

    See :class:`~kornia.Translate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(translation):
        raise TypeError("Input translation type is not a torch.Tensor. Got {}"
                        .format(type(translation)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3])


def scale(tensor: torch.Tensor, scale_factor: torch.Tensor) -> torch.Tensor:
    r"""Scales the input image.

    See :class:`~kornia.Scale` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(scale_factor):
        raise TypeError("Input scale_factor type is not a torch.Tensor. Got {}"
                        .format(type(scale_factor)))

    # compute the tensor center

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    center = torch.tensor([(tensor.size(4) - 1) / 2, (tensor.size(3) - 1) / 2]).expand(tensor.shape[0], -1).to(tensor.device)
    scale_factor = scale_factor.expand(tensor.shape[0])
    scaling_matrix = _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix[..., :2, :3])


def shear(tensor: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
    r"""Shear the tensor.

    See :class:`~kornia.Shear` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(shear):
        raise TypeError("Input shear type is not a torch.Tensor. Got {}"
                        .format(type(shear)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    shear_matrix = _compute_shear_matrix(shear)

    # warp using the affine transform
    return affine(tensor, shear_matrix[..., :2, :3])


