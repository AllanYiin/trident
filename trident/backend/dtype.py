import numpy as np
import typing

from trident.backend.common import get_backend


__all__=[
    "int8", "byte",
    "int16", "short",
    "int32", "intc",
    "int64", "intp",
    "uint8", "ubyte",
    "float16", "half",
    "float32", "single",
    "float64", "double","long"
    "bool_"]


if get_backend() == 'pytorch':
    import torch
    # type definition
    bool_ = torch.bool

    int8 = torch.int8
    byte = torch.int8
    int16 =  torch.int16
    short = torch.int16
    int32 =  torch.int32
    intc =  torch.int32
    int64 =  torch.int64
    intp = torch.int64

    uint8 = torch.uint8
    ubyte = torch.uint8
    float16 = torch.float16
    half = torch.float16
    float32 = torch.float32
    single =  torch.float32
    float64 =  torch.float64
    double = torch.float64
    long = torch.int64


elif get_backend() == 'tensorflow':
    import tensorflow as tf
    bool_ = tf.bool

    int8 = tf.int8
    byte = tf.int8
    int16 = tf.int16
    short = tf.int16
    int32 = tf.int32
    intc = tf.int32
    int64 = tf.int64
    intp = tf.int64

    uint8 = tf.uint8
    ubyte = tf.uint8
    float16 = tf.float16
    half = tf.float16
    float32 = tf.float32
    single = tf.float32
    float64 = tf.float64
    double = tf.float64
    long=tf.int64
else:
    bool_ = np.bool

    int8 = np.int8
    byte = np.int8
    int16 = np.int16
    short = np.int16
    int32 = np.int32
    intc = np.int32
    int64 = np.int64
    intp = np.int64

    uint8 = np.uint8
    ubyte = np.uint8
    float16 = np.float16
    half = np.float16
    float32 = np.float32
    single = np.float32
    float64 = np.float64
    double = np.float64
    long = np.int64

