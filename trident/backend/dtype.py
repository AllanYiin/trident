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
    "float64", "double","long","float",
    "bool"]


if get_backend() == 'pytorch':
    import torch
    # type definition
    bool = torch.bool

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
    float= torch.float32
    complex64 = torch.complex64
    complex128 = torch.complex128
    cfloat = torch.cfloat


elif get_backend() == 'tensorflow':
    import tensorflow as tf
    bool = tf.bool

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
    float =  tf.float32
    complex64 = tf.complex64
    complex128 = tf.complex128
    cfloat = tf.complex64
elif get_backend() == 'onnx':
    from onnx import onnx_pb
    bool = onnx_pb.TensorProto.BOOL
    int8 = onnx_pb.TensorProto.INT8
    byte = onnx_pb.TensorProto.INT8
    int16 = onnx_pb.TensorProto.INT16
    short = onnx_pb.TensorProto.INT16
    int32 = onnx_pb.TensorProto.INT32
    intc = onnx_pb.TensorProto.INT32
    int64 = onnx_pb.TensorProto.INT64
    intp = onnx_pb.TensorProto.INT64

    uint8 = onnx_pb.TensorProto.UINT8
    ubyte = onnx_pb.TensorProto.UINT8
    float16 = onnx_pb.TensorProto.FLOAT1
    half = onnx_pb.TensorProto.FLOAT1
    float32 = onnx_pb.TensorProto.FLOAT
    single = onnx_pb.TensorProto.FLOAT
    float64 = onnx_pb.TensorProto.DOUBLE
    double = onnx_pb.TensorProto.DOUBLE
    long = onnx_pb.TensorProto.INT64
    float = onnx_pb.TensorProto.FLOAT
    complex64 = None
    complex128  =None
    cfloat  =None
else:
    bool = np.bool

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
    float= np.float32
    complex64 = np.complex64
    complex128 = np.complex128
    cfloat = np.complex64

