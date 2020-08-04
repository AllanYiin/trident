"""Activation Layers"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import builtins
import inspect

import string
from functools import partial
from pydoc import locate

import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as af
from torch.nn.parameter import Parameter

from trident.backend.common import get_function, get_class, camel2snake,snake2camel, enforce_singleton
from trident.backend.pytorch_backend import Layer
from trident.backend.pytorch_ops import *

__all__ = ['Identity', 'Sigmoid', 'Tanh', 'Relu', 'Relu6', 'LeakyRelu', 'LeakyRelu6', 'SmoothRelu', 'PRelu', 'Swish',
           'Elu', 'HardSigmoid', 'HardSwish', 'Selu', 'LecunTanh', 'SoftSign', 'SoftPlus', 'HardTanh', 'Logit',
           'LogLog', 'Mish', 'Softmax', 'Gelu', 'GptGelu','SIREN', 'LogSoftmax', 'get_activation']


class Identity(Layer):
    """
    Identity activation Layer
    A placeholder identity operator that is argument-insensitive.

    Examples:
        >>> Identity()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-3.0, -1.0, 0.0, 2.0])

    """
    def __init__(self, name=None):
        super(Identity, self).__init__(name=name)
        self._built = True

    def forward(self, x):
        return x


class Relu(Layer):
    """Rectified Linear Unit activation function.

    With default values, it returns element-wise max(x, 0).
    Otherwise, it follows:

        ```
        f(x) = max_value if x >= max_value
        f(x) = x if threshold <= x < max_value
        f(x) = negative_slope * (x - threshold) otherwise

        ```

    Examples:
        >>> Relu()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, name=None):
        super(Relu, self).__init__(name=name)
        self._built = True

    def forward(self, *x):
        """
        Args:
        x: Input tensor.

        Returns: output tensor

        """
        x = enforce_singleton(x)
        return relu(x)


class Relu6(Layer):
    """Rectified Linear Unit  6 activation function.

    With default values, it returns element-wise min(max(x, 0),6).
    Otherwise, it follows:

            ```
            f(x) = 6 if x >= 6
            f(x) = x if threshold <= x < 6
            f(x) = negative_slope * (x - threshold) otherwise

            ```

    Examples:
        >>> Relu6()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, name=None):
        super(Relu6, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return relu6(x)


class LeakyRelu(Layer):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:

          ```
            f(x) = alpha * x if x < 0
            f(x) = x if x >= 0

          ```

    Examples:
        >>> LeakyRelu()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, alpha=0.2, name=None):
        super(LeakyRelu, self).__init__()
        self.alpha = alpha
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return leaky_relu(x, self.alpha)

    def extra_repr(self):
        s = 'alpha={alpha}'
        return s.format(**self.__dict__)


class LeakyRelu6(Layer):
    """Leaky version of a Rectified Linear Unit.6
    It allows a small gradient when the unit is not active:
          ```
            f(x) = alpha * x if x < 0
            f(x) = x if  6>=x >= 0
            f(x) = 6 if  x > 6

          ```

    Examples:
        >>> LeakyRelu6()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, name=None):
        super(LeakyRelu6, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return leaky_relu6(x)

    def extra_repr(self):
        s = 'alpha={alpha}'
        return s.format(**self.__dict__)


class SmoothRelu(Layer):
    """Smooth_relu activation Layer

    Examples:
        >>> SmoothRelu()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, name=None):
        super(SmoothRelu, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return smooth_relu(x)


class PRelu(Layer):
    """Parametric Rectified Linear Unit.

    It follows:
        ```
        f(x) = alpha * x for x < 0
        f(x) = x for x >= 0

        ```
    where `alpha` is a learned parameters , it's a 1-D array, the length equal 1 or input_filters.

    Args:
        num_parameters:(1 or None)  if None num_parameters will equal to input_filters .
        init (float): initial value of the parameters

    """

    def __init__(self, num_parameters=None, init=0.25, name=None):
        super(PRelu, self).__init__(name=name)
        self.num_parameters = None
        if num_parameters == 1:
            self.num_parameters = num_parameters
        self.init = init
        self.weight = None

    def build(self, input_shape):
        if self._built == False:
            if self.num_parameters is None:
                self.num_parameters = self.input_filters
            self.weight = Parameter(ones((self.num_parameters)) * self.init)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        pos = relu(x)
        reshape_shape =[1]*len(x.shape)
        reshape_shape[1] =self.num_parameters

        neg = self.weight.view(*reshape_shape) * (x - abs(x)) * 0.5
        return pos + neg


class Sigmoid(Layer):
    """Sigmoid activation layer.

    Examples:
        >>> Sigmoid()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, name=None):
        super(Sigmoid, self).__init__()
        self._built = True

    def forward(self, *x):
        """
        Args:
        x: Input tensor.

        Returns: output tensor

        """
        x = enforce_singleton(x)
        return sigmoid(x)


class Tanh(Layer):
    """ Tanh activation layer.

    Examples:
        >>> Tanh()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, name=None):
        super(Tanh, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return tanh(x)


class Swish(Layer):
    """ Self-Gated Activation Function.
    it follows:
        ```
        f(x) =  x * sigmoid(x)

        ```
    References:
        Swish: a Self-Gated Activation Function
        https://arxiv.org/abs/1710.05941v1

    Examples:
        >>> Swish()(to_tensor([[-3.0, -1.0, 0.0, 2.0]])).cpu()
        tensor([[-0.1423, -0.2689,  0.0000,  1.7616]])

    """

    def __init__(self, name=None):
        super(Swish, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return swish(x)


class HardSigmoid(Layer):
    """ Hard sigmoid activation layer.
    it follows:
        ```
        f(x) = relu6(x + 3) / 6

        ```
    Examples:
        >>> HardSigmoid()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, inplace=False, name=None):
        super(HardSigmoid, self).__init__()

        self.inplace = inplace
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_sigmoid(x)


class HardSwish(Layer):
    """Hard swish Activation Function.

    Memory saving version of swish
    it follows:

      ```
        f(x) =  x * hard_sigmoid(x)

      ```

    References:
        Searching for MobileNetV3
        https://arxiv.org/abs/1905.02244

    Examples:
        >>> HardSwish()(to_tensor([[-3.0, -1.0, 0.0, 2.0]])).cpu()
        tensor([[-0.0000, -0.3333,  0.0000,  1.6667]])

    """

    def __init__(self, name=None):
        super(HardSwish, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_swish(x)


class HardTanh(Layer):
    """Hard tanh Activation Function.

    Examples:
        >>> HardTanh()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
    """

    def __init__(self, name=None):
        super(HardTanh, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_tanh(x)


class Selu(Layer):
    """Selu activation function

    Scaled exponential linear unit operation. Computes the element-wise exponential linear
    of ``x``: ``scale * x`` for ``x >= 0`` and ``x``: ``scale * alpha * (exp(x)-1)`` otherwise.
    scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717

    Args:
        x (tensor): input tensor
        name(string, None): name of the layer.

    Returns:The output tensor has the same shape as ``x``

    Examples:
        >>> selu(to_tensor([[-1, -0.5, 0, 1, 2]]))
        tensor([[-1.1113, -0.6918,  0.0000,  1.0507,  2.1014]])

    References:
        paper: https://arxiv.org/abs/1706.02515
        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter

    """
    def __init__(self,  name=None):
        super(Selu, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return selu(x)


class Elu(Layer):
    """Exponential Linear Unit.
         It follows:
         ```
           f(x) =  alpha * (exp(x) - 1.) for x < 0
           f(x) = x for x >= 0
         ```
    """

    def __init__(self, name=None):
        super(Elu, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return elu(x)


class LecunTanh(Layer):
    def __init__(self, name=None):
        super(LecunTanh, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_swish(x)


class SoftSign(Layer):
    def __init__(self, name=None):
        super(SoftSign, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return soft_sign(x)


class SoftPlus(Layer):
    def __init__(self, name=None):
        super(SoftPlus, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return soft_plus(x)


class Logit(Layer):
    def __init__(self, name=None):
        super(Logit, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return logit(x)


class LogLog(Layer):
    """LogLog Activation Function
          it follows:
          ```
            f(x) =  1 - exp(-exp(x))

          ```
        References:
            "Complementary Log-Log and Probit: Activation Functions Implemented in Artificial Neural Networks"
            https://ieeexplore.ieee.org/document/4626755/

        Examples:
            >>> LogLog()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
            tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    """
    def __init__(self, name=None):
        super(LogLog, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return log_log(x)


class Mish(Layer):
    """Self Regularized Non-Monotonic Neural Activation Function
      it follows:
      ```
        f(x) =  x * tanh(softplus(x))

      ```
    References:
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681

    Examples:
        >>> Mish()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    """

    def __init__(self, name=None):
        super().__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return mish(x)


class Softmax(Layer):
    """Softmax activation layer.
    Args
           x: Input tensor.
           axis: Integer, axis along which the softmax normalization is applied.

    Returns
           Tensor, output of softmax transformation.

    Raises
           ValueError: In case `dim(x) == 1`.
    """

    def __init__(self, name=None):
        super(Softmax, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return softmax(x)


class LogSoftmax(Layer):
    def __init__(self, name=None):
        super(LogSoftmax, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return log_softmax(x)


class Gelu(Layer):
    """Gaussian Error Linear Unit.

    it follows:
        ```
        f(x) =x∗Φ(x)
        where \Phi(x)Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

        ```

    References:
        Gaussian Error Linear Units (GELUs)
        https://arxiv.org/abs/1606.08415

    Examples:
        >>> Gelu()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32)>

    """

    def __init__(self, name=None):
        super(Gelu, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return gelu(x)


class GptGelu(Layer):
    r"""For information: OpenAI GPT's GELU is slightly different (and gives
    slightly different results).
    """

    def __init__(self, name=None):
        super(GptGelu, self).__init__()
        self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        return gpt_gelu(x)


class SIREN(Layer):
    """SIREN leverages periodic activation functions for implicit neural representations and demonstrate
    that these networks are ideally suited for representing complex natural signals and their derivatives.

    Their project page can be found here "https://vsitzmann.github.io/siren/"

    For more details please refer to the paper Implicit Neural Representations with PeriodicActivation Functions by
    Sitzmann et. al. (https://arxiv.org/abs/2006.09661)
    """

    def __init__(self, name=None):
        super(SIREN, self).__init__()
        self._built = True
        self.w0=Parameter(data=to_tensor(30.0,requires_grad=True))

    def forward(self, *x):
        x = enforce_singleton(x)
        x=sin(self.w0*x)
        return x


def get_activation(fn_name):
    """

    Args:
        fn_name ():

    Returns:

    Examples:
        >>> print(get_activation('swish'))



    """
    if fn_name is None:
        return None
    fn_modules = ['trident.layers.pytorch_activations', 'trident.backend.pytorch_ops', 'torch.nn.functional']
    try:
        if isinstance(fn_name, str):
            if fn_name.lower() == fn_name:
                if fn_name == 'p_relu' or fn_name == 'prelu':
                    return PRelu()
                activation_fn = get_function(fn_name, ['trident.backend.pytorch_ops',
                                                       'trident.layers.pytorch_activations'] if fn_name in __all__
                else fn_modules)
                return activation_fn
            else:
                try:
                    activation_fn = get_class(snake2camel(fn_name), fn_modules)
                    return activation_fn()
                except Exception:
                    activation_fn = get_class(fn_name, ['trident.backend.pytorch_ops',
                        'trident.layers.pytorch_activations'] if fn_name in __all__ else fn_modules)
                    return activation_fn()
        elif getattr(fn_name, '__module__', None) == 'trident.layers.pytorch_activations':
            if inspect.isfunction(fn_name):
                return fn_name
            elif inspect.isclass(fn_name) and inspect._is_type(fn_name):
                return fn_name()
            elif isinstance(fn_name, Layer):
                return fn_name
        else:
            if callable(fn_name):
                result = inspect.getfullargspec(fn_name)
                if 1 <= len(result.args) <= 2:
                    return fn_name if inspect.isfunction(fn_name) else fn_name()
                else:
                    raise ValueError('Unknown activation function/ class')
    except Exception as e:
        print(e)
        return None
