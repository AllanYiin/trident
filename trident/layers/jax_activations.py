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
import jax
import jax.numpy as jnp
import jaxlib

from trident.backend.common import get_function, get_class, camel2snake,snake2camel, enforce_singleton,TensorShape
from trident.backend.jax_backend import Layer,Parameter
from trident.backend.jax_ops import *

__all__ = ['Identity', 'Sigmoid', 'Tanh','TanhExp','ExpTanh', 'Relu', 'Relu6','SquaredRelu', 'LeakyRelu', 'LeakyRelu6', 'SmoothRelu','CRelu','Silu', 'PRelu', 'Swish',
           'Elu', 'HardSigmoid', 'HardSwish', 'Selu', 'LecunTanh', 'SoftSign', 'SoftPlus', 'HardTanh', 'Logit',
           'LogLog', 'Mish','HardMish', 'Softmax', 'Gelu', 'GptGelu','SIREN', 'LogSoftmax', 'get_activation']


class Identity(Layer):
    """
    Identity activation Layer
    A placeholder identity operator that is argument-insensitive.

    Examples:
        >>> Identity()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-3.0, -1.0, 0.0, 2.0])

    """
    def __init__(self, keep_output=False,name=None):
        super(Identity, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
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

    def __init__(self,inplace=False,keep_output=False, name=None):
        super(Relu, self).__init__(keep_output=keep_output,name=name)
        self._built = True
        self.inplace=inplace

    def forward(self, x, **kwargs):
        """
        Args:
        x: Input tensor.

        Returns: output tensor

        """
        if not hasattr(self,'inplace'):
            self.inplace=False
        # if self.inplace and not x.is_leaf:
        #     return relu_(x)
        # else:
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

    def __init__(self,inplace=False, keep_output=False,name=None):
        super(Relu6, self).__init__(keep_output=keep_output,name=name)
        self._built = True
        self.inplace=inplace

    def forward(self, x, **kwargs):
        if not hasattr(self,'inplace'):
            self.inplace=False
        # if self.inplace and not x.is_leaf:
        #     return clip_(F.relu_(x),max=6)
        # else:
        return clip(relu(x),max=6)


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

    def __init__(self,inplace=False, alpha=0.2, keep_output=False,name=None):
        super(LeakyRelu, self).__init__(keep_output=keep_output,name=name)
        self.alpha = alpha
        self._built = True
        self.inplace=inplace

    def forward(self, x, **kwargs):
        if not hasattr(self,'inplace'):
            self.inplace=False
        # if self.inplace and not x.is_leaf:
        #     return F.leaky_relu(x,self.alpha,inplace=True)
        # else:
        return leaky_relu(x,self.alpha,inplace=False)
        


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

    def __init__(self,inplace=False, alpha=0.2, keep_output=False,name=None):
        super(LeakyRelu6, self).__init__(keep_output=keep_output,name=name)
        self._built = True
        self.alpha = alpha
        self.inplace=inplace

    def forward(self, x, **kwargs):
        if not hasattr(self,'inplace'):
            self.inplace=False
        # if self.inplace and not x.is_leaf:
        #     return clip_(leaky_relu(x, self.alpha, inplace=True),min=-6,max=6)
        # else:
        return clip(leaky_relu(x, self.alpha, inplace=False),min=-6,max=6)

    def extra_repr(self):
        s = 'alpha={alpha}'
        return s.format(**self.__dict__)


class SquaredRelu(Layer):
    """Rectified Linear Unit activation function.

        Primer: Searching for Efficient Transformers for Language Modeling

        ```
        f(x) = max_value**2 if x >= max_value
        f(x) = x**2 if threshold <= x < max_value
        f(x) = 0 otherwise

        ```

    Examples:
        >>> SquaredRelu()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self,inplace=False,keep_output=False, name=None):
        super(SquaredRelu, self).__init__(keep_output=keep_output,name=name)
        self._built = True
        self.inplace=inplace

    def forward(self, x, **kwargs):
        """
        Args:
        x: Input tensor.

        Returns: output tensor

        """
        if not hasattr(self,'inplace'):
            self.inplace=False
        # if self.inplace and not x.is_leaf:
        #     return square(relu_(x))
        # else:
        return square(relu(x))



class SmoothRelu(Layer):
    """Smooth_relu activation Layer

    Examples:
        >>> SmoothRelu()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, keep_output=False,name=None):
        super(SmoothRelu, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return smooth_relu(x)


class CRelu(Layer):
    """Computes Concatenated ReLU.

    Concatenates a ReLU which selects only the positive part of the activation
    with a ReLU which selects only the *negative* part of the activation.
    Note that as a result this non-linearity doubles the depth of the activations.
    Source: [Understanding and Improving Convolutional Neural Networks via
    Concatenated Rectified Linear Units. W. Shang, et
    al.](https://arxiv.org/abs/1603.05201)

    References:
      Understanding and Improving Convolutional Neural Networks via Concatenated
      Rectified Linear Units:
        [Shang et al., 2016](http://proceedings.mlr.press/v48/shang16)
        ([pdf](http://proceedings.mlr.press/v48/shang16.pdf))
    """

    def __init__(self, axis=1,keep_output=False,name=None):
        super(CRelu, self).__init__(keep_output=keep_output,name=name)
        self._built = True
        self.axis=axis

    def forward(self, x, **kwargs):
        return crelu(x,axis=self.axis)


class Silu(Layer):
    """Applies the silu function, element-wise.

    .. math::
        \text{silu}(x) = x * \\sigma(x), \text{where } \\sigma(x) \\text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = Silu()
        >>> input = randn(2)
        >>> output = m(input)
    """

    def __init__(self, keep_output=False,name=None):
        super(Silu, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        return silu(x)

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

    def __init__(self, num_parameters=None, init=0.25, keep_output=False,name=None):
        super(PRelu, self).__init__(keep_output=keep_output,name=name)
        self.num_parameters = None
        if num_parameters == 1:
            self.num_parameters = num_parameters
        self.init = init
        self.weight = None

    def build(self, input_shape:TensorShape):
        if not self._built:
            if self.num_parameters is None:
                self.num_parameters = self.input_filters
            self.weight = Parameter(ones((self.num_parameters)) * self.init)
            self._built = True

    def forward(self, x, **kwargs):
        return p_relu(x, self.weight)



class Sigmoid(Layer):
    """Sigmoid activation layer.

    Examples:
        >>> Sigmoid()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, keep_output=False,name=None):
        super(Sigmoid, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        """
        Args:
        x: Input tensor.

        Returns: output tensor

        """
        
        return sigmoid(x)


class Tanh(Layer):
    """ Tanh activation layer.

    Examples:
        >>> Tanh()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, keep_output=False,name=None):
        super(Tanh, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return tanh(x)


class TanhExp(Layer):
    """ TanhExp activation layer.
     ```
        f(x) =  x*tanh(exp(x))

      ```

    References:
        TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks
        https://arxiv.org/abs/2003.09855

    Examples:
        >>> TanhExp()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, keep_output=False, name='tanhexp'):
        super(TanhExp, self).__init__(keep_output=keep_output, name=name)
        self._built = True

    def forward(self, x, **kwargs):
        return x*tanh(exp(x))


class ExpTanh(Layer):
    """ ExpTanh activation layer.

     ```
        f(x) =  tanh(exp(x))

      ```

    Examples:
        >>> ExpTanh()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, keep_output=False, name='exptanh'):
        super(ExpTanh, self).__init__(keep_output=keep_output, name=name)
        self._built = True

    def forward(self, x, **kwargs):
        return tanh(exp(x))


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

    def __init__(self, keep_output=False,name=None):
        super(Swish, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return swish(x)


class HardSigmoid(Layer):
    """ Hard sigmoid activation layer.
    it follows:
      .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \\le -3, \\
            1 & \text{if~} x \\ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \\end{cases}
    Examples:
        >>> HardSigmoid()(to_tensor([-3.0, -1.0, 0.0, 2.0]))

    """

    def __init__(self, inplace=False, keep_output=False,name=None):
        super(HardSigmoid, self).__init__(keep_output=keep_output,name=name)

        self.inplace = inplace
        self._built = True

    def forward(self, x, **kwargs):
        
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

    def __init__(self, keep_output=False,name=None):
        super(HardSwish, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return hard_swish(x)


class HardTanh(Layer):
    """Hard tanh Activation Function.

    Examples:
        >>> HardTanh()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
    """

    def __init__(self, keep_output=False,name=None):
        super(HardTanh, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
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
    def __init__(self,  keep_output=False,name=None):
        super(Selu, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return selu(x)


class Elu(Layer):
    """Exponential Linear Unit.
         It follows:
         ```
           f(x) =  alpha * (exp(x) - 1.) for x < 0
           f(x) = x for x >= 0
         ```
    """

    def __init__(self, keep_output=False,name=None):
        super(Elu, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return elu(x)


class LecunTanh(Layer):
    def __init__(self, keep_output=False,name=None):
        super(LecunTanh, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return hard_swish(x)


class SoftSign(Layer):
    def __init__(self, keep_output=False,name=None):
        super(SoftSign, self).__init__(keep_output=keep_output,name=name)

    def forward(self, x, **kwargs):
        
        return soft_sign(x)


class SoftPlus(Layer):
    def __init__(self, keep_output=False,name=None):
        super(SoftPlus, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return soft_plus(x)


class Logit(Layer):
    def __init__(self, keep_output=False,name=None):
        super(Logit, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
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
    def __init__(self, keep_output=False,name=None):
        super(LogLog, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
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

    def __init__(self, keep_output=False,name=None):
        super().__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return mish(x)


class HardMish(Layer):
    """Self Regularized Non-Monotonic Neural Activation Function.

    it follows:
    ::

        f(x) =  x * hard_tanh(softplus(x))


    References:
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681

    Examples:
        >>> HardMish()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32)>

    """
    def __init__(self,keep_output=False,name=None):
        super(HardMish, self).__init__(keep_output=keep_output,name=name)
        self._built = True
    def forward(self, x, **kwargs):
        
        return hard_mish(x)

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

    def __init__(self, axis=-1,keep_output=False,name=None):
        super(Softmax, self).__init__(keep_output=keep_output,name=name)
        self.axis=axis
        self._built = True

    def forward(self, x, **kwargs):
        
        return softmax(x,axis=self.axis)


class LogSoftmax(Layer):
    def __init__(self, keep_output=False,name=None):
        super(LogSoftmax, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return log_softmax(x)


class Gelu(Layer):
    """Gaussian Error Linear Unit.

    it follows:
        ```
        f(x) =x∗Φ(x)
        where \\Phi(x)Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

        ```

    References:
        Gaussian Error Linear Units (GELUs)
        https://arxiv.org/abs/1606.08415

    Examples:
        >>> Gelu()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32)>

    """

    def __init__(self, keep_output=False,name=None):
        super(Gelu, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return gelu(x)


class GptGelu(Layer):
    r"""For information: OpenAI GPT's GELU is slightly different (and gives
    slightly different results).
    """

    def __init__(self, keep_output=False,name=None):
        super(GptGelu, self).__init__(keep_output=keep_output,name=name)
        self._built = True

    def forward(self, x, **kwargs):
        
        return gpt_gelu(x)


class SIREN(Layer):
    """SIREN leverages periodic activation functions for implicit neural representations and demonstrate
    that these networks are ideally suited for representing complex natural signals and their derivatives.

    Their project page can be found here "https://vsitzmann.github.io/siren/"

    For more details please refer to the paper Implicit Neural Representations with PeriodicActivation Functions by
    Sitzmann et. al. (https://arxiv.org/abs/2006.09661)
    """

    def __init__(self, w0=1.0,keep_output=False,name=None):
        super(SIREN, self).__init__(keep_output=keep_output,name=name)
        self._built = True
        self.w0 = w0

    def forward(self, x, **kwargs):
        x=sin(self.w0*x)
        return x


def get_activation(fn_name,only_layer=False):
    """

    Args:
        only_layer (bool):
        fn_name ():

    Returns:

    Examples:
        >>> print(get_activation('relu'))
        <function relu at 0x000002CED4F71158>
        >>> print(get_activation('swish'))
        <function swish at 0x000002CED4F71A60>



    """
    if fn_name is None:
        return None
    fn_modules = ['trident.layers.jax_activations', 'trident.backend.jax_ops', 'jax.nn']
    trident_fn_modules = ['trident.layers.jax_activations', 'trident.backend.jax_ops']
    if only_layer:
        fn_modules = ['trident.layers.jax_activations']
        trident_fn_modules = ['trident.layers.jax_activations']
    try:
        if isinstance(fn_name, str):
            if not only_layer and (camel2snake(fn_name)== fn_name or fn_name.lower()== fn_name):
                if fn_name == 'p_relu' or fn_name == 'prelu':
                    return PRelu()
                activation_fn = get_function(fn_name, trident_fn_modules)
                return activation_fn
            else:
                try:
                    activation_fn = get_class(snake2camel(fn_name), fn_modules)
                    return activation_fn()
                except Exception:
                    activation_fn = get_class(fn_name, fn_modules)
                    return activation_fn()
        elif isinstance(fn_name,(Layer)) or getattr(fn_name, '__module__', None) == 'trident.layers.jax_activations':
            if inspect.isfunction(fn_name):
                return partial(fn_name)
            elif inspect.isclass(fn_name) and fn_name.__class__.__name__=="type":
                return fn_name()
            elif isinstance(fn_name, Layer):
                return fn_name
        elif inspect.isfunction(fn_name) and getattr(fn_name, '__module__', None) == 'trident.backend.jax_ops':
            if only_layer:
                activation_layer = get_class(snake2camel(fn_name.__name__), trident_fn_modules)
                return activation_layer()
            else:
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
