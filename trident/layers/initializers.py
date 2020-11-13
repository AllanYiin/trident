
class Initializer(object):
  """Initializer base class: all initializers inherit from this class.
  """

  def __init__(self, name=None, **kwargs):
    super(Initializer, self).__init__(name=name, **kwargs)
  def __call__(self, shape, dtype=None):
    """Returns a tensor object initialized as specified by the initializer.
    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided will return tensor
       of `tf.float32`.

    """
    raise NotImplementedError





class TruncatedNormal(Initializer):
  """Initializer base class: all initializers inherit from this class.
  """
  def __init__(self, stddev=0.02,name=None, **kwargs):
    super(TruncatedNormal, self).__init__(name=name, **kwargs)
    self.stddev=stddev

  def __call__(self, shape, dtype=None):
    """Returns a tensor object initialized as specified by the initializer.
    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided will return tensor
       of `tf.float32`.

    """
    raise NotImplementedError

