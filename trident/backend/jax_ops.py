import jax.numpy as jnp




def relu(x):
    return jnp.maximum(0, x)


def make_onehot(label, num_classes, axis=-1):
    """
    Create a one-hot encoding of x of size k.

    x: array
        The array to be one hot encoded
    k: interger
        The number of classes
    dtype: jnp.dtype, optional(default=float32)
        The dtype to be used on the encoding
    Examples:
    >>> make_onehot(jnp.array([[1, 2],[1, 3]],dtype=jnp.int64), 4, axis=-1)
    tensor([[[0., 1., 1., 0.],
             [0., 1., 0., 1.]],
    <BLANKLINE>
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.]]])

    """
    onehot = jnp.array(label[:, None] == jnp.arange(num_classes), dtype=jnp.float32)
    if axis!=-1:
        axes=np.arange(ndim(a))
        axes[axis]=ndim(a)-1
        axes[-1]=axis
        onehot=jnp.transpose(axes)
    return onehot