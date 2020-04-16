from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 1e-7
_IMAGE_DATA_FORMAT = 'channels_first'

_IMAGE_CHANNEL_ORDER='bgr'


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> trident.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON


def set_epsilon(e):
    """Sets the value of the fuzz factor used in numeric expressions.

    # Arguments
        e: float. New value of epsilon.

    # Example
    ```python
        >>> from trident import backend as K
        >>> K.epsilon()
        1e-07
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    """

    global _EPSILON
    _EPSILON = float(e)


def floatx():
    """Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> trident.backend.floatx()
        'float32'
    ```
    """
    return _FLOATX


def set_floatx(floatx):
    """Sets the default float type.

    # Arguments
        floatx: String, 'float16', 'float32', or 'float64'.

    # Example
    ```python
        >>> from trident import backend as K
        >>> K.floatx()
        'float32'
        >>> K.set_floatx('float16')
        >>> K.floatx()
        'float16'
    ```
    """
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def cast_to_floatx(x):
    """Cast a Numpy array to the default Keras float type.

    # Arguments
        x: Numpy array.

    # Returns
        The same Numpy array, cast to its new type.

    # Example
    ```python
        >>> from trident import backend as K
        >>> K.floatx()
        'float32'
        >>> arr = numpy.array([1.0, 2.0], dtype='float64')
        >>> arr.dtype
        dtype('float64')
        >>> new_arr = K.cast_to_floatx(arr)
        >>> new_arr
        array([ 1.,  2.], dtype=float32)
        >>> new_arr.dtype
        dtype('float32')
    ```
    """
    return np.asarray(x, dtype=_FLOATX)


def image_data_format():
    """Returns the default image data format convention.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> trident.backend.image_data_format()
        'channels_first'
    ```
    """
    return _IMAGE_DATA_FORMAT


def set_image_data_format(data_format):
    """Sets the value of the data format convention.

    # Arguments
        data_format: string. `'channels_first'` or `'channels_last'`.

    # Example
    ```python
        >>> from trident import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```
    """
    global _IMAGE_DATA_FORMAT
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError('Unknown data_format:', data_format)
    _IMAGE_DATA_FORMAT = str(data_format)


def image_channel_order():
    """Returns the default image data format convention.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> trident.backend.image_channel_order()
        'channels_first'
    ```
    """
    return _IMAGE_CHANNEL_ORDER


def set_image_channel_order(channel_order):
    """Sets the value of the data format convention.

    # Arguments
        data_format: string. `'channels_first'` or `'channels_last'`.

    # Example
    ```python
        >>> from trident import backend as K
        >>> K.image_channel_order()
        'channels_first'
        >>> K.set_image_channel_order('channels_last')
        >>> K.image_channel_order()
        'channels_last'
    ```
    """
    global _IMAGE_CHANNEL_ORDER
    if channel_order not in {'bgr', 'rgb'}:
        raise ValueError('Unknown channel_order:', channel_order)
    _IMAGE_CHANNEL_ORDER = str(channel_order)









