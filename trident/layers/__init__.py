
from trident.backend.load_backend import get_backend

__all__ = ['activations','layers','blocks','normalizations','pooling']
if get_backend()=='pytorch':
    from trident.layers import pytorch_activations as activations
    from trident.layers import pytorch_layers as layers
    from trident.layers import pytorch_blocks as blocks
    from trident.layers import pytorch_normalizations as normalizations
    from trident.layers import pytorch_pooling as pooling
elif get_backend()=='tensorflow':
    from trident.layers import tensorflow_activations as activations
    from trident.layers import tensorflow_layers as layers
    from trident.layers import tensorflow_blocks as blocks
    from trident.layers import tensorflow_normalizations as normalizations
    from trident.layers import tensorflow_pooling as pooling

