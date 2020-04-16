class Regularizer(object):

    """Regularizer base class.

    """
    def __call__(self, x):
        return 0.
    @classmethod
    def from_config(cls, config):
        return cls(**config)

