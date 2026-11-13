import warnings
from trident import context
from trident.backend.tensorspec import TensorSpec,TensorShape
cxt=context._context()

if cxt.get_backend()=='pytorch':
    from trident.backend.pytorch_ops import exp,clip,reshape,log,sigmoid,softmax,random_normal

elif cxt.get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import  exp,clip,reshape,log,sigmoid,softmax,random_normal


def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number` or `torch.*Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.*Tensor` instance
    """
    if not all(isinstance(v, torch.Tensor) or isinstance(v, Number) for v in values):
        raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
    if not all([isinstance(v, torch.Tensor) for v in values]):
        options = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        values = [v if isinstance(v, torch.Tensor) else torch.tensor(v, **options)
                  for v in values]
    return torch.broadcast_tensors(*values)


def _standard_normal(shape, dtype, device):
    """Generate a tensor of samples from the standard normal distribution.

    Args:
        shape : shape of the output tensor
        dtype : data type of the output tensor
        device : device to place the output tensor on

    Returns:
        A tensor of samples from the standard normal distribution.

    Note:
        This function uses the `random_normal` function to generate the samples and then moves the tensor to the specified device."""
    return random_normal(shape=shape,dtype=dtype).to(device)



def _sum_rightmost(value, dim):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits, is_binary=False):
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return sigmoid(logits)
    return softmax(logits, axis=-1)


def clamp_probs(probs):
    """Clamp probabilities.

    Args:
        probs: A tensor representing probabilities.

    Returns:
        A tensor with probabilities clamped between a small epsilon value and 1 - epsilon."""
    eps = torch.finfo(probs.dtype).eps
    return clip(probs,min=eps, max=1 - eps)


def probs_to_logits(probs, is_binary=False):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return log(ps_clamped) - log(1-ps_clamped)
    return log(ps_clamped)

class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    has_rsample = False
    has_enumerate_support = False
    _validate_args = False
    support = None
    arg_constraints = {}

    @staticmethod
    def set_default_validate_args(value):
        """Set the default value for validating arguments in the Distribution class.

        Args:
            value: The value to set for validating arguments. Must be either True or False.

        Raises:
            ValueError: If the value is not True or False."""
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value

    def __init__(self, batch_shape:TensorShape=None, event_shape:TensorShape=None, validate_args=None):
        """Initialize a Distribution object.

        Args:
            batch_shape (TensorShape, optional): The shape of the batch dimensions of the distribution. Default is None.
            event_shape (TensorShape, optional): The shape of the event dimensions of the distribution. Default is None.
            validate_args (bool, optional): Whether to validate the arguments of the distribution. Default is None.

        Raises:
            ValueError: If the parameter values are invalid."""
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            for param, constraint in self.arg_constraints.items():
                if constraints.is_dependent(constraint):
                    continue  # skip constraints that cannot be checked
                if param not in self.__dict__ and isinstance(getattr(type(self), param), lazy_property):
                    continue  # skip checking lazily-constructed args
                if not constraint.check(getattr(self, param)).all():
                    raise ValueError("The parameter {} has invalid values".format(param))
        super(Distribution, self).__init__()

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        raise NotImplementedError

    @property
    def batch_shape(self):
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def arg_constraints(self):
        """
        Returns a dictionary from argument names to
        :class:`~torch.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
        raise NotImplementedError

    @property
    def support(self):
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        raise NotImplementedError

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, sample_shape:TensorShape=None):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        return self.rsample(sample_shape)

    def rsample(self, sample_shape:TensorShape=None):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        warnings.warn('sample_n will be deprecated. Use .sample((n,)) instead', UserWarning)
        return self.sample(TensorShape([n]))

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def cdf(self, value):
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def icdf(self, value):
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        raise NotImplementedError

    def perplexity(self):
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return exp(self.entropy())

    def _extended_shape(self, sample_shape:TensorShape=None):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, TensorShape):
            sample_shape = TensorShape(sample_shape)
        return sample_shape + self._batch_shape + self._event_shape

    def _validate_sample(self, value):
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.

        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        if not isinstance(value, torch.Tensor):
            raise ValueError('The value argument to log_prob must be a Tensor')

        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError('The right-most size of value must match event_shape: {} vs {}.'.
                             format(value.size(), self._event_shape))

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError('Value is not broadcastable with batch_shape+event_shape: {} vs {}.'.
                                 format(actual_shape, expected_shape))

        if not self.support.check(value).all():
            raise ValueError('The value argument must be within the support')

    def _get_checked_instance(self, cls, _instance=None):
        """Get a checked instance of a class.

        Args:
            cls: The class to get the checked instance of.
            _instance: An optional instance of the class.

        Returns:
            The checked instance of the class.

        Raises:
            NotImplementedError: If _instance is None and the class defines a custom __init__ method but does not define a custom .expand() method."""
        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError("Subclass {} of {} that defines a custom __init__ method "
                                      "must also define a custom .expand() method.".
                                      format(self.__class__.__name__, cls.__name__))
        return self.__new__(type(self)) if _instance is None else _instance

    def __repr__(self):
        """!!! note

        Failed to generate docs
        """
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ', '.join(['{}: {}'.format(p, self.__dict__[p]
                                if self.__dict__[p].numel() == 1
                                else self.__dict__[p].size()) for p in param_names])
        return self.__class__.__name__ + '(' + args_string + ')'