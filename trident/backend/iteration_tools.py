import collections.abc
import itertools
import warnings
import operator
from collections import namedtuple
from contextlib import suppress
from functools import wraps, reduce, partial

from itertools import chain
from operator import itemgetter
from typing import Iterable, Sized, Union, Callable, Sequence, Any, Tuple, NamedTuple

import numpy as np

__all__ = ['join', 'recursive_conditional_map', 'pam', 'zip_equal', 'head_tail', 'peek', 'lmap', 'dmap', 'zdict']

import six

from trident.backend.common import is_instance


def join(values):
    return ", ".join(map(str, values))


def recursive_conditional_map(xr, f, condition):
    """Walks recursively through iterable data structure ``xr``. Applies ``f`` on objects that satisfy ``condition``."""
    return tuple(f(x) if condition(x) else recursive_conditional_map(x, f, condition) for x in xr)

def accumulate(binop, seq, initial=None):
    """ Repeatedly apply binary function to a sequence, accumulating results
    >>> from operator import add, mul
    >>> list(accumulate(add, [1, 2, 3, 4, 5]))
    [1, 3, 6, 10, 15]
    >>> list(accumulate(mul, [1, 2, 3, 4, 5]))
    [1, 2, 6, 24, 120]
    Accumulate is similar to ``reduce`` and is good for making functions like
    cumulative sum:
    >>> from functools import partial, reduce
    >>> sum    = partial(reduce, add)
    >>> cumsum = partial(accumulate, add)
    Accumulate also takes an optional argument that will be used as the first
    value. This is similar to reduce.
    >>> list(accumulate(add, [1, 2, 3], -1))
    [-1, 0, 2, 5]
    >>> list(accumulate(add, [], 1))
    [1]
    See Also:
        itertools.accumulate :  In standard itertools for Python 3.2+
    """
    seq = iter(seq)
    if initial is None:
        try:
            result = next(seq)
        except StopIteration:
            return
    else:
        result = initial
    yield result
    for elem in seq:
        result = binop(result, elem)
        yield result

def interleave(seqs):
    """ Interleave a sequence of sequences
    >>> list(interleave([[1, 2], [3, 4]]))
    [1, 3, 2, 4]
    >>> ''.join(interleave(('ABC', 'XY')))
    'AXBYC'
    Both the individual sequences and the sequence of sequences may be infinite
    Returns a lazy iterator
    """
    iters = itertools.cycle(map(iter, seqs))
    while True:
        try:
            for itr in iters:
                yield next(itr)
            return
        except StopIteration:
            predicate = partial(operator.is_not, itr)
            iters = itertools.cycle(itertools.takewhile(predicate, iters))


def pam(functions: Iterable[Callable], *args, **kwargs):
    """
    Inverse of `map`. Apply a sequence of callables to fixed arguments.

    Examples
        >>> list(pam([np.sqrt, np.square, np.cbrt], 64))
        [8, 4096, 4]
    """
    for f in functions:
        yield f(*args, **kwargs)


def zip_equal(*args: Union[Sized, Iterable]) -> Iterable[Tuple]:
    """
    zip over the given iterables, but enforce that all of them exhaust simultaneously.

    Examples
        >>> zip_equal([1, 2, 3], [4, 5, 6]) # ok
        >>> zip_equal([1, 2, 3], [4, 5, 6, 7]) # raises ValueError
        # ValueError is raised even if the lengths are not known
        >>> zip_equal([1, 2, 3], map(np.sqrt, [4, 5, 6])) # ok
        >>> zip_equal([1, 2, 3], map(np.sqrt, [4, 5, 6, 7])) # raises ValueError
    """
    if not args:
        return

    lengths = []
    all_lengths = []
    for arg in args:
        try:
            lengths.append(len(arg))
            all_lengths.append(len(arg))
        except TypeError:
            all_lengths.append('?')

    if lengths and not all(x == lengths[0] for x in lengths):
        raise ValueError('The arguments have different lengths: {join(all_lengths)}.')

    iterables = [iter(arg) for arg in args]
    while True:
        result = []
        for it in iterables:
            with suppress(StopIteration):
                result.append(next(it))

        if len(result) != len(args):
            break
        yield tuple(result)

    if len(result) != 0:
        raise ValueError('The iterables did not exhaust simultaneously.')


def head_tail(iterable: Iterable) -> Tuple[Any, Iterable]:
    """
    Split the ``iterable`` into the first and the rest of the elements.

    Examples
        >>> head, tail = head_tail(map(np.square, [1, 2, 3]))
        >>> head, list(tail)
        1, [4, 9]
    """
    iterable = iter(iterable)
    return next(iterable), iterable


def peek(iterable: Iterable) -> Tuple[Any, Iterable]:
    """
    Return the first element from ``iterable`` and the whole iterable.

    Notes
        The incoming ``iterable`` might be mutated, use the returned iterable instead.

    Examples
        >>> original_iterable = map(np.square, [1, 2, 3])
        >>> head, iterable = peek(original_iterable)
        >>> head, list(iterable)
        (1, [1, 4, 9])
    """
    head, tail = head_tail(iterable)
    return head, chain([head], tail)


def lmap(func: Callable, *iterables: Iterable) -> list:
    """Composition of list and map."""
    return list(map(func, *iterables))


def dmap(func: Callable, dictionary: dict, *args, **kwargs):
    """
    Transform the ``dictionary`` by mapping ``func`` over its values.
    ``args`` and ``kwargs`` are passed as additional arguments.

    Examples
        >>> dmap(np.square, {'a': 1, 'b': 2})
        {'a': 1, 'b': 4}
    """
    return {k: func(v, *args, **kwargs) for k, v in dictionary.items()}


def zdict(keys: Iterable, values: Iterable) -> dict:
    """Create a dictionary from ``keys`` and ``values``."""
    return dict(zip_equal(keys, values))


def squeeze_first(inputs):
    """Remove the first dimension in case it is singleton."""
    if len(inputs) == 1:
        inputs = inputs[0]
    return inputs


def flatten(iterable: Iterable, iterable_types: Union[tuple, type] = None) -> list:
    """
    Recursively flattens an ``iterable`` as long as it is an instance of ``iterable_types``.

    Examples
        >>> flatten([1, [2, 3], [[4]]])
        [1, 2, 3, 4]
        >>> flatten([1, (2, 3), [[4]]])
        [1, (2, 3), 4]
        >>> flatten([1, (2, 3), [[4]]], iterable_types=(list, tuple))
        [1, 2, 3, 4]
    """
    if iterable_types is None:
        iterable_types = type(iterable)
    if not isinstance(iterable, iterable_types):
        return [iterable]

    return sum((flatten(value, iterable_types) for value in iterable), [])


def filter_mask(iterable: Iterable, mask: Iterable[bool]) -> Iterable:
    """Filter values from ``iterable`` according to ``mask``."""
    return map(itemgetter(1), filter(itemgetter(0), zip_equal(mask, iterable)))


def extract(sequence: Sequence, indices: Iterable):
    """Extract ``indices`` from ``sequence``."""
    return [sequence[i] for i in indices]


def negate_indices(indices: Iterable, length: int):
    """Return valid indices for a sequence of len ``length`` that are not present in ``indices``."""
    other_indices = np.ones(length, bool)
    other_indices[list(indices)] = False
    return np.where(other_indices)[0]


def make_chunks(iterable: Iterable, chunk_size: int, incomplete: bool = True):
    """
    Group ``iterable`` into chunks of size ``chunk_size``.

    Args
        iterable
        chunk_size
        incomplete
            whether to yield the last chunk in case it has a smaller size.
    """
    chunk = []
    for value in iterable:
        chunk.append(value)
        if len(chunk) == chunk_size:
            yield tuple(chunk)
            chunk = []

    if incomplete and chunk:
        yield chunk


def collect(func: Callable):
    """
    Make a function that returns a list from a function that returns an iterator.
    Examples
        >>> @collect
        >>> def squares(n):
        >>>     for i in range(n):
        >>>         yield i ** 2
        >>>
        >>> squares(3)
        [1, 4, 9]
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))

    wrapper.__annotations__['return'] = list
    return wrapper


def is_sequence(x):
    return isinstance(x, (collections.abc.Sequence, collections.abc.Mapping, collections.abc.MappingView, collections.abc.MutableMapping, tuple, namedtuple)) and not isinstance(x,
                                                                                                                                                                                 str)


def is_sequence_or_composite(x):
    return is_sequence(x) or x.__class__.__name__ == 'CompositeTensor'


def sequence_like(instance, args):
    """Converts the sequence `args` to the same type as `instance`.

    Args:
      instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
          `collections.OrderedDict`, or `composite_tensor.Composite_Tensor`
          or `type_spec.TypeSpec`.
      args: elements to be converted to the `instance` type.

    Returns:
      `args` with the type of `instance`.
    """

    def _get_attrs_items(obj):
        """Returns a list of (name, value) pairs from an attrs instance.

        The list will be sorted by name.

        Args:
          obj: an object.

        Returns:
          A list of (attr_name, attr_value) pairs, sorted by attr_name.
        """
        attrs = getattr(obj.__class__, "__attrs_attrs__")
        attr_names = (a.name for a in attrs)
        return [(attr_name, getattr(obj, attr_name)) for attr_name in attr_names]

    if isinstance(instance, collections.MutableMapping):
        # Pack dictionaries in a deterministic order by sorting the keys.
        # Notice this means that we ignore the original order of `OrderedDict`
        # instances. This is intentional, to avoid potential bugs caused by mixing
        # ordered and plain dicts (e.g., flattening a dict but using a
        # corresponding `OrderedDict` to pack it back).
        result = dict(zip(sorted(instance.keys()), args))
        instance_type = type(instance)
        if instance_type == collections.defaultdict:
            d = collections.defaultdict(instance.default_factory)
        else:
            d = instance_type()
        for key in instance:
            d[key] = result[key]
        return d
    elif isinstance(instance, collections.Mapping):
        result = dict(zip(sorted(instance.keys()), args))
        instance_type = type(instance)
        warnings.warn("Mapping types may not work well with tf.nest. Prefer" " using MutableMapping for {}".format(instance_type), 1)
        try:
            return instance_type((key, result[key]) for key in instance)
        except TypeError as err:
            raise TypeError("Error creating an object of type {} like {}. Note that "
                            "it must accept a single positional argument "
                            "representing an iterable of key-value pairs, in "
                            "addition to self. Cause: {}".format(
                type(instance), instance, err))
    elif isinstance(instance, collections.MappingView):
        # We can't directly construct mapping views, so we create a list instead
        return list(args)
    elif isinstance(instance, NamedTuple):
        if is_instance(instance, 'wrapt.ObjectProxy'):
            instance_type = type(instance.__wrapped__)
        else:
            instance_type = type(instance)
        return instance_type(*args)
    elif instance.__class__.__name__ == 'CompositeTensor':
        assert len(args) == 1
        spec = instance._type_spec  # pylint: disable=protected-access
        return spec._from_components(args[0])  # pylint: disable=protected-access
    elif hasattr(instance, '_type_spec'):
        # Pack a CompositeTensor's components according to a TypeSpec.
        assert len(args) == 1
        return instance._from_components(args[0])  # pylint: disable=protected-access
    elif isinstance(instance, six.moves.range):
        return sequence_like(list(instance), args)
    elif is_instance(instance, 'wrapt.ObjectProxy'):
        # For object proxies, first create the underlying type and then re-wrap it
        # in the proxy type.
        return type(instance)(sequence_like(instance.__wrapped__, args))
    else:
        # Not a namedtuple
        return type(instance)(args)


def yield_value(iterable):
    for _, v in yield_sorted_items(iterable):
        yield v


def yield_sorted_items(iterable):
    """Yield (key, value) pairs for `iterable` in a deterministic order.

    For Sequences, the key will be an int, the array index of a value.
    For Mappings, the key will be the dictionary key.
    For objects (e.g. namedtuples), the key will be the attribute name.

    In all cases, the keys will be iterated in sorted order.

    Args:
      iterable: an iterable.

    Yields:
      The iterable's (key, value) pairs, in order of sorted keys.
    """
    # Ordered to check common structure types (list, tuple, dict) first.
    if isinstance(iterable, list):
        for item in enumerate(iterable):
            yield item
    # namedtuples handled separately to avoid expensive namedtuple check.
    elif type(iterable) == tuple:  # pylint: disable=unidiomatic-typecheck
        for item in enumerate(iterable):
            yield item
    elif isinstance(iterable, (dict, collections.abc.Mapping)):
        # Iterate through dictionaries in a deterministic order by sorting the
        # keys. Notice this means that we ignore the original order of `OrderedDict`
        # instances. This is intentional, to avoid potential bugs caused by mixing
        # ordered and plain dicts (e.g., flattening a dict but using a
        # corresponding `OrderedDict` to pack it back).
        for key in sorted(iterable.keys()):
            yield key, iterable[key]
    elif isinstance(iterable, NamedTuple):
        for field in iterable._fields:
            yield field, getattr(iterable, field)
    elif iterable.__class__.__name__ == 'CompositeTensor' and hasattr(iterable, '_type_spec'):
        type_spec = iterable._type_spec  # pylint: disable=protected-access
        yield type_spec.value_type.__name__, type_spec._to_components(iterable)  # pylint: disable=protected-access
    elif hasattr(iterable, '_type_spec'):
        # Note: to allow CompositeTensors and their TypeSpecs to have matching
        # structures, we need to use the same key string here.
        yield iterable.value_type.__name__, iterable._component_specs  # pylint: disable=protected-access
    else:
        for item in enumerate(iterable):
            yield item

def yield_flat_up_to(shallow_tree, input_tree, is_seq, path=()):
    """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

    Args:
      shallow_tree: Nested structure. Traverse no further than its leaf nodes.
      input_tree: Nested structure. Return the paths and values from this tree.
        Must have the same upper structure as shallow_tree.
      is_seq: Function used to test if a value should be treated as a sequence.
      path: Tuple. Optional argument, only used when recursing. The path from the
        root of the original shallow_tree, down to the root of the shallow_tree
        arg of this recursive call.

    Yields:
      Pairs of (path, value), where path the tuple path of a leaf node in
      shallow_tree, and value is the value of the corresponding node in
      input_tree.
    """
    if not is_seq(shallow_tree):
        yield (path, input_tree)
    else:
        input_tree = dict(yield_sorted_items(input_tree))
        for shallow_key, shallow_subtree in yield_sorted_items(shallow_tree):
            subpath = path + (shallow_key,)
            input_subtree = input_tree[shallow_key]
            for leaf_path, leaf_value in yield_flat_up_to(shallow_subtree,
                                                           input_subtree, is_seq,
                                                           path=subpath):
                yield (leaf_path, leaf_value)


def pack_sequence_as(structure, flat_sequence, expand_composites=False, sequence_fn=None):
    """
    Implements sequence packing, with the option to alter the structure.
  
    Args:
        structure (): 
        flat_sequence (): 
        expand_composites (): 
        sequence_fn (): 

    Returns:
    Examples:
        >>> structure = { 'key3': "", 'key1': "", 'key2': "" }
        >>> flat_sequence = [1,2,3]
        >>> pack_sequence_as(structure, flat_sequence)
        {'key3': 3, 'key1': 1, 'key2': 2}
        >>> structure = (('a','b'), ('c','d','e'), 'f')
        >>> flat_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> pack_sequence_as(structure, flat_sequence)
        ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)
        >>> structure = { "key3": {"c": ('alpha', 'beta'), "a": ('gamma')},"key1": {"e": "val1", "d": "val2"} }
        >>> flat_sequence = ['val2', 'val1', 3.0, 1.0, 2.0]
        >>> pack_sequence_as(structure, flat_sequence)
        {'key3': {'c': (1.0, 2.0), 'a': 3.0}, 'key1': {'e': 'val1', 'd': 'val2'}}

    """

    def _packed_nest_with_indices(structure, flat, index, is_seq, sequence_fn):
        packed = []
        sequence_fn = sequence_fn
        for s in yield_value(structure):
            if is_seq(s):
                new_index, child = _packed_nest_with_indices(s, flat, index, is_seq, sequence_fn)
                packed.append(sequence_fn(s, child))
                index = new_index
            else:
                packed.append(flat[index])
                index += 1
        return index, packed

    is_seq = is_sequence_or_composite if expand_composites else is_sequence
    sequence_fn = sequence_fn or sequence_like

    def truncate(value, length):
        value_str = str(value)
        return value_str[:length] + (value_str[length:] and "...")

    if not is_seq(flat_sequence):
        raise TypeError(
            "Attempted to pack value:\n  {}\ninto a sequence, but found "
            "incompatible type `{}` instead."
                .format(truncate(flat_sequence, 100), type(flat_sequence)))

    if not is_seq(structure):
        if len(flat_sequence) != 1:
            raise ValueError(
                "The target structure is of type `{}`\n  {}\nHowever the input "
                "structure is a sequence ({}) of length {}.\n  {}\nnest cannot "
                "guarantee that it is safe to map one to the other.".format(
                    type(structure), truncate(structure, 100), type(flat_sequence),
                    len(flat_sequence), truncate(flat_sequence, 100)))
        return flat_sequence[0]
    packed = None
    try:
        final_index, packed = _packed_nest_with_indices(structure, flat_sequence, 0, is_seq, sequence_fn)
        if final_index < len(flat_sequence):
            raise IndexError
    except IndexError:
        flat_structure = flatten(structure)
        if len(flat_structure) != len(flat_sequence):
            raise ValueError(
                "Could not pack sequence. Structure had %d elements, but "
                "flat_sequence had %d elements.  Structure: %s, flat_sequence: %s." %
                (len(flat_structure), len(flat_sequence), structure, flat_sequence))
    return sequence_fn(structure, packed)


def flatten_up_to(shallow_tree, input_tree, check_types=True, expand_composites=False):
    """Flattens `input_tree` up to `shallow_tree`.

    Any further depth in structure in `input_tree` is retained as elements in the
    partially flatten output.

    If `shallow_tree` and `input_tree` are not sequences, this returns a
    single-element list: `[input_tree]`.

    Use Case:

    Sometimes we may wish to partially flatten a nested sequence, retaining some
    of the nested structure. We achieve this by specifying a shallow structure,
    `shallow_tree`, we wish to flatten up to.

    The input, `input_tree`, can be thought of as having the same structure layout
    as `shallow_tree`, but with leaf nodes that are themselves tree structures.

    Examples:

    ```python
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]

    flattened_input_tree = flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = flatten_up_to(shallow_tree, shallow_tree)

    # Output is:
    # [[2, 2], [3, 3], [4, 9], [5, 5]]
    # [True, True, False, True]
    ```

    ```python
    input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
    shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

    input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
    input_tree_flattened = flatten(input_tree)

    # Output is:
    # [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
    ```

    Non-Sequence Edge Cases:

    ```python
    flatten_up_to(0, 0)  # Output: [0]
    flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]
    flatten_up_to([0, 1, 2], 0)  # Output: TypeError
    flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]
    ```

    Args:
      shallow_tree: a possibly pruned structure of input_tree.
      input_tree: an arbitrarily nested structure or a scalar object.
        Note, numpy arrays are considered scalars.
      check_types: bool. If True, check that each node in shallow_tree has the
        same type as the corresponding node in input_tree.
      expand_composites: If true, then composite tensors such as
        `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
        component tensors.

    Returns:
      A Python list, the partially flattened version of `input_tree` according to
      the structure of `shallow_tree`.

    Raises:
      TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
      TypeError: If the sequence types of `shallow_tree` are different from
        `input_tree`.
      ValueError: If the sequence lengths of `shallow_tree` are different from
        `input_tree`.

    Examples:
        >>> input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
        >>> shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]
        >>> flatten_up_to(shallow_tree, input_tree)
        [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

    """

    # def assert_shallow_structure(shallow_tree,
    #                              input_tree,
    #                              check_types=True,
    #                              expand_composites=False):
    #     is_seq = is_sequence_or_composite if expand_composites else is_sequence
    #     if is_seq(shallow_tree):
    #         if not is_seq(input_tree):
    #             raise TypeError(
    #                 "If shallow structure is a sequence, input must also be a sequence. "
    #                 "Input has type: %s." % type(input_tree))
    #
    #         if is_instance(shallow_tree, 'wrapt.ObjectProxy'):
    #             shallow_type = type(shallow_tree.__wrapped__)
    #         else:
    #             shallow_type = type(shallow_tree)
    #
    #         if check_types and not isinstance(input_tree, shallow_type):
    #             # Duck-typing means that nest should be fine with two different
    #             # namedtuples with identical name and fields.
    #             shallow_is_namedtuple = _is_namedtuple(shallow_tree, False)
    #             input_is_namedtuple = _is_namedtuple(input_tree, False)
    #             if shallow_is_namedtuple and input_is_namedtuple:
    #                 if not _same_namedtuples(shallow_tree, input_tree):
    #                     raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
    #                         input_type=type(input_tree),
    #                         shallow_type=type(shallow_tree)))
    #
    #             elif ((_is_composite_tensor(shallow_tree) or
    #                    _is_composite_tensor(input_tree)) and
    #                   (_is_type_spec(shallow_tree) or _is_type_spec(input_tree))):
    #                 pass  # Compatibility will be checked below.
    #
    #             elif not (isinstance(shallow_tree, _collections.abc.Mapping) and
    #                       isinstance(input_tree, _collections_abc.Mapping)):
    #                 raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
    #                     input_type=type(input_tree),
    #                     shallow_type=type(shallow_tree)))
    #
    #         if _is_composite_tensor(shallow_tree) or _is_composite_tensor(input_tree):
    #             if not (
    #                     (_is_composite_tensor(input_tree) or _is_type_spec(input_tree)) and
    #                     (_is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree))):
    #                 raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
    #                     input_type=type(input_tree),
    #                     shallow_type=type(shallow_tree)))
    #             type_spec_1 = (shallow_tree if _is_type_spec(shallow_tree) else
    #                            shallow_tree._type_spec)  # pylint: disable=protected-access
    #             type_spec_2 = (input_tree if _is_type_spec(input_tree) else
    #                            input_tree._type_spec)  # pylint: disable=protected-access
    #             try:
    #                 _ = type_spec_1.most_specific_compatible_type(type_spec_2)
    #             except (TypeError, ValueError) as e:
    #                 raise ValueError(
    #                     "Incompatible CompositeTensor TypeSpecs: %s vs. %s -- %s" %
    #                     (type_spec_1, type_spec_2, e))
    #
    #         elif _is_type_spec(shallow_tree):
    #             if not _is_type_spec(input_tree):
    #                 raise TypeError("If shallow structure is a TypeSpec, input must also "
    #                                 "be a TypeSpec.  Input has type: %s."
    #                                 % type(input_tree))
    #         else:
    #             if len(input_tree) != len(shallow_tree):
    #                 raise ValueError(
    #                     _STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
    #                         input_length=len(input_tree), shallow_length=len(shallow_tree)))
    #             elif len(input_tree) < len(shallow_tree):
    #                 raise ValueError(
    #                     _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(
    #                         input_size=len(input_tree), shallow_size=len(shallow_tree)))
    #
    #         if isinstance(shallow_tree, _collections_abc.Mapping):
    #             absent_keys = set(shallow_tree) - set(input_tree)
    #             if absent_keys:
    #                 raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS
    #                                  .format(sorted(absent_keys)))
    #
    #         for shallow_branch, input_branch in zip(yield_value(shallow_tree),
    #                                                 yield_value(input_tree)):
    #             assert_shallow_structure(shallow_branch, input_branch,
    #                                      check_types=check_types,
    #                                      expand_composites=expand_composites)



    is_seq = is_sequence_or_composite if expand_composites else is_sequence
    # assert_shallow_structure(shallow_tree,
    #                          input_tree,
    #                          check_types=check_types,
    #                          expand_composites=expand_composites)
    # Discard paths returned by _yield_flat_up_to.
    return [v for _, v in yield_flat_up_to(shallow_tree, input_tree, is_seq)]
