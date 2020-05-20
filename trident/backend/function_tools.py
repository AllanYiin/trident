

class Record(dict):
    """
    Easy construction of a record (=immutable singleton class) from keyword arguments.

    Example:
        >>> r = Record(x = 13, y = 42)
        >>> r.x
            13

    Args:
        kwargs: keyword arguments to turn into the record members

    Returns:
        A singleton class instance that has all passed kw args as immutable class members.
    """
    def __init__(self, **args_dict):
        super(Record, self).__init__(args_dict)
        self.__dict__.update(args_dict)
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("record has no attribute '{}'".format(key))
        return self[key]
    def __setattr__(self, key, value):
        raise AttributeError('record is immutable')
    def updated_with(self, **kwargs):
        """
        Create a new Record from an existing one with members modified or added.

        Example:
            >>> r = Record(x = 13)
            >>> r.x
                13
            >>> r2 = r.updated_with(x = 42)
            >>> r2.x
                42

        Args:
            kwargs: keyword arguments to turn into the record members

        Returns:
            A singleton class instance that has all passed kw args as immutable class members.
        """
        d = dict(**self)   # make it mutable
        d.update(kwargs)   # merge the new items
        return Record(**d) # lock it up again



def get_python_function_arguments(f):
    """
    Helper to get the parameter names and annotations of a Python function.
    """
    # Note that we only return non-optional arguments (we assume that any optional args are not specified).
    # This allows to, e.g., accept max(a, b, *more, name='') as a binary function
    import sys
    if sys.version_info.major >= 3:
        from inspect import getfullargspec
    else:
        def getfullargspec(f):
            from inspect import getargspec
            annotations = getattr(f, '__annotations__', {})
            #f.__annotations__ = None  # needed when faking it under Python 3 for debugging purposes
            a = getargspec(f)
            #f.__annotations__ = annotations
            return Record(args=a.args, varargs=a.varargs, varkw=a.keywords, defaults=a.defaults, kwonlyargs=[], kwonlydefaults=None, annotations=annotations)
    param_specs = getfullargspec(f)
    annotations = param_specs.annotations
    arg_names = param_specs.args
    defaults = param_specs.defaults # "if this tuple has n elements, they correspond to the last n elements listed in args"
    if defaults:
        arg_names = arg_names[:-len(defaults)] # we allow Function(functions with default arguments), but those args will always have default values since CNTK Functions do not support this
    return (arg_names, annotations)

def map_function_arguments(params, params_dict, *args, **kwargs):
    """
    Helper to determine the argument map for use with various call operations.
    Returns a dictionary from parameters to whatever arguments are passed.
    Accepted are both positional and keyword arguments.
    This mimics Python's argument interpretation, except that keyword arguments are not optional.
    This does not require the arguments to be Variables or Functions. It is also called by train_minibatch() and @Signature.
    """
    # start with positional arguments
    arg_map = dict(zip(params, args))

    # now look up keyword arguments
    if len(kwargs) != 0:
        for name, arg in kwargs.items():  # keyword args are matched by name
            if name not in params_dict:
                raise TypeError("got an unexpected keyword argument '%s'" % name)
            param = params_dict[name]
            if param in arg_map:
                raise SyntaxError("got multiple values for argument '%s'" % name)
            arg_map[param] = arg # add kw argument to dict
    assert len(arg_map) == len(params)

    return arg_map