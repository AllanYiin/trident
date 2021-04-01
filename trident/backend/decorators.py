
def trident_export(*api_names, **kwargs):
    def Decorator(func_or_class):
        func_or_class._TRIDENT_API = api_names
        return func_or_class
    return Decorator


def deprecated(version, substitute):
    """deprecated warning
    Args:
        version (str): version that the operator or function is deprecated.
        substitute (str): the substitute name for deprecated operator or function.
    """

    def decorate(func):
        def wrapper(*args, **kwargs):
            cls = getattr(args[0], "__class__", None) if args else None
            name = cls.__name__ if cls else func.__name__
            print(f"WARNING: '{func.__name__}' is deprecated from version {version} and will be removed in a future version, "
                  f"use '{substitute}' instead.")
            ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorate