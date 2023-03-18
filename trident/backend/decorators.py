from typing import Callable
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



def compact(fun: Callable) -> Callable:
  """Marks the given module method allowing inlined submodules.
  Methods wrapped in @compact can define submodules directly within the method.
  For instance::
    @compact
    __call__(self, x, features):
      x = nn.Dense(features)(x)
      ...
  At most one method in each Module may be wrapped with @compact.
  Args:
    fun: The Module method to mark as compact.
  Returns:
    The given function `fun` marked as compact.
  """
  fun.compact = True  # type: ignore[attr-defined]
  return fun




def signature(fun: Callable) -> Callable:
  """Generate this Callable's signature
  Methods wrapped in @compact can define submodules directly within the method.
  For instance::
    @compact
    __call__(self, x, features):
      x = nn.Dense(features)(x)
      ...
  At most one method in each Module may be wrapped with @compact.
  Args:
    fun: The Module method to mark as compact.
  Returns:
    The given function `fun` marked as compact.
  """
  fun.compact = True  # type: ignore[attr-defined]
  return fun