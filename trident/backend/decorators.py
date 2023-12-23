from typing import Callable
import time
import functools

# Global variable to control the state of the performance timing
PERFORMANCE_TIMING_ENABLED = False

def measure_perf(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if PERFORMANCE_TIMING_ENABLED:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            # 檢查是函數還是類別方法
            if '.' in func.__qualname__:
                # 獲得類別名稱和方法名
                class_name, method_name = func.__qualname__.split('.')
                # 檢查類別是否有 'name' 屬性
                if hasattr(args[0], 'training_name'):
                    class_info = f"{class_name}({getattr(args[0], 'training_name')})"
                else:
                    class_info = class_name
                name = f"{class_info}.{method_name}"
            else:
                name = func.__name__
            print(f"Execution time of {name}: {end_time - start_time} seconds")
            return result
        else:
            # Execute the function normally without timing
            return func(*args, **kwargs)
    return wrapper

class PerformanceTimerContext:
    """Context manager to enable or disable performance timing."""
    def __enter__(self):
        global PERFORMANCE_TIMING_ENABLED
        PERFORMANCE_TIMING_ENABLED = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global PERFORMANCE_TIMING_ENABLED
        PERFORMANCE_TIMING_ENABLED = False


def trident_export(*api_names, **kwargs):
    """A decorator function for exporting Trident APIs.

    Args:
        *api_names: Variable length argument list of API names to be exported.

    Returns:
        The decorated function or class.

    Example usage:
        @trident_export("api_name1", "api_name2")
        def my_function():
            pass"""

    def Decorator(func_or_class):
        """Decorator function.

        Args:
            func_or_class: The function or class to be decorated.

        Returns:
            The decorated function or class."""
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
        """Decorator function to print a warning message for deprecated functions.

        Args:
            func: The function to be decorated

        Returns:
            The decorated function"""

        def wrapper(*args, **kwargs):
            """A wrapper function to deprecate a function.

            Args:
                *args: Variable length argument list
                **kwargs: Arbitrary keyword arguments

            Returns:
                The return value of the wrapped function."""
            cls = getattr(args[0], "__class__", None) if args else None
            name = cls.__name__ if cls else func.__name__
            print(
                f"WARNING: '{func.__name__}' is deprecated from version {version} and will be removed in a future version, "
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
