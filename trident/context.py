import inspect
import json
import os
import sys
import time
import threading
import platform
from collections import OrderedDict
import numpy as np
import locale

_trident_context=None



def sanitize_path(path):
    """

    Args:
        path (str): a path of file or folder

    Returns:
        sanitized path

    """
    if isinstance(path, str):
        return os.path.normpath(path.strip()).replace('\\', '/')
    else:
        return path

def split_path(path:str):
    """split path into folder, filename and ext

    Args:
        path (str): a path of file or folder

    Returns:

    """
    if path is None or len(path) == 0:
        return '', '', ''
    path = sanitize_path(path)
    folder, filename = os.path.split(path)
    ext = ''
    if '.' in filename:
        filename, ext = os.path.splitext(filename)
        # handle double ext, like 'mode.pth.tar'
        filename, ext2 = os.path.splitext(filename)
        ext = ext2 + ext
    else:
        folder = os.path.join(folder, filename)
        filename = ''
    return folder, filename, ext

def make_dir_if_need(path):
    """Check the base folder in input path whether exist, if not , then create it.

    Args:
        path (str): a path of file or folder

    Returns:
        sanitized path

    """
    folder, filename, ext = split_path(path)
    if len(folder) > 0 and not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(e)
            sys.stderr.write('folder:{0} is not valid path'.format(folder))
    return sanitize_path(path)



class _ThreadLocalInfo(threading.local):
    """
    Thread local Info used for store thread local attributes.
    """

    def __init__(self):
        super(_ThreadLocalInfo, self).__init__()
        self._reserve_class_name_in_scope = True

    @property
    def reserve_class_name_in_scope(self):
        """Gets whether to save the network class name in the scope."""
        return self._reserve_class_name_in_scope

    @reserve_class_name_in_scope.setter
    def reserve_class_name_in_scope(self, reserve_class_name_in_scope):
        """Sets whether to save the network class name in the scope."""
        if not isinstance(reserve_class_name_in_scope, bool):
            raise ValueError(
                "Set reserve_class_name_in_scope value must be bool!")
        self._reserve_class_name_in_scope = reserve_class_name_in_scope


class _Context:
    """
    _Context is the environment in which operations are executed
    Note:
        Create a context through instantiating Context object is not recommended.
        should use context() to get the context since Context is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._thread_local_info = _ThreadLocalInfo()
        self._context_handle = OrderedDict()
        self._initial_context()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def _get_trident_dir(self):
        """Get or create trident directory
        1)  read from enviorment variable 'TRIDENT_HOME'
        2) use default directory '~/.trident'
        3) if the directory not exist, create it!

        Returns:
            the  trident directory path

        """
        _trident_dir = ''
        if 'TRIDENT_HOME' in os.environ:
            _trident_dir = os.environ.get('TRIDENT_HOME')
        else:
            _trident_base_dir = os.path.expanduser('~')
            if not os.access(_trident_base_dir, os.W_OK):
                _trident_dir = '/tmp/.trident'
            else:
                _trident_dir = os.path.expanduser('~/.trident')

        _trident_dir = sanitize_path(_trident_dir)
        if not os.path.exists(_trident_dir):
            try:
                os.makedirs(_trident_dir)
            except OSError as e:
                # Except permission denied and potential race conditions
                # in multi-threaded environments.
                print(e)

        return _trident_dir

    def _get_plateform(self):
        """

        Returns:
            check current system os plateform.

        """
        plateform_str = platform.system().lower()
        if 'darwin' in plateform_str:
            return 'mac'
        elif 'linux' in plateform_str:
            return 'linux'
        elif 'win' in plateform_str:
            return 'windows'
        else:
            return plateform_str

    def _initial_context(self):
        self._module_dict = dict()
        self.trident_dir = self._get_trident_dir()
        self.backend = 'pytorch'
        self.enable_tensorboard=False
        self.summary_writer=None
        self.locale = locale.getdefaultlocale()[0].lower()

        self.image_backend = 'opencv'
        self.epoch_equivalent = 1000
        self.floatx = 'float32'
        self.epsilon = 1e-7
        self.working_directory = os.getcwd()
        self.plateform = self._get_plateform()
        self.numpy_print_format = '{0:.4e}'
        self.amp_available = False
        self.is_autocast_enabled = False
        _config_path = os.path.expanduser(os.path.join(self.trident_dir, 'trident.json'))
        _config = {}
        if os.path.exists(_config_path):
            try:
                with open(_config_path) as f:
                    _config = json.load(f)
                    for k, v in _config.items():
                        try:
                            if k == 'floatx':
                                assert v in {'float16', 'float32', 'float64'}
                            if k not in ['trident_dir', 'device', 'working_directory']:
                                self.__setattr__(k, v)
                        except Exception as e:
                            print(e)
            except ValueError as ve:
                print(ve)
        if 'TRIDENT_WORKING_DIR' in os.environ:
            self.working_directory = os.environ['TRIDENT_WORKING_DIR']
            os.chdir(os.environ['TRIDENT_WORKING_DIR'])

        if 'TRIDENT_BACKEND' in os.environ:
            if self.backend != os.environ['TRIDENT_BACKEND']:
                self.backend = os.environ['TRIDENT_BACKEND']

        if not  hasattr(self,'backend') or self.backend is None:
            try:
                import torch
                os.environ['TRIDENT_BACKEND'] = 'pytorch'
            except:
                try:
                    import tensorflow
                    os.environ['TRIDENT_BACKEND'] = 'tensorflow'
                except:
                    pass
        np.set_printoptions(formatter={'float_kind': lambda x: self.numpy_print_format.format(x)})
        self.device = None


    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        if attr == "_context_handle" and value is None:
            raise ValueError("Context handle is none in context!!!")
        return value

    @property
    def module_dict(self):
        return self._module_dict

    def get_module(self, cls_name, module_name='module'):
        """Get the registry record.
        Args:
            module_name ():
            cls_name ():
        Returns:
            class: The corresponding class.
        """
        if module_name not in self._module_dict:
            raise KeyError('{module_name} is not in registry')
        dd = self._module_dict[module_name]
        if cls_name not in dd:
            raise KeyError('{cls_name} is not registered in {module_name}')

        return dd[cls_name]

    def _register_module(self, cls, module_name):
        if not inspect.isclass(cls):
            raise TypeError('module must be a class, ' 'but got {type(cls)}')

        cls_name = cls.__name__
        self._module_dict.setdefault(module_name, dict())
        dd = self._module_dict[module_name]
        if cls_name in dd:
            raise KeyError('{cls_name} is already registered '
                           'in {module_name}')
        dd[cls_name] = cls

    def register_module(self, module_name='module'):

        def _register(cls):
            self._register_module(cls, module_name)
            return cls

        return _register

    def get_backend(self):
        return self.backend

    def try_enable_tensorboard(self, summary_writer):
        self.enable_tensorboard=True
        self.summary_writer = summary_writer

    def regist_data_provider(self,data_provider ):
        if not hasattr(self._thread_local_info,'data_providers'):
            self._thread_local_info.data_providers=OrderedDict()
        self._thread_local_info.data_providers[getattr(data_provider,'uuid')]=data_provider

    def get_data_provider(self):
        return list(self._thread_local_info.data_providers.values())

    def regist_resources(self,resource_name,resource ):
        if not hasattr(self._thread_local_info,'resources'):
            self._thread_local_info.resources=OrderedDict()
        self._thread_local_info.resources[resource_name]=resource
        return self._thread_local_info.resources[resource_name]

    def get_resources(self,resource_name):
        if not hasattr(self._thread_local_info, 'resources'):
            self._thread_local_info.resources = OrderedDict()
        if resource_name in self._thread_local_info.resources:
            return self._thread_local_info.resources[resource_name]
        else:
            return None








def _context():
    """
    Get the global _context, if context is not created, create a new one.
    Returns:
        _Context, the global context in PyNative mode.
    """
    global _trident_context
    if _trident_context is None:
        _trident_context = _Context()
    return _trident_context
