import functools
import gc
import os
import sys
import traceback
import warnings



def is_in_ipython():
    "Is the code running in the ipython environment (jupyter including)"
    program_name = os.path.basename(os.getenv('_', ''))

    if ('jupyter-notebook' in program_name or  # jupyter-notebook
            'ipython' in program_name or  # ipython
            'jupyter' in program_name or  # jupyter
            'JPY_PARENT_PID' in os.environ):  # ipython-notebook
        return True
    else:
        return False
IS_IN_IPYTHON = is_in_ipython()


def is_in_colab():
    if not is_in_ipython(): return False
    try:
        from google import colab
        return True
    except:
        return False
IS_IN_COLAB = is_in_colab()

def is_in_kaggle_kernel():
    if 'kaggle' in os.environ['PYTHONPATH']:
        return True
    else:
        return False






