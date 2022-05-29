import functools
import operator
import threading
from abc import ABC, abstractmethod

from trident.backend.common import import_or_install, get_session

import logging
import os
import re
from argparse import Namespace
import time
from typing import Any, Dict, Optional, Union

from trident.loggers.logger import BaseLogger
from trident.context import split_path, make_dir_if_need, sanitize_path
ctx=get_session()
if ctx.get_backend() == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import Tensor
    from trident.backend.pytorch_ops import *

elif ctx.get_backend() == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import Tensor
    from trident.backend.tensorflow_ops import *





_MLFLOW_AVAILABLE =True
try:
    import mlflow
    from mlflow.tracking import context, MlflowClient
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

except ImportError:
    try:
        import_or_install('mlflow','mlflow')
        import mlflow
        from mlflow.tracking import context, MlflowClient
        from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
    except ImportError:
        _MLFLOW_AVAILABLE=False
        raise ImportError( 'You want to use `mlflow` logger which is not installed yet,'
                ' install it with `pip install mlflow`.')

# before v1.1.0
if hasattr(context, 'resolve_tags'):
    from mlflow.tracking.context import resolve_tags


# since v1.1.0
elif hasattr(context, 'registry'):
    from mlflow.tracking.context.registry import resolve_tags
else:
    def resolve_tags(tags=None):
        return tags
LOCAL_FILE_URI_PREFIX = "file:"




class MLFlowLogger(object):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """
    __singleton_lock = threading.Lock()
    __singleton_instance = None

    # define the classmethod
    @classmethod
    def instance(cls):

        # check for the singleton instance
        if not cls.__singleton_instance:
            with cls.__singleton_lock:
                if not cls.__singleton_instance:
                    cls.__singleton_instance = cls()

                    # return the singleton instance
        return cls.__singleton_instance
    def __init__(self,experiment_name="my-experiment"):
        mlflow.set_tracking_uri('http://{0}:{1}'.format(ctx.mlflow_server, 4040))
        self.client= MlflowClient()
        make_dir_if_need('Log/images')
        self.run=None
        self.file_writer=None
        self.all_writers=None
        self.experiment_name=experiment_name
        experiments = self.client.list_experiments()
        experiment_ids=[e.experiment_id for e in experiments if e.name==self.experiment_name]
        if len(experiment_ids)>0:
            self.experiment_id =experiment_ids[-1]
        else:
            self.experiment_id = self.client.create_experiment(self.experiment_name,)

    def start_run(self,run_id=None,experiment_id=None,run_name=None):
        if experiment_id is None:
            experiment_id=self.experiment_id
        self.run= self.client.create_run(experiment_id)
        mlflow.start_run(run_id=self.run.info.run_id,experiment_id=experiment_id,run_name=run_name)

    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
       pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Add scalar data to summary.

        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar.png
           :scale: 50 %

        """
        self.client.log_metric(run_id=self.run.info.run_id,key=tag, value=to_scalar(scalar_value),timestamp=walltime if walltime is not None else int(time.time()), step=global_step)
        #log_metric(run_id: str, key: str, value: float, timestamp: Optional[int] = None, step: Optional[int] = None) → None


    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        pass

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None):
        pass

    def add_histogram_raw(self, tag, min, max, num, sum, sum_squares,
                          bucket_limits, bucket_counts, global_step=None,
                          walltime=None):
        pass

    def add_image(self, img_path):

        self.client.log_image(run_id=self.run.info.run_id,image=image2array(img_path).astype(np.uint8),artifact_file='Log/images')


    def add_images(self, tag, img_tensor, global_step=None, walltime=None):
        pass

    def add_figure(self, tag, figure, artifact_file=None):
        self.client.log_figure(run_id=self.run.info.run_id,figure=figure,artifact_file=artifact_file)
        #log_figure(run_id: str, figure: Union[matplotlib.figure.Figure, plotly.graph_objects.Figure], artifact_file: str)



    def add_onnx_graph(self, prototxt):
        pass


    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        if self.all_writers is None:
            return
        for writer in self.all_writers.values():
            writer.flush()

    def close(self):
        if self.all_writers is None:
            return  # ignore double close
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None
        self.client.set_terminated(self.run.info.run_id,end_time=time.time())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()