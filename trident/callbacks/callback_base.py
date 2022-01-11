import inspect
import uuid
import warnings
from abc import ABC
from  functools import partial
from types import MethodType
import numpy as np
from tqdm.auto import tqdm

__all__ = ['CallbackBase','StoppingCriterionCallback','EarlyStoppingCriterionCallback','LambdaCallback','UnfreezeModelCallback']

from trident import context
ctx = context._context()
_backend = ctx.get_backend()
if _backend=="pytorch":
    import torch
    from trident.backend.pytorch_backend import *


_valid_when=["on_training_start"
 ,"on_training_end"
 ,"on_epoch_start"
 ,"on_epoch_end"
 ,"on_overall_epoch_end"
 ,"on_batch_start"
 ,"on_batch_end"
 ,"on_overall_batch_end"
 ,"on_data_received"
 ,"on_infer_start"
 ,"on_infer_end"
 ,"on_loss_calculation_start"
 ,"on_loss_calculation_end"
 ,"on_optimization_step_start"
 ,"on_optimization_step_end"
 ,"on_metrics_evaluation_start"
 ,"on_metrics_evaluation_end"
 ,"on_progress_start"
 ,"on_progress_end"
,"on_excution_exception"
 ,"on_model_saving_start"
 ,"on_model_saving_end"
,"on_get_observation_start"
,"on_get_observation_end"
,"on_select_action_start"
,"on_select_action_end"
,"on_get_rewards_start"
,"do_on_get_rewards_end"


             ]

class CallbackBase(ABC):
    """
    Objects of derived classes inject functionality in several points of the training process.
    """

    def __init__(self,is_shared=False):
        self.is_shared=is_shared
        self.uuid=str(uuid.uuid4())[:8].__str__().replace('-', '')

    def __eq__(self, other):
        return self.uuid==other.uuid

    def on_training_start(self, training_context):
        """
        Called at the beginning of the training process.
        :param training_context: Dict containing information regarding the training process.
        """
        pass
    def on_training_end(self, training_context):
        """
        Called at the end of the training process.
        :param training_context: Dict containing information regarding the training process.
        """
        pass

    def on_training_terminated(self, training_context):
        """
        Called at the end of the training process.
        :param training_context: Dict containing information regarding the training process.
        """
        pass

    def on_epoch_start(self, training_context):

        """

        Called at the beginning of a new epoch.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_epoch_end(self, training_context):
        """
        Called at the end of an epoch.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_overall_epoch_end(self, training_context):
        """
        Called after a batch has been processed.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_batch_start(self, training_context):
        """
        Called just before processing a new batch.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_batch_end(self, training_context):
        """
        Called after a batch has been processed.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_overall_batch_end(self, training_context):
        """
        Called after a batch has been processed.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_data_received(self, training_context):
        """
        Called just before processing a new batch.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_infer_start(self, training_context):

        """

        Called just after prediction during training time.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_infer_end(self, training_context):

        """

        Called just after prediction during training time.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_loss_calculation_start(self, training_context):

        """
        Called just after loss calculation.
        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_loss_calculation_end(self, training_context):
        """

        Called just after loss calculation.



        :param training_context: Dict containing information regarding the training process.

        """

        pass


    def on_optimization_step_start(self, training_context):
        """
        Called just before the optimization step.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_optimization_step_end(self, training_context):

        """

        Called just after backward is called.



        :param training_context: Dict containing information regarding the training process.

        """

        pass


    def on_metrics_evaluation_start(self, training_context):
        """
        Called at the beginning of the evaluation step.
        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_metrics_evaluation_end(self, training_context):
        """
        Called at the end of the evaluation step.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_progress_start(self, training_context):
        """
        Called at the beginning of the printing progress step.
        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_progress_end(self, training_context):
        """
        Called at the end of the printing progress step.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_excution_exception(self, training_context):
        """
        Called when the expection occure.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_model_saving_start(self, training_context):
        """
        Called just before the optimization step.



        :param training_context: Dict containing information regarding the training process.

        """

        pass
    def on_model_saving_end(self, training_context):
        """
        Called just before the optimization step.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_get_observation_start(self, training_context):
        pass

    def on_get_observation_end(self, training_context):
        pass

    def on_select_action_start(self, training_context):
        pass

    def on_select_action_end(self, training_context):
        pass

    def on_get_rewards_start(self, training_context):
        pass

    def on_get_rewards_end(self, training_context):
        pass

    

class LambdaCallback(CallbackBase):
    """
    Objects of derived classes inject functionality in several points of the training process.
    """

    def __init__(self,when='on_batch_end',frequency=1,unit='batch',action=None,is_shared=False):
        super(LambdaCallback, self).__init__(is_shared=is_shared)
        self.is_shared=is_shared

        self.frequency = frequency if frequency is not None else 1
        self.action = None
        if action is None:
            raise ValueError("action cannot be None")
        argspec = inspect.getfullargspec(action)
        if 'training_context' in argspec.args and len(argspec.args)==1:
            self.action=action
        else:
            raise ValueError("action should has only-one argment 'training_context")
        if when in _valid_when:
            self.when=when
        else:
            raise ValueError("{0} is not valid event trigger.".format(when))
        self.unit=unit


        def on_trigger(self, training_context):
            steps=training_context['steps']
            epoch=training_context['current_epoch']
            if self.unit=='epoch' and (epoch+1)% (self.frequency )==0:
                if ctx.amp_available== True and ctx.is_autocast_enabled  == True and get_device() == 'cuda':
                    with torch.cuda.amp.autocast():
                        self.action(training_context)
                else:
                    self.action(training_context)
            elif self.unit=='batch' and (steps+1)% (self.frequency)==0:
                if ctx.amp_available == True and ctx.is_autocast_enabled == True and get_device() == 'cuda':
                    with torch.cuda.amp.autocast():
                        self.action(training_context)
                else:
                    self.action(training_context)

        setattr(self,when,MethodType(on_trigger, self))









class StoppingCriterionCallback(CallbackBase):
    def __init__():
        super().__init__()
    pass


class EarlyStoppingCriterionCallback(StoppingCriterionCallback):
    """
    Stops the training process if the results do not get better for a number of epochs.
    """
    def __init__(self, monitor='total_losses',mode='min',patience=5, min_delta=1e-4,stopped_epoch=None):

        """
        Args:
            monitor: quantity to be monitored.
            patience: How many epochs to forgive deteriorating results.
            mode: one of {auto, min, max}. In `min` mode,
                lr will be reduced when the quantity
                monitored has stopped decreasing; in `max`
                mode it will be reduced when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
            min_delta: threshold for measuring the new optimum,
                to only focus on significant changes.
            stopped_epoch:
        """
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.wait = 0
        self.best = 0
        if monitor == 'total_losses':
            mode = 'min'
        self.mode = mode
        self.monitor_op=None
        self.training_items=None
        self.stopped_epoch=stopped_epoch


    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStoppingCriterionCallback Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.wait = 0


    def on_training_start(self, training_context):
        self._reset()
        if 'num_epochs' in training_context:
            stopped_epoch=training_context['num_epochs']
        elif 'total_epoch'  in training_context:
            stopped_epoch=training_context['total_epoch']


    def on_overall_epoch_end(self, training_context):
        results=[]
        self.training_items = training_context['training_items']
        for item in self.training_items:

            history = item.training_context['batch_losses'].get(self.monitor, item.training_context['batch_metrics'].get(self.monitor,
                                                                                                               item.training_context[
                                                                                                                   'batch_losses'][
                                                                                                                   'total_losses']))

            if history is None:
                warnings.warn(
                    'EarlyStoppingCriterionCallback conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (self.monitor, ','.join(item.training_context['batch_metrics'].keys_list)), RuntimeWarning
                )
            else:
                current = np.array(history[-min(5, len(history)):]).mean()

                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                    results.append(False)
                else:
                    self.wait += 1
                    if self.wait >= self.patience  :
                        results.append(True)
                    else:
                        results.append(False)
        if  self.stopped_epoch is not None and self.training_items[0].training_context['steps']>=self.stopped_epoch:
                    results.append(True)


    def on_overall_batch_end(self, training_context):
        results=[]
        self.training_items = training_context['training_items']
        for item in self.training_items:

            history = item.training_context['batch_losses'].get(self.monitor, item.training_context['batch_metrics'].get(self.monitor,
                                                                                                               item.training_context[
                                                                                                                   'batch_losses'][
                                                                                                                   'total_losses']))

            if history is None:
                warnings.warn(
                    'EarlyStoppingCriterionCallback conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (self.monitor, ','.join(item.training_context['batch_metrics'].keys_list)), RuntimeWarning
                )
            else:
                current = np.array(history[-min(5, len(history)):]).mean()

                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                    results.append(False)
                else:
                    self.wait += 1
                    if self.wait >= self.patience  :
                        results.append(True)
                    else:
                        results.append(False)
        if  self.stopped_epoch is not None and self.training_items[0].training_context['steps']>=self.stopped_epoch:
                    results.append(True)


    def on_training_end(self, training_context):
        tqdm.write("Epoch chosen: %d" % self._best_epoch)
        training_context['system'].load_model_state(self._best_state_filepath)


class UnfreezeModelCallback(CallbackBase):
    def __init__(self, frequency: int, unit='epoch',slice_from=None, slice_to=None, module_name=None):
        super().__init__()
        self.unit=unit
        self.frequency=frequency
        self.slice_from=slice_from
        if self.slice_from is None:
            self.slice_from=0
        self.slice_to=slice_to
        self.module_name=module_name

    def unfreeze_model(self,training_context):
        model=training_context["current_model"]

        if self.module_name is not None:
            for name,module in model.named_modules():
                if name==self.module_name or module.name==self.module_name:
                    module.trainable = True
        else:
            if "Sequential" in model.__class__.__name__:
                if self.slice_from == 0 and  self.slice_to is None:
                    model.trainable = True
                elif isinstance(self.slice_from, int) and isinstance(self.slice_to, int):
                    layers=model[self.slice_from:self.slice_to]
                    for layer in layers:
                        layer.trainable = True
                elif isinstance(self.slice_from, int) and self.slice_to is None:
                    layers=model[self.slice_from:]
                    for layer in layers:
                        layer.trainable = True


    def on_batch_end(self, training_context):
        if self.unit == 'batch' and training_context['steps'] == self.frequency:
            self.unfreeze_model(training_context)
    def on_epoch_end(self, training_context):
        if self.unit == 'epoch' and training_context['current_epoch'] == self.frequency:
            self.unfreeze_model(training_context)





