from abc import ABC
from tqdm.auto import tqdm


__all__ = ['CallbackBase','StoppingCriterionCallback','EarlyStoppingCriterionCallback','NumberOfEpochsStoppingCriterionCallback']



class CallbackBase(ABC):
    """
    Objects of derived classes inject functionality in several points of the training process.
    """
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

    def post_loss_calculation(self, training_context):

        """

        Called just after loss calculation.



        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_optimization_step_starting(self, training_context):
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
        Called at the beginning of the evaluation step.
        :param training_context: Dict containing information regarding the training process.

        """

        pass

    def on_progress_end(self, training_context):
        """
        Called at the end of the evaluation step.



        :param training_context: Dict containing information regarding the training process.

        """

        pass


    def on_start_save_model(self, training_context):
        """
        Called just before the optimization step.



        :param training_context: Dict containing information regarding the training process.

        """

        pass


class StoppingCriterionCallback(CallbackBase):

    pass


class EarlyStoppingCriterionCallback(StoppingCriterionCallback):
    """
    Stops the training process if the results do not get better for a number of epochs.
    """
    def __init__(self, patience, evaluation_data_loader_key, evaluator_key, tmp_best_state_filepath):

        """
        :param patience: How many epochs to forgive deteriorating results.
        :param evaluation_data_loader_key: Key of the data-loader dict (provided as an argument to the train method of
            System) that corresponds to the data-set that the early stopping method considers.
        :param evaluator_key: Key of the evaluators dict (provided as an argument to the train method of System) that
            corresponds to the evaluator that the early stopping method considers.
        :param tmp_best_state_filepath: Path where the state of the best so far model will be saved.
        """

        self._patience = patience

        self._evaluation_data_loader_key = evaluation_data_loader_key

        self._evaluator_key = evaluator_key

        self._best_epoch = 0

        self._current_patience = self._patience

        self._best_state_filepath = tmp_best_state_filepath



    def on_training_start(self, training_context):

        self._best_epoch = 0

        self._current_patience = self._patience



    def on_metrics_evaluation_end(self, training_context):
        # current_epoch = training_context['current_epoch']
        # best_results = training_context['metric_history'][list(training_context['metric_history'].keys())[0]][self._best_epoch]
        # current_results = training_context['metric_history'][list(training_context['metric_history'].keys())[0]][current_epoch]
        # if current_results.is_better_than(best_results) or current_epoch == 0:
        #     self._best_epoch = current_epoch
        #     self._current_patience = self._patience
        #     training_context['system'].save_model_state(self._best_state_filepath)
        # else:
        #     self._current_patience -= 1
        # if self._current_patience == -1:
        #     training_context['stop_training'] = True
        pass


    def on_training_end(self, training_context):
        tqdm.write("Epoch chosen: %d" % self._best_epoch)
        training_context['system'].load_model_state(self._best_state_filepath)

class NumberOfEpochsStoppingCriterionCallback(StoppingCriterionCallback):
    """
    Stops the training process after a number of epochs.
    """
    def __init__(self, nb_of_epochs):
        """
        :param nb_of_epochs: Number of epochs to train.
        """
        self._nb_of_epochs = nb_of_epochs


    def on_epoch_end(self, training_context):

        if training_context['current_epoch'] == self._nb_of_epochs - 1:
            training_context['stop_training'] = True




from .AdjustLRCallbacks import *
from .ModelSavingCallbacks import *
from .TrainingTimeRegularizationCallbacks import *