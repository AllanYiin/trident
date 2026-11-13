"""Optional experiment loggers. Imports and remote runs occur only on start."""
import importlib

import torch.distributed as dist


class ExperimentLogger:
    is_shared = False

    def __init__(self, backend, project, **kwargs):
        if backend not in ("wandb", "swanlab"):
            raise ValueError("backend must be wandb or swanlab")
        self.backend, self.project, self.kwargs = backend, project, kwargs
        self.run = None

    def on_training_start(self, context):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        module = importlib.import_module(self.backend)
        self.run = module.init(project=self.project, **self.kwargs)

    def on_optimization_step_end(self, context):
        if self.run is not None:
            self.run.log(context["language_metrics"], step=context["optimizer_updates"])

    def on_training_end(self, context):
        if self.run is not None:
            self.run.finish()
            self.run = None

    on_training_terminated = on_training_end
