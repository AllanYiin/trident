"""Opt-in language item for the existing fluent TrainingPlan interface."""
import copy
from pathlib import Path

import torch

from trident.callbacks.warmup import WarmupScheduler, resolve_warmup
from trident.optims.pytorch_trainer import LanguageModel

from .checkpoint import atomic_torch_save, cpu_state, tokenizer_config
from .generation import generate_tokens
from .lora import adapter_state_dict, inject_lora, load_adapter_state_dict, merge_lora
from .objectives import CausalLMObjective, unwrap_model


class CausalLanguageModel(LanguageModel):
    uses_objective_runner = True

    def __init__(self, network, tokenizer=None, name="language_model"):
        if not isinstance(network, torch.nn.Module):
            raise TypeError("network must be a PyTorch nn.Module")
        super().__init__()
        self._model = network
        object.__setattr__(self, "_language_name", name)
        self.tokenizer = tokenizer
        self.objective = CausalLMObjective()
        self.precision = "fp32"
        self.lora_config = None
        self.extra_losses = {}
        self.extra_metrics = {}
        self.optimizer_updates = 0
        self.microbatches = 0
        self.valid_tokens = 0
        self.history = []
        self._prepared = False
        self._language_scheduler = None
        self._scaler = None
        self._optimizer_config = None

    @property
    def name(self):
        return self._language_name

    def __setattr__(self, name, value):
        if name == "name":
            object.__setattr__(self, "_language_name", value)
        elif name in ("optimizer_updates", "microbatches", "valid_tokens", "history"):
            # ModelBase routes existing context keys into training_context;
            # these are owned counters, mirrored into callbacks only.
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    @property
    def device(self):
        return next(self._model.parameters()).device

    def to(self, device):
        if self._prepared:
            raise RuntimeError("move the model before starting training")
        self._model.to(device)
        return self

    def with_objective(self, objective):
        if not callable(objective) or not hasattr(objective, "prepare"):
            raise TypeError("objective must implement prepare and __call__")
        if self._prepared:
            raise RuntimeError("create a new training item for a new objective/stage")
        self.objective = objective
        return self

    def with_lora(self, rank=8, alpha=16, target_modules=("q_proj", "v_proj"), dropout=0.0):
        if self.optimizer is not None:
            raise RuntimeError("install LoRA before constructing the optimizer")
        self.lora_config = inject_lora(self._model, rank, alpha, target_modules, dropout)
        return self

    def with_optimizer(self, optimizer=torch.optim.AdamW, **kwargs):
        if self._prepared:
            raise RuntimeError("cannot replace optimizer during a training run")
        parameters = [value for value in self._model.parameters() if value.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable parameters")
        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer, None)
            if optimizer is None:
                raise ValueError("unknown torch optimizer")
        if isinstance(optimizer, torch.optim.Optimizer):
            if kwargs:
                raise ValueError("optimizer instance does not accept constructor kwargs")
            installed = {id(value) for group in optimizer.param_groups for value in group["params"]}
            if installed != {id(value) for value in parameters}:
                raise ValueError("optimizer must contain exactly the trainable parameters")
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer(parameters, **kwargs)
        self._optimizer_config = {"type": type(self.optimizer).__module__ + "." + type(self.optimizer).__name__}
        return self

    def with_accumulate_grads(self, accumulation_steps=2):
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
        self.accumulation_steps = accumulation_steps
        return self

    def with_precision(self, precision="fp32"):
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self._prepared:
            raise RuntimeError("set precision before training")
        self.precision = precision
        return self

    def with_automatic_mixed_precision_training(self, dtype="fp16", **kwargs):
        return self.with_precision(dtype)

    def with_grad_clipping(self, clipping_threshold=1.0, **kwargs):
        if clipping_threshold <= 0:
            raise ValueError("clipping_threshold must be positive")
        self.grad_clipping_threshold = float(clipping_threshold)
        self.grad_clipping_by_norm = True
        return self

    def with_gradient_checkpointing(self):
        method = getattr(self._model, "gradient_checkpointing_enable", None)
        if not callable(method):
            raise ValueError("model does not support gradient checkpointing")
        method()
        return self

    def with_learning_rate_scheduler(self, lr_schedule=None, warmup=0, **kwargs):
        if self.optimizer is None:
            raise RuntimeError("configure optimizer before scheduler")
        warmup = resolve_warmup(warmup, self.optimizer)
        if kwargs and not callable(lr_schedule):
            raise TypeError("scheduler options require a scheduler factory")
        if warmup:
            # Native scheduler constructors may immediately set their first LR.
            base_lrs = [float(g['lr'] if callable(lr_schedule) else g.get('initial_lr', g['lr']))
                        for g in self.optimizer.param_groups]
        scheduler = lr_schedule(self.optimizer, **kwargs) if callable(lr_schedule) else lr_schedule
        if scheduler is not None and getattr(scheduler, "optimizer", None) is not self.optimizer:
            raise ValueError("scheduler must own this optimizer")
        self._language_scheduler = WarmupScheduler(warmup, self.optimizer, scheduler, base_lrs) if warmup else scheduler
        self.lr_scheduler = self._language_scheduler
        return self

    def with_loss(self, loss, loss_weight=1.0, name="", **kwargs):
        if kwargs:
            raise TypeError("language extra losses use loss(outputs, batch), weight and name")
        if not callable(loss):
            raise TypeError("loss must be callable")
        self.extra_losses[name or getattr(loss, "__name__", type(loss).__name__)] = (loss, float(loss_weight))
        return self

    def with_metric(self, metric, name="", **kwargs):
        if not callable(metric):
            raise TypeError("metric must be callable")
        self.extra_metrics[name or getattr(metric, "__name__", type(metric).__name__)] = metric
        return self

    def prepare(self):
        if self.optimizer is None:
            raise RuntimeError("configure an optimizer before training")
        if self.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; use fp32 or bf16 on CPU")
        if self.precision == "bf16" and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support bf16")
        self.objective.prepare(self._model, self.device)
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
        self._prepared = True

    def autocast(self):
        return torch.autocast(device_type=self.device.type,
                              dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16,
                              enabled=self.precision != "fp32")

    def compute(self, batch):
        with self.autocast():
            result = self.objective(self._model, batch)
            for name, (function, weight) in self.extra_losses.items():
                extra = function(result.outputs, batch)
                if not torch.is_tensor(extra) or extra.ndim:
                    raise ValueError("extra loss must return a scalar tensor")
                result.loss_sum = result.loss_sum + weight * extra * result.weight
                result.metrics[name] = float(extra.detach())
        with torch.no_grad():
            for name, function in self.extra_metrics.items():
                result.metrics[name] = float(function(result.outputs, batch))
        if result.loss_sum.ndim or not bool(torch.isfinite(result.loss_sum)):
            raise FloatingPointError("objective produced an invalid scalar loss")
        if result.weight < 0:
            raise ValueError("objective weight cannot be negative")
        return result

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return generate_tokens(unwrap_model(self._model), input_ids, attention_mask, **kwargs)

    @classmethod
    def from_huggingface(cls, path, tokenizer=None, local_files_only=True, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        network = AutoModelForCausalLM.from_pretrained(path, local_files_only=local_files_only, **kwargs)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        return cls(network, tokenizer)

    def save_adapter(self, path):
        if self.lora_config is None:
            raise ValueError("no LoRA adapter installed")
        atomic_torch_save({"config": self.lora_config, "state_dict": adapter_state_dict(unwrap_model(self._model))}, path)
        return Path(path)

    def load_adapter(self, path):
        if self.optimizer is not None:
            raise RuntimeError("load adapter before constructing optimizer")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if self.lora_config is None:
            self.with_lora(**state["config"])
        elif self.lora_config != state["config"]:
            raise ValueError("adapter configuration mismatch")
        load_adapter_state_dict(self._model, state["state_dict"])
        return self

    def export(self, directory, format="native", merge_adapter=True):
        from .interop import export_model
        network = copy.deepcopy(unwrap_model(self._model)).cpu()
        if self.lora_config:
            if not merge_adapter:
                raise ValueError("use save_adapter for an unmerged adapter export")
            merge_lora(network)
        return export_model(network, self.tokenizer, directory, format)

    def save_model(self, save_path=None, **kwargs):
        path = save_path or self.training_context.get("save_path")
        if not path:
            raise ValueError("save path is required")
        atomic_torch_save(cpu_state(unwrap_model(self._model).state_dict()), path)
        return Path(path)

    def config_dict(self):
        model = unwrap_model(self._model)
        config = getattr(model, "config", None)
        scheduler = self._language_scheduler
        scheduler_config = scheduler.config_dict() if isinstance(scheduler, WarmupScheduler) else (
            None if scheduler is None else type(scheduler).__name__)
        return {"model_type": type(model).__module__ + "." + type(model).__name__,
                "model": config.to_dict() if hasattr(config, "to_dict") else None,
                "objective": self.objective.config_dict(), "lora": self.lora_config,
                "precision": self.precision, "accumulation_steps": self.accumulation_steps,
                "optimizer": self._optimizer_config,
                "scheduler": scheduler_config,
                "grad_clipping": self.grad_clipping_threshold if self.grad_clipping_by_norm else None,
                "tokenizer": tokenizer_config(self.tokenizer),
                "extra_losses": {name: weight for name, (_, weight) in self.extra_losses.items()},
                "extra_metrics": list(self.extra_metrics)}
