"""LR warmup policies shared by the callback and native optimizer runners."""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, isfinite, pi, sin
from numbers import Integral, Real

from trident.callbacks.callback_base import CallbackBase


@dataclass(frozen=True)
class _UpdateWarmup:
    """Immutable configuration. Each optimizer owns its update counter."""

    steps: int
    start_factor: float = 0.01

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, Integral) or self.steps < 1:
            raise ValueError("warmup steps must be a positive integer")
        if (isinstance(self.start_factor, bool) or not isinstance(self.start_factor, Real)
                or not 0 <= self.start_factor <= 1):
            raise ValueError("warmup start_factor must be between 0 and 1")

    @classmethod
    def from_ratio(cls, ratio, *, total_updates, start_factor=0.01):
        """Use a fraction of a planned optimizer update budget, rounded up."""
        if isinstance(total_updates, bool) or not isinstance(total_updates, Integral) or total_updates < 1:
            raise ValueError("total_updates must be a positive integer")
        if isinstance(ratio, bool) or not isinstance(ratio, Real) or not 0 < ratio <= 1:
            raise ValueError("warmup ratio must be greater than 0 and at most 1")
        # Use the decimal representation so e.g. 7% of 100 is exactly 7.
        return cls(ceil(Fraction(str(ratio)) * int(total_updates)), start_factor)

    def factor_at(self, update):
        """LR multiplier for a zero-based warmup update index."""
        progress = update / (self.steps - 1) if self.steps > 1 else 1.0
        return self._factor(progress)

    def config_dict(self):
        config = dict(steps=int(self.steps), start_factor=float(self.start_factor))
        # Checkpoints written before multiple curves were supported were linear.
        if self.curve != 'linear':
            config['curve'] = self.curve
        return config


@dataclass(frozen=True)
class LinearWarmup(_UpdateWarmup):
    """Increase the LR by equal increments, reaching the base LR on the last update."""

    curve = 'linear'

    def _factor(self, progress):
        return self.start_factor + (1 - self.start_factor) * progress


@dataclass(frozen=True)
class CosineWarmup(_UpdateWarmup):
    """Half-cosine rise with a smooth start and finish."""

    curve = 'cosine'

    def _factor(self, progress):
        return self.start_factor + (1 - self.start_factor) * sin(pi * progress / 2) ** 2


@dataclass(frozen=True)
class ExponentialWarmup(_UpdateWarmup):
    """Multiply the LR by the same ratio between consecutive warmup updates."""

    curve = 'exponential'

    def __post_init__(self):
        super().__post_init__()
        if self.start_factor == 0:
            raise ValueError("exponential warmup requires a positive start_factor")

    def _factor(self, progress):
        return self.start_factor ** (1 - progress)


@dataclass(frozen=True)
class ConstantWarmup(_UpdateWarmup):
    """Hold the lower LR for all warmup updates, then hand off on the next update."""

    curve = 'constant'

    def _factor(self, progress):
        return self.start_factor


def resolve_warmup(warmup, optimizer):
    """Resolve configuration values before constructing the scheduler or checkpoint config."""
    if isinstance(warmup, str):
        name = warmup.strip()
        # ConfigParser preserves quotes around INI values.
        if len(name) >= 2 and name[0] in ('"', "'") and name[-1] == name[0]:
            name = name[1:-1]
        if name.casefold() != 'cosine':
            raise ValueError(f"unknown warmup preset {warmup!r}; expected 'cosine'")
        warmup = CosineWarmup.from_ratio(0.03, total_updates=12_000)
    validate_warmup(warmup, optimizer)
    return warmup


def validate_warmup(warmup, optimizer):
    if not isinstance(warmup, _UpdateWarmup):
        if isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 0:
            raise ValueError("warmup must be a warmup policy or a nonnegative integer epoch count")
        if warmup == 0:
            return
    if optimizer is None:
        raise ValueError("configure the optimizer before warmup")
    if getattr(optimizer, 'use_warmup', False) or any(g.get('warmup', 0) for g in optimizer.param_groups):
        raise ValueError("disable the optimizer's built-in warmup before configuring scheduler warmup")


def set_group_lr(group, lr):
    """Keep tensor-valued optimizer LRs connected to their existing storage."""
    current = group['lr']
    if hasattr(current, 'fill_'):
        current.fill_(lr)
    elif hasattr(current, 'assign'):
        current.assign(lr)
    else:
        group['lr'] = lr


class WarmupScheduler(CallbackBase):
    """Per-optimizer warmup followed by a Trident callback or a native scheduler."""

    _state_fields = ('updates', 'main_updates', 'first_epoch', 'first_batch', 'main_epoch',
                     'main_batch', 'last_main_epoch', 'base_lrs', 'entry_lrs')

    def __init__(self, warmup, optimizer, scheduler=None, base_lrs=None):
        super().__init__()
        self.warmup = warmup
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.native = scheduler is not None and not isinstance(scheduler, CallbackBase)
        self.entry_lrs = [float(g['lr']) for g in optimizer.param_groups]
        self.base_lrs = list(base_lrs) if base_lrs is not None else self.entry_lrs.copy()
        self.updates = 0
        self.main_updates = 0
        self.first_epoch = None
        self.first_batch = None
        self.main_epoch = None
        self.main_batch = None
        self.last_main_epoch = None

    def _write_lrs(self, rates):
        for group, lr in zip(self.optimizer.param_groups, rates):
            set_group_lr(group, lr)

    def _local_context(self, context):
        return dict(context, optimizer=self.optimizer, warmup_base_lrs=self.base_lrs,
                    base_lr=next((lr for lr in self.base_lrs if lr), 0.0),
                    current_lr=self.optimizer.param_groups[0]['lr'],
                    steps=self.main_updates, current_epoch=context['current_epoch'] - self.main_epoch,
                    current_batch=(context['current_batch'] - self.main_batch
                                   if context['current_epoch'] == self.main_epoch else context['current_batch']),
                    total_epoch=context['total_epoch'] - self.main_epoch)

    def _warmup_factor(self, context):
        if isinstance(self.warmup, _UpdateWarmup):
            if self.updates >= self.warmup.steps:
                return None
            return self.warmup.factor_at(self.updates)
        elapsed = context['current_epoch'] - self.first_epoch
        if context.get('only_steps'):
            from trident import context as trident_context
            equivalent = trident_context._context().epoch_equivalent
            elapsed = (context['steps'] - self.first_batch) // equivalent
        if elapsed >= self.warmup:
            return None
        return 0.01 + 0.99 * (elapsed / self.warmup)

    def on_optimization_step_start(self, context):
        epoch = context['current_epoch']
        if context.get('optimizer') is not None and context['optimizer'] is not self.optimizer:
            raise ValueError("configure the LR scheduler again after replacing its optimizer")
        # Loss modules may add optimizer groups after the configuration-time check.
        if self.first_epoch is None or len(self.optimizer.param_groups) != len(self.base_lrs):
            validate_warmup(self.warmup, self.optimizer)
        if self.first_epoch is None:
            self.first_epoch = epoch
            self.first_batch = context['steps']
            if not self.native:
                self.base_lrs = [float(g['lr']) for g in self.optimizer.param_groups]
                self.entry_lrs = self.base_lrs.copy()
        for group in self.optimizer.param_groups[len(self.base_lrs):]:
            self.base_lrs.append(float(group['lr']))
            self.entry_lrs.append(float(group['lr']))

        if self.main_epoch is None:
            factor = self._warmup_factor(context)
            if factor is not None:
                self._write_lrs(lr * factor for lr in self.base_lrs)
            else:
                self.main_epoch = epoch
                self.main_batch = context['current_batch']
                self._write_lrs(self.entry_lrs)
                if self.scheduler is not None and not self.native:
                    local = self._local_context(context)
                    batches = context['total_batch']
                    if batches is not None and batches > 0 and isfinite(batches):
                        window = context.get('gradient_accumulation_steps', 1)
                        future_epochs = local['total_epoch'] - 1
                        remaining = batches - context['current_batch'] - 1
                        local['schedule_total_steps'] = (1 + ceil(remaining / window)
                                                         + future_epochs * ceil(batches / window))
                    self.scheduler.on_training_start(local)
        if self.main_epoch is not None and self.scheduler is not None and not self.native:
            local = self._local_context(context)
            if self.last_main_epoch != epoch:
                self.scheduler.on_epoch_start(local)
            self.scheduler.on_batch_start(local)
            self.scheduler.on_optimization_step_start(local)
        context['current_lr'] = float(self.optimizer.param_groups[0]['lr'])

    def on_optimization_step_end(self, context):
        self.updates += 1
        if self.main_epoch is not None:
            if self.native:
                self.scheduler.step()
            elif self.scheduler is not None:
                local = self._local_context(context)
                self.scheduler.on_optimization_step_end(local)
                self.scheduler.on_batch_end(local)
            self.last_main_epoch = context['current_epoch']
            self.main_updates += 1

    def on_epoch_end(self, context):
        if self.last_main_epoch == context['current_epoch'] and self.scheduler is not None and not self.native:
            self.scheduler.on_epoch_end(self._local_context(context))

    def config_dict(self):
        inner = type(self.scheduler)
        warmup = (self.warmup.config_dict() if isinstance(self.warmup, _UpdateWarmup)
                  else dict(epochs=int(self.warmup)))
        return dict(warmup=warmup, scheduler=inner.__module__ + '.' + inner.__qualname__)

    def state_dict(self):
        state = {key: deepcopy(getattr(self, key)) for key in self._state_fields}
        state.update(config=self.config_dict(), lrs=[float(g['lr']) for g in self.optimizer.param_groups],
                     scheduler=None if self.scheduler is None else self.scheduler.state_dict())
        return state

    def load_state_dict(self, state):
        """Restore after loading the matching optimizer state."""
        if state['config'] != self.config_dict() or len(state['lrs']) != len(self.optimizer.param_groups):
            raise ValueError("warmup/scheduler configuration or optimizer group count mismatch")
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        for key in self._state_fields:
            setattr(self, key, deepcopy(state[key]))
        self._write_lrs(state['lrs'])
