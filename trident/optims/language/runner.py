"""Strict language execution path, selected explicitly by TrainingPlan items."""
import contextlib
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from trident.callbacks.warmup import WarmupScheduler

from .checkpoint import (atomic_torch_save, capture_rng, cpu_state, data_state,
                         loader_config, read_checkpoint, restore_rng)
from .objectives import unwrap_model


def move_batch(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {getattr(key, "name", key): move_batch(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move_batch(item, device) for item in value)
    return value


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _world():
    return dist.get_world_size() if dist.is_initialized() else 1


def _notify(plan, item, event):
    seen = set()
    for callback in list(plan.callbacks) + list(item.callbacks):
        if id(callback) in seen:
            continue
        seen.add(id(callback))
        method = getattr(callback, event, None)
        if callable(method):
            method(plan.__dict__ if getattr(callback, "is_shared", False) else item.training_context)


def _loader(plan):
    if not plan._dataloaders:
        raise ValueError("configure a language data loader")
    loader = plan._dataloaders.value_list[-1]
    # The native iterator is preferable to the legacy TensorSpec-keyed adapter.
    loader = getattr(loader, "iterator", loader)
    if getattr(plan, "_language_batch_size_explicit", False):
        if plan.batch_size < 1:
            raise ValueError("batch size must be positive")
        if isinstance(loader, torch.utils.data.DataLoader):
            if loader.batch_size != plan.batch_size:
                raise ValueError("configure batch_size when constructing a PyTorch DataLoader")
        elif hasattr(loader, "batch_size"):
            loader.batch_size = plan.batch_size
    return loader


def _configure_distributed(plan, item, loader):
    strategy = getattr(plan, "_language_distributed", None)
    if strategy:
        if not dist.is_initialized():
            if int(os.environ.get("WORLD_SIZE", "1")) < 2:
                raise ValueError("DDP requires torchrun or an initialized process group")
            backend = strategy.get("backend") or ("nccl" if item.device.type == "cuda" and os.name != "nt" else "gloo")
            if item.device.type == "cuda":
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
                item._model.to(torch.device("cuda", local_rank))
            dist.init_process_group(backend=backend)
        if not isinstance(item._model, torch.nn.parallel.DistributedDataParallel):
            item._model = torch.nn.parallel.DistributedDataParallel(
                item._model, device_ids=[item.device.index] if item.device.type == "cuda" else None,
                find_unused_parameters=strategy.get("find_unused_parameters", False))
        if hasattr(loader, "rank") and hasattr(loader, "world_size"):
            loader.rank, loader.world_size = _rank(), _world()
        elif not isinstance(getattr(loader, "sampler", None), torch.utils.data.DistributedSampler):
            raise ValueError("DDP loader must be a pipeline Iterator or use DistributedSampler")
        lengths = [None] * _world()
        try:
            length = len(loader)
        except TypeError as error:
            raise ValueError("DDP streaming loaders require a known steps_per_epoch") from error
        dist.all_gather_object(lengths, length)
        if len(set(lengths)) != 1:
            raise ValueError("DDP ranks must yield equal batch counts; adjust sharding/batch size")
    elif dist.is_initialized() and _world() > 1:
        raise ValueError("an active distributed process group requires with_distributed()")


def _distributed_scalar(value, device, op=dist.ReduceOp.SUM):
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if _world() > 1:
        dist.all_reduce(tensor, op=op)
    return float(tensor)


def _token_count(batch, result):
    if "valid_tokens" in result.metrics:
        return int(result.metrics["valid_tokens"])
    masks = [value for key, value in batch.items() if key == "labels" or key.endswith("_labels")]
    return sum(int((value[:, 1:] != -100).sum()) for value in masks)


def _collective_stage(item, label, action):
    """All ranks finish local work before advancing to the next collective."""
    error = None
    result = None
    try:
        result = action()
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    if _distributed_scalar(error is not None, item.device, dist.ReduceOp.MAX):
        if error is not None:
            raise error
        raise RuntimeError(f"{label} failed on another rank")
    return result


def _notify_ranks(plan, item, event):
    _collective_stage(item, event, lambda: _notify(plan, item, event))


def _next_window(iterator, size, item):
    batches = _collective_stage(item, "batch loading", lambda: list(itertools.islice(iterator, size)))
    if _world() > 1:
        minimum = _distributed_scalar(len(batches), item.device, dist.ReduceOp.MIN)
        maximum = _distributed_scalar(len(batches), item.device, dist.ReduceOp.MAX)
        if minimum != maximum:
            raise ValueError("DDP ranks delivered different numbers of microbatches")
    return batches


@contextlib.contextmanager
def _local_ddp_backward(modules):
    # Defer reduction until rank-local forward/backward and callbacks have succeeded.
    with contextlib.ExitStack() as stack:
        for module in modules:
            if isinstance(module, torch.nn.parallel.DistributedDataParallel):
                previous = module.broadcast_buffers
                stack.callback(setattr, module, 'broadcast_buffers', previous)
                module.broadcast_buffers = False
                stack.enter_context(module.no_sync())
        yield


@torch.no_grad()
def _sync_gradients(parameters, device):
    if _world() == 1:
        return
    present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=device)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    buckets = {}
    for parameter, active in zip(parameters, present.tolist()):
        if active:
            buckets.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    for bucket in buckets.values():
        flat = torch.cat([p.grad.reshape(-1) if p.grad is not None else p.new_zeros(p.numel())
                          for p in bucket])
        dist.all_reduce(flat)
        flat.div_(_world())
        for p, value in zip(bucket, flat.split([p.numel() for p in bucket])):
            p.grad = value.view_as(p)


def _routing_loss(statistics):
    """Use global token counts and expert loads while differentiating local probabilities."""
    auxiliary = None
    for module, rows in statistics.items():
        weight = module.gate.weight
        sizes = weight.new_zeros(module.config.num_experts + 1, dtype=torch.float32)
        probability = sizes[:-1].clone()
        for load, likelihood, count in rows:
            sizes[:-1] += load.detach()
            sizes[-1] += count.detach()
            probability = probability + likelihood
        if _world() > 1:
            dist.all_reduce(sizes)
        count = sizes[-1].clamp_min(1)
        loads = sizes[:-1] / (count * module.config.num_experts_per_tok)
        term = (loads * probability).sum() / count
        term = term * (module.config.num_experts * module.config.router_aux_loss_coef * _world())
        auxiliary = term if auxiliary is None else auxiliary + term
    return auxiliary


def _update_window(plan, item, batches):
    """Accumulate local gradients, then synchronize successful windows."""
    from trident.models.pytorch_causal_lm import SparseMoE, routing_statistics

    item._model.train()
    auxiliary_optimizers = getattr(item.objective, "optimizers", lambda: [])()
    optimizers = [item.optimizer] + auxiliary_optimizers
    parameters = list(dict.fromkeys(p for optimizer in optimizers for group in optimizer.param_groups
                                    for p in group['params'] if p.requires_grad))
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    total_weight, loss_sum, tokens = 0.0, 0.0, 0
    expected_weight = _collective_stage(
        item, "normalization", lambda: sum(item.objective.normalization_weight(batch) for batch in batches))
    global_expected_weight = _distributed_scalar(expected_weight, item.device)
    denominator = global_expected_weight / _world() if global_expected_weight else 1.0
    aggregate = {}
    started = time.perf_counter()
    stats = {module: [] for module in item._model.modules() if isinstance(module, SparseMoE)}
    delayed = []

    def process(raw_batch):
        batch = move_batch(raw_batch, item.device)
        item.training_context.update(train_data=batch, current_batch=item.microbatches,
                                     steps=item.microbatches, optimizer=item.optimizer)
        _notify(plan, item, "on_batch_start")
        _notify(plan, item, "on_data_received")
        token = routing_statistics.set(stats) if stats else None
        try:
            result = item.compute(batch)
        finally:
            if token is not None:
                routing_statistics.reset(token)
        if result.weight != item.objective.normalization_weight(batch):
            raise ValueError("objective weight does not match normalization_weight")
        if not torch.isfinite(result.loss_sum.detach()).all():
            raise FloatingPointError("Objective produced nonfinite loss")
        objective = result.loss_sum / denominator
        if stats:
            delayed.append(objective)
        elif objective.requires_grad:
            item._scaler.scale(objective).backward()
        item.training_context["current_loss"] = result.loss.detach()
        item.microbatches += 1
        _notify(plan, item, "on_batch_end")
        return (result.weight, float(result.loss_sum.detach()), _token_count(batch, result),
                {key: float(value) * result.weight for key, value in result.metrics.items()})

    with _local_ddp_backward([item._model, getattr(item.objective, "critic", None)]):
        for raw_batch in batches:
            weight, loss, count, batch_metrics = _collective_stage(
                item, "training microbatch", lambda: process(raw_batch))
            total_weight += weight
            loss_sum += loss
            tokens += count
            for key, value in batch_metrics.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
        if stats and global_expected_weight:
            auxiliary = _routing_loss(stats)
            window_loss = sum(delayed) + auxiliary
            loss_sum += float(auxiliary.detach()) * denominator
            if window_loss.requires_grad:
                _collective_stage(item, "MoE backward", lambda: item._scaler.scale(window_loss).backward())

    _sync_gradients(parameters, item.device)
    active = [optimizer for optimizer in optimizers
              if any(p.grad is not None for group in optimizer.param_groups for p in group['params'])]
    global_weight = _distributed_scalar(total_weight, item.device)
    global_loss = _distributed_scalar(loss_sum, item.device)
    global_tokens = int(_distributed_scalar(tokens, item.device))
    updated = False
    applied_lr = item.optimizer.param_groups[0]["lr"]
    if global_weight and active:
        def unscale():
            for optimizer in active:
                item._scaler.unscale_(optimizer)

        _collective_stage(item, "unscale", unscale)
        finite = all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
        finite = not _distributed_scalar(not finite, item.device, dist.ReduceOp.MAX)
        if finite:
            def prepare_step():
                nonlocal applied_lr
                if item.grad_clipping_by_norm:
                    torch.nn.utils.clip_grad_norm_(parameters, item.grad_clipping_threshold, error_if_nonfinite=True)
                if isinstance(item._language_scheduler, WarmupScheduler) and item.optimizer in active:
                    item._language_scheduler.on_optimization_step_start(item.training_context)
                    applied_lr = item.training_context['current_lr']
                _notify(plan, item, "on_optimization_step_start")

            def step():
                for optimizer in active:
                    item._scaler.step(optimizer)
                item._scaler.update()
                if any(not bool(torch.isfinite(p).all()) for p in parameters):
                    raise FloatingPointError("optimizer produced nonfinite parameters; checkpoint was not saved")
                if item._language_scheduler is not None and item.optimizer in active:
                    if isinstance(item._language_scheduler, WarmupScheduler):
                        item._language_scheduler.on_optimization_step_end(item.training_context)
                    else:
                        item._language_scheduler.step()

            _collective_stage(item, "optimizer preparation", prepare_step)
            _collective_stage(item, "optimizer step", step)
            item.optimizer_updates += 1
            updated = True
        elif item.precision == "fp16":
            item._scaler.update(new_scale=item._scaler.get_scale() / 2)
        else:
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite gradients; optimizer was not updated")
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    item.valid_tokens += global_tokens
    keys = sorted(aggregate)
    if _world() > 1:
        all_keys = [None] * _world()
        dist.all_gather_object(all_keys, keys)
        keys = sorted({key for rank_keys in all_keys for key in rank_keys})
    metrics = {key: _distributed_scalar(aggregate.get(key, 0), item.device) / (global_weight or 1)
               for key in keys}
    metrics.update(loss=global_loss / (global_weight or 1), valid_tokens=global_tokens,
                   optimizer_updates=item.optimizer_updates, microbatches=item.microbatches,
                   lr=(applied_lr if isinstance(item._language_scheduler, WarmupScheduler)
                       else item.optimizer.param_groups[0]["lr"]), updated=updated,
                   tokens_per_second=global_tokens / max(time.perf_counter() - started, 1e-9))
    if item.device.type == "cuda":
        metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(item.device)
    item.history.append(metrics)
    item.training_context.update(language_metrics=metrics, optimizer_updates=item.optimizer_updates,
                                 current_lr=metrics["lr"], valid_tokens=item.valid_tokens)
    def report_update():
        if _rank() == 0:
            from trident import context
            ctx = context._context()
            if ctx.enable_tensorboard and ctx.summary_writer is not None:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        ctx.summary_writer.add_scalar(item.name + "/" + key, value, item.optimizer_updates)
            frequency = plan.print_progress_frequency
            if frequency > 0 and plan.print_progress_unit in ("batch", "step") and item.optimizer_updates % frequency == 0:
                print("{} update={} loss={:.6f} tokens/s={:.1f}".format(item.name, item.optimizer_updates, metrics["loss"], metrics["tokens_per_second"]))
        _notify(plan, item, "on_optimization_step_end")

    if updated:
        _collective_stage(item, "optimizer callbacks", report_update)
    return updated


def save_training_checkpoint(plan, item, loader, epoch, offset, path):
    try:
        if offset and offset >= len(loader):
            epoch, offset = epoch + 1, 0
    except TypeError:
        pass
    local = {"rng": capture_rng(), "data": data_state(loader, epoch, offset)}
    if local["data"]["iterator"] is not None and "batch_offset" in local["data"]["iterator"]:
        local["data"]["iterator"].update(epoch=epoch, batch_offset=offset)
    ranks = [None] * _world()
    if _world() > 1:
        dist.all_gather_object(ranks, local)
    else:
        ranks[0] = local
    error = None
    if _rank() == 0:
        try:
            _notify(plan, item, "on_model_saving_start")
            if any(not bool(torch.isfinite(parameter).all()) for parameter in item._model.parameters()):
                raise FloatingPointError("refusing to save nonfinite model weights")
            state = {"format": "trident.language", "version": 1, "world_size": _world(),
                     "plan_config": getattr(plan, "execution_config", lambda: {})(),
                     "config": item.config_dict(), "model": unwrap_model(item._model).state_dict(),
                     "optimizer": item.optimizer.state_dict(),
                     "scheduler": None if item._language_scheduler is None else item._language_scheduler.state_dict(),
                     "scaler": item._scaler.state_dict(), "objective": item.objective.state_dict(),
                     "optimizer_updates": item.optimizer_updates, "microbatches": item.microbatches,
                     "valid_tokens": item.valid_tokens, "history": item.history, "ranks": ranks}
            atomic_torch_save(cpu_state(state), path)
            if item.tokenizer is not None:
                item.tokenizer.save_pretrained(Path(path).parent / "tokenizer")
            _notify(plan, item, "on_model_saving_end")
        except Exception as caught:
            error = caught
    if _distributed_scalar(error is not None, item.device, dist.ReduceOp.MAX):
        raise error if error is not None else RuntimeError("checkpoint write failed on rank zero")
    plan.last_checkpoint = str(path)


def _load_training_checkpoint(plan, item, loader, path):
    state = read_checkpoint(path)
    if state["config"] != item.config_dict():
        raise ValueError("checkpoint model/objective/optimizer/precision configuration mismatch")
    if state.get("plan_config", {}) != getattr(plan, "execution_config", lambda: {})():
        raise ValueError("checkpoint rollout/update configuration mismatch")
    reshard = getattr(plan, "_language_reshard", False)
    changed_world = state["world_size"] != _world()
    if changed_world and not reshard:
        raise ValueError("exact resume requires the same world size; opt into reshard=True")
    local = state["ranks"][0 if changed_world else _rank()]
    if local["data"]["config"] != loader_config(loader):
        raise ValueError("checkpoint data configuration mismatch")
    unwrap_model(item._model).load_state_dict(state["model"])
    item.optimizer.load_state_dict(state["optimizer"])
    if (item._language_scheduler is None) != (state["scheduler"] is None):
        raise ValueError("checkpoint scheduler configuration mismatch")
    if item._language_scheduler is not None:
        item._language_scheduler.load_state_dict(state["scheduler"])
    item._scaler.load_state_dict(state["scaler"])
    item.objective.load_state_dict(state["objective"])
    for key in ("optimizer_updates", "microbatches", "valid_tokens", "history"):
        setattr(item, key, state[key])
    epoch, offset = local["data"]["epoch"], local["data"]["offset"]
    native_state = local["data"]["iterator"]
    if changed_world:
        # Explicitly restart the interrupted epoch. Never label this exact replay.
        offset = 0
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        torch.manual_seed(42 + _rank() + epoch)
        restored_iterator = False
    else:
        restored_iterator = native_state is not None and callable(getattr(loader, "load_state_dict", None))
        if restored_iterator:
            loader.load_state_dict(native_state)
        restore_rng(local["rng"])
    return epoch, offset, restored_iterator, local["rng"] if not changed_world else None


@torch.no_grad()
def evaluate(item, loader):
    was_training = item._model.training
    item._model.eval()
    total, weight = 0.0, 0.0
    # Validation must not consume training RNG or a persistent iterator cursor.
    rng = capture_rng()
    state_fn = getattr(loader, "state_dict", None)
    state = state_fn() if callable(state_fn) else None
    def local_evaluation():
        nonlocal total, weight
        with _local_ddp_backward([item._model, getattr(item.objective, "critic", None)]):
            for batch in loader:
                result = item.compute(move_batch(batch, item.device))
                total += float(result.loss_sum)
                weight += result.weight

    try:
        _collective_stage(item, "validation", local_evaluation)
    finally:
        item._model.train(was_training)
        restore_rng(rng)
        if state is not None:
            loader.load_state_dict(state)
    total = _distributed_scalar(total, item.device)
    weight = _distributed_scalar(weight, item.device)
    if not weight:
        raise ValueError("validation contains no supervised tokens/samples")
    return {"loss": total / weight}


def run_language_plan(plan, only_steps=False, max_batches=math.inf, **kwargs):
    if len(plan.training_items) != 1:
        raise ValueError("language plans require one policy item; use TrainingPipeline for stages")
    item = plan.training_items.value_list[0]
    if not getattr(item, "uses_objective_runner", False):
        raise TypeError("cannot mix legacy and objective training items")
    loader = _loader(plan)
    if plan.num_epochs < 1:
        raise ValueError("repeat_epochs must be positive")
    if only_steps and (not math.isfinite(max_batches) or max_batches < 1):
        raise ValueError("only_steps requires a finite positive max_batches")
    item.training_context['only_steps'] = only_steps
    _configure_distributed(plan, item, loader)
    if not item._prepared:
        item.prepare()
    epoch, offset, restored_iterator, resume_rng = 0, 0, False, None
    resume = getattr(plan, "_language_resume", None)
    if resume:
        epoch, offset, restored_iterator, resume_rng = _load_training_checkpoint(plan, item, loader, resume)
    checkpointing = getattr(plan, "_language_checkpointing", None)
    limit = getattr(plan, "_language_max_updates", None)
    start_updates = item.optimizer_updates
    try:
        _notify_ranks(plan, item, "on_training_start")
        for current_epoch in range(epoch, plan.num_epochs):
            plan.epochs = current_epoch
            item.training_context["current_epoch"] = current_epoch
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(current_epoch)
            _notify_ranks(plan, item, "on_epoch_start")
            iterator = iter(loader)
            current_offset = offset if current_epoch == epoch else 0
            if current_offset and not restored_iterator:
                # Exact replay for deterministic re-iterable loaders only.
                if isinstance(loader, torch.utils.data.DataLoader) and (loader.num_workers or isinstance(loader.sampler, torch.utils.data.RandomSampler)):
                    raise ValueError("exact mid-epoch resume needs a stateful pipeline Iterator or deterministic loader")
                for _ in range(current_offset):
                    next(iterator)
            if resume_rng is not None:
                restore_rng(resume_rng)
                resume_rng = None
            restored_iterator = False
            while True:
                batches = _next_window(iterator, item.accumulation_steps, item)
                if not batches:
                    break
                updated = _update_window(plan, item, batches)
                current_offset += len(batches)
                plan.steps = item.microbatches
                if checkpointing and updated and item.optimizer_updates % checkpointing["every_updates"] == 0:
                    save_training_checkpoint(plan, item, loader, current_epoch, current_offset,
                                             Path(checkpointing["directory"]) / "latest.pt")
                stop = (limit is not None and item.optimizer_updates - start_updates >= limit) or (only_steps and item.microbatches >= max_batches)
                terminated = _distributed_scalar(plan.is_terminate, item.device, dist.ReduceOp.MAX)
                if stop or terminated:
                    if checkpointing:
                        save_training_checkpoint(plan, item, loader, current_epoch, current_offset,
                                                 Path(checkpointing["directory"]) / "latest.pt")
                    _notify_ranks(plan, item, "on_training_terminated" if terminated else "on_training_end")
                    return item.history
            _notify_ranks(plan, item, "on_epoch_end")
            validation = getattr(plan, "_language_validation_loader", None)
            if validation is not None:
                plan.validation_metrics = evaluate(item, validation)
            if checkpointing:
                save_training_checkpoint(plan, item, loader, current_epoch + 1, 0,
                                         Path(checkpointing["directory"]) / "latest.pt")
            offset = 0
        _notify_ranks(plan, item, "on_training_end")
        return item.history
    except BaseException:
        item.optimizer.zero_grad(set_to_none=True)
        try:
            _notify(plan, item, "on_training_terminated")
        except Exception:
            pass  # Preserve the original failure during teardown.
        raise
