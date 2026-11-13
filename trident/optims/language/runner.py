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


def _update_window(plan, item, batches):
    """Accumulate sums, then normalize by the global valid-token/sample count."""
    item._model.train()
    auxiliary_optimizers = getattr(item.objective, "optimizers", lambda: [])()
    optimizers = [item.optimizer] + auxiliary_optimizers
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    total_weight, loss_sum, tokens = 0.0, 0.0, 0
    expected_weight = sum(item.objective.normalization_weight(batch) for batch in batches)
    global_expected_weight = _distributed_scalar(expected_weight, item.device)
    denominator = max(global_expected_weight / _world(), 1)
    aggregate = {}
    started = time.perf_counter()
    for index, raw_batch in enumerate(batches):
        batch = move_batch(raw_batch, item.device)
        item.training_context.update(train_data=batch, current_batch=item.microbatches,
                                     steps=item.microbatches, optimizer=item.optimizer)
        _notify(plan, item, "on_batch_start")
        _notify(plan, item, "on_data_received")
        with contextlib.ExitStack() as sync:
            if index < len(batches) - 1:
                modules = [item._model, getattr(item.objective, "critic", None)]
                for module in modules:
                    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
                        sync.enter_context(module.no_sync())
            error = None
            try:
                result = item.compute(batch)
                if result.weight != item.objective.normalization_weight(batch):
                    raise ValueError("objective weight does not match normalization_weight")
            except Exception as caught:
                error = caught
            if _distributed_scalar(error is not None, item.device, dist.ReduceOp.MAX):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                raise error if error is not None else RuntimeError("objective failed on another rank")
            total_weight += result.weight
            loss_sum += float(result.loss_sum.detach())
            tokens += _token_count(batch, result)
            for key, value in result.metrics.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value) * result.weight
            # Zero-target microbatches still participate in DDP gradient sync.
            item._scaler.scale(result.loss_sum / denominator).backward()
            item.training_context["current_loss"] = result.loss.detach()
        item.microbatches += 1
        _notify(plan, item, "on_batch_end")
    global_weight = _distributed_scalar(total_weight, item.device)
    global_loss = _distributed_scalar(loss_sum, item.device)
    global_tokens = int(_distributed_scalar(tokens, item.device))
    updated = False
    for optimizer in optimizers:
        item._scaler.unscale_(optimizer)
    parameters = [parameter for optimizer in optimizers for group in optimizer.param_groups
                  for parameter in group["params"] if parameter.requires_grad]
    if global_weight:
        # Loss was normalized before scaling/backward, avoiding FP16 overflow
        # from a large token sum. DDP averages rank-local gradient sums.
        finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters if parameter.grad is not None)
        finite = not _distributed_scalar(not finite, item.device, dist.ReduceOp.MAX)
        if finite:
            if item.grad_clipping_by_norm:
                torch.nn.utils.clip_grad_norm_(parameters, item.grad_clipping_threshold, error_if_nonfinite=True)
            _notify(plan, item, "on_optimization_step_start")
            for optimizer in optimizers:
                item._scaler.step(optimizer)
            item._scaler.update()
            item.optimizer_updates += 1
            if item._language_scheduler is not None:
                item._language_scheduler.step()
            updated = True
        elif item.precision == "fp16":
            # All ranks skip together, including ranks without local overflow.
            item._scaler.update(new_scale=item._scaler.get_scale() / 2)
        else:
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite gradients; optimizer was not updated")
    elif item._scaler.is_enabled():
        item._scaler.update(new_scale=item._scaler.get_scale())
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    if updated and _distributed_scalar(any(not bool(torch.isfinite(parameter).all()) for parameter in parameters),
                                       item.device, dist.ReduceOp.MAX):
        raise FloatingPointError("optimizer produced nonfinite parameters; checkpoint was not saved")
    item.valid_tokens += global_tokens
    metrics = {key: _distributed_scalar(value, item.device) / max(global_weight, 1) for key, value in aggregate.items()}
    metrics.update(loss=global_loss / max(global_weight, 1), valid_tokens=global_tokens,
                   optimizer_updates=item.optimizer_updates, microbatches=item.microbatches,
                   lr=item.optimizer.param_groups[0]["lr"], updated=updated,
                   tokens_per_second=global_tokens / max(time.perf_counter() - started, 1e-9))
    if item.device.type == "cuda":
        metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(item.device)
    item.history.append(metrics)
    item.training_context.update(language_metrics=metrics, optimizer_updates=item.optimizer_updates,
                                 current_lr=metrics["lr"], valid_tokens=item.valid_tokens)
    if updated:
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
    try:
        for batch in loader:
            result = item.compute(move_batch(batch, item.device))
            total += float(result.loss_sum)
            weight += result.weight
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
    _notify(plan, item, "on_training_start")
    try:
        for current_epoch in range(epoch, plan.num_epochs):
            plan.epochs = current_epoch
            item.training_context["current_epoch"] = current_epoch
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(current_epoch)
            _notify(plan, item, "on_epoch_start")
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
                batches = list(itertools.islice(iterator, item.accumulation_steps))
                if not batches:
                    break
                updated = _update_window(plan, item, batches)
                current_offset += len(batches)
                plan.steps = item.microbatches
                if checkpointing and updated and item.optimizer_updates % checkpointing["every_updates"] == 0:
                    save_training_checkpoint(plan, item, loader, current_epoch, current_offset,
                                             Path(checkpointing["directory"]) / "latest.pt")
                stop = (limit is not None and item.optimizer_updates - start_updates >= limit) or (only_steps and item.microbatches >= max_batches)
                if stop or plan.is_terminate:
                    if checkpointing:
                        save_training_checkpoint(plan, item, loader, current_epoch, current_offset,
                                                 Path(checkpointing["directory"]) / "latest.pt")
                    _notify(plan, item, "on_training_terminated" if plan.is_terminate else "on_training_end")
                    return item.history
            _notify(plan, item, "on_epoch_end")
            validation = getattr(plan, "_language_validation_loader", None)
            if validation is not None:
                plan.validation_metrics = evaluate(item, validation)
            if checkpointing:
                save_training_checkpoint(plan, item, loader, current_epoch + 1, 0,
                                         Path(checkpointing["directory"]) / "latest.pt")
            offset = 0
        _notify(plan, item, "on_training_end")
        return item.history
    except BaseException:
        item.optimizer.zero_grad(set_to_none=True)
        _notify(plan, item, "on_training_terminated")
        raise
