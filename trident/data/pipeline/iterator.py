from __future__ import absolute_import, division, print_function

import math
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - Python 3.5
    from collections import Mapping

from .batch import Batch
from .collate import default_collate
from .dataset import Dataset
from .errors import (CollationError, DataPipelineError, MemoryBudgetError,
                     SourceError, TransformError)
from .transforms import SampleTransform, TransformContext


_PROCESS_ITERATOR = None


def _initialize_process_iterator(config):
    global _PROCESS_ITERATOR
    _PROCESS_ITERATOR = Iterator(**config)


def _run_process_map_batch(arguments):
    _PROCESS_ITERATOR.dataset.set_epoch(arguments[2])
    return _PROCESS_ITERATOR._load_map_batch(*arguments)


def estimate_nbytes(value):
    if value.__class__.__name__ == "RaggedBatch" and hasattr(value, "row_splits"):
        return estimate_nbytes(value.values) + estimate_nbytes(value.row_splits)
    if isinstance(value, Mapping):
        return sum(estimate_nbytes(key) + estimate_nbytes(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(estimate_nbytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    nelement = getattr(value, "nelement", None)
    element_size = getattr(value, "element_size", None)
    if callable(nelement) and callable(element_size):
        return int(nelement() * element_size())
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


class Iterator(object):
    """Bounded, deterministic batch executor for map and streaming datasets."""

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,
                 transforms=None, group_transforms=None, collate=None,
                 batch_transforms=None, sample_filter=None, workers=0,
                 prefetch_batches=2, memory_budget_mb=None, seed=0,
                 steps_per_epoch=None, rank=0, world_size=1, tensorizer=None,
                 pin_memory=False, device=None, non_blocking=True,
                 executor="thread", persistent_workers=False,
                 shuffle_buffer_size=1024):
        if not isinstance(dataset, Dataset):
            dataset = Dataset(dataset)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if workers < 0:
            raise ValueError("workers cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid rank/world_size")
        if executor not in ("thread", "process"):
            raise ValueError("executor must be 'thread' or 'process'")
        if executor == "process" and dataset.iterable_style:
            raise ValueError("process executor currently requires a map-style dataset")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if dataset.iterable_style and steps_per_epoch is None:
            self._known_length = False
        else:
            self._known_length = True
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.transforms = tuple(transforms or ())
        self.group_transforms = tuple(group_transforms or ())
        self.collate = collate or default_collate
        self.batch_transforms = tuple(batch_transforms or ())
        self.sample_filter = sample_filter
        self.workers = int(workers)
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.memory_budget_bytes = (None if memory_budget_mb is None else
                                    int(float(memory_budget_mb) * 1024 * 1024))
        self.seed = int(seed)
        self.steps_per_epoch = steps_per_epoch
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.tensorizer = tensorizer
        self.pin_memory = bool(pin_memory)
        self.device = device
        self.non_blocking = bool(non_blocking)
        self.executor_kind = executor
        self.persistent_workers = bool(persistent_workers)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.epoch = 0
        self._resume_batch = 0
        self._executor = None
        self._closed = False

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._resume_batch = 0
        return self

    def _apply_sample(self, sample, sample_id, worker_id, epoch):
        context = TransformContext(
            seed=self.seed, epoch=epoch, rank=self.rank,
            worker_id=worker_id, sample_id=sample_id)
        try:
            result = sample
            for transform in self.transforms:
                if isinstance(transform, SampleTransform):
                    result = transform(result, context=context,
                                       schema=self.dataset.schema)
                else:
                    result = transform(result)
            if self.sample_filter is not None and not self.sample_filter(result):
                return None, context
            return result, context
        except DataPipelineError:
            raise
        except Exception as error:
            raise TransformError("sample transform failed", stage="sample_transform",
                                 sample_id=sample_id, cause=error)

    def _finish_batch(self, samples, sample_ids, batch_number, epoch, worker_id):
        processed = []
        contexts = []
        for sample, sample_id in zip(samples, sample_ids):
            value, context = self._apply_sample(sample, sample_id, worker_id, epoch)
            if value is not None:
                processed.append(value)
                contexts.append(context)
        if not processed:
            return None
        try:
            for transform in self.group_transforms:
                processed = transform(processed, contexts=contexts,
                                      schema=self.dataset.schema)
                if isinstance(processed, Mapping):
                    processed = [processed]
                    contexts = contexts[:1]
        except DataPipelineError:
            raise
        except Exception as error:
            raise TransformError("group transform failed", stage="group_transform",
                                 sample_id=list(sample_ids), cause=error)
        try:
            if getattr(self.collate, "accepts_schema", False):
                batch = self.collate(processed, schema=self.dataset.schema)
            else:
                batch = self.collate(processed)
            if not isinstance(batch, Batch) and isinstance(batch, Mapping):
                batch = Batch(batch, schema=self.dataset.schema)
        except DataPipelineError:
            raise
        except Exception as error:
            raise CollationError("batch collation failed", stage="collate",
                                 sample_id=list(sample_ids), cause=error)
        try:
            for transform in self.batch_transforms:
                batch = transform(batch)
        except DataPipelineError:
            raise
        except Exception as error:
            raise TransformError("batch transform failed", stage="batch_transform",
                                 sample_id=list(sample_ids), cause=error)
        if isinstance(batch, Batch):
            batch.metadata["transform_records"] = [context.records for context in contexts]
            batch.metadata.update(dict(epoch=epoch, batch_number=batch_number,
                                       sample_ids=list(sample_ids)))
        return batch

    def _prepare_delivery(self, batch):
        try:
            if self.tensorizer is not None:
                batch = self.tensorizer(batch)
            if self.pin_memory and hasattr(batch, "pin_memory"):
                batch = batch.pin_memory()
            if self.device is not None and hasattr(batch, "to"):
                batch = batch.to(self.device, non_blocking=self.non_blocking)
            return batch
        except DataPipelineError:
            raise
        except Exception as error:
            raise TransformError("batch delivery failed", stage="delivery", cause=error)

    def _load_map_batch(self, indices, batch_number, epoch, worker_id):
        try:
            samples = self.dataset.get_batch(indices)
        except DataPipelineError:
            raise
        except Exception as error:
            raise SourceError("dataset fetch failed", stage="source",
                              sample_id=list(indices), cause=error)
        return self._finish_batch(samples, indices, batch_number, epoch, worker_id)

    def _map_batch_arguments(self, epoch):
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.world_size > 1:
            indices = indices[self.rank::self.world_size]
        if self.shuffle:
            rng = np.random.RandomState(self.seed + epoch)
            rng.shuffle(indices)
        total_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size:
            total_batches += 1
        if self.steps_per_epoch is not None:
            total_batches = min(total_batches, int(self.steps_per_epoch))
        for batch_number in range(total_batches):
            if batch_number < self._resume_batch:
                continue
            start = batch_number * self.batch_size
            batch_indices = indices[start:start + self.batch_size].tolist()
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            yield (batch_indices, batch_number, epoch,
                   batch_number % max(1, self.workers))

    def _iter_stream_source(self, source, epoch):
        if not self.shuffle:
            for sample in source:
                yield sample
            return
        rng = np.random.RandomState(self.seed + epoch * 1009 + self.rank)
        buffer = []
        for sample in source:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            index = int(rng.randint(len(buffer)))
            yield buffer[index]
            buffer[index] = sample
        while buffer:
            index = int(rng.randint(len(buffer)))
            yield buffer.pop(index)

    def _iterable_batch_arguments(self, epoch):
        source = self.dataset.shard(self.world_size, self.rank) if self.world_size > 1 else self.dataset
        samples = []
        sample_ids = []
        batch_number = 0
        for position, sample in enumerate(self._iter_stream_source(source, epoch)):
            sample_id = getattr(sample, "sample_id", None)
            if sample_id is None:
                sample_id = position
            samples.append(sample)
            sample_ids.append(sample_id)
            if len(samples) == self.batch_size:
                if batch_number >= self._resume_batch:
                    yield (samples, sample_ids, batch_number, epoch,
                           batch_number % max(1, self.workers))
                samples = []
                sample_ids = []
                batch_number += 1
                if self.steps_per_epoch is not None and batch_number >= self.steps_per_epoch:
                    return
        if samples and not self.drop_last and (self.steps_per_epoch is None or
                                               batch_number < self.steps_per_epoch):
            if batch_number >= self._resume_batch:
                yield (samples, sample_ids, batch_number, epoch,
                       batch_number % max(1, self.workers))
    def _validate_batch_memory(self, batch):
        if not self.memory_budget_bytes:
            return estimate_nbytes(batch)
        footprint = max(1, estimate_nbytes(batch))
        if footprint > self.memory_budget_bytes:
            raise MemoryBudgetError(
                "one batch exceeds memory_budget_mb", stage="prefetch")
        return footprint

    def _process_worker_config(self):
        return dict(
            dataset=self.dataset, batch_size=self.batch_size,
            shuffle=False, drop_last=self.drop_last, transforms=self.transforms,
            group_transforms=self.group_transforms, collate=self.collate,
            batch_transforms=self.batch_transforms, sample_filter=self.sample_filter,
            workers=0, prefetch_batches=1, memory_budget_mb=None,
            seed=self.seed, steps_per_epoch=self.steps_per_epoch,
            rank=self.rank, world_size=self.world_size, tensorizer=None,
            pin_memory=False, device=None, non_blocking=self.non_blocking,
            executor="thread", persistent_workers=False,
            shuffle_buffer_size=self.shuffle_buffer_size)

    def _get_executor(self):
        if self._closed:
            raise RuntimeError("Iterator is closed")
        if self._executor is not None:
            return self._executor
        if self.executor_kind == "thread":
            executor = ThreadPoolExecutor(max_workers=self.workers)
        else:
            executor = ProcessPoolExecutor(
                max_workers=self.workers, initializer=_initialize_process_iterator,
                initargs=(self._process_worker_config(),))
        if self.persistent_workers:
            self._executor = executor
        return executor

    def _ordered_prefetch(self, function, arguments):
        if self.workers == 0:
            for args in arguments:
                result = function(*args)
                if result is not None:
                    result = self._prepare_delivery(result)
                    self._validate_batch_memory(result)
                    yield result
            return
        executor = self._get_executor()
        pending = deque()
        arguments = iter(arguments)
        effective_prefetch = self.prefetch_batches
        exhausted = False
        process_mode = self.executor_kind == "process"

        def submit(arguments_item):
            return (executor.submit(_run_process_map_batch, arguments_item)
                    if process_mode else executor.submit(function, *arguments_item))

        try:
            # A budgeted iterator measures one batch before filling the queue;
            # otherwise a large configured prefetch could exceed the budget at startup.
            if self.memory_budget_bytes:
                try:
                    first = submit(next(arguments)).result()
                except StopIteration:
                    exhausted = True
                else:
                    if first is not None:
                        first = self._prepare_delivery(first)
                        footprint = self._validate_batch_memory(first)
                        allowed = max(1, self.memory_budget_bytes // footprint)
                        effective_prefetch = min(self.prefetch_batches, allowed)
                        yield first
            while pending or not exhausted:
                while not exhausted and len(pending) < effective_prefetch:
                    try:
                        pending.append(submit(next(arguments)))
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                result = pending.popleft().result()
                if result is None:
                    continue
                result = self._prepare_delivery(result)
                if self.memory_budget_bytes:
                    footprint = self._validate_batch_memory(result)
                    allowed = max(1, self.memory_budget_bytes // footprint)
                    effective_prefetch = min(self.prefetch_batches, allowed)
                yield result
        finally:
            if not self.persistent_workers:
                executor.shutdown(wait=True)

    def __iter__(self):
        if self._closed:
            raise RuntimeError("Iterator is closed")
        epoch = self.epoch
        completed = False
        self.dataset.set_epoch(epoch)
        try:
            if self.dataset.map_style:
                arguments = self._map_batch_arguments(epoch)
                batches = self._ordered_prefetch(self._load_map_batch, arguments)
            else:
                arguments = self._iterable_batch_arguments(epoch)
                batches = self._ordered_prefetch(self._finish_batch, arguments)
            for batch in batches:
                if isinstance(batch, Batch):
                    self._resume_batch = int(
                        batch.metadata.get("batch_number", self._resume_batch)) + 1
                else:
                    self._resume_batch += 1
                yield batch
            completed = True
        finally:
            if completed:
                self.epoch = epoch + 1
                self._resume_batch = 0
    def __len__(self):
        if self.steps_per_epoch is not None:
            if self.dataset.iterable_style:
                return int(self.steps_per_epoch)
        if self.dataset.iterable_style:
            raise TypeError("streaming Iterator length requires steps_per_epoch")
        local_length = max(0, int(math.ceil(float(len(self.dataset) - self.rank) /
                                            self.world_size)))
        if self.drop_last:
            batches = local_length // self.batch_size
        else:
            batches = int(math.ceil(float(local_length) / self.batch_size))
        return min(batches, int(self.steps_per_epoch)) if self.steps_per_epoch is not None else batches

    def to_torch_dataloader(self, **kwargs):
        from .framework import to_torch_dataloader
        return to_torch_dataloader(self, **kwargs)

    def to_tensorflow_dataset(self, output_signature=None, prefetch=1):
        from .framework import to_tensorflow_dataset
        return to_tensorflow_dataset(
            self, output_signature=output_signature, prefetch=prefetch)

    def state_dict(self):
        state = dict(version=1, epoch=self.epoch,
                     batch_offset=self._resume_batch, seed=self.seed,
                     rank=self.rank, world_size=self.world_size)
        source_state = getattr(self.dataset.source, "state_dict", None)
        if callable(source_state):
            state["source"] = source_state()
        return state

    def load_state_dict(self, state):
        if int(state.get("rank", self.rank)) != self.rank:
            raise ValueError("checkpoint rank does not match Iterator rank")
        if int(state.get("world_size", self.world_size)) != self.world_size:
            raise ValueError("checkpoint world_size does not match Iterator world_size")
        self.epoch = int(state.get("epoch", self.epoch))
        self._resume_batch = int(state.get("batch_offset", 0))
        self.seed = int(state.get("seed", self.seed))
        load_source = getattr(self.dataset.source, "load_state_dict", None)
        if "source" in state and callable(load_source):
            load_source(state["source"])
        return self

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)