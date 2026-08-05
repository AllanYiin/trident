"""WP0 legacy/new characterization benchmark.

This benchmark deliberately uses the public legacy data classes for the
baseline. It is not a microbenchmark of isolated NumPy operations.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trident.data.dataset import (  # noqa: E402
    BboxDataset as LegacyBboxDataset,
    ImageDataset as LegacyImageDataset,
    Iterator as LegacyIterator,
    NumpyDataset as LegacyNumpyDataset,
)
from trident.data.vision_transforms import Resize as LegacyResize  # noqa: E402
from trident.data.pipeline import (  # noqa: E402
    Dataset,
    DatasetSchema,
    FieldSpec,
    Iterator,
    Resize,
)


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.STDOUT).decode("utf-8").strip()
    except Exception:
        return None


def _rss_bytes():
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


class PeakRssMonitor(object):
    def __init__(self):
        self.start_rss = _rss_bytes()
        self.peak_rss = self.start_rss
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.start_rss is not None:
            self._thread = threading.Thread(target=self._sample)
            self._thread.daemon = True
            self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(0.002):
            current = _rss_bytes()
            if current is not None:
                self.peak_rss = max(self.peak_rss, current)

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_delta_mb(self):
        if self.start_rss is None:
            return None
        return (self.peak_rss - self.start_rss) / (1024.0 * 1024.0)


def _image_template(size):
    values = np.arange(size * size * 3, dtype=np.uint32) % 251
    return values.astype(np.uint8).reshape(size, size, 3)


def _box_template(size):
    return np.array([[size * 0.1, size * 0.15, size * 0.7, size * 0.8, 1]],
                    dtype=np.float32)


def _build_legacy(workload, args):
    batch_count = args.samples // args.batch_size
    if workload == "image":
        images = [_image_template(args.image_size)] * args.samples
        image_data = LegacyImageDataset(images, symbol="image")
        image_data.transform_funcs = [LegacyResize((args.output_size, args.output_size),
                                                   keep_aspect=False)]
        iterator = LegacyIterator(data=image_data, batch_size=args.batch_size,
                                  is_shuffle=False, workers=0)
    elif workload == "detection":
        images = [_image_template(args.image_size)] * args.samples
        boxes = [_box_template(args.image_size)] * args.samples
        iterator = LegacyIterator(
            data=LegacyImageDataset(images, symbol="image"),
            label=LegacyBboxDataset(boxes, symbol="boxes"),
            batch_size=args.batch_size, is_shuffle=False, workers=0)
        iterator.paired_transform_funcs = [
            LegacyResize((args.output_size, args.output_size), keep_aspect=False)]
    else:
        tokens = np.arange(args.samples * args.sequence_length, dtype=np.int64).reshape(
            args.samples, args.sequence_length) % 32000
        labels = (np.arange(args.samples, dtype=np.int64) % 2).reshape(-1, 1)
        iterator = LegacyIterator(
            data=LegacyNumpyDataset(tokens, symbol="input_ids"),
            label=LegacyNumpyDataset(labels, symbol="label"),
            batch_size=args.batch_size, is_shuffle=False, workers=0)

    def batches():
        for _ in range(batch_count):
            yield iterator.next()
    return batches(), batch_count * args.batch_size


def _build_pipeline(workload, args):
    if workload in ("image", "detection"):
        image = _image_template(args.image_size)
        rows = []
        for index in range(args.samples):
            row = {"image": image, "label": index % 10}
            if workload == "detection":
                row["boxes"] = _box_template(args.image_size)
            rows.append(row)
        fields = [FieldSpec("image", kind="image", layout="HWC")]
        if workload == "detection":
            fields.append(FieldSpec("boxes", kind="bbox", coordinate_format="xyxy"))
        else:
            fields.append(FieldSpec("label", kind="label"))
        dataset = Dataset(rows, schema=DatasetSchema(fields))
        iterator = Iterator(
            dataset, batch_size=args.batch_size,
            transforms=[Resize((args.output_size, args.output_size))],
            workers=args.workers, prefetch_batches=args.prefetch_batches,
            memory_budget_mb=args.memory_budget_mb)
    else:
        tokens = np.arange(args.samples * args.sequence_length, dtype=np.int64).reshape(
            args.samples, args.sequence_length) % 32000
        rows = [{"input_ids": tokens[index], "label": index % 2}
                for index in range(args.samples)]
        iterator = Iterator(
            Dataset(rows), batch_size=args.batch_size, workers=args.workers,
            prefetch_batches=args.prefetch_batches,
            memory_budget_mb=args.memory_budget_mb)
    return iter(iterator), args.samples


def _batch_size(batch):
    values = batch.values() if isinstance(batch, dict) else batch
    first = next(iter(values))
    return len(first)


def _measure(engine, workload, args):
    builder = _build_legacy if engine == "legacy" else _build_pipeline
    batches, expected_samples = builder(workload, args)
    latencies = []
    consumed = 0
    with PeakRssMonitor() as memory:
        started = time.perf_counter()
        previous = started
        for batch in batches:
            now = time.perf_counter()
            latencies.append(now - previous)
            previous = now
            consumed += _batch_size(batch)
        elapsed = time.perf_counter() - started
    if consumed != expected_samples:
        raise RuntimeError("expected {0} samples, consumed {1}".format(
            expected_samples, consumed))
    return {
        "engine": engine,
        "workload": workload,
        "samples": consumed,
        "seconds": elapsed,
        "samples_per_second": consumed / elapsed,
        "batch_latency_p50_ms": statistics.median(latencies) * 1000.0,
        "batch_latency_p95_ms": float(np.percentile(latencies, 95)) * 1000.0,
        "peak_rss_delta_mb": memory.peak_delta_mb,
    }


def _summarize(runs):
    grouped = {}
    for run in runs:
        key = (run["engine"], run["workload"])
        grouped.setdefault(key, []).append(run)
    summary = []
    for (engine, workload), values in sorted(grouped.items()):
        summary.append({
            "engine": engine,
            "workload": workload,
            "samples_per_second_median": statistics.median(
                value["samples_per_second"] for value in values),
            "batch_latency_p50_ms_median": statistics.median(
                value["batch_latency_p50_ms"] for value in values),
            "peak_rss_delta_mb_max": max(
                value["peak_rss_delta_mb"] or 0 for value in values),
            "repeats": len(values),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=112)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--memory-budget-mb", type=float, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.samples % args.batch_size:
        raise ValueError("WP0 benchmark requires samples divisible by batch-size")

    runs = []
    for _ in range(args.repeats):
        for workload in ("image", "detection", "text"):
            for engine in ("legacy", "pipeline"):
                runs.append(_measure(engine, workload, args))
    result = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "git_revision": _git_revision(),
        },
        "configuration": vars(args),
        "capability_gaps": {
            "legacy_huggingface_streaming": "unsupported",
            "legacy_batched_tokenizer_collator": "unsupported",
        },
        "summary": _summarize(runs),
        "runs": runs,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output_path = os.path.abspath(args.output)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(payload)
            output_file.write("\n")


if __name__ == "__main__":
    main()
