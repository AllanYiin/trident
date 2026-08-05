"""Repeatable smoke benchmark for the shadow data pipeline.

Examples:
    python benchmarks/benchmark_data_pipeline.py --workload all
    python benchmarks/benchmark_data_pipeline.py --workload image --workers 2
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trident.data.pipeline import (  # noqa: E402
    Dataset,
    DatasetSchema,
    FieldSpec,
    Iterator,
    RandomCrop,
    TokenizerCollator,
)


class ImageSource(object):
    def __init__(self, length, image_size):
        self.length = length
        self.template = np.arange(image_size * image_size * 3, dtype=np.uint8).reshape(
            image_size, image_size, 3)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"image": self.template.copy(), "label": index % 10}

    def __getitems__(self, indices):
        return [self[index] for index in indices]


class TextSource(object):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"text": "有限記憶體資料供應-{0}".format(index % 97),
                "label": index % 2}

    def __getitems__(self, indices):
        return [self[index] for index in indices]


class BenchmarkTokenizer(object):
    def __call__(self, texts, padding="longest", return_tensors=None, **kwargs):
        values = [[ord(char) % 32000 for char in text] for text in texts]
        width = max(len(value) for value in values)
        return {
            "input_ids": [value + [0] * (width - len(value)) for value in values],
            "attention_mask": [[1] * len(value) + [0] * (width - len(value))
                               for value in values],
        }


def process_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except ImportError:
        return None


def build_iterator(workload, args):
    if workload == "image":
        schema = DatasetSchema([
            FieldSpec("image", kind="image", layout="HWC"),
            FieldSpec("label", kind="label"),
        ])
        dataset = Dataset(ImageSource(args.samples, args.image_size), schema=schema)
        return Iterator(
            dataset, batch_size=args.batch_size, workers=args.workers,
            prefetch_batches=args.prefetch_batches,
            memory_budget_mb=args.memory_budget_mb,
            transforms=[RandomCrop((args.crop_size, args.crop_size))], seed=7)
    dataset = Dataset(TextSource(args.samples))
    return Iterator(
        dataset, batch_size=args.batch_size, workers=args.workers,
        prefetch_batches=args.prefetch_batches,
        memory_budget_mb=args.memory_budget_mb,
        collate=TokenizerCollator(BenchmarkTokenizer(), truncation=True))


def run_once(workload, args):
    iterator = build_iterator(workload, args)
    batch_times = []
    rows = 0
    rss_before = process_rss_mb()
    start = time.perf_counter()
    last = start
    for batch in iterator:
        now = time.perf_counter()
        batch_times.append(now - last)
        last = now
        first_value = next(iter(batch.values()))
        rows += len(first_value)
    elapsed = time.perf_counter() - start
    rss_after = process_rss_mb()
    return {
        "workload": workload,
        "samples": rows,
        "seconds": elapsed,
        "samples_per_second": rows / elapsed,
        "batch_latency_p50_ms": statistics.median(batch_times) * 1000.0,
        "batch_latency_p95_ms": np.percentile(batch_times, 95) * 1000.0,
        "rss_delta_mb": None if rss_before is None else rss_after - rss_before,
        "workers": args.workers,
        "prefetch_batches": args.prefetch_batches,
        "memory_budget_mb": args.memory_budget_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("image", "text", "all"), default="all")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--memory-budget-mb", type=float, default=512)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    args = parser.parse_args()
    workloads = ("image", "text") if args.workload == "all" else (args.workload,)
    print(json.dumps([run_once(workload, args) for workload in workloads],
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
