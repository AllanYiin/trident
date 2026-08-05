from __future__ import absolute_import, division, print_function

import argparse
import json


REQUIRED_WORKLOADS = frozenset(("image", "detection", "text"))
REQUIRED_ENGINES = frozenset(("legacy", "pipeline"))


def validate(result):
    errors = []
    summary = result.get("summary", [])
    pairs = set((entry.get("engine"), entry.get("workload")) for entry in summary)
    required_pairs = set((engine, workload) for engine in REQUIRED_ENGINES
                         for workload in REQUIRED_WORKLOADS)
    missing = sorted(required_pairs - pairs)
    if missing:
        errors.append("missing engine/workload summaries: {0}".format(missing))
    for entry in summary:
        if entry.get("repeats", 0) < 3:
            errors.append("{0}/{1} has fewer than 3 repeats".format(
                entry.get("engine"), entry.get("workload")))
        if entry.get("samples_per_second_median", 0) <= 0:
            errors.append("{0}/{1} has invalid throughput".format(
                entry.get("engine"), entry.get("workload")))
        if entry.get("batch_latency_p50_ms_median", 0) <= 0:
            errors.append("{0}/{1} has invalid latency".format(
                entry.get("engine"), entry.get("workload")))
    gaps = result.get("capability_gaps", {})
    for name in ("legacy_huggingface_streaming", "legacy_batched_tokenizer_collator"):
        if name not in gaps:
            errors.append("missing capability-gap record: {0}".format(name))
    environment = result.get("environment", {})
    for name in ("python", "platform", "numpy", "git_revision"):
        if not environment.get(name):
            errors.append("missing environment field: {0}".format(name))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    with open(args.result, "r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    errors = validate(result)
    if errors:
        for error in errors:
            print("ERROR: {0}".format(error))
        raise SystemExit(1)
    print("WP0 baseline validation passed: 3 workloads x 2 engines x >=3 repeats")


if __name__ == "__main__":
    main()
