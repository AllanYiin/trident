"""Final source/artifact gate for the new modeling data pipeline."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trident.data import PipelineDataProvider, PipelineDataset, PipelineIterator
from trident.data.pipeline import PIPELINE_API_VERSION

REQUIRED = [
    "docs/zh-tw/data_pipeline_wp0_baseline.md",
    "docs/zh-tw/data_pipeline_wp1_contracts.md",
    "docs/zh-tw/data_pipeline_wp2_adapters.md",
    "docs/zh-tw/data_pipeline_wp3_augmentation.md",
    "docs/zh-tw/data_pipeline_wp4_collation.md",
    "docs/zh-tw/data_pipeline_wp5_executor.md",
    "docs/zh-tw/data_pipeline_wp6_frameworks.md",
    "docs/zh-tw/data_pipeline_migration.md",
    "docs/zh-tw/data_pipeline_wp7_release.md",
    "examples/data_pipeline/_bootstrap.py",
    "examples/data_pipeline/vision_geometry.py",
    "examples/data_pipeline/text_tokenizer.py",
    "examples/data_pipeline/low_memory_training.py",
]


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing pipeline artifacts: {0}".format(missing))
    if PIPELINE_API_VERSION != 1:
        raise SystemExit("unexpected pipeline API version")
    if any(value is None for value in
           (PipelineDataset, PipelineIterator, PipelineDataProvider)):
        raise SystemExit("public pipeline aliases are unavailable")
    print("data pipeline final artifact gate passed ({0} files)".format(len(REQUIRED)))


if __name__ == "__main__":
    main()
