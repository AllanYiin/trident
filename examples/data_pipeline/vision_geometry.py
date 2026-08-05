import _bootstrap  # noqa: F401
import numpy as np

from trident.data.pipeline import (
    Dataset, DatasetSchema, FieldSpec, GeometryCompose, Iterator,
    RandomHorizontalFlip, Resize,
)


schema = DatasetSchema([
    FieldSpec("image", kind="image", layout="HWC"),
    FieldSpec("mask", kind="mask", layout="HWC"),
    FieldSpec("boxes", kind="bbox", coordinate_format="xyxy"),
])
sample = {
    "image": np.zeros((8, 12, 3), np.uint8),
    "mask": np.zeros((8, 12), np.uint8),
    "boxes": np.asarray([[1, 1, 6, 5]], np.float32),
}
geometry = GeometryCompose([Resize((16, 24)), RandomHorizontalFlip(1.0)])
batch = next(iter(Iterator(Dataset([sample], schema=schema),
                           transforms=[geometry], seed=7)))
print(batch["image"].shape, batch["boxes"])
print([record.to_dict() for record in batch.metadata["transform_records"][0]])
