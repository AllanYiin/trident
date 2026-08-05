import unittest

import numpy as np

from trident.data.dataset import (
    BboxDataset as LegacyBboxDataset,
    ImageDataset as LegacyImageDataset,
    Iterator as LegacyIterator,
    NumpyDataset as LegacyNumpyDataset,
)
from trident.data.vision_transforms import Resize as LegacyResize
from trident.data.pipeline import (
    Dataset,
    DatasetSchema,
    FieldSpec,
    Iterator,
    Resize,
)


class TestWp0LegacyCharacterization(unittest.TestCase):
    def test_new_resize_preserves_legacy_geometry_semantics(self):
        image = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
        boxes = np.array([[1.0, 1.0, 3.0, 3.0, 7.0]], dtype=np.float32)

        legacy = LegacyIterator(
            data=LegacyImageDataset([image], symbol="image"),
            label=LegacyBboxDataset([boxes], symbol="boxes"),
            batch_size=1, is_shuffle=False, workers=0)
        legacy.paired_transform_funcs = [LegacyResize((8, 16), keep_aspect=False)]
        legacy_image, legacy_boxes = legacy.next()

        schema = DatasetSchema([
            FieldSpec("image", kind="image", layout="HWC"),
            FieldSpec("boxes", kind="bbox", coordinate_format="xyxy"),
        ])
        pipeline = next(iter(Iterator(
            Dataset([{"image": image, "boxes": boxes}], schema=schema),
            transforms=[Resize((8, 16))])))

        self.assertEqual((1, 3, 8, 16), legacy_image.shape)
        self.assertEqual((1, 8, 16, 3), pipeline["image"].shape)
        np.testing.assert_allclose(legacy_boxes, pipeline["boxes"])

    def test_legacy_rejects_numpy_unicode_but_new_batch_keeps_python_strings(self):
        values = np.array([["中文"], ["資料"]], dtype=np.str_)
        with self.assertRaises(TypeError):
            LegacyIterator(
                data=LegacyNumpyDataset(values, symbol="text"),
                batch_size=2, is_shuffle=False, workers=0)

        pipeline_text = next(iter(Iterator(
            Dataset([{"text": "中文"}, {"text": "資料"}]), batch_size=2)))["text"]
        self.assertEqual(["中文", "資料"], pipeline_text)
        self.assertTrue(all(isinstance(value, str) for value in pipeline_text))

    def test_legacy_cycles_to_fill_last_batch_while_new_keeps_partial_batch(self):
        legacy = LegacyIterator(
            data=LegacyNumpyDataset(np.arange(5).reshape(5, 1), symbol="value"),
            batch_size=2, is_shuffle=False, workers=0)
        legacy_batches = [legacy.next()[0] for _ in range(4)]
        pipeline_batches = list(Iterator(
            Dataset([{"value": value} for value in range(5)]), batch_size=2))

        legacy_values = [batch.reshape(-1).tolist() for batch in legacy_batches]
        self.assertEqual([[0, 1], [0, 1], [2, 3], [4, 0]], legacy_values)
        self.assertEqual([2, 2, 1], [len(batch["value"]) for batch in pipeline_batches])


if __name__ == "__main__":
    unittest.main()
