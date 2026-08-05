import json
import unittest

import numpy as np

from trident.data.pipeline import (
    CopyPaste, CutMix, Dataset, DatasetSchema, FieldSpec, Iterator, MixUp, Mosaic,
    RandomAffine, RandomCrop, RandomPerspective, Resize, SanitizeTargets,
    TransformContext, TransformRecord,
)


class WP3AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.schema = DatasetSchema([
            FieldSpec("image", kind="image", layout="HWC"),
            FieldSpec("mask", kind="mask", layout="HWC"),
            FieldSpec("boxes", kind="bbox", coordinate_format="xyxy"),
            FieldSpec("classes", kind="label", metadata={"linked_to": "boxes"}),
            FieldSpec("keypoints", kind="keypoints"),
        ])
        self.sample = {
            "image": np.arange(64, dtype=np.uint8).reshape(8, 8),
            "mask": np.eye(8, dtype=np.uint8),
            "boxes": np.array([[1, 1, 5, 5]], dtype=np.float32),
            "classes": np.array([3]),
            "keypoints": np.array([[2, 2, 1]], dtype=np.float32),
        }

    def test_affine_is_deterministic_serializable_and_replayable(self):
        transform = RandomAffine(degrees=15, translate=(0.1, 0.1),
                                 scale=(0.9, 1.1), shear=5)
        context = TransformContext(seed=7, epoch=2, sample_id=11)
        output = transform(self.sample, context=context, schema=self.schema)
        record = context.records[0]
        json.dumps(record.to_dict())
        replayed = transform.replay(self.sample, record.to_dict(), schema=self.schema)
        np.testing.assert_array_equal(output["image"], replayed["image"])
        np.testing.assert_allclose(output["boxes"], replayed["boxes"])
        np.testing.assert_allclose(output["keypoints"], replayed["keypoints"])
        self.assertIn("inverse_matrix", record.params)

    def test_worker_count_does_not_change_perspective(self):
        transform = RandomPerspective(distortion=0.2, probability=1.0)
        outputs = []
        for worker_id in (0, 9):
            context = TransformContext(seed=5, epoch=1, rank=0,
                                       worker_id=worker_id, sample_id="same")
            outputs.append(transform(self.sample, context=context, schema=self.schema))
        np.testing.assert_array_equal(outputs[0]["image"], outputs[1]["image"])
        np.testing.assert_allclose(outputs[0]["boxes"], outputs[1]["boxes"])

    def test_sanitize_drops_boxes_and_linked_targets_together(self):
        sample = dict(self.sample)
        sample["boxes"] = np.array([[0, 0, 0, 2], [1, 1, 4, 4]], dtype=np.float32)
        sample["classes"] = np.array([8, 9])
        output = SanitizeTargets(policy="drop")(sample, schema=self.schema)
        np.testing.assert_array_equal(output["classes"], [9])
        self.assertEqual(output["boxes"].shape, (1, 4))

    def test_mixup_and_cutmix_return_mix_metadata(self):
        samples = [
            {"image": np.zeros((8, 8, 3), np.float32), "label": 0},
            {"image": np.ones((8, 8, 3), np.float32), "label": 1},
        ]
        contexts = [TransformContext(seed=3, sample_id=4)]
        mixed = MixUp(alpha=0.4)(samples, contexts=contexts)
        self.assertIn("_mixup", mixed)
        self.assertEqual(mixed["label"][:2], (0, 1))
        cut = CutMix(alpha=1.0)(samples, contexts=contexts)
        self.assertIn("_cutmix", cut)
        self.assertEqual(cut["image"].shape, samples[0]["image"].shape)

    def test_mosaic_and_copy_paste(self):
        mosaic_samples = [{"image": np.full((2, 3, 1), index, np.uint8)}
                          for index in range(4)]
        output = Mosaic()(mosaic_samples)
        self.assertEqual(output["image"].shape, (4, 6, 1))
        base = {"image": np.zeros((2, 2, 1), np.uint8)}
        donor = {"image": np.ones((2, 2, 1), np.uint8),
                 "mask": np.array([[0, 1], [0, 0]], np.uint8)}
        pasted = CopyPaste()([base, donor])
        self.assertEqual(int(pasted["image"].sum()), 1)
        self.assertEqual(pasted["_copypaste"]["pixels"], 1)


    def test_crop_then_sanitize_keeps_linked_targets_aligned(self):
        sample = dict(self.sample)
        sample["boxes"] = np.array([[0, 0, 1, 1], [3, 3, 7, 7]], np.float32)
        sample["classes"] = np.array([4, 5])
        record = TransformRecord("RandomCrop", {
            "x": 2, "y": 2, "crop_h": 4, "crop_w": 4,
            "height": 4, "width": 4,
        })
        cropped = RandomCrop((4, 4)).replay(sample, record, schema=self.schema)
        self.assertEqual(len(cropped["boxes"]), 2)
        cleaned = SanitizeTargets("drop")(cropped, schema=self.schema)
        np.testing.assert_array_equal(cleaned["classes"], [5])

    def test_optical_flow_resize_scales_vectors(self):
        schema = DatasetSchema([
            FieldSpec("image", kind="image", layout="HWC"),
            FieldSpec("flow", kind="optical_flow", layout="HWC"),
        ])
        sample = {"image": np.zeros((2, 4, 3), np.uint8),
                  "flow": np.ones((2, 4, 2), np.float32)}
        output = Resize((4, 12))(sample, schema=schema)
        np.testing.assert_allclose(output["flow"][..., 0], 3.0)
        np.testing.assert_allclose(output["flow"][..., 1], 2.0)

    def test_group_transform_runs_inside_iterator(self):
        rows = [{"image": np.zeros((2, 2, 1), np.float32), "label": 0},
                {"image": np.ones((2, 2, 1), np.float32), "label": 1}]
        batch = next(iter(Iterator(Dataset(rows), batch_size=2,
                                   group_transforms=[MixUp(alpha=0.4)])))
        self.assertEqual(batch["image"].shape, (1, 2, 2, 1))
        self.assertIn("_mixup", batch)


if __name__ == "__main__":
    unittest.main()
