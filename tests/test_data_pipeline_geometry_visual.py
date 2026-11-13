import copy
import unittest
from pathlib import Path

import cv2
import numpy as np

from trident.data.pipeline import (
    DatasetSchema,
    FieldSpec,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFlip,
    RandomPerspective,
    Resize,
    TransformContext,
)


RESULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "test_data"
    / "TransformResults"
)


def _make_geometry_fixture():
    height, width = 160, 240
    image = np.full((height, width, 3), (30, 34, 42), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    for x in range(0, width, 20):
        cv2.line(image, (x, 0), (x, height - 1), (48, 52, 62), 1)
    for y in range(0, height, 20):
        cv2.line(image, (0, y), (width - 1, y), (48, 52, 62), 1)

    cv2.rectangle(mask, (30, 35), (105, 125), 1, -1)
    cv2.circle(mask, (175, 80), 40, 2, -1)
    image[mask == 1] = (58, 154, 92)
    image[mask == 2] = (196, 105, 52)
    cv2.rectangle(image, (30, 35), (105, 125), (105, 230, 145), 2)
    cv2.circle(image, (175, 80), 40, (245, 180, 75), 2)

    return {
        "image": image,
        "mask": mask,
        "boxes": np.array(
            [[30.0, 35.0, 106.0, 126.0], [135.0, 40.0, 216.0, 121.0]],
            dtype=np.float32,
        ),
        "keypoints": np.array(
            [[68.0, 80.0, 1.0], [175.0, 80.0, 1.0]], dtype=np.float32
        ),
    }


def _schema():
    return DatasetSchema(
        [
            FieldSpec("image", kind="image", layout="HWC"),
            FieldSpec("mask", kind="mask", layout="HWC"),
            FieldSpec("boxes", kind="bbox", coordinate_format="xyxy"),
            FieldSpec("keypoints", kind="keypoints"),
        ]
    )


def _transform_cases(sample, schema):
    cases = [("original", copy.deepcopy(sample))]
    transforms = [
        ("resize 120x180", Resize((120, 180))),
        ("horizontal flip", RandomHorizontalFlip(probability=1.0)),
        ("random crop", RandomCrop((120, 180))),
        (
            "random affine",
            RandomAffine(
                degrees=18,
                translate=(0.12, 0.12),
                scale=(0.9, 1.05),
                shear=5,
            ),
        ),
        (
            "random perspective",
            RandomPerspective(distortion=0.12, probability=1.0),
        ),
    ]

    for case_index, (name, transform) in enumerate(transforms, start=1):
        context = TransformContext(seed=2026, epoch=3, sample_id=case_index)
        output = transform(copy.deepcopy(sample), context=context, schema=schema)
        replayed = transform.replay(
            copy.deepcopy(sample), context.records[0], schema=schema
        )
        for field in ("image", "mask", "boxes", "keypoints"):
            np.testing.assert_allclose(output[field], replayed[field])
        cases.append((name, output))
    return cases


def _annotated_image(sample, show_mask):
    image = np.asarray(sample["image"]).copy().astype(np.uint8)
    if show_mask:
        mask = np.asarray(sample["mask"])
        palette = np.array(
            [[0, 0, 0], [30, 225, 95], [245, 130, 35]], dtype=np.uint8
        )
        colors = palette[np.clip(mask.astype(np.int64), 0, len(palette) - 1)]
        foreground = mask > 0
        image[foreground] = (
            image[foreground].astype(np.float32) * 0.55
            + colors[foreground].astype(np.float32) * 0.45
        ).astype(np.uint8)
    else:
        for box in np.asarray(sample["boxes"]):
            x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (50, 220, 255), 2)
        for point in np.asarray(sample["keypoints"]):
            if point.shape[0] > 2 and point[2] <= 0:
                continue
            x, y = np.rint(point[:2]).astype(int)
            cv2.drawMarker(
                image,
                (x, y),
                (255, 80, 220),
                markerType=cv2.MARKER_CROSS,
                markerSize=11,
                thickness=2,
            )
    return image


def _make_contact_sheet(cases, show_mask):
    preview_height, preview_width = 180, 270
    title_height = 28
    cells = []
    for name, sample in cases:
        preview = cv2.resize(
            _annotated_image(sample, show_mask),
            (preview_width, preview_height),
            interpolation=cv2.INTER_NEAREST,
        )
        cell = np.full(
            (preview_height + title_height, preview_width, 3), 245, dtype=np.uint8
        )
        cell[title_height:] = preview
        cv2.putText(
            cell,
            name,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (28, 28, 28),
            1,
            cv2.LINE_AA,
        )
        cells.append(cell)
    return np.concatenate(
        [np.concatenate(cells[:3], axis=1), np.concatenate(cells[3:], axis=1)],
        axis=0,
    )


class TestPipelineGeometryVisualResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = _schema()
        cls.sample = _make_geometry_fixture()
        cls.cases = _transform_cases(cls.sample, cls.schema)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def test_segmentation_fields_stay_paired_and_write_visual_result(self):
        expected_labels = {0, 1, 2}
        for _, sample in self.cases:
            self.assertTrue(set(np.unique(sample["mask"])).issubset(expected_labels))
            self.assertEqual(sample["image"].shape[:2], sample["mask"].shape[:2])

        output_path = RESULT_DIR / "pipeline_geometry_segmentation_paired.png"
        self.assertTrue(cv2.imwrite(str(output_path), _make_contact_sheet(self.cases, True)))
        self.assertGreater(output_path.stat().st_size, 0)

    def test_position_fields_stay_paired_and_write_visual_result(self):
        expected_mask_labels = (1, 2)
        for _, sample in self.cases:
            mask = np.asarray(sample["mask"])
            height, width = mask.shape[:2]
            for point, expected_label in zip(sample["keypoints"], expected_mask_labels):
                x, y = np.rint(point[:2]).astype(int)
                if not (0 <= x < width and 0 <= y < height):
                    continue
                radius = 2
                neighborhood = mask[
                    max(0, y - radius) : min(height, y + radius + 1),
                    max(0, x - radius) : min(width, x + radius + 1),
                ]
                self.assertIn(expected_label, neighborhood)

        output_path = RESULT_DIR / "pipeline_geometry_bbox_keypoints_paired.png"
        self.assertTrue(cv2.imwrite(str(output_path), _make_contact_sheet(self.cases, False)))
        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
