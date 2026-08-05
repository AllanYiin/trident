import unittest

import numpy as np

from trident.data.pipeline import (
    DataPipelineError,
    DataProvider,
    Dataset,
    DatasetSchema,
    FieldSpec,
    GeometryCompose,
    Iterator,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    TokenizerCollator,
)


class VectorSource(object):
    def __init__(self, values):
        self.values = values
        self.batch_calls = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __getitems__(self, indices):
        self.batch_calls += 1
        return [self.values[index] for index in indices]


class FakeHFDataset(object):
    __module__ = "datasets.arrow_dataset"
    features = {"text": "string", "label": "int64"}
    column_names = ["text", "label"]

    def __init__(self):
        self.rows = [
            {"text": "甲", "label": 0},
            {"text": "乙", "label": 1},
            {"text": "丙", "label": 2},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, list):
            return {
                "text": [self.rows[item]["text"] for item in index],
                "label": [self.rows[item]["label"] for item in index],
            }
        return self.rows[index]


class FakeTokenizer(object):
    def __init__(self):
        self.calls = []

    def __call__(self, texts, padding=None, return_tensors=None, **kwargs):
        self.calls.append(list(texts))
        width = max(len(text) for text in texts)
        ids = []
        masks = []
        for text in texts:
            row = [ord(char) for char in text]
            ids.append(row + [0] * (width - len(row)))
            masks.append([1] * len(row) + [0] * (width - len(row)))
        return {"input_ids": ids, "attention_mask": masks}


class StreamSource(object):
    def __init__(self, count):
        self.count = count

    def __iter__(self):
        for value in range(self.count):
            yield {"value": value}


class TestDataPipeline(unittest.TestCase):
    def test_map_dataset_uses_batched_fetch_and_native_mapping(self):
        source = VectorSource([{"value": index, "text": "字{0}".format(index)}
                               for index in range(5)])
        batches = list(Iterator(Dataset(source), batch_size=2))
        self.assertEqual(3, len(batches))
        self.assertGreater(source.batch_calls, 0)
        np.testing.assert_array_equal(batches[0]["value"], np.array([0, 1]))
        self.assertEqual(["字0", "字1"], batches[0]["text"])

    def test_huggingface_adapter_does_not_materialize_dataset(self):
        dataset = Dataset.from_huggingface(FakeHFDataset())
        samples = dataset.get_batch([0, 2])
        self.assertEqual(["甲", "丙"], [sample["text"] for sample in samples])
        self.assertEqual([0, 2], [sample["label"] for sample in samples])

    def test_tokenizer_is_called_once_per_batch_and_text_stays_native(self):
        tokenizer = FakeTokenizer()
        dataset = Dataset([
            {"text": "中文", "label": 1},
            {"text": "資料流", "label": 0},
        ])
        batch = next(iter(Iterator(
            dataset, batch_size=2,
            collate=TokenizerCollator(tokenizer, text_field="text"))))
        self.assertEqual([["中文", "資料流"]], tokenizer.calls)
        self.assertNotIn("text", batch)
        self.assertEqual((2, 3), batch["input_ids"].shape)
        np.testing.assert_array_equal(batch["label"], np.array([1, 0]))

    def test_geometry_parameters_are_shared_across_spatial_fields(self):
        schema = DatasetSchema([
            FieldSpec("image", kind="image", layout="HWC"),
            FieldSpec("mask", kind="mask", layout="HWC"),
            FieldSpec("boxes", kind="bbox", coordinate_format="xyxy"),
        ])
        sample = {
            "image": np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3),
            "mask": np.ones((4, 8), dtype=np.uint8),
            "boxes": np.array([[1.0, 1.0, 3.0, 3.0, 7.0]], dtype=np.float32),
        }
        transform = GeometryCompose([
            Resize((8, 16)),
            RandomHorizontalFlip(probability=1.0),
        ])
        dataset = Dataset([sample], schema=schema)
        batch = next(iter(Iterator(dataset, transforms=[transform])))
        self.assertEqual((1, 8, 16, 3), batch["image"].shape)
        self.assertEqual((1, 8, 16), batch["mask"].shape)
        np.testing.assert_allclose(
            batch["boxes"][0, 0], np.array([10.0, 2.0, 14.0, 6.0, 7.0]))
        self.assertEqual(2, len(batch.metadata["transform_records"][0]))

    def test_augmentation_is_stable_when_worker_count_changes(self):
        schema = DatasetSchema([FieldSpec("image", kind="image", layout="HWC")])
        rows = [{"image": np.arange(100, dtype=np.uint8).reshape(10, 10)}
                for _ in range(8)]
        transform = RandomCrop((6, 6))
        serial = list(Iterator(Dataset(rows, schema=schema), batch_size=2,
                               transforms=[transform], seed=42, workers=0))
        threaded = list(Iterator(Dataset(rows, schema=schema), batch_size=2,
                                 transforms=[transform], seed=42, workers=2,
                                 prefetch_batches=2))
        for left, right in zip(serial, threaded):
            np.testing.assert_array_equal(left["image"], right["image"])

    def test_streaming_dataset_supports_steps_and_rank_sharding(self):
        dataset = Dataset(StreamSource(10), map_style=False)
        iterator = Iterator(dataset, batch_size=2, steps_per_epoch=2,
                            rank=1, world_size=2)
        batches = list(iterator)
        self.assertEqual(2, len(batches))
        np.testing.assert_array_equal(batches[0]["value"], np.array([1, 3]))
        np.testing.assert_array_equal(batches[1]["value"], np.array([5, 7]))

    def test_memory_budget_rejects_a_batch_that_cannot_fit(self):
        dataset = Dataset([{"value": np.zeros((1024,), dtype=np.float32)}])
        iterator = Iterator(dataset, workers=1, memory_budget_mb=0.0001)
        with self.assertRaises(DataPipelineError):
            list(iterator)

    def test_data_provider_owns_splits_and_builds_iterators(self):
        provider = DataProvider(train=Dataset([{"x": 1}, {"x": 2}]))
        batch = next(iter(provider.iter("train", batch_size=2)))
        np.testing.assert_array_equal(batch["x"], np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
