import os
import tempfile
import unittest

import numpy as np

from trident.data.dataset import NumpyDataset as LegacyNumpyDataset
from trident.data.pipeline import Dataset, DatasetSchema, FieldSpec, Iterator, Sample


class FakeHFMap(object):
    __module__ = "datasets.arrow_dataset"
    features = {"text": "string", "label": "int64"}
    column_names = ["text", "label"]

    def __init__(self, rows, operations=None):
        self.rows = list(rows)
        self.operations = list(operations or ())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, list):
            return dict((key, [self.rows[row][key] for row in index])
                        for key in self.rows[0])
        return dict(self.rows[index])

    def with_format(self, type=None, columns=None, output_all_columns=False, **kwargs):
        return FakeHFMap(self.rows, self.operations + [("format", type, columns)])

    def with_transform(self, transform, columns=None, output_all_columns=False):
        rows = []
        for row in self.rows:
            batch = dict((key, [value]) for key, value in row.items())
            transformed = transform(batch)
            rows.append(dict((key, value[0]) for key, value in transformed.items()))
        return FakeHFMap(rows, self.operations + [("transform", columns)])

    def select_columns(self, columns):
        return FakeHFMap([
            dict((name, row[name]) for name in columns) for row in self.rows
        ], self.operations + [("columns", tuple(columns))])


class FakeHFIterable(object):
    __module__ = "datasets.iterable_dataset"
    features = {"value": "int64"}
    column_names = ["value"]

    def __init__(self, rows):
        self.rows = list(rows)
        self.epochs = []

    def __iter__(self):
        return iter([dict(row) for row in self.rows])

    def shard(self, num_shards, index):
        return FakeHFIterable(self.rows[index::num_shards])

    def shuffle(self, seed=0, buffer_size=None):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(self.rows))
        return FakeHFIterable([self.rows[index] for index in order])

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class TestDataPipelineAdapters(unittest.TestCase):
    def test_columnar_source_keeps_columns_and_checks_lengths(self):
        values = np.arange(6).reshape(3, 2)
        labels = np.array([0, 1, 0])
        dataset = Dataset.from_columns({"values": values, "labels": labels})

        self.assertIs(values, dataset.source.columns["values"])
        self.assertEqual(3, len(dataset))
        np.testing.assert_array_equal(dataset[2]["values"], np.array([4, 5]))
        with self.assertRaises(ValueError):
            Dataset.from_columns({"a": [1], "b": [1, 2]})

    def test_iterable_factory_is_reusable_and_preserves_source_ids(self):
        calls = []

        def factory():
            calls.append(True)
            return ({"value": value} for value in range(4))

        dataset = Dataset.from_iterable(factory)
        first = list(Iterator(dataset, batch_size=2, steps_per_epoch=2))
        second = list(Iterator(dataset, batch_size=2, steps_per_epoch=2))
        self.assertEqual(2, len(calls))
        self.assertEqual([0, 1], first[0].metadata["sample_ids"])
        np.testing.assert_array_equal(first[1]["value"], second[1]["value"])

    def test_folder_source_loads_lazily_and_can_derive_labels(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            class_a = os.path.join(root, "class_a")
            class_b = os.path.join(root, "class_b")
            os.makedirs(class_a)
            os.makedirs(class_b)
            paths = [os.path.join(class_a, "a.txt"), os.path.join(class_b, "b.txt")]
            for path, value in zip(paths, ("A", "B")):
                with open(path, "w", encoding="utf-8") as output:
                    output.write(value)

            loaded = []

            def loader(path):
                loaded.append(path)
                with open(path, "r", encoding="utf-8") as source:
                    return source.read()

            dataset = Dataset.from_folder(
                root, patterns="*.txt", loader=loader, field="text",
                label_from_parent=True)
            self.assertEqual([], loaded)
            sample = dataset[0]
            self.assertEqual(1, len(loaded))
            self.assertIn(sample["text"], ("A", "B"))
            self.assertIn(sample["label"], ("class_a", "class_b"))
            self.assertTrue(os.path.isabs(sample["path"]))

    def test_legacy_adapter_is_lazy_and_uses_symbols(self):
        data = LegacyNumpyDataset(np.arange(6).reshape(3, 2), symbol="input")
        labels = LegacyNumpyDataset(np.arange(3).reshape(3, 1), symbol="target")
        dataset = Dataset.from_legacy(data=data, label=labels)
        sample = dataset[2]
        self.assertEqual(["input", "target"], list(sample.keys()))
        np.testing.assert_array_equal(sample["input"], np.array([4, 5]))

    def test_huggingface_map_views_and_batched_indices_remain_lazy(self):
        source = FakeHFMap([
            {"text": "甲", "label": 0},
            {"text": "乙", "label": 1},
            {"text": "丙", "label": 2},
        ])
        schema = DatasetSchema([
            FieldSpec("text", kind="text", dtype="str"),
            FieldSpec("label", kind="label", dtype="int64", shape=()),
        ])
        dataset = Dataset.from_huggingface(source, schema=schema)
        formatted = dataset.with_format("numpy", columns=["label"])
        selected = formatted.select_columns(["text"])
        transformed = dataset.with_huggingface_transform(
            lambda batch: {"text": [value + "!" for value in batch["text"]],
                           "label": batch["label"]})

        samples = dataset.get_batch([2, 0])
        self.assertEqual([2, 0], [sample.sample_id for sample in samples])
        self.assertEqual(["丙", "甲"], [sample["text"] for sample in samples])
        self.assertEqual(["text"], list(selected[0].keys()))
        self.assertEqual("甲!", transformed[0]["text"])
        self.assertEqual("format", formatted.source.operations[-1][0])

    def test_huggingface_streaming_shuffle_epoch_and_rank_shards(self):
        source = FakeHFIterable([{"value": value} for value in range(10)])
        dataset = Dataset.from_huggingface(source)
        self.assertTrue(dataset.iterable_style)

        rank_zero = list(Iterator(dataset, batch_size=2, steps_per_epoch=3,
                                  rank=0, world_size=2))
        rank_one = list(Iterator(dataset, batch_size=2, steps_per_epoch=3,
                                 rank=1, world_size=2))
        zero_values = np.concatenate([batch["value"] for batch in rank_zero]).tolist()
        one_values = np.concatenate([batch["value"] for batch in rank_one]).tolist()
        self.assertEqual([0, 2, 4, 6, 8], zero_values)
        self.assertEqual([1, 3, 5, 7, 9], one_values)
        self.assertFalse(set(zero_values) & set(one_values))
        self.assertEqual([0, 0], source.epochs)

        shuffled_a = [row["value"] for row in dataset.shuffle(seed=7, buffer_size=4)]
        shuffled_b = [row["value"] for row in dataset.shuffle(seed=7, buffer_size=4)]
        self.assertEqual(shuffled_a, shuffled_b)
        self.assertNotEqual(list(range(10)), shuffled_a)


if __name__ == "__main__":
    unittest.main()
