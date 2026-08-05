import json
import unittest

import numpy as np

from trident.data.pipeline import (
    Batch,
    CollationError,
    DataPipelineError,
    Dataset,
    DatasetSchema,
    FieldSpec,
    Iterator,
    Sample,
    SchemaValidationError,
    SourceError,
)


class BrokenSource(object):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise OSError("broken source")


class TestDataPipelineContracts(unittest.TestCase):
    def test_schema_round_trip_is_ordered_and_fingerprinted(self):
        schema = DatasetSchema([
            FieldSpec("tokens", kind="sequence", dtype=np.int64,
                      shape=(None,), variable_axes=(0,), pad_value=0,
                      metadata={"role": "input"}),
            FieldSpec("label", kind="label", dtype="int64", shape=(),
                      metadata={"role": "target"}),
        ])
        encoded = json.loads(json.dumps(schema.to_dict()))
        restored = DatasetSchema.from_dict(encoded)

        self.assertEqual(schema, restored)
        self.assertEqual(["tokens", "label"], list(restored.keys()))
        self.assertEqual(schema.fingerprint, restored.fingerprint)
        self.assertEqual(["label"], [field.name for field in restored.fields_by_role("target")])

    def test_schema_validates_required_dtype_shape_and_strict_fields(self):
        schema = DatasetSchema([
            FieldSpec("image", kind="image", dtype="uint8", shape=(4, 8, 3)),
            FieldSpec("caption", kind="text", dtype="str", required=False),
        ])
        valid = {"image": np.zeros((4, 8, 3), dtype=np.uint8)}
        self.assertIs(valid, schema.validate(valid))

        with self.assertRaises(SchemaValidationError):
            schema.validate({})
        with self.assertRaises(SchemaValidationError):
            schema.validate({"image": np.zeros((4, 8, 3), dtype=np.float32)})
        with self.assertRaises(SchemaValidationError):
            schema.validate({"image": np.zeros((8, 4, 3), dtype=np.uint8)})
        with self.assertRaises(SchemaValidationError):
            schema.validate(dict(valid, unexpected=1), strict=True)

    def test_dataset_returns_sample_with_stable_identity(self):
        dataset = Dataset([{"value": 3}, {"value": 5}])
        sample = dataset[1]
        self.assertIsInstance(sample, Sample)
        self.assertEqual(1, sample.sample_id)
        self.assertEqual(5, sample["value"])

        copied = sample.copy()
        copied.metadata["source"] = "unit-test"
        self.assertEqual(1, copied.sample_id)
        self.assertNotIn("source", sample.metadata)

    def test_dataset_enforces_schema_at_source_boundary(self):
        schema = DatasetSchema([FieldSpec("value", dtype="float32", shape=(2,))])
        dataset = Dataset([{"value": np.array([1, 2], dtype=np.int64)}], schema=schema)
        with self.assertRaises(SchemaValidationError) as captured:
            dataset[0]
        self.assertEqual("schema_validation_error", captured.exception.error_code)
        self.assertEqual("value", captured.exception.field)

    def test_batch_copy_preserves_schema_without_sharing_metadata(self):
        schema = DatasetSchema([FieldSpec("value")])
        batch = Batch({"value": np.array([1, 2])}, schema=schema,
                      metadata={"epoch": 2})
        copied = batch.copy()
        copied.metadata["epoch"] = 3
        self.assertIs(schema, copied.schema)
        self.assertEqual(2, batch.metadata["epoch"])
        self.assertEqual(3, copied.metadata["epoch"])

    def test_source_and_collation_failures_are_structured(self):
        with self.assertRaises(SourceError) as source_error:
            list(Iterator(Dataset(BrokenSource()), batch_size=1))
        self.assertEqual("source", source_error.exception.stage)
        self.assertIn("broken source", source_error.exception.to_dict()["cause"])

        def broken_collator(samples):
            raise ValueError("cannot collate")

        with self.assertRaises(CollationError) as collate_error:
            list(Iterator(Dataset([{"value": 1}]), collate=broken_collator))
        self.assertEqual("collation_error", collate_error.exception.error_code)
        self.assertEqual("collate", collate_error.exception.stage)
        self.assertIsInstance(collate_error.exception, DataPipelineError)


if __name__ == "__main__":
    unittest.main()
