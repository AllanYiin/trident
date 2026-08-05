import unittest

import numpy as np

from trident.data.pipeline import (
    DataProvider,
    Dataset,
    DatasetSchema,
    FieldSpec,
    TrainingPlanAdapter,
)


class TestDataPipelineCompatibility(unittest.TestCase):
    def test_training_adapter_keeps_legacy_assumptions_at_boundary(self):
        schema = DatasetSchema([
            FieldSpec("input", kind="data"),
            FieldSpec("target", kind="label", metadata={"role": "target"}),
        ])
        provider = DataProvider(train=Dataset([
            {"input": np.array([1.0]), "target": 0},
            {"input": np.array([2.0]), "target": 1},
        ], schema=schema))
        adapter = provider.for_trident_trainer(batch_size=2)

        self.assertIsInstance(adapter, TrainingPlanAdapter)
        self.assertEqual("input", adapter.traindata.data.symbol)
        self.assertEqual("target", adapter.traindata.label.symbol)
        self.assertEqual(1, len(adapter.batch_sampler))

        legacy_batch = next(iter(adapter))
        names = [field.name for field in legacy_batch.keys()]
        self.assertEqual(["input", "target"], names)
        np.testing.assert_array_equal(legacy_batch.value_list[1], np.array([0, 1]))

    def test_training_adapter_requires_explicit_contract_without_schema(self):
        provider = DataProvider(train=Dataset([{"input": 1, "target": 2}]))
        with self.assertRaises(ValueError):
            provider.for_trident_trainer(batch_size=1)
        adapter = provider.for_trident_trainer(
            batch_size=1, input_fields=["input"], target_fields=["target"])
        self.assertEqual("input", adapter.traindata.data.symbol)


if __name__ == "__main__":
    unittest.main()
