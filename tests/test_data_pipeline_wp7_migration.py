import unittest
import warnings

import numpy as np

from trident.data import PipelineDataProvider, PipelineDataset, PipelineIterator
from trident.data.pipeline import (
    DataProvider, Dataset, Iterator, PIPELINE_API_VERSION,
    migrate_legacy_provider,
)


class LegacyField(object):
    def __init__(self, symbol, values):
        self.symbol = symbol
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class LegacyIterator(object):
    def __init__(self):
        self.data = LegacyField("x", [1, 2, 3])
        self.label = LegacyField("target", [2, 4, 6])
        self.unpair = None


class LegacyProvider(object):
    dataset_name = "legacy-regression"
    traindata = LegacyIterator()
    validdata = None
    testdata = None


class WP7MigrationTests(unittest.TestCase):
    def test_public_aliases_are_explicit_and_versioned(self):
        self.assertIs(PipelineDataset, Dataset)
        self.assertIs(PipelineIterator, Iterator)
        self.assertIs(PipelineDataProvider, DataProvider)
        self.assertEqual(PIPELINE_API_VERSION, 1)

    def test_legacy_provider_bridge_is_lazy_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            provider = migrate_legacy_provider(LegacyProvider())
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        batch = next(iter(provider.iter("train", batch_size=2)))
        np.testing.assert_array_equal(batch["x"], [1, 2])
        np.testing.assert_array_equal(batch["target"], [2, 4])
        self.assertEqual(provider.metadata["migrated_from"], "LegacyProvider")


if __name__ == "__main__":
    unittest.main()
