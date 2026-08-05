import time
import unittest

import numpy as np

from trident.data.pipeline import Dataset, Iterator, MemoryBudgetError


def slow_identity(sample):
    time.sleep((5 - sample["value"]) * 0.002)
    return sample


class WP5ExecutorTests(unittest.TestCase):
    def test_thread_prefetch_preserves_batch_order(self):
        rows = [{"value": index} for index in range(6)]
        batches = list(Iterator(Dataset(rows), batch_size=1, workers=3,
                                prefetch_batches=3, transforms=[slow_identity]))
        self.assertEqual([int(batch["value"][0]) for batch in batches], list(range(6)))

    def test_mid_epoch_checkpoint_resumes_at_next_batch(self):
        rows = [{"value": index} for index in range(8)]
        iterator = Iterator(Dataset(rows), batch_size=2, shuffle=True, seed=17)
        active = iter(iterator)
        first = next(active)
        state = iterator.state_dict()
        active.close()
        resumed = Iterator(Dataset(rows), batch_size=2, shuffle=True, seed=999)
        resumed.load_state_dict(state)
        remainder = list(resumed)
        values = list(first["value"])
        for batch in remainder:
            values.extend(batch["value"].tolist())
        self.assertEqual(sorted(values), list(range(8)))
        self.assertEqual(len(values), 8)
        self.assertEqual(resumed.epoch, 1)

    def test_streaming_shuffle_buffer_is_reproducible_and_epoch_aware(self):
        def factory():
            return ({"value": index} for index in range(20))

        def order(epoch):
            iterator = Iterator(Dataset.from_iterable(factory), batch_size=4,
                                shuffle=True, shuffle_buffer_size=5, seed=23,
                                steps_per_epoch=5).set_epoch(epoch)
            return np.concatenate([batch["value"] for batch in iterator]).tolist()

        first = order(0)
        self.assertEqual(first, order(0))
        self.assertNotEqual(first, order(1))
        self.assertEqual(sorted(first), list(range(20)))

    def test_persistent_thread_executor_is_reused_and_closeable(self):
        iterator = Iterator(Dataset([{"value": i} for i in range(4)]),
                            batch_size=2, workers=2, persistent_workers=True)
        list(iterator)
        executor = iterator._executor
        list(iterator)
        self.assertIs(iterator._executor, executor)
        iterator.close()
        self.assertIsNone(iterator._executor)
        with self.assertRaises(RuntimeError):
            list(iterator)

    def test_process_executor_map_style_smoke(self):
        iterator = Iterator(Dataset([{"value": i} for i in range(6)]),
                            batch_size=2, workers=2, executor="process",
                            prefetch_batches=2)
        try:
            batches = list(iterator)
        finally:
            iterator.close()
        self.assertEqual(np.concatenate([batch["value"] for batch in batches]).tolist(),
                         list(range(6)))

    def test_memory_budget_rejects_single_oversized_batch(self):
        iterator = Iterator(Dataset([{"image": np.zeros((128, 128), np.float32)}]),
                            memory_budget_mb=0.001)
        with self.assertRaises(MemoryBudgetError):
            list(iterator)


if __name__ == "__main__":
    unittest.main()
