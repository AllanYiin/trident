import pathlib
import unittest


class TestReduceSumIterableCompatibility(unittest.TestCase):
    def test_reduce_sum_falls_back_to_builtin_sum_for_iterable(self):
        source = pathlib.Path('trident/backend/pytorch_ops.py').read_text(encoding='utf-8')
        self.assertIn('if not is_tensor(x):', source)
        self.assertIn('if isinstance(x, Iterable):', source)
        self.assertIn('return builtins.sum(x)', source)


if __name__ == '__main__':
    unittest.main()
