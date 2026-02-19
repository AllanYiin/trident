import pathlib
import unittest


class TestSummaryOutputShapeGuard(unittest.TestCase):
    def test_summary_initializes_output_shape(self):
        source = pathlib.Path('trident/backend/pytorch_backend.py').read_text(encoding='utf-8')
        self.assertIn('summary[m_key]["output_shape"] = None', source)

    def test_summary_uses_safe_get_for_output_shape_print(self):
        source = pathlib.Path('trident/backend/pytorch_backend.py').read_text(encoding='utf-8')
        self.assertIn('summary[layer].get("output_shape", None)', source)


if __name__ == '__main__':
    unittest.main()
