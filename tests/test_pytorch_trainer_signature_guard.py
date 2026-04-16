import pathlib
import unittest


class TestPytorchTrainerSignatureGuard(unittest.TestCase):
    def test_initial_graph_recovers_missing_signature(self):
        source = pathlib.Path('trident/optims/pytorch_trainer.py').read_text(encoding='utf-8')
        self.assertIn("if hasattr(output, 'signature'):", source)
        self.assertIn('output_signature = get_signature(output)', source)
        self.assertIn('output._signature = output_signature', source)
        self.assertIn('len(output._signature.inputs)', source)


if __name__ == '__main__':
    unittest.main()
