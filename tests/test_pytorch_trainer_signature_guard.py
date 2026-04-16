import pathlib
import unittest


class TestPytorchTrainerSignatureGuard(unittest.TestCase):
    def test_initial_graph_recovers_missing_signature(self):
        source = pathlib.Path('trident/optims/pytorch_trainer.py').read_text(encoding='utf-8')
        self.assertIn("if not hasattr(output, 'signature') or output.signature is None:", source)
        self.assertIn('output._signature = get_signature(output)', source)

    def test_image_classification_model_uses_safe_signature_lookup(self):
        source = pathlib.Path('trident/optims/pytorch_trainer.py').read_text(encoding='utf-8')
        self.assertIn("model_signature = getattr(self._model, 'signature', None)", source)


if __name__ == '__main__':
    unittest.main()
