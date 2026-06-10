import pathlib
import unittest


class TestPytorchLoadCompatibility(unittest.TestCase):
    def test_backend_load_defaults_to_weights_only_true(self):
        source = pathlib.Path('trident/backend/pytorch_backend.py').read_text(encoding='utf-8')
        self.assertIn('def load(f, weights_only=True):', source)
        self.assertIn("torch.load(f, map_location=torch.device('cpu'), weights_only=weights_only)", source)

    def test_word2vec_load_explicitly_opts_into_legacy_module_checkpoint(self):
        source = pathlib.Path('trident/models/pytorch_embedded.py').read_text(encoding='utf-8')
        self.assertIn("load(os.path.join(dirname, 'word2vec_chinese.pth'), weights_only=False)", source)


if __name__ == '__main__':
    unittest.main()
