import pathlib
import unittest


class TestTrainingSaveGuardrails(unittest.TestCase):
    def test_training_end_no_longer_forces_save_model(self):
        source = pathlib.Path('trident/optims/trainers.py').read_text(encoding='utf-8')
        self.assertIn('def do_on_training_end(self):', source)
        self.assertNotIn('item.save_model()', source)

    def test_pytorch_save_model_skips_when_nan_or_inf_detected(self):
        source = pathlib.Path('trident/optims/pytorch_trainer.py').read_text(encoding='utf-8')
        self.assertIn('nan/inf detected in model weights. Skip saving.', source)
        self.assertIn('if is_abnormal:', source)
        self.assertIn('return False', source)

    def test_pytorch_save_model_uses_atomic_replace(self):
        source = pathlib.Path('trident/optims/pytorch_trainer.py').read_text(encoding='utf-8')
        self.assertIn('os.replace(temp_path, target_path)', source)


if __name__ == '__main__':
    unittest.main()
