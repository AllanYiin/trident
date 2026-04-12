import pathlib
import unittest


class TestGradientClippingAccumulationTiming(unittest.TestCase):
    def test_trainer_marks_accumulation_state_on_optimizer(self):
        source = pathlib.Path('trident/optims/pytorch_trainer.py').read_text(encoding='utf-8')
        self.assertIn("setattr(self.optimizer, '_trident_is_accumulating_gradients', accumulate_grads)", source)

    def test_agc_checks_accumulation_state_before_clipping(self):
        source = pathlib.Path('trident/optims/pytorch_optimizers.py').read_text(encoding='utf-8')
        self.assertIn('def _should_apply_adaptive_gradient_clipping(self):', source)
        self.assertIn("return not getattr(self, '_trident_is_accumulating_gradients', False)", source)
        self.assertNotIn('if self.use_adaptive_gradient_clipping:', source)
        self.assertIn('if self._should_apply_adaptive_gradient_clipping():', source)


if __name__ == '__main__':
    unittest.main()
