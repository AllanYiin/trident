import unittest

import numpy as np

from trident.data.pipeline import (
    FieldCollator, PaddingCollator, RaggedBatch, RaggedCollator,
    Seq2SeqCollator, TorchTensorizer,
)


class FakeSeq2SeqTokenizer(object):
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, sources, text_target=None, padding=None,
                 return_tensors=None, **kwargs):
        self.calls.append((list(sources), list(text_target)))
        return {
            "input_ids": [[11, 12, 0], [21, 0, 0]],
            "attention_mask": [[1, 1, 0], [1, 0, 0]],
            "labels": [[31, 32, 0], [41, 0, 0]],
        }


class WP4CollationTests(unittest.TestCase):
    def test_ragged_batch_uses_flat_values_and_row_splits(self):
        samples = [{"tokens": [1, 2]}, {"tokens": [3]}, {"tokens": []}]
        batch = RaggedCollator(["tokens"])(samples)
        self.assertIsInstance(batch["tokens"], RaggedBatch)
        np.testing.assert_array_equal(batch["tokens"].values, [1, 2, 3])
        np.testing.assert_array_equal(batch["tokens"].row_splits, [0, 2, 3, 3])
        padded, mask = batch["tokens"].to_padded(pad_value=-1)
        np.testing.assert_array_equal(padded, [[1, 2], [3, -1], [-1, -1]])
        np.testing.assert_array_equal(mask, [[1, 1], [1, 0], [0, 0]])

    def test_padding_collator_can_emit_attention_style_mask(self):
        samples = [{"tokens": [1, 2, 3]}, {"tokens": [4]}]
        batch = PaddingCollator(
            {"tokens": {"pad_value": 0, "return_mask": True}},
            pad_to_multiple_of=4)(samples)
        self.assertEqual(batch["tokens"].shape, (2, 4))
        np.testing.assert_array_equal(batch["tokens_mask"],
                                      [[1, 1, 1, 0], [1, 0, 0, 0]])

    def test_field_specific_collator_does_not_force_other_ragged_values(self):
        samples = [{"x": [1], "text": "中文"},
                   {"x": [2, 3], "text": "資料"}]
        batch = FieldCollator({"x": RaggedBatch.from_sequences})(samples)
        self.assertIsInstance(batch["x"], RaggedBatch)
        self.assertEqual(batch["text"], ["中文", "資料"])

    def test_seq2seq_tokenizes_once_and_masks_label_padding(self):
        tokenizer = FakeSeq2SeqTokenizer()
        samples = [{"text": "甲", "target_text": "A", "id": 1},
                   {"text": "乙", "target_text": "B", "id": 2}]
        batch = Seq2SeqCollator(tokenizer)(samples)
        self.assertEqual(len(tokenizer.calls), 1)
        np.testing.assert_array_equal(batch["labels"],
                                      [[31, 32, -100], [41, -100, -100]])
        np.testing.assert_array_equal(batch["id"], [1, 2])
        self.assertNotIn("text", batch)

    def test_torch_tensorizer_preserves_ragged_structure(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        batch = RaggedCollator(["tokens"])([{"tokens": [1, 2]}, {"tokens": [3]}])
        converted = TorchTensorizer()(batch)
        self.assertIsInstance(converted["tokens"], RaggedBatch)
        self.assertIsInstance(converted["tokens"].values, torch.Tensor)
        self.assertIsInstance(converted["tokens"].row_splits, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
