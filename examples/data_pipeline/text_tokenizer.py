import _bootstrap  # noqa: F401
from trident.data.pipeline import Dataset, Iterator, TokenizerCollator


class SmallTokenizer(object):
    def __call__(self, texts, padding="longest", return_tensors=None, **kwargs):
        token_ids = [[ord(character) % 1000 for character in text] for text in texts]
        width = max(map(len, token_ids))
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in token_ids],
            "attention_mask": [[1] * len(row) + [0] * (width - len(row))
                               for row in token_ids],
        }


rows = [{"text": "中文", "label": 0}, {"text": "資料管線", "label": 1}]
batch = next(iter(Iterator(
    Dataset(rows), batch_size=2,
    collate=TokenizerCollator(SmallTokenizer(), text_field="text"))))
print(batch)
