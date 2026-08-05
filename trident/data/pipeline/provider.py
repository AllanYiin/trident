from __future__ import absolute_import, division, print_function

from .dataset import Dataset


class DataProvider(object):
    """Owns named dataset splits and creates configured Iterators."""

    def __init__(self, train=None, valid=None, test=None, dataset_name="",
                 metadata=None, **splits):
        self.dataset_name = dataset_name
        self.metadata = dict(metadata or {})
        self._splits = {}
        for name, value in (("train", train), ("valid", valid), ("test", test)):
            if value is not None:
                self.add_split(name, value)
        for name, value in splits.items():
            if value is not None:
                self.add_split(name, value)

    def add_split(self, name, dataset):
        if not isinstance(dataset, Dataset):
            dataset = Dataset(dataset, name=name)
        self._splits[name] = dataset
        return self

    def split(self, name):
        if name not in self._splits:
            raise KeyError("unknown dataset split: {0}".format(name))
        return self._splits[name]

    def iter(self, split="train", **kwargs):
        from .iterator import Iterator
        return Iterator(self.split(split), **kwargs)

    def for_trident_trainer(self, split="train", input_fields=None,
                            target_fields=None, unpaired_fields=None, **iterator_kwargs):
        from .compat import TrainingPlanAdapter
        iterator = self.iter(split, **iterator_kwargs)
        return TrainingPlanAdapter(
            iterator, input_fields=input_fields, target_fields=target_fields,
            unpaired_fields=unpaired_fields)

    @property
    def train(self):
        return self._splits.get("train")

    @property
    def valid(self):
        return self._splits.get("valid")

    @property
    def test(self):
        return self._splits.get("test")

    def keys(self):
        return self._splits.keys()

    def __getitem__(self, name):
        return self.split(name)
