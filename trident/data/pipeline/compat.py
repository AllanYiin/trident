from __future__ import absolute_import, division, print_function

from trident.backend.common import OrderedDict


_DEFAULT_TARGET_KINDS = frozenset((
    "label", "bbox", "mask", "depth", "keypoints", "landmarks", "polygon",
))


class _CompatField(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, _CompatField) and self.name == other.name


class _SymbolGroup(object):
    def __init__(self, symbols):
        if not symbols:
            self.symbol = ""
        elif len(symbols) == 1:
            self.symbol = symbols[0]
        else:
            self.symbol = tuple(symbols)

    def __len__(self):
        return 0 if self.symbol == "" else (len(self.symbol) if isinstance(self.symbol, tuple) else 1)


class _CompatTrainData(object):
    def __init__(self, input_fields, target_fields, unpaired_fields, template):
        self.data = _SymbolGroup(input_fields)
        self.label = _SymbolGroup(target_fields)
        self.unpair = _SymbolGroup(unpaired_fields)
        self.data_template = template
        self.batch_sampler = None


class TrainingPlanAdapter(object):
    """Narrow adapter from the new Iterator to the legacy TrainingPlan contract.

    The adapter intentionally contains all knowledge of the old TensorSpec-keyed
    batches. The new Dataset/Iterator/DataProvider remain independent of it.
    """

    def __init__(self, iterator, input_fields=None, target_fields=None,
                 unpaired_fields=None):
        self.iterator = iterator
        schema = iterator.dataset.schema
        if input_fields is None and target_fields is None:
            if len(schema) == 0:
                raise ValueError(
                    "TrainingPlanAdapter needs a schema or explicit input_fields/target_fields")
            input_fields = []
            target_fields = []
            unpaired_fields = []
            for field in schema:
                role = field.metadata.get("role")
                if role == "unpaired":
                    unpaired_fields.append(field.name)
                elif role == "target" or (role is None and field.kind in _DEFAULT_TARGET_KINDS):
                    target_fields.append(field.name)
                else:
                    input_fields.append(field.name)
        self.input_fields = list(input_fields or ())
        self.target_fields = list(target_fields or ())
        self.unpaired_fields = list(unpaired_fields or ())
        names = self.input_fields + self.target_fields + self.unpaired_fields
        self._fields = OrderedDict((name, _CompatField(name)) for name in names)
        template = OrderedDict((self._fields[name], None) for name in names)
        self.traindata = _CompatTrainData(
            self.input_fields, self.target_fields, self.unpaired_fields, template)
        self.traindata.batch_sampler = self
        self.testdata = None
        self.signature = object()
        self._batch_transform_funcs = []
        self._mode = "dict"
        self._cursor = None

    @property
    def batch_sampler(self):
        return self

    @property
    def batch_size(self):
        return self.iterator.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.iterator.batch_size = int(value)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    def _wrap(self, batch):
        output = OrderedDict()
        for name, value in batch.items():
            field = self._fields.get(name)
            if field is None:
                field = _CompatField(name)
                self._fields[name] = field
            output[field] = value
        return output

    def __iter__(self):
        for batch in self.iterator:
            yield self._wrap(batch)

    def __len__(self):
        return len(self.iterator)

    def next(self):
        if self._cursor is None:
            self._cursor = iter(self)
        return next(self._cursor)

    def __next__(self):
        return self.next()

    def next_test(self):
        return None
