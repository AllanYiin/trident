"""Class frequencies come from the dataset, outside the differentiable loss."""
from collections import Counter

import numpy as np

from trident.backend.common import get_session
from trident.backend.tensorspec import ObjectType, object_type_inference
from trident.data.dataset import BboxDataset, LabelDataset, MaskDataset, TextSequenceDataset, ZipDataset


def dataset_class_weights(loss, target, num_classes):
    providers = getattr(get_session()._thread_local_info, 'data_providers', {})
    if not providers:
        return None
    dataset = list(providers.values())[-1].traindata.label
    if isinstance(dataset, ZipDataset):
        candidates = [ds for ds in dataset.items if ds.symbol in loss.signature.inputs]
        if not candidates:
            allowed = getattr(loss, 'valid_target_object_type', ())
            candidates = [ds for ds in dataset.items if ds.object_type in allowed
                          and ds.object_type == object_type_inference(target)]
        dataset = candidates[-1] if candidates else None
    if dataset is None:
        return None
    counts = getattr(dataset, '_label_statistics', None)
    if counts is None:
        if isinstance(dataset, TextSequenceDataset):
            counts = {dataset.text2index[word]: count for word, count in dataset.vocabs_frequency.items()
                      if word in dataset.text2index}
        elif isinstance(dataset, LabelDataset) or dataset.object_type == ObjectType.classification_label:
            counts = dict(zip(*np.unique(dataset.items, return_counts=True)))
        else:
            is_box = isinstance(dataset, BboxDataset) or dataset.object_type in (ObjectType.absolute_bbox, ObjectType.relative_bbox)
            is_mask = isinstance(dataset, MaskDataset) or dataset.object_type in (ObjectType.label_mask, ObjectType.color_mask, ObjectType.binary_mask)
            if not (is_box or is_mask):
                return None
            indices = range(len(dataset))
            if is_mask and len(dataset) > 1000:
                indices = np.random.choice(len(dataset), 1000, replace=False)
            counts = Counter()
            for idx in indices:
                values = dataset[idx][:, 4] if is_box else dataset[idx]
                counts.update(dict(zip(*np.unique(values, return_counts=True))))
        dataset._label_statistics = counts
    frequencies = np.ones(num_classes, np.float64)
    for label, n in counts.items():
        if 0 <= int(label) < num_classes:
            frequencies[int(label)] = max(1, n)
    if isinstance(dataset, TextSequenceDataset):
        order = np.argsort(-frequencies)
        median = frequencies[order[np.searchsorted(np.cumsum(frequencies[order]), frequencies.sum() / 2)]]
        inverse = frequencies.mean() / frequencies
        weights = np.where(frequencies >= median, inverse, np.sqrt(inverse))
        weights[frequencies <= 10] = 1.
        weights[0] = .01
    else:
        weights = 1 / np.sqrt(frequencies)
        weights /= weights.mean()
    return weights.astype(np.float32)
