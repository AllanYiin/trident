from __future__ import absolute_import, division, print_function

import warnings

from .dataset import Dataset
from .provider import DataProvider


PIPELINE_API_VERSION = 1


def migrate_legacy_provider(provider, schema_by_split=None,
                            validate_samples=True):
    """Build a new provider around legacy train/test iterator-like sources.

    This is an explicit bridge, not a global monkey patch. The old provider is
    left untouched and values continue to be loaded lazily through LegacySource.
    """
    warnings.warn(
        "migrate_legacy_provider is a transition bridge; prefer constructing "
        "pipeline Dataset splits directly for new code.",
        DeprecationWarning, stacklevel=2)
    schema_by_split = dict(schema_by_split or {})
    candidates = (
        ("train", getattr(provider, "traindata", None)),
        ("valid", getattr(provider, "validdata", None)),
        ("test", getattr(provider, "testdata", None)),
    )
    splits = {}
    for name, source in candidates:
        if source is not None:
            splits[name] = Dataset.from_legacy(
                data=getattr(source, "data", source),
                label=getattr(source, "label", None),
                unpair=getattr(source, "unpair", None),
                schema=schema_by_split.get(name), name=name,
                validate_samples=validate_samples)
    if not splits:
        raise ValueError("legacy provider exposes no train/valid/test data")
    return DataProvider(dataset_name=getattr(provider, "dataset_name", ""),
                        metadata={"migrated_from": provider.__class__.__name__},
                        **splits)
