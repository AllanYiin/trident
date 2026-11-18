"""Publish a set of checkpoint files only after all serializations succeed."""
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock


@contextmanager
def staged_checkpoints(*destinations):
    """Writers use the staged filenames. A failed replacement restores earlier files."""
    targets = tuple(Path(name) for name in destinations)
    directory = targets[0].parent
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        drafts = [Path(scratch) / target.name for target in targets]
        yield drafts
        for draft in drafts:
            with draft.open('rb+') as stream:
                os.fsync(stream.fileno())
        with FileLock(str(directory / '.trident-checkpoint.lock')):
            previous = {}
            installed = []
            try:
                for target, draft in zip(targets, drafts):
                    if target != targets[-1] and target.exists():
                        backup = Path(scratch) / (target.name + '.old')
                        try:
                            os.link(target, backup)
                        except OSError:
                            shutil.copy2(target, backup)
                        previous[target] = backup
                    os.replace(draft, target)
                    installed.append(target)
            except BaseException:
                for target in reversed(installed):
                    if target in previous:
                        os.replace(previous[target], target)
                    else:
                        target.unlink()
                raise
