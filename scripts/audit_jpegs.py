"""Find JPEG files that OpenCV/libjpeg reports as damaged before training.

Run: python tools/audit_jpegs.py PATH [--report bad-jpegs.txt]
Exit status 1 means at least one file needs replacement or exclusion.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys


JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif"}


def check_one(path):
    # libjpeg writes diagnostics to native stderr. Isolating each decode in a
    # process gives those diagnostics an unambiguous file path and avoids
    # changing stderr for other training workers.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--check-one", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    diagnostic = result.stderr.strip()
    if result.returncode or diagnostic:
        return path, diagnostic or "decoder exited with status {0}".format(result.returncode)
    return None


def audit(root, workers=4):
    root = Path(root)
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(path for path in root.rglob("*")
                       if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS)
    else:
        raise ValueError("Path does not exist: {0}".format(root))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        problems = [item for item in pool.map(check_one, paths) if item is not None]
    return len(paths), problems


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check-one":
        import cv2
        import numpy as np

        path = Path(sys.argv[2])
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if cv2.imdecode(data, cv2.IMREAD_UNCHANGED) is None:
                print("OpenCV could not decode image", file=sys.stderr)
                return 1
        except Exception as exc:
            print("{0}: {1}".format(type(exc).__name__, exc), file=sys.stderr)
            return 1
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JPEG file or dataset directory")
    parser.add_argument("--report", type=Path, help="write damaged file paths to UTF-8 text")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    count, problems = audit(args.path, args.workers)
    for path, diagnostic in problems:
        print("{0}: {1}".format(path, diagnostic))
    if args.report is not None:
        args.report.write_text("".join("{0}\n".format(path) for path, _ in problems), encoding="utf-8")
    print("Checked {0} JPEG files; {1} damaged.".format(count, len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
