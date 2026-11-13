import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_jpegs.py"
spec = importlib.util.spec_from_file_location("audit_jpegs", SCRIPT)
audit_jpegs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_jpegs)


class TestJpegAudit(unittest.TestCase):
    def test_reports_corrupt_file_without_flagging_valid_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))[1].tobytes()
            (root / "valid.jpg").write_bytes(encoded)
            damaged = root / "damaged.jpg"
            damaged.write_bytes(encoded[:len(encoded) // 2])

            count, problems = audit_jpegs.audit(root, workers=2)

            self.assertEqual(count, 2)
            self.assertEqual([path for path, _ in problems], [damaged])
            self.assertIn("could not decode", problems[0][1])


if __name__ == "__main__":
    unittest.main()
