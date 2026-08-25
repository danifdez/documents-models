import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lib.execution.code_identity import _tree_fingerprint, code_fingerprint


class CodeIdentityTest(unittest.TestCase):
    def setUp(self):
        _tree_fingerprint.cache_clear()

    def tearDown(self):
        _tree_fingerprint.cache_clear()

    def test_fingerprint_changes_only_for_versioned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = root / "common"
            common.mkdir()
            (common / "code_identity.json").write_text(
                json.dumps({
                    "schema": 1,
                    "versionScope": ["common/**", "tasks/**"],
                    "exclude": ["**/__pycache__/**", "**/*.pyc"],
                }),
                encoding="utf-8",
            )
            task = root / "tasks" / "demo.py"
            task.parent.mkdir()
            task.write_text("VALUE = 1\n", encoding="utf-8")
            ignored = root / "notes.txt"
            ignored.write_text("first\n", encoding="utf-8")

            first = code_fingerprint(root)
            ignored.write_text("second\n", encoding="utf-8")
            _tree_fingerprint.cache_clear()
            unchanged = code_fingerprint(root)
            task.write_text("VALUE = 2\n", encoding="utf-8")
            _tree_fingerprint.cache_clear()
            changed = code_fingerprint(root)

        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(first, unchanged)
        self.assertNotEqual(first, changed)

    def test_accepts_only_a_canonical_controlled_fingerprint(self):
        expected = "sha256:" + "a" * 64
        with patch.dict(
            "os.environ", {"DOCUMENTS_CODE_FINGERPRINT": expected}
        ):
            self.assertEqual(code_fingerprint(), expected)
        with patch.dict(
            "os.environ", {"DOCUMENTS_CODE_FINGERPRINT": "tree-a"}
        ):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                code_fingerprint()


if __name__ == "__main__":
    unittest.main()
