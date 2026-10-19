"""Check release integrity failures and safe archive extraction."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from download_distillation_v8 import extract_archive, verify_archive


class DownloadTests(unittest.TestCase):
    def test_corrupt_archive_is_rejected_even_with_matching_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "release.zip"
            path.write_bytes(b"changed")
            expected = dict(bytes=7, sha256=hashlib.sha256(b"correct").hexdigest())
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_archive(path, expected)
            with self.assertRaisesRegex(ValueError, "size"):
                verify_archive(path, dict(expected, bytes=8))

    def test_traversal_is_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            archive = out / "release.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("dataset/good.txt", "ok")
                source.writestr("dataset/../outside.txt", "escape")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_archive(archive, out, "dataset")
            self.assertFalse((out / "dataset").exists())
            self.assertFalse((out / "outside.txt").exists())

    def test_existing_dataset_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            archive = out / "release.zip"
            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("dataset/train.jsonl", "original")
            target = extract_archive(archive, out, "dataset")
            (target / "train.jsonl").write_text("local edits")
            with self.assertRaisesRegex(ValueError, "already exists"):
                extract_archive(archive, out, "dataset")
            self.assertEqual((target / "train.jsonl").read_text(), "local edits")


if __name__ == "__main__":
    unittest.main()
