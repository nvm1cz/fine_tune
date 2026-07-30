import tempfile
import unittest
from pathlib import Path

from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint


class KaggleDatasetCheckpointTests(unittest.TestCase):
    def test_restore_and_publish_only_checkpoint_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote"
            output = root / "output"
            remote.mkdir()
            (remote / "trials.jsonl").write_text('{"status":"completed"}\n')
            (remote / "placeholder.txt").write_text("placeholder")
            uploads = []

            def download(handle, **kwargs):
                self.assertEqual(handle, "owner/checkpoint")
                return str(remote)

            def upload(handle, local_dir, **kwargs):
                uploads.append({
                    "handle": handle,
                    "files": sorted(path.name for path in Path(local_dir).iterdir()),
                    "note": kwargs["version_notes"],
                })

            checkpoint = KaggleDatasetCheckpoint(
                "owner/checkpoint",
                download_fn=download,
                upload_fn=upload,
            )
            restored = checkpoint.restore(output)
            self.assertEqual(restored, ["trials.jsonl"])
            self.assertFalse((output / "placeholder.txt").exists())

            (output / "progress.json").write_text("{}")
            checkpoint.publish(output, "screen 1/10")
            self.assertEqual(uploads[0]["handle"], "owner/checkpoint")
            self.assertEqual(
                uploads[0]["files"], ["progress.json", "trials.jsonl"]
            )
            self.assertEqual(uploads[0]["note"], "screen 1/10")

    def test_invalid_handle_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KaggleDatasetCheckpoint("missing-owner")


if __name__ == "__main__":
    unittest.main()
