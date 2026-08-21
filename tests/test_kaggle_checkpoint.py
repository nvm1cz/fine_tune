import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint


class KaggleDatasetCheckpointTests(unittest.TestCase):
    def test_restore_and_publish_only_checkpoint_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote"
            output = root / "output"
            remote.mkdir()
            (remote / "trials.jsonl").write_text('{"status":"completed"}\n')
            (remote / "scenario_best_configs.json").write_text('{"configs":{}}\n')
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
            self.assertEqual(restored, ["scenario_best_configs.json", "trials.jsonl"])
            self.assertFalse((output / "placeholder.txt").exists())

            (output / "progress.json").write_text("{}")
            (output / "scenario_registry_eulc_pso.csv").write_text("scenario_key\n")
            checkpoint.publish(output, "screen 1/10")
            self.assertEqual(uploads[0]["handle"], "owner/checkpoint")
            self.assertEqual(
                uploads[0]["files"],
                ["progress.json", "scenario_best_configs.json",
                 "scenario_registry_eulc_pso.csv", "trials.jsonl"],
            )
            self.assertEqual(uploads[0]["note"], "screen 1/10")

    def test_invalid_handle_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KaggleDatasetCheckpoint("missing-owner")

    def test_restore_can_pin_a_historical_dataset_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote"
            output = root / "output"
            remote.mkdir()
            (remote / "pso_matrix_manifest.json").write_text("{}")
            handles = []

            def download(handle, **kwargs):
                handles.append(handle)
                return str(remote)

            checkpoint = KaggleDatasetCheckpoint(
                "owner/checkpoint",
                download_fn=download,
                upload_fn=lambda *args, **kwargs: None,
            )
            checkpoint.restore(output, version=278)
            self.assertEqual(handles, ["owner/checkpoint/versions/278"])
            self.assertTrue((output / "pso_matrix_manifest.json").exists())

            with self.assertRaisesRegex(ValueError, "positive integer"):
                checkpoint.restore(output, version=0)

    def test_restore_rebuilds_zip_when_kaggle_expands_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote"
            expanded = remote / "pso_matrix_results"
            scenario = expanded / "dense_n150_p6400_e1p0"
            scenario.mkdir(parents=True)
            (scenario / "result.json").write_text('{"schema_version":3}')
            (expanded / "pso_matrix_manifest.json").write_text(
                '{"completed_count":3}'
            )
            output = root / "output"

            checkpoint = KaggleDatasetCheckpoint(
                "owner/checkpoint",
                download_fn=lambda *args, **kwargs: str(remote),
                upload_fn=lambda *args, **kwargs: None,
            )
            restored = checkpoint.restore(output, version=278)
            self.assertIn("pso_matrix_results.zip", restored)
            with zipfile.ZipFile(output / "pso_matrix_results.zip") as archive:
                self.assertIn(
                    "dense_n150_p6400_e1p0/result.json", archive.namelist()
                )

    def test_production_backend_disables_notebook_attach_resolver(self) -> None:
        fake = types.SimpleNamespace(
            dataset_download=lambda *args, **kwargs: "downloaded",
            dataset_upload=lambda *args, **kwargs: None,
        )
        checkpoint = KaggleDatasetCheckpoint("owner/checkpoint")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISABLE_KAGGLE_CACHE", None)
            with patch.dict(sys.modules, {"kagglehub": fake}):
                download, upload = checkpoint._backend()
            self.assertEqual(os.environ["DISABLE_KAGGLE_CACHE"], "true")
        self.assertEqual(download(), "downloaded")
        self.assertIsNone(upload())


if __name__ == "__main__":
    unittest.main()
