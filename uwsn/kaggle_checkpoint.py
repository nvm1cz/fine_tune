from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable


CHECKPOINT_FILENAMES = {
    "trials.jsonl",
    "progress.json",
    "progress.log",
    "manifest.json",
    "best_summary.json",
    "benchmark_results.jsonl",
    "completed_cases.jsonl",
    "case_summary.csv",
    "round_metrics.csv",
    "benchmark_manifest.json",
    "recovery_results.jsonl",
    "recovery_summary.csv",
    "recovery_manifest.json",
    "ofat_manifest.json",
    "ofat_trials.jsonl",
    "ofat_history.jsonl",
    "scenario_ofat_manifest.json",
    "scenario_ofat_trials.jsonl",
    "scenario_ofat_history.jsonl",
    "scenario_best_configs.json",
    "pso_matrix_manifest.json",
    "pso_matrix_results.zip",
}


class KaggleDatasetCheckpoint:
    """Synchronize resumable tuning artifacts with one Kaggle Dataset."""

    def __init__(
        self,
        handle: str,
        *,
        retries: int = 3,
        download_fn: Callable[..., str] | None = None,
        upload_fn: Callable[..., None] | None = None,
    ) -> None:
        if handle.count("/") != 1 or any(not part for part in handle.split("/")):
            raise ValueError("checkpoint dataset must have the form owner/dataset-slug")
        self.handle = handle
        self.retries = max(1, int(retries))
        self._download_fn = download_fn
        self._upload_fn = upload_fn

    def _backend(self) -> tuple[Callable[..., str], Callable[..., None]]:
        if self._download_fn is not None and self._upload_fn is not None:
            return self._download_fn, self._upload_fn
        # Kaggle's notebook-native resolver tries to attach a Dataset. Batch and
        # scheduled sessions cannot attach a new Dataset, so force KaggleHub's
        # authenticated HTTP resolver, which also retrieves the latest version.
        os.environ["DISABLE_KAGGLE_CACHE"] = "true"
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "kagglehub is required when --checkpoint-dataset is used"
            ) from exc
        return kagglehub.dataset_download, kagglehub.dataset_upload

    @staticmethod
    def _is_artifact(path: Path) -> bool:
        return (
            path.name in CHECKPOINT_FILENAMES
            or path.name.startswith("rankings_") and path.suffix == ".csv"
            or path.name.startswith("best_config_") and path.suffix == ".yaml"
            or path.name.startswith("scenario_registry_") and path.suffix in {".csv", ".json"}
        )

    def restore(self, output_dir: Path, *, version: int | None = None) -> list[str]:
        download, _ = self._backend()
        source_handle = (
            self.handle if version is None
            else f"{self.handle}/versions/{int(version)}"
        )
        if version is not None and int(version) <= 0:
            raise ValueError("checkpoint version must be a positive integer")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(
                download(
                    source_handle,
                    output_dir=temporary,
                    force_download=True,
                )
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            restored = []
            for path in source.rglob("*"):
                if path.is_file() and self._is_artifact(path):
                    shutil.copy2(path, output_dir / path.name)
                    restored.append(path.name)
        print(
            f"[checkpoint] restored {len(restored)} file(s) from {source_handle}",
            flush=True,
        )
        return sorted(restored)

    def publish(self, output_dir: Path, note: str) -> None:
        _, upload = self._backend()
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            artifacts = [
                path for path in output_dir.iterdir()
                if path.is_file() and self._is_artifact(path)
            ]
            if not artifacts:
                raise RuntimeError("no checkpoint artifacts are available to publish")
            for path in artifacts:
                shutil.copy2(path, snapshot / path.name)
            last_error: Exception | None = None
            for attempt in range(1, self.retries + 1):
                try:
                    upload(self.handle, str(snapshot), version_notes=note)
                    print(
                        f"[checkpoint] uploaded {len(artifacts)} file(s) "
                        f"to {self.handle}",
                        flush=True,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt < self.retries:
                        time.sleep(2 ** (attempt - 1))
            raise RuntimeError(
                f"failed to publish checkpoint to {self.handle} "
                f"after {self.retries} attempts"
            ) from last_error
