#!/usr/bin/env python
"""Build a five-fold prediction registry from five explicit completed run paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "revision0710_prediction_registry_v1"
PREDICTION_COLUMNS = (
    "participant_id",
    "organ_system",
    "actual_age",
    "predicted_age",
    "outer_fold",
    "split_role",
)
ROLE_FILES = (
    ("validation", "validation_predictions_long.parquet"),
    ("test", "test_predictions_long.parquet"),
)


class RegistryBuildError(ValueError):
    """Raised when an explicit run path is incomplete or inconsistent."""


def _reject_glob(path: Path, *, label: str) -> None:
    if any(token in str(path) for token in ("*", "?", "[", "]")):
        raise RegistryBuildError(f"{label} must be explicit and contain no glob tokens")


def _explicit_directory(path: Path, *, label: str) -> Path:
    _reject_glob(path, label=label)
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise RegistryBuildError(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise RegistryBuildError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryBuildError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryBuildError(f"{label} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry_relative_path(path: Path, registry_path: Path) -> str:
    return Path(os.path.relpath(path, start=registry_path.parent)).as_posix()


def _validate_prediction_file(path: Path, *, fold: int, role: str) -> int:
    if not path.is_file():
        raise RegistryBuildError(f"missing {role} long predictions for fold {fold}: {path}")
    try:
        frame = pd.read_parquet(path, columns=list(PREDICTION_COLUMNS))
    except Exception as exc:
        raise RegistryBuildError(f"cannot read required columns from {path}: {exc}") from exc
    if frame.empty:
        raise RegistryBuildError(f"prediction file is empty: {path}")
    if frame[list(PREDICTION_COLUMNS)].isna().to_numpy().any():
        raise RegistryBuildError(f"prediction file contains null required values: {path}")
    observed_folds = pd.to_numeric(frame["outer_fold"], errors="coerce")
    if observed_folds.isna().any() or not observed_folds.eq(fold).all():
        raise RegistryBuildError(f"prediction file outer_fold does not equal {fold}: {path}")
    if not frame["split_role"].astype(str).eq(role).all():
        raise RegistryBuildError(f"prediction file split_role does not equal {role!r}: {path}")
    for column in ("actual_age", "predicted_age"):
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise RegistryBuildError(f"prediction file {column} is not finite numeric: {path}")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RegistryBuildError(f"prediction file has duplicate participant-organ rows: {path}")
    return int(len(frame))


def build_registry(*, method: str, run_dirs: Sequence[Path], output: Path) -> dict[str, object]:
    if not method.strip():
        raise RegistryBuildError("method must be non-empty")
    if len(run_dirs) != 5:
        raise RegistryBuildError("provide exactly five --run-dir values, one for each outer fold")
    _reject_glob(output, label="output")
    output = output.expanduser().resolve()

    runs_by_fold: dict[int, tuple[Path, dict[str, object], dict[str, object]]] = {}
    for index, raw_run_dir in enumerate(run_dirs):
        run_dir = _explicit_directory(raw_run_dir, label=f"run_dir[{index}]")
        config = _read_json(run_dir / "run_config.json", label=f"run_dir[{index}].run_config")
        artifacts = _read_json(
            run_dir / "artifact_hashes.json",
            label=f"run_dir[{index}].artifact_hashes",
        )
        if config.get("model_name") != method:
            raise RegistryBuildError(
                f"run_dir[{index}] model_name mismatch: expected={method!r}, "
                f"observed={config.get('model_name')!r}"
            )
        fold = config.get("outer_fold")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5):
            raise RegistryBuildError(f"run_dir[{index}] outer_fold must be an integer in 0..4")
        if fold in runs_by_fold:
            raise RegistryBuildError(f"duplicate run directories for outer_fold={fold}")
        runs_by_fold[fold] = (run_dir, config, artifacts)
    if set(runs_by_fold) != set(range(5)):
        raise RegistryBuildError(f"run directories do not cover outer folds 0..4: {sorted(runs_by_fold)}")

    run_entries: list[dict[str, object]] = []
    prediction_entries: list[dict[str, object]] = []
    for fold in range(5):
        run_dir, _, artifact_hashes = runs_by_fold[fold]
        run_entries.append(
            {
                "outer_fold": fold,
                "path": _registry_relative_path(run_dir, output),
                "run_config_sha256": _sha256_file(run_dir / "run_config.json"),
            }
        )
        for role, filename in ROLE_FILES:
            prediction_path = run_dir / filename
            rows = _validate_prediction_file(prediction_path, fold=fold, role=role)
            observed_hash = _sha256_file(prediction_path)
            artifact_key = f"{role}_predictions_long_parquet"
            expected_hash = artifact_hashes.get(artifact_key)
            if expected_hash != observed_hash:
                raise RegistryBuildError(
                    f"{artifact_key} hash disagrees with artifact_hashes.json for fold {fold}: "
                    f"expected={expected_hash!r}, observed={observed_hash}"
                )
            prediction_entries.append(
                {
                    "outer_fold": fold,
                    "split_role": role,
                    "path": _registry_relative_path(prediction_path, output),
                    "sha256": observed_hash,
                    "rows": rows,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "selection": "five explicit completed run directories; no discovery or globbing",
        "run_directories": run_entries,
        "prediction_files": prediction_entries,
    }


def write_registry(payload: dict[str, object], *, output: Path, overwrite: bool) -> None:
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise RegistryBuildError(f"output already exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_registry(method=args.method, run_dirs=args.run_dir, output=args.output)
    write_registry(payload, output=args.output, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
