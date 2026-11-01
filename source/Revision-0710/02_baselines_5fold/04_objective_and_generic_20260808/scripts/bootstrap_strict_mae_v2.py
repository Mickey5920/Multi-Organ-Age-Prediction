from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REVISION_ROOT = EXPERIMENT_ROOT.parents[1]
OUTPUT_ROOT = EXPERIMENT_ROOT / "04_aligned_v2_evaluation_20260809" / "paired_bootstrap"
REGISTRY_ROOT = EXPERIMENT_ROOT / "03_evaluation" / "registries"
EHR_OOF = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "02_fivefold_frozen_huber1"
    / "02_summary"
    / "oof_predictions.parquet"
)
METHODS = (
    "linear_svr",
    "mlp",
    "cnn",
    "resnet",
    "transformer",
    "retain_prospective",
    "behrt_style_prospective",
    "medbert_style_prospective",
    "ehrformer_style_prospective",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_prediction_paths(method: str) -> list[Path]:
    registry_path = REGISTRY_ROOT / f"{method}.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    paths = []
    for entry in payload["prediction_files"]:
        if entry["split_role"] != "test":
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = registry_path.parent / path
        path = path.resolve()
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"prediction hash mismatch: {method} {path}")
        paths.append(path)
    if len(paths) != 5:
        raise RuntimeError(f"expected five test prediction files for {method}")
    return paths


def load_ehr_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "participant_id",
        "organ_system",
        "actual_age",
        "predicted_age",
        "strict_healthy_reference_60m",
        "outer_fold",
    ]
    frame = pd.read_parquet(EHR_OOF, columns=columns)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame = frame.loc[frame["strict_healthy_reference_60m"].eq(1)].copy()
    frame = frame.sort_values(["participant_id", "organ_system"]).reset_index(drop=True)
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("EHR-AGE strict predictions contain duplicate participant-organ rows")
    counts = frame.groupby("participant_id")["organ_system"].nunique()
    if not counts.eq(7).all():
        raise RuntimeError("EHR-AGE strict participants do not have seven organ rows")
    participant = (
        frame.assign(
            ehr_absolute_error=(frame["predicted_age"] - frame["actual_age"]).abs()
        )
        .groupby("participant_id", as_index=False)
        .agg(
            outer_fold=("outer_fold", "first"),
            ehr_mae=("ehr_absolute_error", "mean"),
            rows=("organ_system", "size"),
        )
    )
    if not participant["rows"].eq(7).all():
        raise RuntimeError("participant-level EHR-AGE strict aggregation is incomplete")
    return frame, participant.drop(columns="rows")


def method_participant_mae(method: str, reference: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for path in test_prediction_paths(method):
        parts.append(
            pd.read_parquet(
                path,
                columns=[
                    "participant_id",
                    "organ_system",
                    "actual_age",
                    "predicted_age",
                    "outer_fold",
                ],
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame["participant_id"] = frame["participant_id"].astype(str)
    keys = reference[["participant_id", "organ_system"]]
    frame = keys.merge(
        frame,
        on=["participant_id", "organ_system"],
        how="left",
        validate="one_to_one",
    )
    if frame[["actual_age", "predicted_age", "outer_fold"]].isna().any().any():
        raise RuntimeError(f"strict prediction coverage failed for {method}")
    if not np.allclose(frame["actual_age"], reference["actual_age"], atol=1e-5, rtol=0.0):
        raise RuntimeError(f"actual age mismatch for {method}")
    frame["absolute_error"] = (frame["predicted_age"] - frame["actual_age"]).abs()
    participant = frame.groupby("participant_id", as_index=False).agg(
        outer_fold=("outer_fold", "first"),
        method_mae=("absolute_error", "mean"),
        rows=("organ_system", "size"),
    )
    if not participant["rows"].eq(7).all():
        raise RuntimeError(f"participant-level strict aggregation failed for {method}")
    return participant.drop(columns="rows").rename(columns={"method_mae": method})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10082026)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_rows, participant = load_ehr_reference()
    for method in METHODS:
        values = method_participant_mae(method, reference_rows)
        participant = participant.merge(
            values,
            on=["participant_id", "outer_fold"],
            how="left",
            validate="one_to_one",
        )
    if participant.isna().any().any():
        raise RuntimeError("paired participant MAE matrix contains missing values")
    for method in METHODS:
        participant[f"advantage_vs_{method}"] = participant[method] - participant["ehr_mae"]
    advantage_columns = [f"advantage_vs_{method}" for method in METHODS]
    point = participant[advantage_columns].mean()
    fold_matrices = {
        fold: participant.loc[participant["outer_fold"].eq(fold), advantage_columns].to_numpy(
            dtype=np.float64
        )
        for fold in range(5)
    }
    rng = np.random.default_rng(args.seed)
    replicates = np.empty((args.n_resamples, len(advantage_columns)), dtype=np.float64)
    total_n = len(participant)
    for replicate in range(args.n_resamples):
        total = np.zeros(len(advantage_columns), dtype=np.float64)
        for fold in range(5):
            matrix = fold_matrices[fold]
            draw = rng.integers(0, len(matrix), size=len(matrix))
            total += matrix[draw].sum(axis=0)
        replicates[replicate] = total / total_n
    replicate_frame = pd.DataFrame(replicates, columns=advantage_columns)
    replicate_frame.insert(0, "replicate", np.arange(args.n_resamples))
    replicate_frame.to_csv(OUTPUT_ROOT / "strict_mae_replicates.csv", index=False)
    rows = []
    for column in advantage_columns:
        values = replicate_frame[column].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": column,
                "estimate_years": float(point[column]),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_probability_gt_0": float(np.mean(values > 0.0)),
                "valid_resamples": args.n_resamples,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "strict_mae_paired_bootstrap_ci.csv", index=False)
    manifest = {
        "schema_version": "revision0710.baseline_extension.strict_mae_bootstrap.v1",
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "bootstrap_unit": "strict-Health participant within outer fold",
        "n_participants": int(total_n),
        "n_rows": int(len(reference_rows)),
        "methods": list(METHODS),
        "ehr_oof": {"path": str(EHR_OOF.resolve()), "sha256": sha256_file(EHR_OOF)},
        "registries": {
            method: {
                "path": str((REGISTRY_ROOT / f"{method}.json").resolve()),
                "sha256": sha256_file(REGISTRY_ROOT / f"{method}.json"),
            }
            for method in METHODS
        },
    }
    (OUTPUT_ROOT / "strict_mae_bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
