from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_fold0_current import atomic_json, sha256_file, utc_now
from run_frozen_outer_fold import ORGAN_SYSTEMS


HERE = Path(__file__).resolve().parent
MAIN_OOF = (
    HERE
    / "02_fivefold_frozen_huber1"
    / "02_summary"
    / "oof_predictions.parquet"
)
BASELINE_REGISTRY = (
    HERE.parents[1]
    / "05_downstream"
    / "01_prediction_registries"
    / "tier_a"
    / "retain.json"
)
OUTPUT_ROOT = HERE / "02_fivefold_frozen_huber1" / "03_paired_bootstrap"
N_RESAMPLES = 1000
SEED = 7172026


def _load_baseline_test() -> pd.DataFrame:
    registry = json.loads(BASELINE_REGISTRY.read_text(encoding="utf-8"))
    frames = []
    for entry in registry["prediction_files"]:
        raw_path = Path(entry["path"] if isinstance(entry, dict) else entry)
        path = (
            raw_path
            if raw_path.is_absolute()
            else (BASELINE_REGISTRY.parent / raw_path).resolve()
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        if isinstance(entry, dict) and entry.get("sha256"):
            if sha256_file(path) != str(entry["sha256"]):
                raise RuntimeError(f"RETAIN prediction hash changed: {path}")
        frame = pd.read_parquet(
            path,
            columns=[
                "participant_id",
                "organ_system",
                "outer_fold",
                "split_role",
                "predicted_age",
            ],
        )
        frame = frame.loc[frame["split_role"].astype(str).eq("test")].copy()
        if not frame.empty:
            frames.append(frame)
    baseline = pd.concat(frames, ignore_index=True)
    baseline["participant_id"] = baseline["participant_id"].astype(str)
    baseline["organ_system"] = baseline["organ_system"].astype(str)
    if baseline.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("RETAIN OOF test predictions contain duplicate rows")
    return baseline.drop(columns="split_role")


def _participant_contributions(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "abs_error": np.abs(prediction - target).sum(axis=1),
        "sse": np.square(prediction - target).sum(axis=1),
        "prediction": prediction.sum(axis=1),
        "prediction2": np.square(prediction).sum(axis=1),
        "target_prediction": (target * prediction).sum(axis=1),
    }


def _weighted_metrics(
    counts: np.ndarray,
    target_sums: dict[str, np.ndarray],
    prediction_sums: dict[str, np.ndarray],
    row_n: int,
) -> dict[str, float]:
    sum_y = float(np.dot(counts, target_sums["target"]))
    sum_y2 = float(np.dot(counts, target_sums["target2"]))
    sum_p = float(np.dot(counts, prediction_sums["prediction"]))
    sum_p2 = float(np.dot(counts, prediction_sums["prediction2"]))
    sum_yp = float(np.dot(counts, prediction_sums["target_prediction"]))
    centered_y = sum_y2 - sum_y * sum_y / row_n
    centered_p = sum_p2 - sum_p * sum_p / row_n
    covariance = sum_yp - sum_y * sum_p / row_n
    return {
        "mae": float(np.dot(counts, prediction_sums["abs_error"]) / row_n),
        "pearson": float(covariance / np.sqrt(centered_y * centered_p)),
        "r2": float(
            1.0 - np.dot(counts, prediction_sums["sse"]) / centered_y
        ),
    }


def run_bootstrap() -> Path:
    main = pd.read_parquet(
        MAIN_OOF,
        columns=[
            "participant_id",
            "organ_system",
            "outer_fold",
            "actual_age",
            "predicted_age",
            "strict_healthy_reference_60m",
        ],
    )
    main["participant_id"] = main["participant_id"].astype(str)
    main["organ_system"] = main["organ_system"].astype(str)
    main = main.loc[main["strict_healthy_reference_60m"].eq(1)].copy()
    baseline = _load_baseline_test()
    paired = main.merge(
        baseline,
        on=["participant_id", "organ_system", "outer_fold"],
        how="inner",
        suffixes=("_ehr_age", "_retain"),
        validate="one_to_one",
    )
    if len(paired) != len(main):
        raise RuntimeError("EHR-AGE and RETAIN strict OOF rows are not exactly paired")
    order = {organ: index for index, organ in enumerate(ORGAN_SYSTEMS)}
    paired["_organ_order"] = paired["organ_system"].map(order)
    paired = paired.sort_values(
        ["outer_fold", "participant_id", "_organ_order"], kind="mergesort"
    )
    participant = paired.drop_duplicates("participant_id")[
        ["participant_id", "outer_fold"]
    ].reset_index(drop=True)
    n = len(participant)
    if len(paired) != n * len(ORGAN_SYSTEMS):
        raise RuntimeError("paired age bootstrap lost the seven-organ cluster")
    target = paired["actual_age"].to_numpy(dtype=np.float64).reshape(n, -1)
    ehr_age = paired["predicted_age_ehr_age"].to_numpy(dtype=np.float64).reshape(
        n, -1
    )
    retain = paired["predicted_age_retain"].to_numpy(dtype=np.float64).reshape(n, -1)
    target_sums = {
        "target": target.sum(axis=1),
        "target2": np.square(target).sum(axis=1),
    }
    contributions = {
        "EHR-AGE": _participant_contributions(target, ehr_age),
        "RETAIN": _participant_contributions(target, retain),
    }
    fold_positions = {
        int(fold): np.flatnonzero(participant["outer_fold"].to_numpy() == fold)
        for fold in range(5)
    }
    row_n = n * len(ORGAN_SYSTEMS)
    point_counts = np.ones(n, dtype=np.float64)
    point = {
        model: _weighted_metrics(point_counts, target_sums, values, row_n)
        for model, values in contributions.items()
    }
    rng = np.random.default_rng(SEED)
    replicate_rows: list[dict[str, Any]] = []
    for replicate in range(N_RESAMPLES):
        counts = np.zeros(n, dtype=np.float64)
        for positions in fold_positions.values():
            draws = rng.choice(positions, size=len(positions), replace=True)
            counts += np.bincount(draws, minlength=n)
        values = {
            model: _weighted_metrics(counts, target_sums, sums, row_n)
            for model, sums in contributions.items()
        }
        for metric, direction in (
            ("mae", "lower"),
            ("pearson", "higher"),
            ("r2", "higher"),
        ):
            raw = values["EHR-AGE"][metric] - values["RETAIN"][metric]
            improvement = -raw if direction == "lower" else raw
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "direction": direction,
                    "retain": values["RETAIN"][metric],
                    "ehr_age": values["EHR-AGE"][metric],
                    "ehr_age_minus_retain": raw,
                    "directional_improvement": improvement,
                }
            )
    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for metric, direction in (
        ("mae", "lower"),
        ("pearson", "higher"),
        ("r2", "higher"),
    ):
        values = replicates.loc[
            replicates["metric"].eq(metric), "directional_improvement"
        ].to_numpy(dtype=np.float64)
        raw = point["EHR-AGE"][metric] - point["RETAIN"][metric]
        improvement = -raw if direction == "lower" else raw
        summary_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "retain": point["RETAIN"][metric],
                "ehr_age": point["EHR-AGE"][metric],
                "ehr_age_minus_retain": raw,
                "directional_improvement": improvement,
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_probability_improvement_gt_0": float(
                    np.mean(values > 0.0)
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    replicates.to_csv(OUTPUT_ROOT / "age_replicates.csv.gz", index=False)
    summary.to_csv(OUTPUT_ROOT / "age_paired_bootstrap_ci.csv", index=False)
    atomic_json(
        OUTPUT_ROOT / "age_bootstrap_manifest.json",
        {
            "schema_version": "revision0710.ehr_age.codebook_v2.age_bootstrap.v1",
            "status": "COMPLETE",
            "completed_utc": utc_now(),
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "bootstrap_unit": "participant within outer fold; seven organ rows retained",
            "strict_participants": n,
            "strict_rows": int(len(paired)),
            "main_oof": {
                "path": str(MAIN_OOF.resolve()),
                "sha256": sha256_file(MAIN_OOF),
            },
            "baseline_registry": {
                "path": str(BASELINE_REGISTRY.resolve()),
                "sha256": sha256_file(BASELINE_REGISTRY),
            },
        },
    )
    print(summary.to_string(index=False))
    return OUTPUT_ROOT


if __name__ == "__main__":
    print(run_bootstrap())
