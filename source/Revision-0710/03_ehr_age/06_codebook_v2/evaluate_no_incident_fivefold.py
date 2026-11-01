from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REVISION_ROOT = WORKSPACE_ROOT / "Revision-0710"
COMMON_DIR = REVISION_ROOT / "05_downstream" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from crossfit_metrics import (  # noqa: E402
    apply_linear_age_correction,
    apply_linear_age_correction_to_validation,
    evaluate_cox_cell,
    fit_linear_age_correction,
)
from run_fold0_current import atomic_json, sha256_file, utc_now  # noqa: E402
from run_frozen_outer_fold import (  # noqa: E402
    BASELINE_ENVELOPE,
    ORGAN_SYSTEMS,
    OUTCOMES_PATH,
    organ_baseline_values,
)
from train_no_incident_outer_fold import OUTPUT_ROOT  # noqa: E402


SUMMARY_ROOT = OUTPUT_ROOT.parent / "02_summary"
MAIN_SUMMARY = (
    Path(__file__).resolve().parent
    / "02_fivefold_frozen_huber1"
    / "02_summary"
    / "fivefold_summary.json"
)


def age_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target = frame["actual_age"].to_numpy(dtype=np.float64)
    score = frame["predicted_age"].to_numpy(dtype=np.float64)
    error = score - target
    denominator = np.square(target - target.mean()).sum()
    return {
        "n_rows": int(len(frame)),
        "n_participants": int(frame["participant_id"].nunique()),
        "mae": float(np.abs(error).mean()),
        "pearson": float(np.corrcoef(target, score)[0, 1]),
        "r2": float(1.0 - np.square(error).sum() / denominator),
    }


def attach_outcomes(prediction: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    frame = prediction.merge(
        outcomes,
        on=["participant_id", "organ_system"],
        how="left",
        validate="one_to_one",
    )
    if frame[["event", "at_risk", "strict_healthy_reference_60m"]].isna().any().any():
        raise RuntimeError("no-incident evaluation did not receive complete v2 outcomes")
    frame["duration"] = frame["duration_months"].fillna(0.0)
    return frame


def build_evaluation() -> Path:
    outcomes = pd.read_parquet(
        OUTCOMES_PATH,
        columns=[
            "participant_id",
            "organ_system",
            "duration_months",
            "event",
            "at_risk",
            "strict_healthy_reference_60m",
        ],
    )
    outcomes["participant_id"] = outcomes["participant_id"].astype(str)
    outcomes["organ_system"] = outcomes["organ_system"].astype(str)
    envelope = json.loads(BASELINE_ENVELOPE.read_text(encoding="utf-8"))
    organ_baseline = organ_baseline_values(envelope)
    cells: list[dict[str, Any]] = []
    outer_frames: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = []
    for fold in range(5):
        root = OUTPUT_ROOT / f"outer_fold_{fold}"
        complete_path = root / "TRAINING_COMPLETE.json"
        calibration_path = root / "calibration_predictions.parquet"
        outer_path = root / "outer_test_predictions.parquet"
        if not all(path.is_file() for path in (complete_path, calibration_path, outer_path)):
            raise FileNotFoundError(f"no-incident outer fold {fold} is incomplete")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "COMPLETE" or complete.get(
            "incident_outcome_file_read"
        ) is not False:
            raise RuntimeError(f"no-incident training receipt failed for fold {fold}")
        calibration = attach_outcomes(pd.read_parquet(calibration_path), outcomes)
        outer = attach_outcomes(pd.read_parquet(outer_path), outcomes)
        correction = fit_linear_age_correction(calibration)
        calibration_corrected = apply_linear_age_correction_to_validation(
            calibration, correction
        )
        outer_corrected = apply_linear_age_correction(outer, correction)
        outer_corrected["outer_fold"] = int(fold)
        outer_frames.append(outer_corrected)
        for organ in ORGAN_SYSTEMS:
            validation_organ = calibration_corrected.loc[
                calibration_corrected["organ_system"].astype(str).eq(organ)
                & calibration_corrected["at_risk"].eq(1)
            ].copy()
            outer_organ = outer_corrected.loc[
                outer_corrected["organ_system"].astype(str).eq(organ)
                & outer_corrected["at_risk"].eq(1)
            ].copy()
            cell = asdict(
                evaluate_cox_cell(
                    validation_organ,
                    outer_organ,
                    fold=fold,
                    organ_system=organ,
                )
            )
            cell["beats_formal_baseline"] = bool(
                float(cell["delta_c"]) > organ_baseline[organ]
            )
            cells.append(cell)
        input_rows.append(
            {
                "outer_fold": fold,
                "training_complete_path": str(complete_path.resolve()),
                "training_complete_sha256": sha256_file(complete_path),
                "calibration_predictions_sha256": sha256_file(calibration_path),
                "outer_predictions_sha256": sha256_file(outer_path),
            }
        )

    oof = pd.concat(outer_frames, ignore_index=True).sort_values(
        ["outer_fold", "participant_id", "organ_system"], kind="mergesort"
    )
    if oof.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("no-incident OOF predictions contain duplicate rows")
    if oof["participant_id"].nunique() != 456_773 or len(oof) != 3_197_411:
        raise RuntimeError("no-incident OOF prediction coverage is incomplete")
    strict = oof.loc[oof["strict_healthy_reference_60m"].eq(1)].copy()
    strict_metrics = age_metrics(strict)
    cell_frame = pd.DataFrame(cells).sort_values(["fold", "organ_system"])
    organ_frame = (
        cell_frame.groupby("organ_system", observed=True)
        .agg(
            mean_c_base=("c_base", "mean"),
            mean_c_plus=("c_plus", "mean"),
            mean_delta_c=("delta_c", "mean"),
            positive_folds=("delta_c", lambda values: int((values > 0).sum())),
        )
        .reindex(ORGAN_SYSTEMS)
        .reset_index()
    )
    risk = {
        "n_cells": int(len(cell_frame)),
        "macro_c_base": float(organ_frame["mean_c_base"].mean()),
        "macro_c_plus": float(organ_frame["mean_c_plus"].mean()),
        "macro_delta_c": float(organ_frame["mean_delta_c"].mean()),
        "positive_organs": int((organ_frame["mean_delta_c"] > 0).sum()),
        "organs": organ_frame.to_dict(orient="records"),
    }
    main = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    main_age = main["age_metrics"]["strict_healthy_reference_60m"]
    main_risk = main["full_risk"]
    comparison = {
        "main_minus_no_incident": {
            "strict_mae": float(main_age["mae"] - strict_metrics["mae"]),
            "strict_pearson": float(
                main_age["pearson"] - strict_metrics["pearson"]
            ),
            "strict_r2": float(main_age["r2"] - strict_metrics["r2"]),
            "full_risk_macro_delta_c": float(
                main_risk["macro_delta_c"] - risk["macro_delta_c"]
            ),
        },
        "interpretation": (
            "Positive Pearson/R2/delta-C and negative MAE differences favor the main "
            "incident-supervised EHR-AGE model."
        ),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_ROOT / ".oof_predictions.parquet.tmp"
    oof.to_parquet(temporary, index=False)
    temporary.replace(SUMMARY_ROOT / "oof_predictions.parquet")
    cell_frame.to_csv(SUMMARY_ROOT / "cox_fold_organ_cells.csv", index=False)
    organ_frame.to_csv(SUMMARY_ROOT / "cox_organ_summary.csv", index=False)
    summary = {
        "schema_version": "revision0710.ehr_age.no_incident.evaluation.v1",
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "training_process_read_incident_outcomes": False,
        "evaluation_process_read_incident_outcomes": True,
        "strict_age": strict_metrics,
        "full_risk": risk,
        "comparison_to_main": comparison,
        "training_inputs": input_rows,
        "evaluation_outcomes": {
            "path": str(OUTCOMES_PATH.resolve()),
            "sha256": sha256_file(OUTCOMES_PATH),
        },
    }
    atomic_json(SUMMARY_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return SUMMARY_ROOT


if __name__ == "__main__":
    print(build_evaluation())
