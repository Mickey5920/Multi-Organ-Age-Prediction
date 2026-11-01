from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from run_fold0_current import atomic_json, sha256_file, utc_now
from run_frozen_outer_fold import (
    BASELINE_TARGETS,
    FIVEFOLD_ROOT,
    FOLD0_METRICS,
    MATCHED_FOLD0_METRICS,
    MATCHED_ROOT,
    MATCHED_RUN_ROOT,
    MATCHED_SELECTION_PATH,
    ORGAN_SYSTEMS,
    RUN_ROOT,
    SELECTION_PATH,
    official_baseline_values,
)


HERE = Path(__file__).resolve().parent
FOLD_MANIFEST = (
    HERE.parents[1] / "01_data_rebuild" / "03_folds" / "fold_manifest_5fold.parquet"
)


def _age_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
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


def _source_paths(
    *, fold0_metrics: Path, run_root: Path
) -> list[tuple[int, Path, Path]]:
    rows = [
        (0, fold0_metrics.parent / "fold0_predictions.parquet", fold0_metrics)
    ]
    for fold in range(1, 5):
        root = run_root / f"outer_fold_{fold}"
        rows.append((fold, root / "outer_predictions.parquet", root / "metrics.json"))
    return rows


def _read_and_validate_predictions(
    sources: list[tuple[int, Path, Path]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metric_payloads: list[dict[str, Any]] = []
    fold_manifest = pd.read_parquet(
        FOLD_MANIFEST, columns=["participant_id", "fold_id"]
    )
    fold_manifest["participant_id"] = fold_manifest["participant_id"].astype(str)
    expected_by_fold = {
        int(fold): set(group["participant_id"])
        for fold, group in fold_manifest.groupby("fold_id", observed=True)
    }
    required = {
        "participant_id",
        "organ_system",
        "actual_age",
        "predicted_age",
        "corrected_predicted_age",
        "strict_healthy_reference_60m",
        "sex",
        "duration",
        "event",
        "at_risk",
        "model_risk_score",
    }
    for fold, prediction_path, metrics_path in sources:
        if not prediction_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"outer fold {fold} is incomplete")
        frame = pd.read_parquet(prediction_path)
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"outer fold {fold} predictions lack {sorted(missing)}")
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame["organ_system"] = frame["organ_system"].astype(str)
        if set(frame["participant_id"]) != expected_by_fold[fold]:
            raise RuntimeError(f"outer fold {fold} participant coverage changed")
        if len(frame) != len(expected_by_fold[fold]) * len(ORGAN_SYSTEMS):
            raise RuntimeError(f"outer fold {fold} does not have seven organ rows")
        if frame.duplicated(["participant_id", "organ_system"]).any():
            raise RuntimeError(f"outer fold {fold} predictions contain duplicates")
        if set(frame["organ_system"]) != set(ORGAN_SYSTEMS):
            raise RuntimeError(f"outer fold {fold} changed the organ axis")
        frame["outer_fold"] = int(fold)
        frames.append(frame)
        metric_payloads.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    oof = pd.concat(frames, ignore_index=True)
    if oof["participant_id"].nunique() != len(fold_manifest):
        raise RuntimeError("five-fold OOF participant coverage is incomplete")
    if oof.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("five-fold OOF contains duplicate participant-organ rows")
    order = {organ: index for index, organ in enumerate(ORGAN_SYSTEMS)}
    oof["_organ_order"] = oof["organ_system"].map(order).astype(np.int8)
    oof = oof.sort_values(
        ["outer_fold", "participant_id", "_organ_order"], kind="mergesort"
    ).drop(columns="_organ_order")
    return oof.reset_index(drop=True), metric_payloads


def _risk_tables(
    metric_payloads: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    cells: list[dict[str, Any]] = []
    for fold, payload in enumerate(metric_payloads):
        risk = payload["full_risk"]
        for organ in ORGAN_SYSTEMS:
            cell = dict(risk["organs"][organ])
            cell["outer_fold"] = int(fold)
            cell["organ_system"] = organ
            cells.append(cell)
    cell_frame = pd.DataFrame(cells).sort_values(["outer_fold", "organ_system"])
    organ_frame = (
        cell_frame.groupby("organ_system", observed=True)
        .agg(
            mean_c_base=("c_base", "mean"),
            mean_c_plus=("c_plus", "mean"),
            mean_delta_c=("delta_c", "mean"),
            sd_delta_c=("delta_c", "std"),
            positive_folds=("delta_c", lambda values: int((values > 0).sum())),
        )
        .reindex(ORGAN_SYSTEMS)
        .reset_index()
    )
    macro = {
        "n_cells": int(len(cell_frame)),
        "macro_c_base": float(organ_frame["mean_c_base"].mean()),
        "macro_c_plus": float(organ_frame["mean_c_plus"].mean()),
        "macro_delta_c": float(organ_frame["mean_delta_c"].mean()),
        "positive_organs": int((organ_frame["mean_delta_c"] > 0).sum()),
    }
    return cell_frame, organ_frame, macro


def _comparison(
    strict: Mapping[str, Any], risk: Mapping[str, Any]
) -> dict[str, Any]:
    targets = json.loads(BASELINE_TARGETS.read_text(encoding="utf-8"))
    baseline = official_baseline_values(targets)
    deltas = {
        "strict_mae_reduction_years": baseline["strict_mae"]
        - float(strict["mae"]),
        "strict_pearson_gain": float(strict["pearson"])
        - baseline["strict_pearson"],
        "strict_r2_gain": float(strict["r2"]) - baseline["strict_r2"],
        "full_risk_macro_delta_c_gain": float(risk["macro_delta_c"])
        - baseline["full_risk_macro_delta_c"],
    }
    passes = {key: value > 0.0 for key, value in deltas.items()}
    return {
        "baseline_values": baseline,
        "delta_vs_formal_baseline": deltas,
        "strictly_better": {
            "strict_mae": passes["strict_mae_reduction_years"],
            "strict_pearson": passes["strict_pearson_gain"],
            "strict_r2": passes["strict_r2_gain"],
            "full_risk_macro_delta_c": passes[
                "full_risk_macro_delta_c_gain"
            ],
        },
        "all_four_strictly_better": bool(all(passes.values())),
    }


def build_summary(model: str = "ehr_age_huber1") -> Path:
    if model == "ehr_age_huber1":
        analysis_root = FIVEFOLD_ROOT
        run_root = RUN_ROOT
        selection_path = SELECTION_PATH
        fold0_metrics = FOLD0_METRICS
        method_column = "ehr_age"
        schema_model = "ehr_age_huber1"
    elif model == "retain_organquery_joint":
        analysis_root = MATCHED_ROOT
        run_root = MATCHED_RUN_ROOT
        selection_path = MATCHED_SELECTION_PATH
        fold0_metrics = MATCHED_FOLD0_METRICS
        method_column = "matched_retain"
        schema_model = "retain_organquery_joint"
    else:
        raise ValueError(f"unsupported model: {model}")
    summary_root = analysis_root / "02_summary"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    sources = _source_paths(fold0_metrics=fold0_metrics, run_root=run_root)
    oof, metric_payloads = _read_and_validate_predictions(sources)
    summary_root.mkdir(parents=True, exist_ok=True)
    temporary = summary_root / ".oof_predictions.parquet.tmp"
    oof.to_parquet(temporary, index=False)
    temporary.replace(summary_root / "oof_predictions.parquet")

    strict_oof = oof.loc[oof["strict_healthy_reference_60m"].eq(1)].copy()
    strict = _age_metrics(strict_oof)
    all_age = _age_metrics(oof)
    fold_rows = []
    for fold, group in oof.groupby("outer_fold", observed=True):
        fold_strict = group.loc[group["strict_healthy_reference_60m"].eq(1)]
        fold_rows.append({"outer_fold": int(fold), **_age_metrics(fold_strict)})
    fold_metrics = pd.DataFrame(fold_rows).sort_values("outer_fold")
    fold_metrics.to_csv(summary_root / "strict_fold_metrics.csv", index=False)

    organ_rows = []
    for organ, group in strict_oof.groupby("organ_system", observed=True):
        organ_rows.append({"organ_system": str(organ), **_age_metrics(group)})
    organ_metrics = (
        pd.DataFrame(organ_rows)
        .set_index("organ_system")
        .reindex(ORGAN_SYSTEMS)
        .reset_index()
    )
    organ_metrics.to_csv(summary_root / "strict_organ_metrics.csv", index=False)

    risk_cells, risk_organs, risk_macro = _risk_tables(metric_payloads)
    risk_cells.to_csv(summary_root / "cox_fold_organ_cells.csv", index=False)
    risk_organs.to_csv(summary_root / "cox_organ_summary.csv", index=False)
    comparison = _comparison(strict, risk_macro)
    comparison_rows = pd.DataFrame(
        [
            {
                "metric": "strict_mae",
                method_column: strict["mae"],
                "baseline": comparison["baseline_values"]["strict_mae"],
                "ehr_age_minus_baseline": strict["mae"]
                - comparison["baseline_values"]["strict_mae"],
                "better": comparison["strictly_better"]["strict_mae"],
            },
            {
                "metric": "strict_pearson",
                method_column: strict["pearson"],
                "baseline": comparison["baseline_values"]["strict_pearson"],
                "ehr_age_minus_baseline": strict["pearson"]
                - comparison["baseline_values"]["strict_pearson"],
                "better": comparison["strictly_better"]["strict_pearson"],
            },
            {
                "metric": "strict_r2",
                method_column: strict["r2"],
                "baseline": comparison["baseline_values"]["strict_r2"],
                "ehr_age_minus_baseline": strict["r2"]
                - comparison["baseline_values"]["strict_r2"],
                "better": comparison["strictly_better"]["strict_r2"],
            },
            {
                "metric": "full_risk_macro_delta_c",
                method_column: risk_macro["macro_delta_c"],
                "baseline": comparison["baseline_values"][
                    "full_risk_macro_delta_c"
                ],
                "ehr_age_minus_baseline": risk_macro["macro_delta_c"]
                - comparison["baseline_values"]["full_risk_macro_delta_c"],
                "better": comparison["strictly_better"][
                    "full_risk_macro_delta_c"
                ],
            },
        ]
    )
    comparison_rows.to_csv(summary_root / "comparison_to_baselines.csv", index=False)

    summary = {
        "schema_version": (
            f"revision0710.ehr_age.codebook_v2.{schema_model}.fivefold_summary.v1"
        ),
        "model": model,
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
        },
        "oof": {
            "participant_n": int(oof["participant_id"].nunique()),
            "participant_organ_rows": int(len(oof)),
            "outer_folds": sorted(oof["outer_fold"].astype(int).unique().tolist()),
        },
        "age_metrics": {
            "strict_healthy_reference_60m": strict,
            "all_participants": all_age,
        },
        "full_risk": {
            **risk_macro,
            "organs": risk_organs.to_dict(orient="records"),
        },
        "comparison_to_six_formal_baselines": comparison,
        "outer_sources": [
            {
                "outer_fold": fold,
                "predictions_path": str(prediction.resolve()),
                "predictions_sha256": sha256_file(prediction),
                "metrics_path": str(metrics.resolve()),
                "metrics_sha256": sha256_file(metrics),
            }
            for fold, prediction, metrics in sources
        ],
    }
    atomic_json(summary_root / "fivefold_summary.json", summary)
    print(json.dumps(summary["age_metrics"]["strict_healthy_reference_60m"], indent=2))
    print(json.dumps(summary["full_risk"], indent=2))
    print(json.dumps(comparison, indent=2))
    return summary_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize frozen five-fold OOF results.")
    parser.add_argument(
        "--model",
        choices=("ehr_age_huber1", "retain_organquery_joint"),
        default="ehr_age_huber1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(build_summary(parse_args().model))
