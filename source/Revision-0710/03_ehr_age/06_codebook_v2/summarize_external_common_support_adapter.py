from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_fold0_current import atomic_json, sha256_file, utc_now
from run_frozen_outer_fold import ORGAN_SYSTEMS
from summarize_frozen_fivefold import _age_metrics, _risk_tables
from train_external_common_support_adapter import OUTPUT_ROOT, VARIANTS


HERE = Path(__file__).resolve().parent
FOLD_MANIFEST = (
    HERE.parents[1] / "01_data_rebuild/03_folds/fold_manifest_5fold.parquet"
)
MAIN_SUMMARY = (
    HERE / "02_fivefold_frozen_huber1/02_summary/fivefold_summary.json"
)


def summarize(variant: str) -> Path:
    run_root = OUTPUT_ROOT / variant / "01_outer_folds"
    summary_root = OUTPUT_ROOT / variant / "02_summary"
    fold_manifest = pd.read_parquet(
        FOLD_MANIFEST, columns=["participant_id", "fold_id"]
    )
    fold_manifest["participant_id"] = fold_manifest["participant_id"].astype(str)
    expected = {
        int(fold): set(group["participant_id"])
        for fold, group in fold_manifest.groupby("fold_id", observed=True)
    }
    frames = []
    metric_payloads: list[dict[str, Any]] = []
    inputs = []
    for fold in range(5):
        directory = run_root / f"outer_fold_{fold}"
        predictions_path = directory / "outer_predictions.parquet"
        metrics_path = directory / "metrics.json"
        if not predictions_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"{variant} outer fold {fold} is incomplete")
        frame = pd.read_parquet(predictions_path)
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame["organ_system"] = frame["organ_system"].astype(str)
        if set(frame["participant_id"]) != expected[fold]:
            raise RuntimeError(f"{variant} fold {fold} participant coverage changed")
        if len(frame) != len(expected[fold]) * len(ORGAN_SYSTEMS):
            raise RuntimeError(f"{variant} fold {fold} lost organ rows")
        if frame.duplicated(["participant_id", "organ_system"]).any():
            raise RuntimeError(f"{variant} fold {fold} contains duplicate rows")
        frame["outer_fold"] = fold
        frames.append(frame)
        metric_payloads.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        inputs.append(
            {
                "outer_fold": fold,
                "predictions_path": str(predictions_path.resolve()),
                "predictions_sha256": sha256_file(predictions_path),
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
            }
        )
    oof = pd.concat(frames, ignore_index=True)
    if oof["participant_id"].nunique() != len(fold_manifest):
        raise RuntimeError(f"{variant} OOF participant coverage is incomplete")
    if oof.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError(f"{variant} OOF rows are not unique")
    strict_oof = oof.loc[oof["strict_healthy_reference_60m"].eq(1)].copy()
    strict = _age_metrics(strict_oof)
    risk_cells, risk_organs, risk = _risk_tables(metric_payloads)
    main = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    main_age = main["age_metrics"]["strict_healthy_reference_60m"]
    main_risk = main["full_risk"]
    comparison = {
        "strict_mae_adapter_minus_main": float(strict["mae"] - main_age["mae"]),
        "strict_pearson_adapter_minus_main": float(
            strict["pearson"] - main_age["pearson"]
        ),
        "strict_r2_adapter_minus_main": float(strict["r2"] - main_age["r2"]),
        "full_risk_delta_c_adapter_minus_main": float(
            risk["macro_delta_c"] - main_risk["macro_delta_c"]
        ),
    }
    summary_root.mkdir(parents=True, exist_ok=True)
    temporary = summary_root / ".oof_predictions.parquet.tmp"
    oof.to_parquet(temporary, index=False)
    temporary.replace(summary_root / "oof_predictions.parquet")
    risk_cells.to_csv(summary_root / "cox_fold_organ_cells.csv", index=False)
    risk_organs.to_csv(summary_root / "cox_organ_summary.csv", index=False)
    fold_rows = []
    for fold, group in oof.groupby("outer_fold", observed=True):
        fold_rows.append(
            {
                "outer_fold": int(fold),
                **_age_metrics(
                    group.loc[group["strict_healthy_reference_60m"].eq(1)]
                ),
            }
        )
    pd.DataFrame(fold_rows).to_csv(
        summary_root / "strict_fold_metrics.csv", index=False
    )
    payload = {
        "schema_version": "revision0710.external_common_support_adapter.summary.v1",
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "variant": variant,
        "adaptation_scope": (
            "UKB train folds only; unlabeled aggregate external feature coverage "
            "used; no external participant-level values, ages, or outcomes read"
        ),
        "oof": {
            "participants": int(oof["participant_id"].nunique()),
            "participant_organ_rows": int(len(oof)),
        },
        "strict_age": strict,
        "full_risk": {
            **risk,
            "organs": risk_organs.to_dict(orient="records"),
        },
        "comparison_to_main_internal_model": comparison,
        "inputs": inputs,
    }
    atomic_json(summary_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return summary_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(summarize(parse_args().variant))
