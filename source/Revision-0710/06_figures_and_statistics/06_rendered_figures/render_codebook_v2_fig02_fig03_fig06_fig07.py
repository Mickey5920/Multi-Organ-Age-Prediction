from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "Revision-0710").is_dir())
REVISION = ROOT / "Revision-0710"
SOURCE_SCRIPT = HERE / "render_corrected_canonical_fig02_fig03_fig06_fig07.py"
OOF = (
    REVISION
    / "03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1/"
    "02_summary/oof_predictions.parquet"
)
MAIN_SUMMARY_ROOT = (
    REVISION / "03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1/02_summary"
)
BASELINE_ROOT = REVISION / "05_downstream/11_codebook_v2"
MASKS = (
    REVISION
    / "01_data_rebuild/07_corrected_primary_v1/02_outcomes_60m/"
    "participant_followup_and_evaluation_masks.parquet"
)
V2_LABELS = REVISION / "01_data_rebuild/09_v2_labels/icd_participant_labels_v2.parquet"
V2_OUTCOMES = REVISION / "01_data_rebuild/09_v2_labels/organ_outcomes_60m.parquet"
OUTPUT_ROOT = REVISION / "Final_Fig_0716/Codebook_v2_Huber1"
AGE_RECORDS = OUTPUT_ROOT / "00_inputs/codebook_v2_age_records.parquet"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("fig020307_runtime", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_age_records(module: Any, core: pd.DataFrame) -> None:
    if AGE_RECORDS.is_file():
        return
    followup = pd.read_parquet(
        MASKS,
        columns=[
            "participant_id",
            "has_prevalent_any_icd",
            "strict_healthy_reference_60m",
        ],
    )
    labels = pd.read_parquet(
        V2_LABELS,
        columns=[
            "participant_id",
            *[f"has_prevalent_{organ}" for organ in module.ORGANS],
        ],
    )
    outcomes = pd.read_parquet(
        V2_OUTCOMES,
        columns=["participant_id", "has_prevalent_target", "at_risk", "event"],
    )
    for frame in (followup, labels, outcomes):
        frame["participant_id"] = frame["participant_id"].astype(str)
    any_future = (
        outcomes.loc[
            ~outcomes["has_prevalent_target"].astype(bool)
            & outcomes["at_risk"].eq(1)
            & outcomes["event"].eq(1)
        ]
        .groupby("participant_id", observed=True)
        .size()
        .gt(0)
        .rename("any_future_event")
    )
    participant = (
        followup.set_index("participant_id")
        .join(labels.set_index("participant_id"), how="inner")
        .join(any_future, how="left")
    )
    participant["any_future_event"] = participant["any_future_event"].fillna(False)
    organ_columns = [f"has_prevalent_{organ}" for organ in module.ORGANS]
    participant["Health"] = (
        ~participant["has_prevalent_any_icd"].astype(bool)
        & (
            participant["strict_healthy_reference_60m"].eq(1)
            | participant["any_future_event"].astype(bool)
        )
    )
    participant["Other_Disease"] = (
        participant["has_prevalent_any_icd"].astype(bool)
        & ~participant[organ_columns].any(axis=1)
    )
    for organ in module.ORGANS:
        participant[organ] = participant[f"has_prevalent_{organ}"].astype(bool)
    flags = participant[["Health", "Other_Disease", *module.ORGANS]].reset_index()
    records = core.merge(flags, on="participant_id", how="inner", validate="many_to_one")
    if len(records) != len(core):
        raise RuntimeError("Fig2/3/6/7 records lost OOF rows")
    AGE_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    temporary = AGE_RECORDS.with_suffix(".tmp.parquet")
    records.to_parquet(temporary, index=False)
    temporary.replace(AGE_RECORDS)


def baseline_age_tables(
    module: Any, mask: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = mask.loc[
        mask["strict_healthy_reference_60m"].eq(1),
        ["participant_id", "chronological_age"],
    ].copy()
    strict["participant_id"] = strict["participant_id"].astype(str)
    age_map = strict.set_index("participant_id")["chronological_age"]
    strict_ids = set(age_map.index)
    fold_rows: list[dict[str, Any]] = []
    organ_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    formal = pd.read_csv(BASELINE_ROOT / "02_summary/baseline_method_table.csv").set_index(
        "method"
    )
    for method, *_ in module.METHODS[:-1]:
        evaluation = BASELINE_ROOT / "01_method_results" / method
        manifest = json.loads(
            (evaluation / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        registry_path = Path(manifest["prediction_registry"])
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        cox = pd.read_csv(evaluation / "cox_fold_organ_cells.csv")
        pooled_parts = []
        for item in registry["prediction_files"]:
            if item["split_role"] != "test":
                continue
            fold = int(item["outer_fold"])
            raw = Path(item["path"])
            path = raw if raw.is_absolute() else (registry_path.parent / raw).resolve()
            prediction = pd.read_parquet(
                path,
                columns=[
                    "participant_id",
                    "organ_system",
                    "predicted_age",
                    "outer_fold",
                ],
            )
            prediction["participant_id"] = prediction["participant_id"].astype(str)
            prediction = prediction.loc[prediction["participant_id"].isin(strict_ids)].copy()
            prediction["actual_age"] = prediction["participant_id"].map(age_map)
            pooled_parts.append(prediction[["organ_system", "actual_age", "predicted_age"]])
            metric = module.metric_block(prediction)
            delta = float(cox.loc[cox["fold"].eq(fold), "delta_c"].mean())
            fold_rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "n_rows": metric["n"],
                    "mae": metric["mae"],
                    "pearson": metric["pearson"],
                    "r2": metric["r2"],
                    "full_risk_macro_delta_c": delta,
                }
            )
            for organ in module.ORGANS:
                organ_metric = module.metric_block(
                    prediction.loc[prediction["organ_system"].eq(organ)]
                )
                organ_delta = float(
                    cox.loc[
                        cox["fold"].eq(fold) & cox["organ_system"].eq(organ),
                        "delta_c",
                    ].iloc[0]
                )
                organ_rows.append(
                    {
                        "method": method,
                        "fold": fold,
                        "organ_system": organ,
                        "n_rows": organ_metric["n"],
                        "mae": organ_metric["mae"],
                        "pearson": organ_metric["pearson"],
                        "r2": organ_metric["r2"],
                        "delta_c": organ_delta,
                    }
                )
        pooled = module.metric_block(pd.concat(pooled_parts, ignore_index=True))
        reference = formal.loc[method]
        pooled_rows.append(
            {
                "method": method,
                "n_rows": pooled["n"],
                "mae": pooled["mae"],
                "pearson": pooled["pearson"],
                "r2": pooled["r2"],
                "formal_mae": reference["strict_mae"],
                "formal_pearson": reference["strict_pearson"],
                "formal_r2": reference["strict_r2"],
                "max_abs_difference": max(
                    abs(pooled["mae"] - reference["strict_mae"]),
                    abs(pooled["pearson"] - reference["strict_pearson"]),
                    abs(pooled["r2"] - reference["strict_r2"]),
                ),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(organ_rows), pd.DataFrame(pooled_rows)


def add_ehr_age_tables(
    module: Any,
    core: pd.DataFrame,
    fold: pd.DataFrame,
    organ: pd.DataFrame,
    pooled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = core.loc[core["strict_healthy_reference_60m"].eq(1)].copy()
    cox = pd.read_csv(MAIN_SUMMARY_ROOT / "cox_fold_organ_cells.csv")
    fold_rows = []
    organ_rows = []
    for fold_id in range(5):
        selected = strict.loc[strict["outer_fold"].eq(fold_id)]
        metric = module.metric_block(selected)
        fold_rows.append(
            {
                "method": "ehr_age",
                "fold": fold_id,
                "n_rows": metric["n"],
                "mae": metric["mae"],
                "pearson": metric["pearson"],
                "r2": metric["r2"],
                "full_risk_macro_delta_c": float(
                    cox.loc[cox["outer_fold"].eq(fold_id), "delta_c"].mean()
                ),
            }
        )
        for system in module.ORGANS:
            organ_metric = module.metric_block(
                selected.loc[selected["organ_system"].eq(system)]
            )
            organ_rows.append(
                {
                    "method": "ehr_age",
                    "fold": fold_id,
                    "organ_system": system,
                    "n_rows": organ_metric["n"],
                    "mae": organ_metric["mae"],
                    "pearson": organ_metric["pearson"],
                    "r2": organ_metric["r2"],
                    "delta_c": float(
                        cox.loc[
                            cox["outer_fold"].eq(fold_id)
                            & cox["organ_system"].eq(system),
                            "delta_c",
                        ].iloc[0]
                    ),
                }
            )
    pooled_metric = module.metric_block(strict)
    official = json.loads(
        (MAIN_SUMMARY_ROOT / "fivefold_summary.json").read_text(encoding="utf-8")
    )["age_metrics"]["strict_healthy_reference_60m"]
    pooled_row = {
        "method": "ehr_age",
        "n_rows": pooled_metric["n"],
        "mae": pooled_metric["mae"],
        "pearson": pooled_metric["pearson"],
        "r2": pooled_metric["r2"],
        "formal_mae": official["mae"],
        "formal_pearson": official["pearson"],
        "formal_r2": official["r2"],
        "max_abs_difference": max(
            abs(pooled_metric["mae"] - official["mae"]),
            abs(pooled_metric["pearson"] - official["pearson"]),
            abs(pooled_metric["r2"] - official["r2"]),
        ),
    }
    return (
        pd.concat([fold, pd.DataFrame(fold_rows)], ignore_index=True),
        pd.concat([organ, pd.DataFrame(organ_rows)], ignore_index=True),
        pd.concat([pooled, pd.DataFrame([pooled_row])], ignore_index=True),
    )


def main() -> None:
    module = load_module()
    module.OUTPUT_ROOT = OUTPUT_ROOT
    module.SOURCE_ROOT = OUTPUT_ROOT
    module.AGE_RECORDS = AGE_RECORDS
    module.MASKS = MASKS
    module.BASELINE_ROOT = BASELINE_ROOT
    core = pd.read_parquet(OOF)
    core["participant_id"] = core["participant_id"].astype(str)
    build_age_records(module, core)
    module.baseline_age_tables = lambda mask: baseline_age_tables(module, mask)
    module.add_ehr_age_tables = lambda data, fold, organ, pooled: add_ehr_age_tables(
        module, data, fold, organ, pooled
    )
    strict = core.loc[core["strict_healthy_reference_60m"].eq(1)].copy()
    module.render_fig2(strict)
    module.render_fig3(core)
    module.render_fig6(core)
    del core
    module.render_fig7()
    manifest = {
        "schema_version": "revision0710.codebook_v2.fig02_03_06_07.v1",
        "status": "PASS",
        "oof": str(OOF),
        "oof_sha256": module.sha256_file(OOF),
        "figures": ["Fig02", "Fig03", "Fig06", "Fig07"],
        "style": "Revision-0626/0708 visual functions reused without redesign",
    }
    module.write_json(OUTPUT_ROOT / "FIG02_FIG03_FIG06_FIG07_V2_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
