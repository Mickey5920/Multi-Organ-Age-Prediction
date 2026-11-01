from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REVISION_ROOT = EXPERIMENT_ROOT.parents[1]
OUTPUT_ROOT = EXPERIMENT_ROOT / "04_aligned_v2_evaluation_20260809"
METHOD_ROOT = OUTPUT_ROOT / "methods"
EHR_SUMMARY = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "02_fivefold_frozen_huber1"
    / "02_summary"
    / "fivefold_summary.json"
)
V2_OUTCOMES = (
    REVISION_ROOT / "01_data_rebuild" / "09_v2_labels" / "organ_outcomes_60m.parquet"
)
PARTICIPANT_MASKS = (
    REVISION_ROOT
    / "01_data_rebuild"
    / "07_corrected_primary_v1"
    / "02_outcomes_60m"
    / "participant_followup_and_evaluation_masks.parquet"
)
FOLD_ROLES = REVISION_ROOT / "01_data_rebuild" / "03_folds" / "outer_fold_roles.parquet"

METHOD_GROUPS = {
    "linear_svr": "generic_age_only",
    "mlp": "generic_age_only",
    "cnn": "generic_age_only",
    "resnet": "generic_age_only",
    "transformer": "generic_age_only",
    "retain_prospective": "objective_matched",
    "behrt_style_prospective": "objective_matched",
    "medbert_style_prospective": "objective_matched",
    "ehrformer_style_prospective": "objective_matched",
    "retain": "existing_age_only",
    "behrt_style": "existing_age_only",
    "medbert_style": "existing_age_only",
    "ehrformer_style": "existing_age_only",
    "retain_organquery": "existing_age_only_organquery",
    "behrt_style_organquery": "existing_age_only_organquery",
    "medbert_style_organquery": "existing_age_only_organquery",
    "ehrformer_style_organquery": "existing_age_only_organquery",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ehr_row() -> dict[str, object]:
    payload = json.loads(EHR_SUMMARY.read_text(encoding="utf-8"))
    strict = payload["age_metrics"]["strict_healthy_reference_60m"]
    all_participants = payload["age_metrics"]["all_participants"]
    cox = payload["full_risk"]
    return {
        "method": "EHR-AGE",
        "group": "primary",
        "strict_mae": strict["mae"],
        "strict_pearson": strict["pearson"],
        "strict_r2": strict["r2"],
        "all_mae": all_participants["mae"],
        "all_pearson": all_participants["pearson"],
        "all_r2": all_participants["r2"],
        "macro_c_base": cox["macro_c_base"],
        "macro_c_plus": cox["macro_c_plus"],
        "macro_delta_c": cox["macro_delta_c"],
        "positive_organs": cox["positive_organs"],
        "summary_path": str(EHR_SUMMARY.resolve()),
        "outcomes_sha256": sha256_file(V2_OUTCOMES),
    }


def method_row(method: str, group: str) -> dict[str, object]:
    path = METHOD_ROOT / method / "evaluation_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = METHOD_ROOT / method / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strict = payload["age_metrics"]["strict_healthy_reference_60m"]
    all_participants = payload["age_metrics"]["all"]
    cox = payload["cox"]
    return {
        "method": method,
        "group": group,
        "strict_mae": strict["mae"],
        "strict_pearson": strict["pearson"],
        "strict_r2": strict["r2"],
        "all_mae": all_participants["mae"],
        "all_pearson": all_participants["pearson"],
        "all_r2": all_participants["r2"],
        "macro_c_base": cox["macro_c_base"],
        "macro_c_plus": cox["macro_c_plus"],
        "macro_delta_c": cox["macro_delta_c"],
        "positive_organs": cox["positive_organs"],
        "summary_path": str(path.resolve()),
        "outcomes_sha256": sha256_file(Path(manifest["organ_outcomes"])),
    }


def main() -> None:
    rows = [ehr_row()]
    for method, group in METHOD_GROUPS.items():
        rows.append(method_row(method, group))
    frame = pd.DataFrame(rows)
    expected_outcome_hash = sha256_file(V2_OUTCOMES)
    if set(frame["outcomes_sha256"]) != {expected_outcome_hash}:
        raise RuntimeError("not all aligned summaries use the locked v2 outcomes")
    baseline_rows = frame.loc[frame["method"].ne("EHR-AGE")]
    if baseline_rows["macro_c_base"].max() - baseline_rows["macro_c_base"].min() > 1e-12:
        raise RuntimeError("baseline macro C-base values are not identical")
    frame.to_csv(OUTPUT_ROOT / "comparison_summary_aligned_v2.csv", index=False)
    objective_effect_rows = []
    for base in ("retain", "behrt_style", "medbert_style", "ehrformer_style"):
        age_only = frame.loc[frame["method"].eq(base)].iloc[0]
        prospective = frame.loc[frame["method"].eq(f"{base}_prospective")].iloc[0]
        objective_effect_rows.append(
            {
                "backbone": base,
                "age_only_strict_mae": age_only["strict_mae"],
                "objective_matched_strict_mae": prospective["strict_mae"],
                "strict_mae_change": prospective["strict_mae"] - age_only["strict_mae"],
                "age_only_macro_delta_c": age_only["macro_delta_c"],
                "objective_matched_macro_delta_c": prospective["macro_delta_c"],
                "macro_delta_c_gain": prospective["macro_delta_c"] - age_only["macro_delta_c"],
            }
        )
    pd.DataFrame(objective_effect_rows).to_csv(
        OUTPUT_ROOT / "objective_matched_effect_aligned_v2.csv", index=False
    )
    payload = {
        "schema_version": "revision0710.baseline_extension.aligned_v2_summary.v1",
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "locked_inputs": {
            "v2_outcomes": {"path": str(V2_OUTCOMES.resolve()), "sha256": expected_outcome_hash},
            "participant_masks": {
                "path": str(PARTICIPANT_MASKS.resolve()),
                "sha256": sha256_file(PARTICIPANT_MASKS),
            },
            "fold_roles": {"path": str(FOLD_ROLES.resolve()), "sha256": sha256_file(FOLD_ROLES)},
        },
        "audit": {
            "methods": len(frame),
            "baseline_macro_c_base_identical": True,
            "baseline_macro_c_base": float(baseline_rows["macro_c_base"].iloc[0]),
            "all_outcome_hashes_match": True,
        },
        "rows": frame.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "comparison_summary_aligned_v2.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
