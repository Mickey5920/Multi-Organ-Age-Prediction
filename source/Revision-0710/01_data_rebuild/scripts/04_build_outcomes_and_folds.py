from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leakage_safe_core import MONTH_DAYS, assign_stratified_folds
from outcome_table_core import build_organ_outcomes, strict_healthy_reference_mask
from source_manifest_core import derive_observed_followup_end, normalize_participant_id


ROOT = Path(__file__).resolve().parents[3]
REVISION_DIR = ROOT / "Revision-0710"
REBUILD_DIR = REVISION_DIR / "01_data_rebuild"
DEFAULT_CONFIG = REBUILD_DIR / "00_config" / "source_paths.json"
DEFAULT_INDEX_DIR = REBUILD_DIR / "01_index_and_events"
DEFAULT_OUTCOME_DIR = REBUILD_DIR / "02_outcomes_60m"
DEFAULT_FOLD_DIR = REBUILD_DIR / "03_folds"
SYSTEMS = [
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
]


def stable_binary_role(seed: str, participant_id: str) -> str:
    value = int(hashlib.sha256(f"{seed}|{participant_id}".encode("utf-8")).hexdigest()[:16], 16)
    return "calibration" if value % 2 == 0 else "screen"


def normalize_sex(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("missing").str.strip().str.lower()
    return text.replace({"f": "female", "m": "male", "": "missing"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build censor-aware organ outcomes and outcome-blind folds.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--outcome-dir", type=Path, default=DEFAULT_OUTCOME_DIR)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    horizon_months = int(config["horizon_months"])
    args.outcome_dir.mkdir(parents=True, exist_ok=True)
    args.fold_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Joining cohort, source follow-up, ICD, and GP manifests", flush=True)
    cohort = pd.read_parquet(args.index_dir / "age_model_cohort.parquet")
    followup = pd.read_parquet(args.index_dir / "followup_source_dates.parquet")
    icd_columns = [
        "participant_id",
        "index_date",
        "last_icd_date",
        "first_incident_any_date",
        "has_prevalent_any_icd",
        *[f"has_prevalent_{system}" for system in SYSTEMS],
        *[f"first_incident_date_{system}" for system in SYSTEMS],
    ]
    gp_columns = ["participant_id", "index_date", "last_gp_date", "model_input_gp_count"]
    icd = pd.read_parquet(args.index_dir / "icd_participant_summary.parquet", columns=icd_columns)
    gp = pd.read_parquet(args.index_dir / "gp_participant_summary.parquet", columns=gp_columns)
    for frame in [cohort, followup, icd, gp]:
        frame["participant_id"] = normalize_participant_id(frame["participant_id"])

    index_checks = (
        cohort[["participant_id", "index_date"]]
        .merge(followup[["participant_id", "index_date"]], on="participant_id", suffixes=("_cohort", "_followup"))
        .merge(icd[["participant_id", "index_date"]], on="participant_id")
        .merge(gp[["participant_id", "index_date"]], on="participant_id", suffixes=("_icd", "_gp"))
    )
    index_mismatch = (
        pd.to_datetime(index_checks["index_date_cohort"]).ne(pd.to_datetime(index_checks["index_date_followup"]))
        | pd.to_datetime(index_checks["index_date_cohort"]).ne(pd.to_datetime(index_checks["index_date_icd"]))
        | pd.to_datetime(index_checks["index_date_cohort"]).ne(pd.to_datetime(index_checks["index_date_gp"]))
    )
    if index_mismatch.any():
        raise ValueError(f"Index-date disagreement in {int(index_mismatch.sum())} participants")

    participant = cohort.merge(
        followup.drop(columns="index_date"), on="participant_id", how="left", validate="one_to_one"
    )
    participant = participant.merge(
        icd.drop(columns="index_date"), on="participant_id", how="left", validate="one_to_one"
    )
    participant = participant.merge(
        gp.drop(columns="index_date"), on="participant_id", how="left", validate="one_to_one"
    )
    date_candidates = ["date1_followup_end", "last_icd_date", "last_gp_date", "gp_deduction_date"]
    participant["observed_followup_end"] = derive_observed_followup_end(
        participant,
        candidate_columns=date_candidates,
        lost_column="lost_to_followup_date",
    )
    participant["strict_healthy_reference_60m"] = strict_healthy_reference_mask(
        participant,
        horizon_months=horizon_months,
    ).astype(np.int8)
    administrative_end = participant["index_date"] + pd.to_timedelta(
        horizon_months * MONTH_DAYS, unit="D"
    )
    participant["followup_ascertained_60m"] = (
        participant["observed_followup_end"].notna()
        & participant["observed_followup_end"].ge(administrative_end)
    ).astype(np.int8)
    participant["death_within_60m"] = (
        participant["death_date"].notna()
        & participant["death_date"].gt(participant["index_date"])
        & participant["death_date"].le(administrative_end)
    ).astype(np.int8)

    print("[2/5] Constructing seven cause-specific outcome tables", flush=True)
    any_prevalent_included = participant[
        [f"has_prevalent_{system}" for system in SYSTEMS]
    ].fillna(False).any(axis=1)
    outcome_parts: list[pd.DataFrame] = []
    for system in SYSTEMS:
        organ_input = pd.DataFrame(
            {
                "participant_id": participant["participant_id"],
                "index_date": participant["index_date"],
                "observed_followup_end": participant["observed_followup_end"],
                "death_date": participant["death_date"],
                "has_prevalent_target": participant[f"has_prevalent_{system}"],
                "first_incident_target_date": participant[f"first_incident_date_{system}"],
            }
        )
        out = build_organ_outcomes(organ_input, horizon_months=horizon_months)
        out["organ_system"] = system
        out["index_date"] = participant["index_date"].to_numpy()
        out["first_incident_target_date"] = participant[f"first_incident_date_{system}"].to_numpy()
        out["has_prevalent_target"] = participant[f"has_prevalent_{system}"].fillna(False).astype(np.int8).to_numpy()
        out["strict_healthy_reference_60m"] = participant["strict_healthy_reference_60m"].to_numpy()
        out["full_incident_risk_set_60m"] = out["at_risk"].astype(np.int8)
        other_prevalent = any_prevalent_included & ~participant[f"has_prevalent_{system}"].fillna(False)
        out["cohort_state_60m"] = np.select(
            [
                out["has_prevalent_target"].eq(1),
                out["event"].eq(1),
                out["strict_healthy_reference_60m"].eq(1),
                out["at_risk"].eq(0),
                other_prevalent,
            ],
            [
                "current_target_disease",
                "baseline_free_future_target_event",
                "strict_disease_free_through_followup",
                "insufficient_followup_or_ascertainment",
                "target_disease_free_other_system_disease",
            ],
            default="target_disease_free_no_included_disease",
        )
        outcome_parts.append(out)
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    bad_duration = outcomes["at_risk"].eq(1) & (
        outcomes["duration_months"].isna()
        | outcomes["duration_months"].le(0)
        | outcomes["duration_months"].gt(horizon_months + 0.05)
    )
    bad_event = outcomes["event"].eq(1) & (
        outcomes["at_risk"].ne(1) | outcomes["has_prevalent_target"].eq(1)
    )
    if bad_duration.any() or bad_event.any():
        raise ValueError(
            f"Outcome consistency failed: bad_duration={int(bad_duration.sum())}, bad_event={int(bad_event.sum())}"
        )

    participant_output_columns = [
        "participant_id",
        "index_date",
        "chronological_age",
        "sex",
        "phenotype_missing_count",
        "phenotype_missing_fraction",
        "death_date",
        "lost_to_followup_date",
        "date1_followup_end",
        "last_icd_date",
        "last_gp_date",
        "gp_deduction_date",
        "observed_followup_end",
        "followup_ascertained_60m",
        "death_within_60m",
        "has_prevalent_any_icd",
        "first_incident_any_date",
        "strict_healthy_reference_60m",
    ]
    participant[participant_output_columns].to_parquet(
        args.outcome_dir / "participant_followup_and_evaluation_masks.parquet",
        index=False,
        compression="zstd",
    )
    outcomes.to_parquet(
        args.outcome_dir / "organ_outcomes_60m.parquet", index=False, compression="zstd"
    )
    outcome_counts = (
        outcomes.groupby(["organ_system", "cohort_state_60m"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    outcome_counts.to_csv(args.outcome_dir / "organ_cohort_state_counts_60m.csv", index=False)

    print("[3/5] Assigning outcome-blind participant folds", flush=True)
    fold_source = participant[
        ["participant_id", "chronological_age", "sex", "phenotype_missing_count"]
    ].copy()
    fold_source["sex_norm"] = normalize_sex(fold_source["sex"])
    fold_source["age_decile"] = pd.qcut(
        fold_source["chronological_age"], 10, labels=False, duplicates="drop"
    ).astype(int)
    fold_source["missingness_bin"] = pd.cut(
        fold_source["phenotype_missing_count"],
        bins=[-1, 0, 5, 10, 19],
        labels=["0", "1-5", "6-10", "11-19"],
        include_lowest=True,
    ).astype("string")
    fold_source["stratum"] = (
        fold_source["sex_norm"]
        + "|age"
        + fold_source["age_decile"].astype(str)
        + "|missing"
        + fold_source["missingness_bin"]
    )
    assigned = assign_stratified_folds(
        fold_source[["participant_id", "stratum"]],
        n_folds=5,
        seed=str(config["fold_seed"]),
    )[["participant_id", "fold_id"]]
    fold_manifest = fold_source.merge(assigned, on="participant_id", how="left", validate="one_to_one")
    fold_manifest = fold_manifest.sort_values(["fold_id", "participant_id"], kind="mergesort")
    fold_manifest.to_parquet(args.fold_dir / "fold_manifest_5fold.parquet", index=False, compression="zstd")
    fold_manifest.to_csv(args.fold_dir / "fold_manifest_5fold.csv.gz", index=False, compression="gzip")

    outer_parts = []
    for outer_fold in range(5):
        view = fold_manifest[["participant_id", "fold_id"]].copy()
        view["outer_fold"] = np.int8(outer_fold)
        view["split_role"] = np.where(
            view["fold_id"].eq(outer_fold),
            "test",
            np.where(view["fold_id"].eq((outer_fold + 1) % 5), "validation", "train"),
        )
        outer_parts.append(view)
    outer_roles = pd.concat(outer_parts, ignore_index=True)
    outer_roles.to_parquet(args.fold_dir / "outer_fold_roles.parquet", index=False, compression="zstd")
    fold0_validation = fold_manifest.loc[
        fold_manifest["fold_id"].eq(1), ["participant_id", "fold_id", "stratum"]
    ].copy()
    fold0_validation["inner_role"] = [
        stable_binary_role(f"{config['fold_seed']}|fold0-validation", participant_id)
        for participant_id in fold0_validation["participant_id"]
    ]
    fold0_validation.to_parquet(
        args.fold_dir / "fold0_validation_calibration_screen.parquet",
        index=False,
        compression="zstd",
    )

    print("[4/5] Auditing fold and outcome balance", flush=True)
    balance = (
        fold_manifest.groupby("fold_id")
        .agg(
            n=("participant_id", "size"),
            age_mean=("chronological_age", "mean"),
            age_sd=("chronological_age", "std"),
            age_min=("chronological_age", "min"),
            age_max=("chronological_age", "max"),
            missingness_mean=("phenotype_missing_count", "mean"),
        )
        .reset_index()
    )
    sex_balance = pd.crosstab(fold_manifest["fold_id"], fold_manifest["sex_norm"]).add_prefix("sex_n__")
    balance = balance.merge(sex_balance, left_on="fold_id", right_index=True, how="left")
    balance.to_csv(args.fold_dir / "fold_baseline_balance.csv", index=False)
    outcome_balance = (
        outcomes[["participant_id", "organ_system", "at_risk", "event"]]
        .merge(fold_manifest[["participant_id", "fold_id"]], on="participant_id", validate="many_to_one")
        .groupby(["organ_system", "fold_id"])
        .agg(risk_set_n=("at_risk", "sum"), event_n=("event", "sum"))
        .reset_index()
    )
    outcome_balance["event_rate"] = outcome_balance["event_n"] / outcome_balance["risk_set_n"]
    outcome_balance.to_csv(args.fold_dir / "fold_outcome_balance_60m.csv", index=False)

    role_counts = outer_roles.groupby(["participant_id", "split_role"]).size().unstack(fill_value=0)
    role_gate = bool(
        role_counts.get("test", pd.Series(dtype=int)).eq(1).all()
        and role_counts.get("validation", pd.Series(dtype=int)).eq(1).all()
        and role_counts.get("train", pd.Series(dtype=int)).eq(3).all()
    )
    outcome_audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "participant_n": int(len(participant)),
        "organ_outcome_rows": int(len(outcomes)),
        "strict_healthy_reference_n": int(participant["strict_healthy_reference_60m"].sum()),
        "full_incident_risk_rows": int(outcomes["at_risk"].sum()),
        "incident_event_rows": int(outcomes["event"].sum()),
        "competing_death_rows": int(outcomes["competing_death"].sum()),
        "bad_duration_rows": int(bad_duration.sum()),
        "bad_event_rows": int(bad_event.sum()),
    }
    fold_audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if role_gate else "FAIL",
        "participant_n": int(len(fold_manifest)),
        "unique_participant_n": int(fold_manifest["participant_id"].nunique()),
        "fold_sizes": {str(key): int(value) for key, value in fold_manifest.groupby("fold_id").size().items()},
        "assignment_fields": ["sex_norm", "chronological_age_decile", "phenotype_missingness_bin"],
        "future_outcome_fields_used_for_assignment": [],
        "outer_role_gate": role_gate,
        "fold0_validation_inner_counts": {
            str(key): int(value) for key, value in fold0_validation["inner_role"].value_counts().items()
        },
    }
    (args.outcome_dir / "outcome_rebuild_audit.json").write_text(
        json.dumps(outcome_audit, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (args.fold_dir / "fold_rebuild_audit.json").write_text(
        json.dumps(fold_audit, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print("[5/5] Outcome and fold gates complete", flush=True)
    print(json.dumps({"outcomes": outcome_audit, "folds": fold_audit}, indent=2), flush=True)
    if not role_gate:
        raise RuntimeError("Fold role audit failed")


if __name__ == "__main__":
    main()
