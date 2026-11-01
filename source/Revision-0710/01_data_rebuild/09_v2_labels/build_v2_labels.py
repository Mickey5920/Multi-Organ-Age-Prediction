from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REBUILD = ROOT / "Revision-0710" / "01_data_rebuild"
BASE_INDEX = REBUILD / "01_index_and_events"
CURRENT_ACCEPTED = REBUILD / "07_corrected_primary_v1"
BASE_OUTCOME = CURRENT_ACCEPTED / "02_outcomes_60m"
BASE_FOLDS = REBUILD / "03_folds"
CODEBOOK_DIR = REBUILD / "08_icd_codebook_v2"
RAW_ICD = Path(
    "E:/MPU/Project/Dataset-0911/Process/data/"
    "1-combined_ICD_Time_birth_phenotype_age_all-1124.csv"
)
SCRIPTS = REBUILD / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from outcome_table_core import MONTH_DAYS, build_organ_outcomes  # noqa: E402


ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_participant(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def parse_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def collapse_counts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["participant_id"])
    frame = pd.concat(parts, ignore_index=True)
    columns = [column for column in frame.columns if column != "participant_id"]
    return frame.groupby("participant_id", as_index=False, sort=False)[columns].sum()


def collapse_dates(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["participant_id"])
    frame = pd.concat(parts, ignore_index=True)
    return frame.groupby("participant_id", as_index=False, sort=False).min()


def build_subsequent_outcomes(
    frame: pd.DataFrame,
    *,
    horizon_months: int = 60,
) -> pd.DataFrame:
    index_date = pd.to_datetime(frame["index_date"], errors="coerce")
    observed = pd.to_datetime(frame["observed_followup_end"], errors="coerce")
    death = pd.to_datetime(frame["death_date"], errors="coerce")
    target = pd.to_datetime(frame["first_subsequent_target_date"], errors="coerce")
    prevalent = frame["has_prevalent_target"].fillna(False).astype(bool)
    administrative_end = index_date + pd.to_timedelta(horizon_months * MONTH_DAYS, unit="D")

    observed_valid = observed.notna() & observed.gt(index_date)
    target_valid = target.notna() & target.gt(index_date)
    death_valid = death.notna() & death.gt(index_date)
    observed_explicitly_invalid = observed.notna() & ~observed_valid
    endpoint_evidence = (target_valid | death_valid) & ~observed_explicitly_invalid

    observation_limit = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    observation_limit.loc[observed_valid] = pd.concat(
        [observed.loc[observed_valid], administrative_end.loc[observed_valid]], axis=1
    ).min(axis=1)
    fallback = observed.isna() & endpoint_evidence
    observation_limit.loc[fallback] = administrative_end.loc[fallback]

    death_observed = death_valid & observation_limit.notna() & death.le(observation_limit)
    target_observed = (
        target_valid
        & observation_limit.notna()
        & target.le(observation_limit)
        & (~death_observed | target.le(death))
    )
    at_risk = prevalent & observation_limit.notna() & observation_limit.gt(index_date)
    event = at_risk & target_observed
    competing_death = at_risk & ~event & death_observed
    end_date = observation_limit.copy()
    end_date.loc[competing_death] = death.loc[competing_death]
    end_date.loc[event] = target.loc[event]
    duration = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    duration.loc[at_risk] = (
        (end_date.loc[at_risk] - index_date.loc[at_risk]).dt.days / MONTH_DAYS
    ).clip(lower=1e-6)
    reason = pd.Series("not_prevalent_or_no_followup", index=frame.index, dtype="string")
    reason.loc[at_risk] = "observed_followup_end"
    reason.loc[at_risk & end_date.eq(administrative_end)] = "administrative_60m"
    reason.loc[competing_death] = "death_before_subsequent_code"
    reason.loc[event] = "subsequent_organ_code"
    return pd.DataFrame(
        {
            "subsequent_event": event.astype(np.int8),
            "subsequent_duration_months": duration,
            "subsequent_competing_death": competing_death.astype(np.int8),
            "subsequent_at_risk": at_risk.astype(np.int8),
            "subsequent_censor_reason": reason,
            "subsequent_outcome_end_date": end_date,
        }
    )


def scan_labels(
    raw_path: Path,
    cohort: pd.DataFrame,
    code_audit: pd.DataFrame,
    *,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cohort_ids = set(cohort["participant_id"])
    index_map = cohort.set_index("participant_id")["index_date"]
    primary_map = code_audit.set_index("icd_code_normalized")["v2_primary_organ"].to_dict()
    baseline_map = (
        code_audit.set_index("icd_code_normalized")["baseline_chronic_eligible"]
        .astype(bool)
        .to_dict()
    )
    incident_map = (
        code_audit.set_index("icd_code_normalized")["incident_endpoint_eligible"]
        .astype(bool)
        .to_dict()
    )
    subsequent_map = (
        code_audit.set_index("icd_code_normalized")["recurrent_endpoint_eligible"]
        .astype(bool)
        .to_dict()
    )
    count_parts: list[pd.DataFrame] = []
    date_parts: list[pd.DataFrame] = []
    source_rows = 0
    cohort_rows = 0
    valid_rows = 0
    preindex_eligible_rows = 0
    postindex_eligible_rows = 0

    reader = pd.read_csv(
        raw_path,
        usecols=["Participant ID", "ICD", "record_time"],
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        source_rows += len(chunk)
        chunk = chunk.rename(
            columns={"Participant ID": "participant_id", "ICD": "code", "record_time": "event_date"}
        )
        chunk["participant_id"] = normalize_participant(chunk["participant_id"])
        chunk = chunk.loc[chunk["participant_id"].isin(cohort_ids)].copy()
        cohort_rows += len(chunk)
        chunk["index_date"] = chunk["participant_id"].map(index_map)
        chunk["event_date"] = parse_date(chunk["event_date"])
        chunk["code"] = (
            chunk["code"].fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        valid = chunk["code"].ne("") & chunk["event_date"].notna()
        valid_rows += int(valid.sum())
        chunk = chunk.loc[valid].copy()
        relative_days = (chunk["event_date"] - chunk["index_date"]).dt.days
        chunk["primary_organ"] = chunk["code"].map(primary_map).fillna("")
        chunk["baseline_eligible"] = chunk["code"].map(baseline_map).fillna(False).astype(bool)
        chunk["incident_eligible"] = chunk["code"].map(incident_map).fillna(False).astype(bool)
        chunk["subsequent_eligible"] = chunk["code"].map(subsequent_map).fillna(False).astype(bool)

        count_frame = pd.DataFrame({"participant_id": chunk["participant_id"]})
        date_frame = pd.DataFrame({"participant_id": chunk["participant_id"]})
        for organ in ORGANS:
            organ_match = chunk["primary_organ"].eq(organ)
            prevalent = organ_match & chunk["baseline_eligible"] & relative_days.le(0)
            incident = organ_match & chunk["incident_eligible"] & relative_days.gt(0)
            subsequent = organ_match & chunk["subsequent_eligible"] & relative_days.gt(0)
            count_frame[f"prevalent_{organ}_count"] = prevalent.astype(np.int32)
            count_frame[f"postindex_{organ}_count"] = incident.astype(np.int32)
            date_frame[f"first_incident_date_{organ}"] = chunk["event_date"].where(incident)
            date_frame[f"first_subsequent_date_{organ}"] = chunk["event_date"].where(subsequent)
            preindex_eligible_rows += int(prevalent.sum())
            postindex_eligible_rows += int(incident.sum())
        count_parts.append(count_frame.groupby("participant_id", as_index=False, sort=False).sum())
        date_parts.append(date_frame.groupby("participant_id", as_index=False, sort=False).min())
        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"chunks={chunk_number}; source_rows={source_rows:,}; cohort_rows={cohort_rows:,}",
                flush=True,
            )

    labels = cohort[["participant_id", "index_date"]].merge(
        collapse_counts(count_parts), on="participant_id", how="left", validate="one_to_one"
    )
    labels = labels.merge(
        collapse_dates(date_parts), on="participant_id", how="left", validate="one_to_one"
    )
    count_columns = [column for column in labels if column.endswith("_count")]
    labels[count_columns] = labels[count_columns].fillna(0).astype(np.int32)
    for organ in ORGANS:
        labels[f"has_prevalent_{organ}"] = labels[f"prevalent_{organ}_count"].gt(0)
    scan = {
        "source_rows_scanned": int(source_rows),
        "age_cohort_source_rows": int(cohort_rows),
        "valid_age_cohort_icd_rows": int(valid_rows),
        "preindex_main_organ_rows": int(preindex_eligible_rows),
        "postindex_main_endpoint_rows": int(postindex_eligible_rows),
    }
    return labels, scan


def build_outputs(
    labels: pd.DataFrame,
    participant: pd.DataFrame,
    old_outcomes: pd.DataFrame,
    *,
    horizon_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = participant.merge(labels.drop(columns="index_date"), on="participant_id", validate="one_to_one")
    any_organ = joined[[f"has_prevalent_{organ}" for organ in ORGANS]].any(axis=1)
    outcome_parts: list[pd.DataFrame] = []
    old_any_prevalent_ids = set(
        old_outcomes.loc[old_outcomes["has_prevalent_target"].eq(1), "participant_id"]
    )
    group_rows: list[dict[str, object]] = [
        {
            "scope": "global",
            "organ_system": "all",
            "group": "strict_Health",
            "v1_n": int(joined["strict_healthy_reference_60m"].sum()),
            "v2_n": int(joined["strict_healthy_reference_60m"].sum()),
        },
        {
            "scope": "global",
            "organ_system": "all",
            "group": "Other_Disease_no_v2_organ_disease",
            "v1_n": int(
                (
                    joined["has_prevalent_any_icd"].astype(bool)
                    & ~joined["participant_id"].isin(old_any_prevalent_ids)
                ).sum()
            ),
            "v2_n": int((joined["has_prevalent_any_icd"].astype(bool) & ~any_organ).sum()),
        },
    ]
    old_indexed = old_outcomes.set_index(["participant_id", "organ_system"]).sort_index()
    for organ in ORGANS:
        organ_input = pd.DataFrame(
            {
                "participant_id": joined["participant_id"],
                "index_date": joined["index_date"],
                "observed_followup_end": joined["observed_followup_end"],
                "death_date": joined["death_date"],
                "has_prevalent_target": joined[f"has_prevalent_{organ}"],
                "first_incident_target_date": joined[f"first_incident_date_{organ}"],
                "first_subsequent_target_date": joined[f"first_subsequent_date_{organ}"],
            }
        )
        out = build_organ_outcomes(organ_input, horizon_months=horizon_months)
        subsequent = build_subsequent_outcomes(organ_input, horizon_months=horizon_months)
        out = pd.concat([out, subsequent], axis=1)
        out["organ_system"] = organ
        out["index_date"] = joined["index_date"].to_numpy()
        out["first_incident_target_date"] = joined[f"first_incident_date_{organ}"].to_numpy()
        out["first_subsequent_target_date"] = joined[f"first_subsequent_date_{organ}"].to_numpy()
        out["has_prevalent_target"] = joined[f"has_prevalent_{organ}"].astype(np.int8).to_numpy()
        out["strict_healthy_reference_60m"] = joined["strict_healthy_reference_60m"].astype(np.int8).to_numpy()
        other_prevalent = any_organ & ~joined[f"has_prevalent_{organ}"]
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

        old_organ = old_indexed.xs(organ, level="organ_system")
        group_rows.extend(
            [
                {
                    "scope": "organ_specific",
                    "organ_system": organ,
                    "group": "Disease",
                    "v1_n": int(old_organ["has_prevalent_target"].sum()),
                    "v2_n": int(joined[f"has_prevalent_{organ}"].sum()),
                },
                {
                    "scope": "organ_specific",
                    "organ_system": organ,
                    "group": "Health_target_free_other_organ_disease",
                    "v1_n": int(old_organ["cohort_state_60m"].eq("target_disease_free_other_system_disease").sum()),
                    "v2_n": int(other_prevalent.sum()),
                },
                {
                    "scope": "outcome",
                    "organ_system": organ,
                    "group": "incident_risk_set",
                    "v1_n": int(old_organ["at_risk"].sum()),
                    "v2_n": int(out["at_risk"].sum()),
                },
                {
                    "scope": "outcome",
                    "organ_system": organ,
                    "group": "incident_event_60m",
                    "v1_n": int(old_organ["event"].sum()),
                    "v2_n": int(out["event"].sum()),
                },
                {
                    "scope": "outcome",
                    "organ_system": organ,
                    "group": "subsequent_event_60m_among_prevalent",
                    "v1_n": np.nan,
                    "v2_n": int(out["subsequent_event"].sum()),
                },
            ]
        )
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    old_event_ids = set(old_outcomes.loc[old_outcomes["event"].eq(1), "participant_id"])
    new_event_ids = set(outcomes.loc[outcomes["event"].eq(1), "participant_id"])
    baseline_icd_free = ~joined["has_prevalent_any_icd"].astype(bool)
    strict = joined["strict_healthy_reference_60m"].astype(bool)
    group_rows.insert(
        1,
        {
            "scope": "global",
            "organ_system": "all",
            "group": "Health_display_strict_or_future_v2_organ_event",
            "v1_n": int((strict | (baseline_icd_free & joined["participant_id"].isin(old_event_ids))).sum()),
            "v2_n": int((strict | (baseline_icd_free & joined["participant_id"].isin(new_event_ids))).sum()),
        },
    )
    groups = pd.DataFrame(group_rows)
    groups["delta_n"] = groups["v2_n"] - groups["v1_n"]
    return outcomes, groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Build codebook-v2 organ labels and outcomes.")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--raw-icd", type=Path, default=RAW_ICD)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--horizon-months", type=int, default=60)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "cohort": BASE_INDEX / "age_model_cohort.parquet",
        "participant_masks": BASE_OUTCOME / "participant_followup_and_evaluation_masks.parquet",
        "old_outcomes": BASE_OUTCOME / "organ_outcomes_60m.parquet",
        "folds": BASE_FOLDS / "fold_manifest_5fold.parquet",
        "codebook": CODEBOOK_DIR / "codebook_v2.csv",
        "observed_code_audit": CODEBOOK_DIR / "observed_code_audit.csv",
        "raw_icd": args.raw_icd,
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)

    cohort = pd.read_parquet(required["cohort"], columns=["participant_id", "index_date"])
    cohort["participant_id"] = normalize_participant(cohort["participant_id"])
    cohort["index_date"] = pd.to_datetime(cohort["index_date"], errors="coerce")
    code_audit = pd.read_csv(required["observed_code_audit"])
    participant = pd.read_parquet(required["participant_masks"])
    participant["participant_id"] = normalize_participant(participant["participant_id"])
    participant["has_prevalent_any_icd"] = participant["has_prevalent_any_icd"].astype(bool)
    old_outcomes = pd.read_parquet(required["old_outcomes"])
    old_outcomes["participant_id"] = normalize_participant(old_outcomes["participant_id"])

    labels, scan = scan_labels(args.raw_icd, cohort, code_audit, chunksize=args.chunksize)
    outcomes, groups = build_outputs(
        labels,
        participant,
        old_outcomes,
        horizon_months=args.horizon_months,
    )

    if len(labels) != 456_773 or labels["participant_id"].nunique() != len(labels):
        raise RuntimeError("Participant coverage failed")
    if len(outcomes) != len(labels) * len(ORGANS):
        raise RuntimeError("Participant-organ coverage failed")
    if int(participant["strict_healthy_reference_60m"].sum()) != 102_129:
        raise RuntimeError("Strict Health changed despite mapping-independent definition")
    if outcomes.loc[outcomes["event"].eq(1), "first_incident_target_date"].le(
        outcomes.loc[outcomes["event"].eq(1), "index_date"]
    ).any():
        raise RuntimeError("Incident event at or before index")
    bad_duration = outcomes["at_risk"].eq(1) & (
        outcomes["duration_months"].isna()
        | outcomes["duration_months"].le(0)
        | outcomes["duration_months"].gt(args.horizon_months + 0.05)
    )
    bad_subsequent = outcomes["subsequent_at_risk"].eq(1) & (
        outcomes["subsequent_duration_months"].isna()
        | outcomes["subsequent_duration_months"].le(0)
        | outcomes["subsequent_duration_months"].gt(args.horizon_months + 0.05)
    )
    if bad_duration.any() or bad_subsequent.any():
        raise RuntimeError(
            f"Duration gate failed: incident={int(bad_duration.sum())}, subsequent={int(bad_subsequent.sum())}"
        )

    label_path = args.output_dir / "icd_participant_labels_v2.parquet"
    outcome_path = args.output_dir / "organ_outcomes_60m.parquet"
    group_path = args.output_dir / "group_counts.csv"
    labels.to_parquet(label_path, index=False, compression="zstd")
    outcomes.to_parquet(outcome_path, index=False, compression="zstd")
    groups.to_csv(group_path, index=False)

    fold_hash = sha256_file(required["folds"])
    audit = {
        "schema_version": "revision0710.v2_labels.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "definition": {
            "baseline": "primary-organ code with baseline_chronic_eligible=1 at or before index",
            "incident": "first primary-organ code with incident_endpoint_eligible=1 after index among baseline target-organ disease-free participants",
            "subsequent": "first eligible post-index organ code among participants with prevalent target-organ disease; source contains first occurrence by code, so this is not claimed as same-code recurrence",
            "strict_health": "mapping-independent any-ICD-free reference with complete 60-month ascertainment",
            "health_display": "strict Health union baseline-any-ICD-free participants with at least one eligible seven-organ event within 60 months",
        },
        "scan": scan,
        "counts": {
            "participant_n": int(len(labels)),
            "participant_organ_rows": int(len(outcomes)),
            "strict_health_n": 102_129,
            "health_display_n": int(
                groups.loc[
                    groups["group"].eq("Health_display_strict_or_future_v2_organ_event"),
                    "v2_n",
                ].iloc[0]
            ),
            "incident_risk_rows": int(outcomes["at_risk"].sum()),
            "incident_event_rows_60m": int(outcomes["event"].sum()),
            "subsequent_risk_rows": int(outcomes["subsequent_at_risk"].sum()),
            "subsequent_event_rows_60m": int(outcomes["subsequent_event"].sum()),
            "bad_incident_duration_rows": int(bad_duration.sum()),
            "bad_subsequent_duration_rows": int(bad_subsequent.sum()),
        },
        "leakage": {
            "model_input_cache_modified": False,
            "fold_manifest_modified": False,
            "fold_manifest_sha256": fold_hash,
            "future_codes_used_only_for_outcomes": True,
            "strict_health_recomputed_from_future_organ_mapping": False,
        },
        "inputs": {name: file_record(path) for name, path in required.items()},
        "outputs": {
            path.name: file_record(path) for path in (label_path, outcome_path, group_path)
        },
    }
    audit_path = args.output_dir / "rebuild_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
