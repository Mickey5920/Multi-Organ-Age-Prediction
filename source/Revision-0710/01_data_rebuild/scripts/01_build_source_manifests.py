from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from source_manifest_core import build_age_model_cohort, normalize_participant_id


ROOT = Path(__file__).resolve().parents[3]
REVISION_DIR = ROOT / "Revision-0710"
DEFAULT_CONFIG = REVISION_DIR / "01_data_rebuild" / "00_config" / "source_paths.json"
DEFAULT_OUTPUT = REVISION_DIR / "01_data_rebuild" / "01_index_and_events"
MONTH_LOOKUP = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
MONTH_LOOKUP.update({name.lower(): number for number, name in enumerate(calendar.month_abbr) if name})


def parse_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def load_assessment(path: Path) -> pd.DataFrame:
    columns = [
        "Participant ID",
        "Date of attending assessment centre | Instance 0",
        "Month of birth",
        "Year of birth",
    ]
    frame = pd.read_csv(path, usecols=columns, dtype=str, low_memory=False)
    out = pd.DataFrame()
    out["participant_id"] = normalize_participant_id(frame["Participant ID"])
    out["index_date"] = parse_date(frame["Date of attending assessment centre | Instance 0"])
    month_text = frame["Month of birth"].astype("string").str.strip().str.lower()
    month = pd.to_numeric(frame["Month of birth"], errors="coerce").fillna(month_text.map(MONTH_LOOKUP))
    year = pd.to_numeric(frame["Year of birth"], errors="coerce")
    out["birth_date"] = pd.to_datetime(
        {"year": year, "month": month, "day": 15}, errors="coerce"
    )
    return out


def load_phenotype(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    excluded = {"Participant ID", "Sex", "Age at recruitment"}
    feature_columns = [column for column in frame.columns if column not in excluded]
    frame = frame.rename(
        columns={
            "Participant ID": "participant_id",
            "Sex": "sex",
            "Age at recruitment": "reported_age_at_recruitment",
        }
    )
    frame["participant_id"] = normalize_participant_id(frame["participant_id"])
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["reported_age_at_recruitment"] = pd.to_numeric(
        frame["reported_age_at_recruitment"], errors="coerce"
    )
    return frame, feature_columns


def load_death(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    frame.columns = frame.columns.str.strip()
    id_column = (
        "Participant ID(participant - eid)"
        if "Participant ID(participant - eid)" in frame.columns
        else "Participant ID(death - eid)"
    )
    out = pd.DataFrame(
        {
            "participant_id": normalize_participant_id(frame[id_column]),
            "death_date": parse_date(frame["Date of death"]),
        }
    )
    return out[out["participant_id"].ne("")].groupby("participant_id", as_index=False)["death_date"].min()


def load_visit_followup(path: Path) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    id_column = "Participant ID" if "Participant ID" in header else "eid"
    contact = [column for column in header if "Date of last personal contact with UK Biobank" in column]
    assessment = [column for column in header if "Date of attending assessment centre" in column]
    lost = [column for column in header if "Date lost to follow-up" in column]
    columns = [id_column, *contact, *assessment, *lost]
    frame = pd.read_csv(path, usecols=columns, dtype=str, low_memory=False)
    frame["participant_id"] = normalize_participant_id(frame[id_column])
    for column in [*contact, *assessment, *lost]:
        frame[column] = parse_date(frame[column])
    out = pd.DataFrame({"participant_id": frame["participant_id"]})
    out["ukb_last_contact_date"] = frame[contact].max(axis=1) if contact else pd.NaT
    out["last_assessment_date"] = frame[assessment].max(axis=1) if assessment else pd.NaT
    out["lost_to_followup_date"] = frame[lost].min(axis=1) if lost else pd.NaT
    out["date1_followup_end"] = out[["ukb_last_contact_date", "last_assessment_date"]].max(axis=1)
    return out[out["participant_id"].ne("")]


def load_gp_registration(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    frame["participant_id"] = normalize_participant_id(frame["Participant ID"])
    frame["gp_registration_date"] = parse_date(frame.get("Registration date"))
    frame["gp_deduction_date"] = parse_date(frame.get("Deduction date"))
    return (
        frame[frame["participant_id"].ne("")]
        .groupby("participant_id", as_index=False)
        .agg(
            gp_registration_date=("gp_registration_date", "min"),
            gp_deduction_date=("gp_deduction_date", "max"),
        )
    )


def source_identity(path: Path, *, sha256: bool) -> dict[str, object]:
    stat = path.stat()
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        identity["sha256"] = digest.hexdigest()
    return identity


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the future-independent Revision-0710 age cohort.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hash-sources", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        name: Path(config[name])
        for name in [
            "assessment_csv",
            "phenotype_csv",
            "death_csv",
            "visit_followup_csv",
            "gp_registration_csv",
        ]
    }
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing configured sources: {missing_paths}")

    print("[1/5] Loading assessment index", flush=True)
    assessment = load_assessment(paths["assessment_csv"])
    print("[2/5] Loading raw 78-feature phenotype table", flush=True)
    phenotype, feature_columns = load_phenotype(paths["phenotype_csv"])
    expected_features = int(config["expected_phenotype_features"])
    if len(feature_columns) != expected_features:
        raise ValueError(f"Expected {expected_features} phenotype features, found {len(feature_columns)}")

    print("[3/5] Building future-independent age cohort", flush=True)
    cohort = build_age_model_cohort(
        assessment,
        phenotype,
        feature_columns=feature_columns,
        max_missing_count=int(config["max_phenotype_missing_count"]),
    )
    expected_cohort = int(config["expected_age_model_cohort"])
    if len(cohort) != expected_cohort and not args.allow_count_mismatch:
        raise ValueError(f"Expected age cohort {expected_cohort:,}, found {len(cohort):,}")

    cohort_ids = set(cohort["participant_id"])
    raw_phenotype = phenotype[phenotype["participant_id"].isin(cohort_ids)].copy()
    raw_phenotype = raw_phenotype.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    missing_mask = raw_phenotype[["participant_id"]].copy()
    for column in feature_columns:
        missing_mask[column] = raw_phenotype[column].isna()

    print("[4/5] Loading death and follow-up source dates", flush=True)
    followup = (
        cohort[["participant_id", "index_date"]]
        .merge(load_death(paths["death_csv"]), on="participant_id", how="left", validate="one_to_one")
        .merge(load_visit_followup(paths["visit_followup_csv"]), on="participant_id", how="left", validate="one_to_one")
        .merge(load_gp_registration(paths["gp_registration_csv"]), on="participant_id", how="left", validate="one_to_one")
    )

    print("[5/5] Writing immutable source manifests", flush=True)
    write_parquet(cohort, output / "age_model_cohort.parquet")
    write_parquet(raw_phenotype, output / "raw_phenotypes.parquet")
    write_parquet(missing_mask, output / "phenotype_missing_mask.parquet")
    write_parquet(followup, output / "followup_source_dates.parquet")
    (output / "phenotype_feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    age_diff = cohort["chronological_age"] - raw_phenotype.set_index("participant_id").loc[
        cohort["participant_id"], "reported_age_at_recruitment"
    ].to_numpy()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "age_model_cohort_future_independent": True,
        "source_participants": int(assessment["participant_id"].nunique()),
        "phenotype_participants": int(phenotype["participant_id"].nunique()),
        "phenotype_feature_count": len(feature_columns),
        "age_model_cohort_n": int(len(cohort)),
        "missing_count_max_in_cohort": int(cohort["phenotype_missing_count"].max()),
        "chronological_age_min": float(cohort["chronological_age"].min()),
        "chronological_age_max": float(cohort["chronological_age"].max()),
        "chronological_minus_reported_age_mean": float(np.nanmean(age_diff)),
        "chronological_minus_reported_age_max_abs": float(np.nanmax(np.abs(age_diff))),
        "future_outcome_fields_used_for_age_eligibility": [],
    }
    (output / "age_cohort_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    identities = {
        name: source_identity(path, sha256=args.hash_sources) for name, path in paths.items()
    }
    (output / "source_identities.json").write_text(
        json.dumps(identities, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
