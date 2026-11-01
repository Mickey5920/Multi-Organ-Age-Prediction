from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = ROOT / "Submit" / "EHR-AGE-Supplement Table.xlsx"
DEFAULT_CURRENT_MAPPING = (
    ROOT
    / "Revision-0710"
    / "01_data_rebuild"
    / "01_index_and_events"
    / "icd_mapping_export.csv"
)
DEFAULT_OBSERVED_CODES = (
    ROOT
    / "Revision-0710"
    / "01_data_rebuild"
    / "01_index_and_events"
    / "observed_icd_code_system_matches.csv"
)
SYSTEM_ORDER = [
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
]
SUBMISSION_SYSTEMS = {
    "Cardiovascular": "cardiovascular",
    "Pulmonary": "pulmonary",
    "Musculoskeletal": "musculoskeletal",
    "Immune": "immune",
    "Renal": "renal",
    "Hepatic": "hepatic",
    "Metabolic": "metabolic",
}
MAPPING_COLUMNS = [
    "organ_system",
    "icd_prefix_normalized",
    "mapped_organ_systems",
    "is_multi_mapped_prefix",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_icd(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .str.strip()
    )


def _system_sort_value(series: pd.Series) -> pd.Series:
    order = {system: index for index, system in enumerate(SYSTEM_ORDER)}
    return series.map(order).fillna(len(order)).astype(int)


def sort_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_system_order"] = _system_sort_value(result["organ_system"])
    result["_prefix_length"] = result["icd_prefix_normalized"].str.len()
    result = result.sort_values(
        ["_system_order", "_prefix_length", "icd_prefix_normalized"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return result.drop(columns=["_system_order", "_prefix_length"]).reset_index(drop=True)


def add_exact_multimap_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=["mapped_organ_systems", "is_multi_mapped_prefix"], errors="ignore").copy()
    grouped = (
        result.groupby("icd_prefix_normalized", sort=False)["organ_system"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("mapped_organ_systems")
    )
    result = result.merge(grouped, on="icd_prefix_normalized", how="left", validate="many_to_one")
    result["is_multi_mapped_prefix"] = result["mapped_organ_systems"].str.contains(";", regex=False)
    return result


def load_submission(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(
        workbook,
        sheet_name="S1-System ICD",
        header=1,
        dtype="string",
        keep_default_na=False,
    )
    expected_columns = ["System", "Disease Category", "ICD Code", "Code Description"]
    if raw.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected S1 columns: {raw.columns.tolist()}")
    raw.insert(0, "source_excel_row", range(3, 3 + len(raw)))

    long = raw.copy()
    long["source_system_label"] = long["System"].replace("", pd.NA).ffill()
    long["organ_system"] = long["source_system_label"].map(SUBMISSION_SYSTEMS)
    if long["organ_system"].isna().any():
        labels = sorted(long.loc[long["organ_system"].isna(), "source_system_label"].dropna().unique())
        raise ValueError(f"Unrecognized submission systems: {labels}")
    categories = long["Disease Category"].replace("", pd.NA)
    long["disease_category"] = categories.groupby(long["organ_system"], sort=False).ffill().fillna("")
    long["icd_code_verbatim"] = long["ICD Code"].astype("string").fillna("")
    long["icd_prefix_normalized"] = normalize_icd(long["icd_code_verbatim"])
    if long["icd_prefix_normalized"].eq("").any():
        rows = long.loc[long["icd_prefix_normalized"].eq(""), "source_excel_row"].tolist()
        raise ValueError(f"Blank ICD codes in S1 rows: {rows}")
    long["code_description"] = long["Code Description"].astype("string").fillna("")
    long = long[
        [
            "source_excel_row",
            "source_system_label",
            "organ_system",
            "disease_category",
            "icd_code_verbatim",
            "icd_prefix_normalized",
            "code_description",
        ]
    ]
    long = add_exact_multimap_fields(long)

    duplicate_pairs = long.duplicated(["organ_system", "icd_prefix_normalized"], keep=False)
    if duplicate_pairs.any():
        raise ValueError("Submitted S1 contains duplicate system-prefix pairs")
    mapping = sort_mapping(long[MAPPING_COLUMNS])
    return raw, long, mapping


def load_current_mapping(path: Path) -> pd.DataFrame:
    current = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = sorted(set(MAPPING_COLUMNS) - set(current.columns))
    if missing:
        raise ValueError(f"Current mapping is missing columns: {missing}")
    current = current[MAPPING_COLUMNS].copy()
    current["organ_system"] = current["organ_system"].str.strip().str.lower()
    current["icd_prefix_normalized"] = normalize_icd(current["icd_prefix_normalized"])
    unexpected = sorted(set(current["organ_system"]) - set(SYSTEM_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected current mapping systems: {unexpected}")
    if current.duplicated(["organ_system", "icd_prefix_normalized"]).any():
        raise ValueError("Current mapping contains duplicate system-prefix pairs")
    return sort_mapping(add_exact_multimap_fields(current))


def build_corrected_primary(
    current: pd.DataFrame, submission_long: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["organ_system", "icd_prefix_normalized"]
    malformed = current["organ_system"].eq("renal") & current["icd_prefix_normalized"].eq("N299C64")
    if int(malformed.sum()) != 1:
        raise ValueError(f"Expected exactly one renal N299C64 row, found {int(malformed.sum())}")

    primary = current.loc[~malformed, ["organ_system", "icd_prefix_normalized"]].copy()
    repairs = pd.DataFrame(
        [
            {"organ_system": "renal", "icd_prefix_normalized": "N299"},
            {"organ_system": "renal", "icd_prefix_normalized": "C64"},
        ]
    )
    primary = pd.concat([primary, repairs], ignore_index=True).drop_duplicates(key)
    primary = sort_mapping(add_exact_multimap_fields(primary))

    submission_metadata = submission_long[
        [
            "organ_system",
            "icd_prefix_normalized",
            "source_excel_row",
            "icd_code_verbatim",
            "disease_category",
            "code_description",
        ]
    ].copy()
    enriched = primary.merge(submission_metadata, on=key, how="left", validate="one_to_one")
    current_keys = set(map(tuple, current[key].itertuples(index=False, name=None)))
    submission_keys = set(map(tuple, submission_metadata[key].itertuples(index=False, name=None)))

    def provenance(row: pd.Series) -> str:
        pair = (row["organ_system"], row["icd_prefix_normalized"])
        if pair in current_keys and pair in submission_keys:
            return "current_revision0710_and_submission"
        if pair in current_keys:
            return "current_revision0710_only_not_in_submission_table"
        if pair in submission_keys:
            return "implementation_correction_from_submission"
        return "implementation_correction_from_mapping_source"

    enriched["source_provenance"] = enriched.apply(provenance, axis=1)
    enriched["correction_note"] = ""
    enriched.loc[
        enriched["icd_prefix_normalized"].eq("N299") & enriched["organ_system"].eq("renal"),
        "correction_note",
    ] = "restored after splitting concatenated Python literals N29.9 and C64"
    enriched.loc[
        enriched["icd_prefix_normalized"].eq("C64") & enriched["organ_system"].eq("renal"),
        "correction_note",
    ] = "restored after splitting concatenated Python literals N29.9 and C64"
    enriched = enriched[
        MAPPING_COLUMNS
        + [
            "source_provenance",
            "source_excel_row",
            "icd_code_verbatim",
            "disease_category",
            "code_description",
            "correction_note",
        ]
    ]
    return primary, enriched


def build_single_assignment_sensitivity(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cardiovascular_c49 = primary["organ_system"].eq("cardiovascular") & primary[
        "icd_prefix_normalized"
    ].str.startswith("C49")
    metabolic_bone = primary["organ_system"].eq("metabolic") & primary[
        "icd_prefix_normalized"
    ].str.startswith(("M80", "M81", "M83"))
    excluded = primary.loc[cardiovascular_c49 | metabolic_bone, ["organ_system", "icd_prefix_normalized"]].copy()
    excluded["sensitivity_reason"] = ""
    excluded.loc[cardiovascular_c49.loc[excluded.index], "sensitivity_reason"] = (
        "assign C49 family only to musculoskeletal"
    )
    excluded.loc[metabolic_bone.loc[excluded.index], "sensitivity_reason"] = (
        "assign M80/M81/M83 families only to musculoskeletal"
    )
    sensitivity = primary.loc[~(cardiovascular_c49 | metabolic_bone), ["organ_system", "icd_prefix_normalized"]]
    sensitivity = sort_mapping(add_exact_multimap_fields(sensitivity.copy()))
    return sensitivity, sort_mapping(excluded)


def mapping_diff(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    key = ["organ_system", "icd_prefix_normalized"]
    old_pairs = set(map(tuple, old[key].itertuples(index=False, name=None)))
    new_pairs = set(map(tuple, new[key].itertuples(index=False, name=None)))
    rows: list[dict[str, str]] = []
    for organ_system, prefix in sorted(old_pairs - new_pairs):
        reason = "removed malformed concatenated prefix" if prefix == "N299C64" else "removed"
        rows.append(
            {
                "change_type": "removed",
                "organ_system": organ_system,
                "icd_prefix_normalized": prefix,
                "reason": reason,
            }
        )
    for organ_system, prefix in sorted(new_pairs - old_pairs):
        reason = (
            "restored token split from N29.9 followed by C64 without a comma"
            if organ_system == "renal" and prefix in {"N299", "C64"}
            else "added"
        )
        rows.append(
            {
                "change_type": "added",
                "organ_system": organ_system,
                "icd_prefix_normalized": prefix,
                "reason": reason,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["change_type", "organ_system", "icd_prefix_normalized", "reason"],
    )


def submission_primary_diff(submission: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    diff = mapping_diff(submission, primary)
    if diff.empty:
        return diff
    diff["reason"] = diff.apply(
        lambda row: (
            "pre-existing Revision-0710 runtime-only mapping; retained without semantic adjudication"
            if row["change_type"] == "added"
            and not (
                row["organ_system"] == "renal"
                and row["icd_prefix_normalized"] in {"N299", "C64"}
            )
            else row["reason"]
        ),
        axis=1,
    )
    return diff


def malformed_audit(versioned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for version, frame in versioned.items():
        for row in frame[["organ_system", "icd_prefix_normalized"]].itertuples(index=False):
            prefix = row.icd_prefix_normalized
            reasons: list[str] = []
            if re.fullmatch(r"[A-Z][0-9]{2}[A-Z0-9]{0,4}", prefix) is None:
                reasons.append("outside_normalized_icd_prefix_shape")
            if len(re.findall(r"[A-Z]", prefix)) > 1:
                reasons.append("contains_multiple_alpha_stems")
            if reasons:
                rows.append(
                    {
                        "version": version,
                        "organ_system": row.organ_system,
                        "icd_prefix_normalized": prefix,
                        "audit_reason": ";".join(reasons),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["version", "organ_system", "icd_prefix_normalized", "audit_reason"],
    )


def exact_duplicate_audit(versioned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, frame in versioned.items():
        counts = (
            frame.groupby(["organ_system", "icd_prefix_normalized"], as_index=False)
            .size()
            .rename(columns={"size": "duplicate_count"})
        )
        counts = counts[counts["duplicate_count"].gt(1)]
        for row in counts.itertuples(index=False):
            rows.append(
                {
                    "version": version,
                    "organ_system": row.organ_system,
                    "icd_prefix_normalized": row.icd_prefix_normalized,
                    "duplicate_count": int(row.duplicate_count),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["version", "organ_system", "icd_prefix_normalized", "duplicate_count"],
    )


def exact_multimap_audit(versioned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, frame in versioned.items():
        grouped = frame.groupby("icd_prefix_normalized")["organ_system"].agg(
            lambda values: sorted(set(values))
        )
        for prefix, systems in grouped.items():
            if len(systems) > 1:
                rows.append(
                    {
                        "version": version,
                        "icd_prefix_normalized": prefix,
                        "mapped_organ_systems": ";".join(systems),
                        "organ_system_count": len(systems),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["version", "icd_prefix_normalized", "mapped_organ_systems", "organ_system_count"],
    ).sort_values(["version", "icd_prefix_normalized"], kind="mergesort", ignore_index=True)


def prefix_overlap_audit(versioned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for version, frame in versioned.items():
        prefix_systems = (
            frame.groupby("icd_prefix_normalized")["organ_system"]
            .agg(lambda values: sorted(set(values)))
            .to_dict()
        )
        seen: set[tuple[str, str, str, str]] = set()
        for longer, longer_systems in prefix_systems.items():
            for length in range(3, len(longer) + 1):
                shorter = longer[:length]
                if shorter not in prefix_systems:
                    continue
                for system_long in longer_systems:
                    for system_short in prefix_systems[shorter]:
                        if system_long == system_short:
                            continue
                        pair_a = (system_short, shorter)
                        pair_b = (system_long, longer)
                        if pair_b < pair_a:
                            pair_a, pair_b = pair_b, pair_a
                        key = (*pair_a, *pair_b)
                        if key in seen:
                            continue
                        seen.add(key)
                        if pair_a[1] == pair_b[1]:
                            relation = "exact_prefix_in_multiple_systems"
                        elif pair_b[1].startswith(pair_a[1]):
                            relation = "prefix_a_is_prefix_of_b"
                        else:
                            relation = "prefix_b_is_prefix_of_a"
                        rows.append(
                            {
                                "version": version,
                                "organ_system_a": pair_a[0],
                                "prefix_a": pair_a[1],
                                "organ_system_b": pair_b[0],
                                "prefix_b": pair_b[1],
                                "relation": relation,
                                "shared_match_root": shorter,
                            }
                        )
    return pd.DataFrame(
        rows,
        columns=[
            "version",
            "organ_system_a",
            "prefix_a",
            "organ_system_b",
            "prefix_b",
            "relation",
            "shared_match_root",
        ],
    ).sort_values(
        ["version", "organ_system_a", "prefix_a", "organ_system_b", "prefix_b"],
        kind="mergesort",
        ignore_index=True,
    )


def system_count_audit(versioned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, frame in versioned.items():
        counts = frame["organ_system"].value_counts()
        for system in SYSTEM_ORDER:
            rows.append(
                {
                    "version": version,
                    "organ_system": system,
                    "mapping_rows": int(counts.get(system, 0)),
                }
            )
        rows.append(
            {"version": version, "organ_system": "TOTAL", "mapping_rows": int(len(frame))}
        )
    return pd.DataFrame(rows)


def classify_observed_codes(
    observed_codes_path: Path,
    current: pd.DataFrame,
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    observed = pd.read_csv(observed_codes_path, dtype="string", keep_default_na=False)
    expected_columns = {"icd_code_normalized", "mapped_organ_systems"}
    missing = sorted(expected_columns - set(observed.columns))
    if missing:
        raise ValueError(f"Observed-code export is missing columns: {missing}")
    observed = observed.copy()
    observed["icd_code_normalized"] = normalize_icd(observed["icd_code_normalized"])
    if observed["icd_code_normalized"].eq("").any():
        raise ValueError("Observed-code export contains a blank normalized ICD code")
    if observed["icd_code_normalized"].duplicated().any():
        raise ValueError("Observed-code export contains duplicate normalized ICD codes")

    def classify(mapping: pd.DataFrame) -> pd.Series:
        prefixes = {
            system: tuple(
                mapping.loc[mapping["organ_system"].eq(system), "icd_prefix_normalized"].tolist()
            )
            for system in SYSTEM_ORDER
        }
        return observed["icd_code_normalized"].map(
            lambda code: ";".join(
                system for system in SYSTEM_ORDER if prefixes[system] and code.startswith(prefixes[system])
            )
        )

    current_classification = classify(current)
    stored_current = observed["mapped_organ_systems"].astype("string").fillna("")
    verification_mismatch = current_classification.ne(stored_current)
    if verification_mismatch.any():
        examples = observed.loc[
            verification_mismatch, ["icd_code_normalized", "mapped_organ_systems"]
        ].head(10)
        raise RuntimeError(
            "Current mapping does not reproduce the stored observed-code classification: "
            f"{examples.to_dict(orient='records')}"
        )

    primary_classification = classify(primary)
    changed = current_classification.ne(primary_classification)
    delta = pd.DataFrame(
        {
            "icd_code_normalized": observed.loc[changed, "icd_code_normalized"],
            "current_mapped_organ_systems": current_classification.loc[changed],
            "primary_mapped_organ_systems": primary_classification.loc[changed],
        }
    ).reset_index(drop=True)

    def system_delta(row: pd.Series, operation: str) -> str:
        old = {value for value in row["current_mapped_organ_systems"].split(";") if value}
        new = {value for value in row["primary_mapped_organ_systems"].split(";") if value}
        values = new - old if operation == "added" else old - new
        return ";".join(system for system in SYSTEM_ORDER if system in values)

    delta["added_organ_systems"] = delta.apply(system_delta, axis=1, operation="added")
    delta["removed_organ_systems"] = delta.apply(system_delta, axis=1, operation="removed")
    delta["change_type"] = delta.apply(
        lambda row: (
            "newly_mapped"
            if not row["current_mapped_organ_systems"] and row["primary_mapped_organ_systems"]
            else "newly_unmapped"
            if row["current_mapped_organ_systems"] and not row["primary_mapped_organ_systems"]
            else "mapping_changed"
        ),
        axis=1,
    )
    delta = delta.sort_values("icd_code_normalized", kind="mergesort", ignore_index=True)
    summary = {
        "observed_unique_codes": int(len(observed)),
        "current_classification_verification_mismatches": int(verification_mismatch.sum()),
        "observed_codes_with_classification_delta": int(len(delta)),
        "observed_codes_newly_mapped": int(delta["change_type"].eq("newly_mapped").sum()),
        "observed_codes_newly_unmapped": int(delta["change_type"].eq("newly_unmapped").sum()),
    }
    return delta, summary


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", na_rep="")


def build(
    output_root: Path,
    workbook: Path,
    current_mapping_path: Path,
    observed_codes_path: Path = DEFAULT_OBSERVED_CODES,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    raw, submission_long, submission_mapping = load_submission(workbook)
    current = load_current_mapping(current_mapping_path)
    primary, primary_long = build_corrected_primary(current, submission_long)
    sensitivity, sensitivity_exclusions = build_single_assignment_sensitivity(primary)
    observed_delta, observed_summary = classify_observed_codes(
        observed_codes_path, current, primary
    )

    snapshots_dir = output_root / "00_source_snapshots"
    submission_dir = output_root / "01_v0_submission"
    primary_dir = output_root / "02_v1_corrected_primary"
    sensitivity_dir = output_root / "03_sensitivity"
    audit_dir = output_root / "04_audit"

    write_csv(current, snapshots_dir / "current_revision0710_mapping_export.csv")
    write_csv(raw, submission_dir / "s1_system_icd_raw.csv")
    write_csv(submission_long, submission_dir / "codebook_long.csv")
    write_csv(submission_mapping, submission_dir / "mapping.csv")
    write_csv(primary_long, primary_dir / "codebook_long.csv")
    write_csv(primary, primary_dir / "mapping.csv")
    write_csv(sensitivity, sensitivity_dir / "mapping_single_assignment.csv")
    write_csv(sensitivity_exclusions, sensitivity_dir / "exclusions_from_primary.csv")

    versioned = {
        "current_revision0710": current,
        "v0_submission": submission_mapping,
        "v1_corrected_primary": primary,
        "sensitivity_single_assignment": sensitivity,
    }
    old_new = mapping_diff(current, primary)
    submission_diff = submission_primary_diff(submission_mapping, primary)
    malformed = malformed_audit(versioned)
    duplicates = exact_duplicate_audit(versioned)
    exact_multimap = exact_multimap_audit(versioned)
    prefix_overlap = prefix_overlap_audit(versioned)
    system_counts = system_count_audit(versioned)

    observed_changes = set(
        map(
            tuple,
            old_new[["change_type", "organ_system", "icd_prefix_normalized"]].itertuples(
                index=False, name=None
            ),
        )
    )
    expected_changes = {
        ("removed", "renal", "N299C64"),
        ("added", "renal", "N299"),
        ("added", "renal", "C64"),
    }
    if observed_changes != expected_changes:
        raise RuntimeError(f"Frozen implementation-only diff failed: {observed_changes}")
    if malformed["version"].eq("v1_corrected_primary").any():
        raise RuntimeError("Corrected primary still contains a malformed prefix")
    if duplicates["version"].eq("v1_corrected_primary").any():
        raise RuntimeError("Corrected primary contains duplicate system-prefix rows")

    write_csv(old_new, audit_dir / "old_new_diff.csv")
    write_csv(submission_diff, audit_dir / "submission_to_primary_diff.csv")
    write_csv(malformed, audit_dir / "malformed_prefix_audit.csv")
    write_csv(duplicates, audit_dir / "exact_duplicate_audit.csv")
    write_csv(exact_multimap, audit_dir / "exact_multimap_audit.csv")
    write_csv(prefix_overlap, audit_dir / "prefix_overlap_multimap_audit.csv")
    write_csv(system_counts, audit_dir / "system_counts.csv")
    write_csv(observed_delta, audit_dir / "observed_code_classification_delta.csv")

    audit_summary = {
        "current_mapping_rows": int(len(current)),
        "v0_submission_rows": int(len(submission_mapping)),
        "v1_corrected_primary_rows": int(len(primary)),
        "sensitivity_rows": int(len(sensitivity)),
        "old_new_added_rows": int(old_new["change_type"].eq("added").sum()),
        "old_new_removed_rows": int(old_new["change_type"].eq("removed").sum()),
        "current_malformed_rows": int(malformed["version"].eq("current_revision0710").sum()),
        "primary_malformed_rows": int(malformed["version"].eq("v1_corrected_primary").sum()),
        "primary_exact_duplicate_rows": int(
            duplicates["version"].eq("v1_corrected_primary").sum()
        ),
        "primary_exact_multimap_prefixes": int(
            exact_multimap["version"].eq("v1_corrected_primary").sum()
        ),
        "primary_prefix_overlap_rows": int(
            prefix_overlap["version"].eq("v1_corrected_primary").sum()
        ),
        "sensitivity_excluded_rows": int(len(sensitivity_exclusions)),
        **observed_summary,
    }
    audit_summary_path = audit_dir / "audit_summary.json"
    audit_summary_path.write_text(
        json.dumps(audit_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    stable_relative_files = [
        Path("00_source_snapshots/current_revision0710_mapping_export.csv"),
        Path("01_v0_submission/s1_system_icd_raw.csv"),
        Path("01_v0_submission/codebook_long.csv"),
        Path("01_v0_submission/mapping.csv"),
        Path("02_v1_corrected_primary/codebook_long.csv"),
        Path("02_v1_corrected_primary/mapping.csv"),
        Path("03_sensitivity/mapping_single_assignment.csv"),
        Path("03_sensitivity/exclusions_from_primary.csv"),
        Path("04_audit/old_new_diff.csv"),
        Path("04_audit/submission_to_primary_diff.csv"),
        Path("04_audit/malformed_prefix_audit.csv"),
        Path("04_audit/exact_duplicate_audit.csv"),
        Path("04_audit/exact_multimap_audit.csv"),
        Path("04_audit/prefix_overlap_multimap_audit.csv"),
        Path("04_audit/system_counts.csv"),
        Path("04_audit/observed_code_classification_delta.csv"),
        Path("04_audit/audit_summary.json"),
    ]
    if output_root.resolve() == HERE.resolve():
        stable_relative_files.extend(
            [Path("build_codebook.py"), Path("README.md"), Path("tests/test_build_codebook.py")]
        )
    generated_files = [output_root / relative for relative in stable_relative_files]
    missing_stable_files = [str(path) for path in generated_files if not path.is_file()]
    if missing_stable_files:
        raise RuntimeError(f"Stable manifest files are missing: {missing_stable_files}")
    manifest = {
        "schema_version": "icd-codebook-v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": (
            "Versioned codebook artifact only; no existing mapping, outcome, canonical data, "
            "figure source, checkpoint, or training job was modified."
        ),
        "primary_policy": (
            "Preserve the current Revision-0710 mapping and repair only the malformed Python "
            "literal concatenation N29.9 + C64."
        ),
        "inputs": {
            "submission_workbook": {
                "path": str(workbook.resolve()),
                "sha256": sha256_file(workbook),
                "sheet": "S1-System ICD",
            },
            "current_revision0710_mapping": {
                "path": str(current_mapping_path.resolve()),
                "sha256": sha256_file(current_mapping_path),
            },
            "observed_icd_code_system_matches": {
                "path": str(observed_codes_path.resolve()),
                "sha256": sha256_file(observed_codes_path),
            },
        },
        "counts": audit_summary,
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "outputs": {
            str(path.relative_to(output_root)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated_files
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the versioned EHR-AGE ICD codebook v1 artifact.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--current-mapping", type=Path, default=DEFAULT_CURRENT_MAPPING)
    parser.add_argument("--observed-codes", type=Path, default=DEFAULT_OBSERVED_CODES)
    parser.add_argument("--output-root", type=Path, default=HERE)
    args = parser.parse_args()
    manifest = build(args.output_root, args.workbook, args.current_mapping, args.observed_codes)
    print(json.dumps(manifest["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
