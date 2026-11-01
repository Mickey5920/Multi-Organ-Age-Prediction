from __future__ import annotations

import argparse
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leakage_safe_core import audit_preindex_events
from source_manifest_core import normalize_participant_id


ROOT = Path(__file__).resolve().parents[3]
REVISION_DIR = ROOT / "Revision-0710"
DEFAULT_CONFIG = REVISION_DIR / "01_data_rebuild" / "00_config" / "source_paths.json"
DEFAULT_INDEX_DIR = REVISION_DIR / "01_data_rebuild" / "01_index_and_events"
SYSTEMS = [
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
]


def parse_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def normalize_icd(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .str.strip()
    )


def load_icd_mapping(path: Path) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    namespace = runpy.run_path(str(path))
    raw = namespace.get("chronic_diseases")
    if not isinstance(raw, dict):
        raise ValueError(f"No chronic_diseases dictionary in {path}")
    mapping: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, str]] = []
    for system in SYSTEMS:
        normalized = normalize_icd(pd.Series(list(raw.get(system, [])), dtype="string"))
        prefixes = tuple(sorted(set(normalized[normalized.ne("")]), key=lambda value: (-len(value), value)))
        mapping[system] = prefixes
        rows.extend(
            {"organ_system": system, "icd_prefix_normalized": prefix}
            for prefix in prefixes
        )
    map_frame = pd.DataFrame(rows)
    mapped_systems = (
        map_frame.groupby("icd_prefix_normalized")["organ_system"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("mapped_organ_systems")
    )
    map_frame = map_frame.merge(mapped_systems, on="icd_prefix_normalized", how="left")
    map_frame["is_multi_mapped_prefix"] = map_frame["mapped_organ_systems"].str.contains(";")
    return mapping, map_frame


def load_icd_mapping_csv(path: Path) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {"organ_system", "icd_prefix_normalized"}
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Mapping CSV missing required columns {missing_columns}: {path}")

    frame = raw[["organ_system", "icd_prefix_normalized"]].copy()
    frame["organ_system"] = frame["organ_system"].astype("string").str.strip()
    frame["icd_prefix_normalized"] = normalize_icd(frame["icd_prefix_normalized"])
    if frame.empty:
        raise ValueError(f"Mapping CSV is empty: {path}")
    if frame["organ_system"].eq("").any():
        raise ValueError(f"Mapping CSV contains an empty organ_system: {path}")
    if frame["icd_prefix_normalized"].eq("").any():
        raise ValueError(f"Mapping CSV contains an empty ICD prefix: {path}")

    observed_systems = set(frame["organ_system"])
    missing_systems = sorted(set(SYSTEMS) - observed_systems)
    unknown_systems = sorted(observed_systems - set(SYSTEMS))
    if missing_systems or unknown_systems:
        raise ValueError(
            "Mapping CSV must contain exactly the seven organ systems; "
            f"missing={missing_systems}, unknown={unknown_systems}: {path}"
        )

    duplicate_pairs = frame.duplicated(["organ_system", "icd_prefix_normalized"], keep=False)
    if duplicate_pairs.any():
        duplicates = (
            frame.loc[duplicate_pairs, ["organ_system", "icd_prefix_normalized"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Mapping CSV contains duplicate organ-system/prefix rows {duplicates}: {path}")

    mapping: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, str]] = []
    for system in SYSTEMS:
        prefixes = tuple(
            sorted(
                frame.loc[frame["organ_system"].eq(system), "icd_prefix_normalized"],
                key=lambda value: (-len(value), value),
            )
        )
        mapping[system] = prefixes
        rows.extend(
            {"organ_system": system, "icd_prefix_normalized": prefix}
            for prefix in prefixes
        )

    map_frame = pd.DataFrame(rows)
    mapped_systems = (
        map_frame.groupby("icd_prefix_normalized")["organ_system"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("mapped_organ_systems")
    )
    map_frame = map_frame.merge(mapped_systems, on="icd_prefix_normalized", how="left")
    map_frame["is_multi_mapped_prefix"] = map_frame["mapped_organ_systems"].str.contains(";")
    return mapping, map_frame


def code_bitmask(code: str, mapping: dict[str, tuple[str, ...]]) -> int:
    if not code:
        return 0
    result = 0
    for index, system in enumerate(SYSTEMS):
        if code.startswith(mapping[system]):
            result |= 1 << index
    return result


def collapse_count_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["participant_id"])
    frame = pd.concat(parts, ignore_index=True)
    columns = [column for column in frame.columns if column != "participant_id"]
    return frame.groupby("participant_id", as_index=False, sort=False)[columns].sum()


def collapse_date_parts(parts: list[pd.DataFrame], operation: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["participant_id"])
    frame = pd.concat(parts, ignore_index=True)
    grouped = frame.groupby("participant_id", as_index=False, sort=False)
    if operation == "min":
        return grouped.min()
    if operation == "max":
        return grouped.max()
    raise ValueError(f"Unsupported date operation: {operation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild canonical strict-preindex ICD events from long source rows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--mapping-csv", type=Path)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--allow-incomplete-source", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = Path(config["icd_long_csv"])
    if args.mapping_csv is not None:
        mapping_path = args.mapping_csv
        if not mapping_path.is_absolute():
            mapping_path = ROOT / mapping_path
        mapping_loader = load_icd_mapping_csv
    else:
        mapping_path = Path(config["icd_mapping_script"])
        if not mapping_path.is_absolute():
            mapping_path = ROOT / mapping_path
        mapping_loader = load_icd_mapping
    if not source.exists() or not mapping_path.exists():
        raise FileNotFoundError(f"Missing ICD source or mapping: {source}, {mapping_path}")

    cohort = pd.read_parquet(
        args.index_dir / "age_model_cohort.parquet",
        columns=["participant_id", "index_date"],
    )
    cohort["participant_id"] = normalize_participant_id(cohort["participant_id"])
    cohort["index_date"] = pd.to_datetime(cohort["index_date"], errors="coerce")
    index_map = cohort.set_index("participant_id")["index_date"]
    mapping, mapping_frame = mapping_loader(mapping_path)
    mapping_frame.to_csv(args.index_dir / "icd_mapping_export.csv", index=False)

    usecols = ["Participant ID", "ICD", "Disease_Description_Feature", "record_time"]
    count_parts: list[pd.DataFrame] = []
    last_date_parts: list[pd.DataFrame] = []
    first_event_parts: list[pd.DataFrame] = []
    preindex_parts: list[pd.DataFrame] = []
    bitmask_cache: dict[str, int] = {"": 0}
    observed_codes: set[str] = set()
    total_rows = 0
    cohort_rows = 0
    global_counts = {
        "empty_icd_rows": 0,
        "undated_nonempty_icd_rows": 0,
        "missing_index_rows": 0,
        "strict_preindex_rows_before_dedup": 0,
        "same_day_rows": 0,
        "post_index_rows": 0,
    }

    reader = pd.read_csv(source, usecols=usecols, dtype=str, chunksize=args.chunksize, low_memory=False)
    for chunk_number, chunk in enumerate(reader, start=1):
        if args.max_chunks is not None and chunk_number > args.max_chunks:
            break
        row_count = len(chunk)
        source_rows = np.arange(total_rows + 2, total_rows + row_count + 2, dtype=np.int64)
        total_rows += row_count
        chunk = chunk.rename(
            columns={
                "Participant ID": "participant_id",
                "ICD": "code",
                "Disease_Description_Feature": "description",
                "record_time": "event_date",
            }
        )
        chunk["source_row_number"] = source_rows
        chunk["participant_id"] = normalize_participant_id(chunk["participant_id"])
        chunk["index_date"] = chunk["participant_id"].map(index_map)
        missing_index = chunk["index_date"].isna()
        global_counts["missing_index_rows"] += int(missing_index.sum())
        chunk = chunk.loc[~missing_index].copy()
        cohort_rows += len(chunk)
        chunk["event_date"] = parse_date(chunk["event_date"])
        chunk["code"] = normalize_icd(chunk["code"])
        chunk["description"] = chunk["description"].astype("string").fillna("").str.strip()
        empty_code = chunk["code"].eq("")
        undated = chunk["event_date"].isna()
        valid = ~empty_code & ~undated
        global_counts["empty_icd_rows"] += int(empty_code.sum())
        global_counts["undated_nonempty_icd_rows"] += int((~empty_code & undated).sum())
        relative_days = (chunk["event_date"] - chunk["index_date"]).dt.days
        model_input = valid & relative_days.lt(0)
        prevalent = valid & relative_days.le(0)
        same_day = valid & relative_days.eq(0)
        incident = valid & relative_days.gt(0)
        global_counts["strict_preindex_rows_before_dedup"] += int(model_input.sum())
        global_counts["same_day_rows"] += int(same_day.sum())
        global_counts["post_index_rows"] += int(incident.sum())

        unique_codes = chunk.loc[valid, "code"].unique().tolist()
        observed_codes.update(unique_codes)
        for code in unique_codes:
            if code not in bitmask_cache:
                bitmask_cache[code] = code_bitmask(code, mapping)
        masks = chunk["code"].map(bitmask_cache).fillna(0).astype(np.uint16)

        count_frame = pd.DataFrame(
            {
                "participant_id": chunk["participant_id"],
                "source_row_count": np.ones(len(chunk), dtype=np.int32),
                "valid_icd_count": valid.astype(np.int32),
                "empty_icd_count": empty_code.astype(np.int32),
                "undated_nonempty_icd_count": (~empty_code & undated).astype(np.int32),
                "model_input_icd_count": model_input.astype(np.int32),
                "prevalent_any_icd_count": prevalent.astype(np.int32),
                "same_day_icd_count": same_day.astype(np.int32),
                "incident_any_icd_count": incident.astype(np.int32),
            }
        )
        first_event = pd.DataFrame(
            {
                "participant_id": chunk["participant_id"],
                "first_incident_any_date": chunk["event_date"].where(incident),
            }
        )
        for index, system in enumerate(SYSTEMS):
            matched = (masks & np.uint16(1 << index)).ne(0)
            count_frame[f"prevalent_{system}_count"] = (prevalent & matched).astype(np.int32)
            count_frame[f"incident_{system}_count"] = (incident & matched).astype(np.int32)
            first_event[f"first_incident_date_{system}"] = chunk["event_date"].where(incident & matched)
        count_parts.append(count_frame.groupby("participant_id", as_index=False, sort=False).sum())
        first_event_parts.append(first_event.groupby("participant_id", as_index=False, sort=False).min())
        last_date_parts.append(
            pd.DataFrame(
                {
                    "participant_id": chunk["participant_id"],
                    "last_icd_date": chunk["event_date"].where(valid),
                }
            ).groupby("participant_id", as_index=False, sort=False).max()
        )

        selected = chunk.loc[
            model_input,
            ["participant_id", "event_date", "code", "description", "source_row_number"],
        ].copy()
        selected["source"] = "hospital_icd"
        selected["relative_days"] = relative_days.loc[model_input].astype(np.int32)
        selected["attention_mask"] = np.int8(1)
        preindex_parts.append(selected)
        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"ICD chunks={chunk_number}; source rows={total_rows:,}; "
                f"cohort rows={cohort_rows:,}; strict preindex={global_counts['strict_preindex_rows_before_dedup']:,}",
                flush=True,
            )

    if args.max_chunks is None and total_rows != 7_071_243 and not args.allow_incomplete_source:
        raise ValueError(f"Expected 7,071,243 ICD source rows, scanned {total_rows:,}")

    counts = collapse_count_parts(count_parts)
    first_events = collapse_date_parts(first_event_parts, "min")
    last_dates = collapse_date_parts(last_date_parts, "max")
    summary = cohort.merge(counts, on="participant_id", how="left", validate="one_to_one")
    summary = summary.merge(first_events, on="participant_id", how="left", validate="one_to_one")
    summary = summary.merge(last_dates, on="participant_id", how="left", validate="one_to_one")
    count_columns = [column for column in summary.columns if column.endswith("_count")]
    summary[count_columns] = summary[count_columns].fillna(0).astype(np.int32)
    summary["has_model_input_icd"] = summary["model_input_icd_count"].gt(0)
    summary["has_prevalent_any_icd"] = summary["prevalent_any_icd_count"].gt(0)
    summary["has_incident_any_icd"] = summary["incident_any_icd_count"].gt(0)
    for system in SYSTEMS:
        summary[f"has_prevalent_{system}"] = summary[f"prevalent_{system}_count"].gt(0)

    canonical = pd.concat(preindex_parts, ignore_index=True)
    canonical = canonical.sort_values(
        ["participant_id", "event_date", "code", "description", "source", "source_row_number"],
        kind="mergesort",
    )
    before_dedup = len(canonical)
    tuple_columns = ["participant_id", "event_date", "code", "description", "source"]
    canonical = canonical.drop_duplicates(tuple_columns, keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(canonical)
    canonical.to_parquet(
        args.index_dir / "canonical_preindex_icd.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_parquet(
        args.index_dir / "icd_participant_summary.parquet",
        index=False,
        compression="zstd",
    )
    observed_code_frame = pd.DataFrame(
        {
            "icd_code_normalized": sorted(observed_codes),
        }
    )
    observed_code_frame["organ_bitmask"] = observed_code_frame["icd_code_normalized"].map(bitmask_cache)
    observed_code_frame["mapped_organ_systems"] = observed_code_frame["organ_bitmask"].map(
        lambda value: ";".join(
            system for index, system in enumerate(SYSTEMS) if int(value) & (1 << index)
        )
    )
    observed_code_frame.to_csv(args.index_dir / "observed_icd_code_system_matches.csv", index=False)

    leakage = audit_preindex_events(canonical)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": leakage["status"],
        "source": str(source.resolve()),
        "source_rows_scanned": int(total_rows),
        "age_cohort_source_rows": int(cohort_rows),
        **global_counts,
        "strict_preindex_rows_after_exact_dedup": int(len(canonical)),
        "exact_tuple_duplicates_removed": int(duplicates_removed),
        "participants_with_model_input_icd": int(summary["has_model_input_icd"].sum()),
        "participants_with_prevalent_any_icd": int(summary["has_prevalent_any_icd"].sum()),
        "observed_unique_icd_codes": int(len(observed_codes)),
        "mapped_unique_icd_codes": int(observed_code_frame["mapped_organ_systems"].ne("").sum()),
        "leakage_gate": leakage,
    }
    (args.index_dir / "icd_rebuild_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError("ICD leakage audit failed")


if __name__ == "__main__":
    main()
