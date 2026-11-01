from __future__ import annotations

import argparse
import json
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


def parse_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def collapse_count_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(parts, ignore_index=True)
    columns = [column for column in frame.columns if column != "participant_id"]
    return frame.groupby("participant_id", as_index=False, sort=False)[columns].sum()


def collapse_last_dates(parts: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(parts, ignore_index=True)
    return frame.groupby("participant_id", as_index=False, sort=False)["last_gp_date"].max()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict-preindex GP prescription events.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--allow-incomplete-source", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = Path(config["gp_prescription_csv"])
    retention_limit = int(config["gp_source_retention_limit"])
    expected_rows = int(config["expected_gp_source_rows"])
    if retention_limit <= 0:
        raise ValueError("gp_source_retention_limit must be positive")
    if not source.exists():
        raise FileNotFoundError(source)

    cohort = pd.read_parquet(
        args.index_dir / "age_model_cohort.parquet",
        columns=["participant_id", "index_date"],
    )
    cohort["participant_id"] = normalize_participant_id(cohort["participant_id"])
    cohort["index_date"] = pd.to_datetime(cohort["index_date"], errors="coerce")
    index_map = cohort.set_index("participant_id")["index_date"]

    count_parts: list[pd.DataFrame] = []
    last_date_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    total_rows = 0
    cohort_rows = 0
    global_counts = {
        "empty_drug_rows": 0,
        "undated_nonempty_drug_rows": 0,
        "missing_index_rows": 0,
        "strict_preindex_rows_before_dedup": 0,
        "same_day_rows": 0,
        "post_index_rows": 0,
    }

    usecols = ["Participant ID", "Drug name", "Date prescription was issued"]
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
                "Drug name": "description",
                "Date prescription was issued": "event_date",
            }
        )
        chunk["source_row_number"] = source_rows
        chunk["participant_id"] = normalize_participant_id(chunk["participant_id"])
        chunk["index_date"] = chunk["participant_id"].map(index_map)
        missing_index = chunk["index_date"].isna()
        global_counts["missing_index_rows"] += int(missing_index.sum())
        chunk = chunk.loc[~missing_index].copy()
        cohort_rows += len(chunk)
        chunk["description"] = chunk["description"].astype("string").fillna("").str.strip()
        chunk["event_date"] = parse_date(chunk["event_date"])
        empty_drug = chunk["description"].eq("")
        undated = chunk["event_date"].isna()
        valid = ~empty_drug & ~undated
        relative_days = (chunk["event_date"] - chunk["index_date"]).dt.days
        preindex = valid & relative_days.lt(0)
        same_day = valid & relative_days.eq(0)
        post_index = valid & relative_days.gt(0)
        global_counts["empty_drug_rows"] += int(empty_drug.sum())
        global_counts["undated_nonempty_drug_rows"] += int((~empty_drug & undated).sum())
        global_counts["strict_preindex_rows_before_dedup"] += int(preindex.sum())
        global_counts["same_day_rows"] += int(same_day.sum())
        global_counts["post_index_rows"] += int(post_index.sum())

        counts = pd.DataFrame(
            {
                "participant_id": chunk["participant_id"],
                "gp_source_row_count": np.ones(len(chunk), dtype=np.int32),
                "valid_gp_count": valid.astype(np.int32),
                "model_input_gp_count": preindex.astype(np.int32),
                "same_day_gp_count": same_day.astype(np.int32),
                "post_index_gp_count": post_index.astype(np.int32),
                "empty_drug_count": empty_drug.astype(np.int32),
                "undated_nonempty_drug_count": (~empty_drug & undated).astype(np.int32),
            }
        )
        count_parts.append(counts.groupby("participant_id", as_index=False, sort=False).sum())
        last_date_parts.append(
            pd.DataFrame(
                {
                    "participant_id": chunk["participant_id"],
                    "last_gp_date": chunk["event_date"].where(valid),
                }
            ).groupby("participant_id", as_index=False, sort=False).max()
        )

        selected = chunk.loc[
            preindex,
            ["participant_id", "event_date", "description", "source_row_number"],
        ].copy()
        selected["code"] = ""
        selected["source"] = "gp_prescription"
        selected["relative_days"] = relative_days.loc[preindex].astype(np.int32)
        selected["attention_mask"] = np.int8(1)
        selected = selected.sort_values(
            ["participant_id", "event_date", "description", "source_row_number"],
            kind="mergesort",
        )
        selected = selected.drop_duplicates(
            ["participant_id", "event_date", "description", "source"], keep="first"
        )
        selected = selected.groupby("participant_id", sort=False, group_keys=False).tail(retention_limit)
        candidate_parts.append(selected)
        if chunk_number == 1 or chunk_number % 5 == 0:
            print(
                f"GP chunks={chunk_number}; source rows={total_rows:,}; cohort rows={cohort_rows:,}; "
                f"strict preindex={global_counts['strict_preindex_rows_before_dedup']:,}",
                flush=True,
            )

    if args.max_chunks is None and total_rows != expected_rows and not args.allow_incomplete_source:
        raise ValueError(f"Expected {expected_rows:,} GP source rows, scanned {total_rows:,}")

    counts = collapse_count_parts(count_parts)
    last_dates = collapse_last_dates(last_date_parts)
    summary = cohort.merge(counts, on="participant_id", how="left", validate="one_to_one")
    summary = summary.merge(last_dates, on="participant_id", how="left", validate="one_to_one")
    count_columns = [column for column in summary.columns if column.endswith("_count")]
    summary[count_columns] = summary[count_columns].fillna(0).astype(np.int32)
    summary["has_model_input_gp"] = summary["model_input_gp_count"].gt(0)

    canonical = pd.concat(candidate_parts, ignore_index=True)
    canonical = canonical.sort_values(
        ["participant_id", "event_date", "description", "source", "source_row_number"],
        kind="mergesort",
    )
    before_dedup = len(canonical)
    canonical = canonical.drop_duplicates(
        ["participant_id", "event_date", "description", "source"], keep="first"
    )
    canonical = canonical.groupby("participant_id", sort=False, group_keys=False).tail(retention_limit)
    canonical = canonical.sort_values(
        ["participant_id", "event_date", "description", "source_row_number"], kind="mergesort"
    ).reset_index(drop=True)
    duplicates_removed = before_dedup - len(canonical)
    selected_counts = canonical.groupby("participant_id").size()
    raw_counts = summary.set_index("participant_id")["model_input_gp_count"]
    truncated = raw_counts.gt(retention_limit)

    canonical.to_parquet(
        args.index_dir / f"canonical_preindex_gp_l{retention_limit}.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_parquet(
        args.index_dir / "gp_participant_summary.parquet",
        index=False,
        compression="zstd",
    )
    leakage = audit_preindex_events(canonical)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": leakage["status"],
        "source": str(source.resolve()),
        "source_rows_scanned": int(total_rows),
        "age_cohort_source_rows": int(cohort_rows),
        **global_counts,
        "retention_limit": retention_limit,
        "retained_rows": int(len(canonical)),
        "candidate_or_exact_duplicates_removed": int(duplicates_removed),
        "participants_with_model_input_gp": int(summary["has_model_input_gp"].sum()),
        "participants_truncated": int(truncated.sum()),
        "raw_rows_beyond_retention_limit": int((raw_counts - retention_limit).clip(lower=0).sum()),
        "max_selected_rows_per_participant": int(selected_counts.max()) if len(selected_counts) else 0,
        "leakage_gate": leakage,
    }
    (args.index_dir / "gp_rebuild_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError("GP leakage audit failed")


if __name__ == "__main__":
    main()
