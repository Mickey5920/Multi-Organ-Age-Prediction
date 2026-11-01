from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
REVISION_ROOT = REPO_ROOT / "Revision-0710"
OUTPUT_ROOT = (
    REVISION_ROOT
    / "Final_Fig_0716"
    / "Reviewer_Statistics"
    / "20_r3_11_temporal_history_coverage"
)
SOURCE_DATA = OUTPUT_ROOT / "Source_Data"

INDEX_ROOT = REVISION_ROOT / "01_data_rebuild" / "01_index_and_events"
CACHE_ROOT = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
ICD_SUMMARY = INDEX_ROOT / "icd_participant_summary.parquet"
GP_SUMMARY = INDEX_ROOT / "gp_participant_summary.parquet"
BATCH_MANIFEST = CACHE_ROOT / "batch_manifest.json"
MODALITY_AUDIT = CACHE_ROOT / "modality_starvation_audit.json"
CACHE_CONTRACT = REVISION_ROOT / "00_contract" / "CACHE_CONTRACT.md"

EXPECTED_PARTICIPANTS = 456_773
PRIMARY_LENGTH = 32
STORAGE_LENGTH = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "role": role,
        "path": path.as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def distribution(values: pd.Series, stage: str, definition: str) -> dict[str, object]:
    array = values.to_numpy(dtype=np.float64)
    quantiles = np.quantile(
        array,
        [0.00, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99, 1.00],
    )
    return {
        "stage": stage,
        "definition": definition,
        "n": int(len(array)),
        "sum": int(array.sum()),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q80": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q95": float(quantiles[6]),
        "q99": float(quantiles[7]),
        "max": float(quantiles[8]),
        "zero_n": int((array == 0).sum()),
    }


def describe(values: pd.Series) -> dict[str, object]:
    array = values.to_numpy(dtype=np.float64)
    quantiles = np.quantile(array, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": int(len(array)),
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)

    icd = pd.read_parquet(
        ICD_SUMMARY,
        columns=["participant_id", "model_input_icd_count"],
    )
    gp = pd.read_parquet(
        GP_SUMMARY,
        columns=["participant_id", "model_input_gp_count"],
    )
    for frame in (icd, gp):
        frame["participant_id"] = frame["participant_id"].astype(str)
    preindex = icd.merge(gp, on="participant_id", validate="one_to_one")
    preindex["raw_preindex_candidate_count"] = (
        preindex["model_input_icd_count"].astype(np.int64)
        + preindex["model_input_gp_count"].astype(np.int64)
    )

    batch_manifest = json.loads(BATCH_MANIFEST.read_text(encoding="utf-8"))
    modality_audit = json.loads(MODALITY_AUDIT.read_text(encoding="utf-8"))
    participant_ids: list[np.ndarray] = []
    primary_counts: list[np.ndarray] = []
    storage_counts: list[np.ndarray] = []
    primary_oldest_days: list[np.ndarray] = []
    storage_oldest_days: list[np.ndarray] = []
    shard_records: list[dict[str, object]] = []

    for entry in batch_manifest:
        path = CACHE_ROOT / entry["path"]
        actual_sha = sha256(path)
        if actual_sha != entry["sha256"]:
            raise RuntimeError(f"cache shard hash mismatch: {path}")
        with np.load(path) as shard:
            participant_id = shard["participant_id"].astype(str)
            primary_count = shard["event_count"].astype(np.int64)
            storage_count = shard["storage_event_count"].astype(np.int64)
            primary_mask = shard["event_valid_mask"].astype(bool)
            storage_mask = shard["storage_event_valid_mask"].astype(bool)
            primary_time = shard["event_relative_days"].astype(np.int64)
            storage_time = shard["storage_event_relative_days"].astype(np.int64)

            if not np.array_equal(primary_count, primary_mask.sum(axis=1)):
                raise RuntimeError(f"primary count/mask mismatch: {path}")
            if not np.array_equal(storage_count, storage_mask.sum(axis=1)):
                raise RuntimeError(f"storage count/mask mismatch: {path}")
            if np.any(primary_mask & (primary_time >= 0)):
                raise RuntimeError(f"non-preindex primary event: {path}")
            if np.any(storage_mask & (storage_time >= 0)):
                raise RuntimeError(f"non-preindex storage event: {path}")

            participant_ids.append(participant_id)
            primary_counts.append(primary_count)
            storage_counts.append(storage_count)
            primary_oldest_days.append(
                np.where(primary_mask, -primary_time, -1).max(axis=1)
            )
            storage_oldest_days.append(
                np.where(storage_mask, -storage_time, -1).max(axis=1)
            )
        shard_records.append(
            {
                "role": f"cache_shard_fold{entry['fold_id']}_batch{entry['batch_in_fold']}",
                "path": path.as_posix(),
                "sha256": actual_sha,
                "bytes": path.stat().st_size,
                "rows": int(entry["n"]),
            }
        )

    cache = pd.DataFrame(
        {
            "participant_id": np.concatenate(participant_ids),
            "primary_count": np.concatenate(primary_counts),
            "storage_count": np.concatenate(storage_counts),
            "primary_oldest_days": np.concatenate(primary_oldest_days),
            "storage_oldest_days": np.concatenate(storage_oldest_days),
        }
    )
    if len(cache) != EXPECTED_PARTICIPANTS or not cache["participant_id"].is_unique:
        raise RuntimeError("cache participant coverage failed")
    frame = preindex.merge(cache, on="participant_id", validate="one_to_one")
    if len(frame) != EXPECTED_PARTICIPANTS:
        raise RuntimeError("source summary and cache participant coverage differ")
    if not frame["primary_count"].le(frame["storage_count"]).all():
        raise RuntimeError("primary count exceeds storage count")
    if not frame["storage_count"].le(frame["raw_preindex_candidate_count"]).all():
        raise RuntimeError("storage count exceeds raw candidate count")
    if int(frame["primary_count"].max()) != PRIMARY_LENGTH:
        raise RuntimeError("primary length is not 32")
    if int(frame["storage_count"].max()) != STORAGE_LENGTH:
        raise RuntimeError("storage length is not 64")

    distributions = pd.DataFrame(
        [
            distribution(
                frame["raw_preindex_candidate_count"],
                "raw_preindex_candidates",
                "strictly pre-index ICD plus GP candidate rows before source-specific deduplication/retention",
            ),
            distribution(
                frame["storage_count"],
                "source_normalized_storage64",
                "source-normalized cache-eligible events after deduplication and up-to-64 storage selection",
            ),
            distribution(
                frame["primary_count"],
                "primary_model_input32",
                "events presented to the primary model after the fixed 32-event selection",
            ),
        ]
    )

    truncated = frame["storage_count"].gt(PRIMARY_LENGTH)
    any_raw = frame["raw_preindex_candidate_count"].gt(0)
    any_storage = frame["storage_count"].gt(0)
    primary_over_raw = (
        frame.loc[any_raw, "primary_count"]
        / frame.loc[any_raw, "raw_preindex_candidate_count"]
    )
    primary_over_storage = (
        frame.loc[any_storage, "primary_count"]
        / frame.loc[any_storage, "storage_count"]
    )
    truncated_primary_over_raw = (
        frame.loc[truncated, "primary_count"]
        / frame.loc[truncated, "raw_preindex_candidate_count"]
    )
    truncated_primary_over_storage = (
        frame.loc[truncated, "primary_count"]
        / frame.loc[truncated, "storage_count"]
    )

    participant_summary = {
        "participant_n": int(len(frame)),
        "any_raw_preindex_candidate_n": int(any_raw.sum()),
        "any_raw_preindex_candidate_fraction": float(any_raw.mean()),
        "raw_candidate_count_gt32_n": int(
            frame["raw_preindex_candidate_count"].gt(PRIMARY_LENGTH).sum()
        ),
        "raw_candidate_count_gt32_fraction": float(
            frame["raw_preindex_candidate_count"].gt(PRIMARY_LENGTH).mean()
        ),
        "primary_truncation_n": int(truncated.sum()),
        "primary_truncation_fraction": float(truncated.mean()),
        "no_primary_truncation_n": int((~truncated).sum()),
        "no_primary_truncation_fraction": float((~truncated).mean()),
        "primary_count_eq32_n": int(frame["primary_count"].eq(PRIMARY_LENGTH).sum()),
        "primary_count_eq32_fraction": float(
            frame["primary_count"].eq(PRIMARY_LENGTH).mean()
        ),
    }
    retention_summary = {
        "event_level_storage_over_raw": float(
            frame["storage_count"].sum()
            / frame["raw_preindex_candidate_count"].sum()
        ),
        "event_level_primary_over_raw": float(
            frame["primary_count"].sum()
            / frame["raw_preindex_candidate_count"].sum()
        ),
        "event_level_primary_over_storage": float(
            frame["primary_count"].sum() / frame["storage_count"].sum()
        ),
        "participant_level_primary_over_raw_nonzero_mean": float(
            primary_over_raw.mean()
        ),
        "participant_level_primary_over_raw_nonzero_median": float(
            primary_over_raw.median()
        ),
        "participant_level_primary_over_storage_nonzero_mean": float(
            primary_over_storage.mean()
        ),
        "participant_level_primary_over_storage_nonzero_median": float(
            primary_over_storage.median()
        ),
        "truncated_primary_over_raw_mean": float(
            truncated_primary_over_raw.mean()
        ),
        "truncated_primary_over_raw_median": float(
            truncated_primary_over_raw.median()
        ),
        "truncated_primary_over_storage_mean": float(
            truncated_primary_over_storage.mean()
        ),
        "truncated_primary_over_storage_median": float(
            truncated_primary_over_storage.median()
        ),
    }

    truncated_detail = {
        "raw_preindex_candidate_count": describe(
            frame.loc[truncated, "raw_preindex_candidate_count"]
        ),
        "source_normalized_storage_count": describe(
            frame.loc[truncated, "storage_count"]
        ),
        "primary_oldest_retained_lookback_years": describe(
            frame.loc[truncated, "primary_oldest_days"] / 365.25
        ),
        "storage_oldest_retained_lookback_years": describe(
            frame.loc[truncated, "storage_oldest_days"] / 365.25
        ),
    }

    summary = {
        "schema_version": "r3_11.temporal_history_coverage.v1",
        "count_stage_definitions": {
            "raw_preindex_candidates": "strictly pre-index ICD plus GP candidate rows before source-specific deduplication/retention",
            "source_normalized_storage64": "cache-eligible events after deduplication and the up-to-64 source-balanced storage rule",
            "primary_model_input32": "events presented to the primary model after the fixed 32-event source-balanced selection",
        },
        "participant_summary": participant_summary,
        "retention_summary": retention_summary,
        "truncated_participant_detail": truncated_detail,
        "selection_contract": {
            "storage_length": STORAGE_LENGTH,
            "primary_length": PRIMARY_LENGTH,
            "storage_minimum_when_available": {"ICD": 16, "GP": 16},
            "primary_minimum_when_available": {"ICD": 8, "GP": 8},
            "remaining_slots": "most recent events overall",
            "selection_uses_age": False,
            "selection_uses_health_status": False,
            "selection_uses_followup_outcome": False,
            "selection_uses_model_prediction": False,
        },
    }

    expected_exact = {
        "raw_sum": 18_139_823,
        "storage_sum": 7_015_500,
        "primary_sum": 4_592_686,
        "primary_truncation_n": 89_649,
        "no_primary_truncation_n": 367_124,
        "primary_count_eq32_n": 90_841,
        "zero_primary_n": 121_508,
    }
    actual_exact = {
        "raw_sum": int(frame["raw_preindex_candidate_count"].sum()),
        "storage_sum": int(frame["storage_count"].sum()),
        "primary_sum": int(frame["primary_count"].sum()),
        "primary_truncation_n": participant_summary["primary_truncation_n"],
        "no_primary_truncation_n": participant_summary["no_primary_truncation_n"],
        "primary_count_eq32_n": participant_summary["primary_count_eq32_n"],
        "zero_primary_n": int(frame["primary_count"].eq(0).sum()),
    }
    if actual_exact != expected_exact:
        raise RuntimeError(
            f"frozen temporal-history totals changed: {actual_exact} != {expected_exact}"
        )
    if participant_summary["primary_truncation_n"] != int(
        modality_audit["participants_truncated_at_primary_length"]
    ):
        raise RuntimeError("primary truncation count differs from modality audit")
    if int(frame["primary_count"].sum()) != sum(
        int(entry["active_events"]) for entry in batch_manifest
    ):
        raise RuntimeError("primary event total differs from batch manifest")
    if int(frame["storage_count"].sum()) != sum(
        int(entry["storage_active_events"]) for entry in batch_manifest
    ):
        raise RuntimeError("storage event total differs from batch manifest")

    distribution_path = SOURCE_DATA / "event_count_distribution.csv"
    truncation_path = SOURCE_DATA / "truncation_and_history_coverage.csv"
    summary_path = SOURCE_DATA / "temporal_history_coverage_summary.json"
    distributions.to_csv(distribution_path, index=False, float_format="%.12f")
    pd.DataFrame(
        [
            {
                "primary_truncation_n": participant_summary["primary_truncation_n"],
                "primary_truncation_fraction": participant_summary[
                    "primary_truncation_fraction"
                ],
                "no_primary_truncation_n": participant_summary[
                    "no_primary_truncation_n"
                ],
                "no_primary_truncation_fraction": participant_summary[
                    "no_primary_truncation_fraction"
                ],
                "truncated_raw_count_median": truncated_detail[
                    "raw_preindex_candidate_count"
                ]["median"],
                "truncated_primary_oldest_lookback_years_q25": truncated_detail[
                    "primary_oldest_retained_lookback_years"
                ]["q25"],
                "truncated_primary_oldest_lookback_years_median": truncated_detail[
                    "primary_oldest_retained_lookback_years"
                ]["median"],
                "truncated_primary_oldest_lookback_years_q75": truncated_detail[
                    "primary_oldest_retained_lookback_years"
                ]["q75"],
                "participant_primary_over_raw_nonzero_mean": retention_summary[
                    "participant_level_primary_over_raw_nonzero_mean"
                ],
                "participant_primary_over_raw_nonzero_median": retention_summary[
                    "participant_level_primary_over_raw_nonzero_median"
                ],
                "event_level_primary_over_raw": retention_summary[
                    "event_level_primary_over_raw"
                ],
                "event_level_primary_over_storage": retention_summary[
                    "event_level_primary_over_storage"
                ],
            }
        ]
    ).to_csv(truncation_path, index=False, float_format="%.12f")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    qa = {
        "status": "PASS",
        "participant_coverage": True,
        "unique_participants": True,
        "source_cache_id_alignment": True,
        "primary_count_le_storage_count": True,
        "storage_count_le_raw_candidate_count": True,
        "primary_max_32": True,
        "storage_max_64": True,
        "count_mask_parity_all_shards": True,
        "strict_preindex_time_gate_all_shards": True,
        "batch_manifest_hashes_verified": True,
        "batch_manifest_event_totals_match": True,
        "modality_audit_truncation_count_match": True,
        "frozen_expected_totals_match": True,
        "model_refit_for_this_audit": False,
        "statistical_model_fit_for_this_audit": False,
    }
    qa_path = OUTPUT_ROOT / "QA_REPORT.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    output_records = [
        file_record(distribution_path, "event_count_distribution", rows=len(distributions)),
        file_record(truncation_path, "truncation_and_history_coverage", rows=1),
        file_record(summary_path, "temporal_history_coverage_summary"),
        file_record(qa_path, "qa_report"),
    ]
    input_records = [
        file_record(ICD_SUMMARY, "participant_preindex_icd_counts", rows=len(icd)),
        file_record(GP_SUMMARY, "participant_preindex_gp_counts", rows=len(gp)),
        file_record(BATCH_MANIFEST, "cache_batch_manifest", rows=len(batch_manifest)),
        file_record(MODALITY_AUDIT, "modality_starvation_audit"),
        file_record(CACHE_CONTRACT, "fixed_length_selection_contract"),
        *shard_records,
    ]
    manifest = {
        "schema_version": "r3_11.temporal_history_coverage.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "reviewer_R3_11_fixed_length_temporal_history_coverage_audit",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "contract": {
            "participant_n": EXPECTED_PARTICIPANTS,
            "primary_length": PRIMARY_LENGTH,
            "storage_length": STORAGE_LENGTH,
            "model_refit_for_this_audit": False,
            "statistical_model_fit_for_this_audit": False,
        },
        "key_results": summary,
        "inputs": input_records,
        "script": file_record(SCRIPT_PATH, "r3_11_temporal_history_coverage_builder"),
        "outputs": output_records,
    }
    manifest_path = OUTPUT_ROOT / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: {manifest_path}")


if __name__ == "__main__":
    main()
