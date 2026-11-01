from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
V8_DIR = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/00_plan/"
    "external_validation_prefreeze_v8"
)
ARTIFACT_DIR = V8_DIR / "artifacts_v8"
WORK_DIR = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/04_external_validation/"
    "core_0716"
)
FINAL_DIR = ROOT / "Revision-0710/Final_Fig_0716"
CONTRACT_PATH = WORK_DIR / "CORE_CONTRACT_0716.json"
STATUS_PATH = WORK_DIR / "RUN_STATUS.json"

if str(V8_DIR) not in sys.path:
    sys.path.insert(0, str(V8_DIR))

import execute_external_validation_v8_runtime as v8  # noqa: E402


MIMIC_PRIMARY = v8.MIMIC_PRIMARY_VARIANT
NHANES_PRIMARY = v8.NHANES_PRIMARY_VARIANT
ORGANS = list(v8.ORGANS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def update_status(stage: str, status: str, **details: Any) -> None:
    payload: dict[str, Any]
    if STATUS_PATH.exists():
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": "revision0710.external_validation.core_0716.status.v1",
            "created_at": now_iso(),
            "stages": {},
        }
    payload["updated_at"] = now_iso()
    payload["stages"][stage] = {
        "status": status,
        "updated_at": now_iso(),
        **details,
    }
    write_json(STATUS_PATH, payload)


def load_registry() -> dict[str, Any]:
    return json.loads(
        (ARTIFACT_DIR / "SOURCE_REGISTRY_V8.json").read_text(encoding="utf-8")
    )


def source_paths() -> dict[str, Path]:
    return {
        row["source_id"]: Path(row["absolute_path"])
        for row in load_registry()["external_sources"]
    }


def locked_paths() -> dict[str, tuple[dict[str, Any], Path]]:
    output: dict[str, tuple[dict[str, Any], Path]] = {}
    for row in load_registry()["locked_artifacts"]:
        path = Path(row["path"])
        if row["path_kind"] == "repo_relative":
            path = ROOT / path
        output[row["artifact_id"]] = (row, path)
    return output


def stage_cache() -> dict[str, Path]:
    directory = WORK_DIR / "shared/02_label_blind_cache"
    return {
        "directory": directory,
        "manifest": directory / "MANIFEST.json",
        "inference_artifact_dir": directory / "inference_worker_artifacts",
        "inference_registry": directory
        / "inference_worker_artifacts/INFERENCE_LOCKED_ARTIFACT_REGISTRY_V8.json",
        "mimic_density": directory
        / "mimic_participant_density_label_blind.parquet",
        "nhanes_release_slot_qc": directory / "nhanes_release_slot_qc.csv",
    }


def stage_predictions() -> dict[str, Path]:
    directory = WORK_DIR / "shared/04_inference_work"
    return {
        "directory": directory,
        "raw_predictions": directory / "raw_per_checkpoint_predictions.csv.gz",
        "locked_registry": stage_cache()["inference_registry"],
        "vocabulary_audit": directory / "vocabulary_prefix_audit.csv",
        "vocabulary_extension_manifest": directory
        / "VOCABULARY_EXTENSION_MANIFEST.json",
    }


def stage_sealed() -> dict[str, Path]:
    directory = stage_predictions()["directory"]
    return {
        "per_checkpoint": directory / "sealed_per_checkpoint_predictions.parquet",
        "ensemble": directory / "sealed_ensemble_predictions.parquet",
        "receipt": directory / "PREDICTION_SEAL_RECEIPT.json",
        "vocabulary_audit": directory / "vocabulary_prefix_audit.csv",
        "vocabulary_extension_manifest": directory
        / "VOCABULARY_EXTENSION_MANIFEST.json",
    }


def freeze_contract() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "Fig13").mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "Fig14").mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    missing: list[str] = []
    size_mismatch: list[str] = []
    for row in registry["external_sources"]:
        path = Path(row["absolute_path"])
        if not path.is_file():
            missing.append(row["source_id"])
        elif path.stat().st_size != int(row["bytes"]):
            size_mismatch.append(row["source_id"])
    locks = locked_paths()
    checkpoint_rows = []
    for fold in range(5):
        row, path = locks[f"checkpoint_fold_{fold}"]
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise AssertionError(f"checkpoint fold {fold} hash mismatch")
        checkpoint_rows.append(
            {
                "outer_fold": fold,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed,
                "age_anchor": v8.CHECKPOINT_AGE_ANCHORS[fold][1],
            }
        )
    if missing or size_mismatch:
        raise FileNotFoundError(
            f"external source registry mismatch: missing={missing}, size={size_mismatch}"
        )
    contract = {
        "schema_version": "revision0710.external_validation.core_0716.contract.v1",
        "status": "FROZEN_BEFORE_EXTERNAL_MODEL_INFERENCE",
        "created_at": now_iso(),
        "model_policy": {
            "checkpoint_selection": "fixed UKB five-fold EHR-AGE E55 checkpoints",
            "external_fine_tuning": False,
            "external_distillation": False,
            "external_adapter_training": False,
            "external_checkpoint_or_panel_selection": False,
            "ensemble": "unweighted mean of five frozen checkpoints",
        },
        "mimic_primary": {
            "release": "MIMIC-IV 3.1",
            "landmark": "second valid adult survived-discharge admission",
            "age_transfer_population": "integer age 40-73 inclusive",
            "input_window": "strictly before landmark discharge",
            "input_variant": MIMIC_PRIMARY,
            "incident_horizons_months": [6, 12],
            "competing_event": "death",
            "recurrent_analysis": "sensitivity only",
            "external_gap_dichotomization": False,
        },
        "nhanes_primary": {
            "releases": [
                "P_2017_MARCH2020_PREPANDEMIC",
                "L_AUG2021_AUG2023",
            ],
            "release_pooling": False,
            "age_population": "MEC examined age 40-73 inclusive",
            "input_variant": NHANES_PRIMARY,
            "event_sequence": "32 inactive padding slots",
            "raw_zero_shot": "primary",
            "release_local_crossfit_recalibration": "sensitivity",
            "survey_weighted_questionnaire_OR": "exploratory association",
        },
        "figures": {
            "Fig13": "MIMIC transportability and incident clinical utility",
            "Fig14": "NHANES release-specific phenotype transportability",
        },
        "source_registry_path": str(
            ARTIFACT_DIR / "SOURCE_REGISTRY_V8.json"
        ),
        "source_registry_sha256": sha256_file(
            ARTIFACT_DIR / "SOURCE_REGISTRY_V8.json"
        ),
        "external_source_count": len(registry["external_sources"]),
        "external_sources_verified_by_frozen_size_and_path": True,
        "external_sources_sha256_from_frozen_registry": True,
        "checkpoints": checkpoint_rows,
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "script": str(Path(__file__).resolve()),
        },
    }
    write_json(CONTRACT_PATH, contract)
    update_status(
        "00_freeze",
        "COMPLETE",
        contract=str(CONTRACT_PATH),
        contract_sha256=sha256_file(CONTRACT_PATH),
    )
    print(json.dumps({"contract": str(CONTRACT_PATH), "status": "COMPLETE"}))


def build_cache() -> None:
    cache = stage_cache()
    if cache["manifest"].is_file():
        update_status(
            "01_cache",
            "COMPLETE_REUSED",
            manifest=str(cache["manifest"]),
            manifest_sha256=sha256_file(cache["manifest"]),
        )
        print(json.dumps({key: str(value) for key, value in cache.items()}))
        return
    update_status("01_cache", "RUNNING")
    cache_dir = cache["directory"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = source_paths()
    mimic, admissions, mimic_flow = v8._participant_cohort(sources)
    preindex_table = cache_dir / "mimic_diagnoses_preindex_allowlist.parquet"
    preindex_receipt = cache_dir / "PREINDEX_DIAGNOSIS_MATERIALIZATION_V8.json"
    if not preindex_table.is_file() or not preindex_receipt.is_file():
        preindex = v8._materialize_preindex_diagnosis_allowlist(
            mimic,
            admissions,
            sources["mimic_diagnoses"],
            preindex_table,
        )
    else:
        preindex = {"table": preindex_table, "receipt": preindex_receipt}
    mimic_events, mimic_drop_audit = v8._mimic_diagnosis_events(
        mimic,
        admissions,
        sources,
        Path(preindex["table"]),
    )
    event_columns = list(mimic_events.columns)
    rx_events = pd.DataFrame(columns=event_columns)
    mimic_raw, mimic_missing, mimic_bare_raw, mimic_bare_missing, mimic_coverage = (
        v8._build_mimic_features(mimic, ARTIFACT_DIR, sources)
    )
    nhanes, nhanes_raw, nhanes_missing, nhanes_coverage = v8._build_nhanes_features(
        ARTIFACT_DIR, sources
    )
    nhanes_design = v8._build_nhanes_survey_design_universe(sources)

    prepared = v8.prepare_variant_timeline(mimic_events, MIMIC_PRIMARY)
    prepared["participant_id"] = prepared["participant_id"].astype(str)
    prepared["event_time"] = pd.to_datetime(prepared["event_time"], errors="raise")
    prepared["landmark_time"] = pd.to_datetime(
        prepared["landmark_time"], errors="raise"
    )
    grouped = prepared.groupby("participant_id", sort=False).agg(
        pre_crop_event_count=("event_time", "size"),
        first_event_time=("event_time", "min"),
        landmark_time=("landmark_time", "max"),
    )
    density = pd.DataFrame(
        {"subject_id": mimic["subject_id"].astype(str).to_numpy()}
    ).merge(
        grouped.reset_index().rename(columns={"participant_id": "subject_id"}),
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    density["pre_crop_event_count"] = (
        density["pre_crop_event_count"].fillna(0).astype(np.int64)
    )
    density["history_span_days"] = (
        (
            pd.to_datetime(density["landmark_time"])
            - pd.to_datetime(density["first_event_time"])
        )
        .dt.total_seconds()
        .div(86400.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    selected = v8.crop_variant_timeline(mimic_events, MIMIC_PRIMARY)
    post_counts = (
        selected.assign(participant_id=selected["participant_id"].astype(str))
        .groupby("participant_id", sort=False)
        .size()
    )
    density["input_variant"] = MIMIC_PRIMARY
    density["post_crop_event_count"] = (
        density["subject_id"].map(post_counts).fillna(0).astype(np.int64)
    )
    density["observed_phenotype_slots"] = (~mimic_missing).sum(axis=1).astype(
        np.int16
    )
    density["density_stratum"] = np.select(
        [
            density["pre_crop_event_count"].le(7),
            density["pre_crop_event_count"].between(8, 31),
        ],
        ["low_0_7", "medium_8_31"],
        default="high_32_plus",
    )
    density["density_stratum_rule"] = (
        "pre_crop_events:low=0-7;medium=8-31;high>=32"
    )
    density["l32_mask_policy"] = (
        "left_aligned_active_events_then_inactive_padding_to_32"
    )
    density["label_or_outcome_fields_read"] = False
    density = density.drop(columns=["first_event_time", "landmark_time"]).sort_values(
        ["subject_id", "input_variant"], kind="mergesort"
    )
    if (density["post_crop_event_count"] > 32).any():
        raise AssertionError("MIMIC primary density exceeds L32")

    nhanes_qc = v8.build_nhanes_release_slot_qc(
        participants=nhanes,
        raw=nhanes_raw,
        missing=nhanes_missing,
        release_slot_map=pd.read_csv(
            ARTIFACT_DIR / "NHANES_78_SLOT_RELEASE_MAP_V8.csv",
            keep_default_na=False,
        ),
    )
    v8.require_nhanes_qc_pass(nhanes_qc)
    v8._write_parquet(cache_dir / "mimic_participants.parquet", mimic)
    v8._write_parquet(cache_dir / "nhanes_participants.parquet", nhanes)
    v8._write_parquet(
        cache_dir / "nhanes_survey_design_universe.parquet", nhanes_design
    )
    v8._write_parquet(cache_dir / "mimic_icd_events_preindex.parquet", mimic_events)
    v8._write_parquet(
        cache_dir / "mimic_inpatient_prescription_proxy_preindex.parquet", rx_events
    )
    v8._write_parquet(
        cache_dir / "mimic_vocabulary_drop_audit.parquet", mimic_drop_audit
    )
    v8._write_parquet(cache_dir / "mimic_feature_coverage.parquet", mimic_coverage)
    v8._write_parquet(cache_dir / "nhanes_feature_coverage.parquet", nhanes_coverage)
    v8._write_parquet(
        cache_dir / "mimic_participant_density_label_blind.parquet", density
    )
    v8._write_csv(cache_dir / "nhanes_release_slot_qc.csv", nhanes_qc)
    np.savez_compressed(
        cache_dir / "phenotypes.npz",
        mimic_raw=mimic_raw,
        mimic_missing=mimic_missing,
        mimic_bare_raw=mimic_bare_raw,
        mimic_bare_missing=mimic_bare_missing,
        nhanes_raw=nhanes_raw,
        nhanes_missing=nhanes_missing,
    )
    bridge = cache_dir / "inference_bridge"
    bridge.mkdir(parents=True, exist_ok=False)
    v8._write_csv_gzip(bridge / "mimic_participants.csv.gz", mimic)
    v8._write_csv_gzip(bridge / "nhanes_participants.csv.gz", nhanes)
    v8._write_csv_gzip(bridge / "mimic_icd_events_preindex.csv.gz", mimic_events)
    v8._write_csv_gzip(
        bridge / "mimic_inpatient_prescription_proxy_preindex.csv.gz", rx_events
    )
    v8._write_csv_gzip(
        bridge / "frozen_event_code_vocab.csv.gz",
        pd.read_parquet(
            ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/"
            "event_code_vocab.parquet"
        ),
    )
    v8._write_csv_gzip(
        bridge / "frozen_event_text_vocab.csv.gz",
        pd.read_parquet(
            ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/"
            "event_text_vocab.parquet"
        ),
    )
    registry = load_registry()
    inference_ids = {
        "event_code_vocab",
        "event_text_vocab",
        "embedding_tables",
        "ehr_age_model",
        "private_retain_age",
        "replication_contract",
        "clip_weight",
        *{f"checkpoint_fold_{fold}" for fold in range(5)},
        *{f"phenotype_transform_fold_{fold}" for fold in range(5)},
        *{f"correction_fold_{fold}" for fold in range(5)},
    }
    inference_records = [
        row
        for row in registry["locked_artifacts"]
        if row["artifact_id"] in inference_ids
    ]
    if {row["artifact_id"] for row in inference_records} != inference_ids:
        raise AssertionError("inference artifact registry is incomplete")
    inference_artifact_dir = cache_dir / "inference_worker_artifacts"
    inference_artifact_dir.mkdir(parents=True, exist_ok=False)
    inference_registry = (
        inference_artifact_dir / "INFERENCE_LOCKED_ARTIFACT_REGISTRY_V8.json"
    )
    write_json(
        inference_registry,
        {
            "schema_version": "revision0710.external_validation.core_0716.inference_registry.v1",
            "status": "NO_EXTERNAL_SOURCE_PATHS_IN_INFERENCE_PROCESS",
            "external_sources": [],
            "locked_artifacts": inference_records,
        },
    )
    density_bridge = (
        cache_dir / "inference_bridge/mimic_participant_density_label_blind.csv.gz"
    )
    density.to_csv(
        density_bridge,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    manifest = {
        "schema_version": "revision0710.external_validation.core_0716.label_blind_cache.v1",
        "status": "PRIMARY_ICD_ONLY_LABEL_BLIND_CACHE_COMPLETE",
        "questionnaire_files_read": [],
        "postindex_diagnosis_rows_read_into_cache": 0,
        "diagnosis_source_streamed_with_preindex_hadm_filter": True,
        "prescription_source_read": False,
        "prescription_proxy_bridge_rows": 0,
        "mimic_input_variants": [MIMIC_PRIMARY],
        "nhanes_input_variants": [NHANES_PRIMARY],
        "mimic_participants": len(mimic),
        "mimic_flow_counts_before_outcome_read": mimic_flow,
        "nhanes_participants": len(nhanes),
        "nhanes_full_examined_design_rows": len(nhanes_design),
        "nhanes_40_73_is_survey_domain_not_design_filter": True,
        "mimic_density_participant_variant_rows": len(density),
        "nhanes_release_slot_qc_rows": len(nhanes_qc),
        "files": {},
    }
    for path in sorted(value for value in cache_dir.rglob("*") if value.is_file()):
        if path.name == "MANIFEST.json":
            continue
        relative = path.relative_to(cache_dir).as_posix()
        manifest["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(cache["manifest"], manifest)
    built = cache
    update_status(
        "01_cache",
        "COMPLETE",
        manifest=str(cache["manifest"]),
        manifest_sha256=sha256_file(cache["manifest"]),
        density_bridge=str(density_bridge),
    )
    print(json.dumps({key: str(value) for key, value in built.items()}))


def _tensor(value: np.ndarray, device: Any, torch: Any) -> Any:
    return torch.from_numpy(value).to(device=device, non_blocking=True)


def _prediction_frames(
    *,
    result: Any,
    participants: pd.DataFrame,
    cohort: str,
    release_ids: np.ndarray,
    landmark_policy: str,
    variant: str,
    fold: int,
    checkpoint_sha256: str,
    cache_manifest_sha256: str,
) -> list[pd.DataFrame]:
    predicted = result.predicted_age.detach().float().cpu().numpy()
    base_age = result.base_age.detach().float().cpu().numpy()
    offset = result.risk_age_offset.detach().float().cpu().numpy()
    risk = result.risk_score.detach().float().cpu().numpy()
    frames = []
    participant_ids = participants["participant_id"].astype(str).to_numpy()
    sex = participants["sex_male"].to_numpy(dtype=np.int8)
    age = participants["chronological_age"].to_numpy(dtype=np.float32)
    for organ_index, organ in enumerate(ORGANS):
        frames.append(
            pd.DataFrame(
                {
                    "cohort": cohort,
                    "participant_id": participant_ids,
                    "release_id": release_ids,
                    "landmark_policy": landmark_policy,
                    "input_variant": variant,
                    "organ_system": organ,
                    "checkpoint_outer_fold": fold,
                    "sex_male": sex,
                    "chronological_age": age,
                    "predicted_age_raw": predicted[:, organ_index],
                    "base_age_raw": base_age,
                    "risk_age_offset": offset[:, organ_index],
                    "risk_score": risk[:, organ_index],
                    "checkpoint_sha256": checkpoint_sha256,
                    "external_cache_manifest_sha256": cache_manifest_sha256,
                }
            )
        )
    return frames


def gpu_inference(batch_size: int) -> None:
    import torch

    predictions = stage_predictions()
    if predictions["raw_predictions"].is_file() and (
        predictions["directory"] / "INFERENCE_MANIFEST.json"
    ).is_file():
        update_status(
            "02_inference",
            "COMPLETE_REUSED",
            manifest=str(predictions["directory"] / "INFERENCE_MANIFEST.json"),
        )
        print(json.dumps({key: str(value) for key, value in predictions.items()}))
        return
    if predictions["directory"].exists():
        raise FileExistsError(
            f"partial inference directory requires audit: {predictions['directory']}"
        )
    update_status(
        "02_inference",
        "RUNNING",
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        batch_size=batch_size,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the streamlined inference stage")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")

    def gpu_encode_extension_rows(
        texts: list[str] | tuple[str, ...], *, model: Any, tokenizer: Any
    ) -> np.ndarray:
        model = model.to(device).eval()
        output: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), 256):
                tokens = tokenizer(list(texts[start : start + 256])).to(device)
                output.append(
                    model.encode_text(tokens)
                    .float()
                    .cpu()
                    .numpy()
                    .astype("<f2")
                )
        if not output:
            return np.empty((0, 512), dtype="<f2")
        return np.ascontiguousarray(np.concatenate(output, axis=0), dtype="<f2")

    v8._encode_extension_rows = gpu_encode_extension_rows
    cache = stage_cache()
    cache_dir = cache["directory"]
    inference_dir = predictions["directory"]
    inference_dir.mkdir(parents=True, exist_ok=False)
    registry = json.loads(cache["inference_registry"].read_text(encoding="utf-8"))
    if registry.get("external_sources") != []:
        raise AssertionError("inference registry contains raw external source paths")
    bridge = cache_dir / "inference_bridge"
    mimic = pd.read_csv(bridge / "mimic_participants.csv.gz")
    mimic["participant_id"] = mimic["subject_id"].astype(str)
    nhanes = pd.read_csv(bridge / "nhanes_participants.csv.gz")
    nhanes["participant_id"] = nhanes["participant_id"].astype(str)
    icd = pd.read_csv(bridge / "mimic_icd_events_preindex.csv.gz")
    icd["event_time"] = pd.to_datetime(icd["event_time"], errors="raise")
    icd["landmark_time"] = pd.to_datetime(icd["landmark_time"], errors="raise")
    code_to_id, text_to_id, code_embedding, text_embedding, coverage = (
        v8._extend_vocab_and_embeddings(
            ROOT, icd, registry, inference_dir, cache_dir
        )
    )
    v8._write_csv(inference_dir / "vocabulary_prefix_audit.csv", coverage)
    phenotypes = np.load(cache_dir / "phenotypes.npz")
    selected = v8.crop_variant_timeline(icd, MIMIC_PRIMARY)
    packed = v8._pack_events(
        selected,
        mimic["participant_id"].tolist(),
        code_to_id,
        text_to_id,
    )
    density = pd.read_csv(
        bridge / "mimic_participant_density_label_blind.csv.gz",
        dtype={"subject_id": str},
    )
    expected = (
        density.loc[density["input_variant"].eq(MIMIC_PRIMARY)]
        .set_index("subject_id")
        .reindex(mimic["participant_id"])["post_crop_event_count"]
        .to_numpy(dtype=np.int64)
    )
    observed = packed["event_valid_mask"].sum(axis=1).astype(np.int64)
    if not np.array_equal(expected, observed):
        raise AssertionError("packed MIMIC event count differs from cache density")
    locked = {row["artifact_id"]: row for row in registry["locked_artifacts"]}
    cache_sha = sha256_file(cache["manifest"])
    nhanes_packed = {
        "event_code_id": np.zeros((len(nhanes), 32), dtype=np.int64),
        "event_text_id": np.zeros((len(nhanes), 32), dtype=np.int64),
        "event_relative_days": np.zeros((len(nhanes), 32), dtype=np.int32),
        "event_type_id": np.zeros((len(nhanes), 32), dtype=np.int8),
        "event_valid_mask": np.zeros((len(nhanes), 32), dtype=bool),
    }
    raw_path = predictions["raw_predictions"]
    wrote_header = False
    row_count = 0
    started = now_iso()
    with gzip.open(raw_path, "wt", encoding="utf-8", newline="") as handle:
        for cohort_name, participants, raw_key, missing_key, event_pack, variant in (
            (
                "MIMIC_IV_3_1",
                mimic,
                "mimic_raw",
                "mimic_missing",
                packed,
                MIMIC_PRIMARY,
            ),
            (
                "NHANES",
                nhanes,
                "nhanes_raw",
                "nhanes_missing",
                nhanes_packed,
                NHANES_PRIMARY,
            ),
        ):
            raw = phenotypes[raw_key].astype(np.float32)
            missing = phenotypes[missing_key].astype(bool)
            for fold in range(5):
                transform_path = ROOT / locked[f"phenotype_transform_fold_{fold}"]["path"]
                transform = np.load(transform_path)
                transformed = (
                    (np.where(missing, transform["median"], raw) - transform["mean"])
                    / transform["std"]
                ).astype(np.float32)
                checkpoint = locked[f"checkpoint_fold_{fold}"]
                model = v8._restore_fixed_model(
                    ROOT,
                    ROOT / checkpoint["path"],
                    code_embedding,
                    text_embedding,
                    outer_fold=fold,
                ).to(device)
                model.eval()
                with torch.inference_mode():
                    for start in range(0, len(participants), batch_size):
                        stop = min(start + batch_size, len(participants))
                        result = model(
                            _tensor(transformed[start:stop], device, torch),
                            _tensor(missing[start:stop], device, torch),
                            _tensor(
                                participants["sex_male"].to_numpy(dtype=np.float32)[
                                    start:stop
                                ],
                                device,
                                torch,
                            ),
                            _tensor(event_pack["event_code_id"][start:stop], device, torch),
                            _tensor(event_pack["event_text_id"][start:stop], device, torch),
                            _tensor(
                                event_pack["event_relative_days"][start:stop],
                                device,
                                torch,
                            ),
                            _tensor(event_pack["event_type_id"][start:stop], device, torch),
                            _tensor(event_pack["event_valid_mask"][start:stop], device, torch),
                        )
                        part = participants.iloc[start:stop].copy()
                        if cohort_name == "MIMIC_IV_3_1":
                            releases = np.full(len(part), None, dtype=object)
                            landmark = "mimic_second_valid_adult_discharge"
                        else:
                            releases = part["release_id"].astype(str).to_numpy()
                            landmark = "nhanes_mec_exam"
                        for frame in _prediction_frames(
                            result=result,
                            participants=part,
                            cohort=cohort_name,
                            release_ids=releases,
                            landmark_policy=landmark,
                            variant=variant,
                            fold=fold,
                            checkpoint_sha256=checkpoint["sha256"],
                            cache_manifest_sha256=cache_sha,
                        ):
                            frame.to_csv(
                                handle,
                                index=False,
                                header=not wrote_header,
                                lineterminator="\n",
                            )
                            wrote_header = True
                            row_count += len(frame)
                del model
                torch.cuda.empty_cache()
    expected_rows = (len(mimic) + len(nhanes)) * len(ORGANS) * 5
    if row_count != expected_rows:
        raise AssertionError(
            f"inference row count mismatch: {row_count} != {expected_rows}"
        )
    manifest_path = inference_dir / "INFERENCE_MANIFEST.json"
    manifest = {
        "schema_version": "revision0710.external_validation.core_0716.inference.v1",
        "status": "FIVE_FIXED_CHECKPOINT_GPU_INFERENCE_COMPLETE",
        "started_at": started,
        "completed_at": now_iso(),
        "device": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "mimic_participants": len(mimic),
        "nhanes_participants": len(nhanes),
        "organs": ORGANS,
        "checkpoint_folds": list(range(5)),
        "rows": row_count,
        "raw_predictions": str(raw_path),
        "raw_predictions_sha256": sha256_file(raw_path),
        "outcome_fields_read": [],
    }
    write_json(manifest_path, manifest)
    update_status(
        "02_inference",
        "COMPLETE",
        manifest=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
    )
    print(json.dumps({key: str(value) for key, value in predictions.items()}))


def seal_predictions() -> None:
    sealed = stage_sealed()
    if sealed["receipt"].is_file():
        update_status(
            "03_seal",
            "COMPLETE_REUSED",
            receipt=str(sealed["receipt"]),
            receipt_sha256=sha256_file(sealed["receipt"]),
        )
        print(json.dumps({key: str(value) for key, value in sealed.items()}))
        return
    update_status("03_seal", "RUNNING")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = v8.seal_prediction_tables(
        stage=WORK_DIR,
        predictions=stage_predictions(),
        contract=contract,
    )
    update_status(
        "03_seal",
        "COMPLETE",
        receipt=str(result["receipt"]),
        receipt_sha256=sha256_file(Path(result["receipt"])),
    )
    print(json.dumps({key: str(value) for key, value in result.items()}))


def build_outcomes() -> None:
    output_dir = WORK_DIR / "shared/05_outcomes_after_prediction_seal"
    receipt = output_dir / "OUTCOME_READ_RECEIPT.json"
    if receipt.is_file():
        update_status(
            "04_outcomes",
            "COMPLETE_REUSED",
            receipt=str(receipt),
            receipt_sha256=sha256_file(receipt),
        )
        print(json.dumps({"receipt": str(receipt)}))
        return
    update_status("04_outcomes", "RUNNING")
    v8.MIMIC_VARIANTS = [MIMIC_PRIMARY]
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = v8.build_outcomes_after_prediction_seal(
        artifact_dir=ARTIFACT_DIR,
        stage=WORK_DIR,
        sources=source_paths(),
        prediction_seal=stage_sealed(),
        contract=contract,
    )
    update_status(
        "04_outcomes",
        "COMPLETE",
        receipt=str(result["receipt"]),
        receipt_sha256=sha256_file(Path(result["receipt"])),
    )
    print(json.dumps({key: str(value) for key, value in result.items()}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("freeze", "cache", "infer", "seal", "outcomes")
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze_contract()
    elif args.stage == "cache":
        build_cache()
    elif args.stage == "infer":
        gpu_inference(args.batch_size)
    elif args.stage == "seal":
        seal_predictions()
    else:
        build_outcomes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
