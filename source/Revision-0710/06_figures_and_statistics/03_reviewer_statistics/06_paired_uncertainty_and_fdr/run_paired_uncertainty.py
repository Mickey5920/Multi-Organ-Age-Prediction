#!/usr/bin/env python
"""Lock inputs and run Revision-0710 paired participant uncertainty analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parents[2]
REPO_ROOT = REVISION_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paired_uncertainty import (  # noqa: E402
    ORGANS,
    ContractError,
    apply_bh_fdr,
    apply_cox_payload,
    apply_frozen_linear_correction,
    apply_sealed_cox_model,
    bootstrap_age_from_stats,
    build_stratified_draw_weights,
    build_age_analysis_frame,
    fit_frozen_cox_nuisance,
    paired_age_point_estimates,
    participant_age_sufficient_statistics,
    resolve_registry_prediction_files,
    sha256_file,
    summarize_paired_replicates,
    weighted_harrell_c_batch,
)


CONTRACT_PATH = SCRIPT_DIR / "analysis_contract.json"
INPUT_MANIFEST_PATH = SCRIPT_DIR / "00_input_lock/INPUT_MANIFEST.json"
AGE_DIR = SCRIPT_DIR / "01_age"
RISK_CACHE_DIR = SCRIPT_DIR / "02_risk_score_cache"
RISK_INPUT_MANIFEST_PATH = RISK_CACHE_DIR / "00_risk_input_lock/RISK_INPUT_MANIFEST.json"
RISK_REFERENCE_DIR = RISK_CACHE_DIR / "01_reference"
RISK_BASELINE_DIR = RISK_CACHE_DIR / "02_baselines"
RISK_BOOTSTRAP_DIR = SCRIPT_DIR / "03_risk_bootstrap"
BASELINE_REGISTRIES = {
    "retain": REVISION_DIR / "05_downstream/01_prediction_registries/tier_a/retain.json",
    "behrt_style": REVISION_DIR / "05_downstream/01_prediction_registries/tier_a/behrt_style.json",
    "medbert_style": REVISION_DIR / "05_downstream/01_prediction_registries/tier_a/medbert_style.json",
    "retain_organquery": REVISION_DIR / "05_downstream/01_prediction_registries/tier_b/retain_organquery.json",
    "behrt_style_organquery": REVISION_DIR / "05_downstream/01_prediction_registries/tier_b/behrt_style_organquery.json",
    "medbert_style_organquery": REVISION_DIR / "05_downstream/01_prediction_registries/tier_b/medbert_style_organquery.json",
}


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return payload


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _file_record(path: Path, *, kind: str) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"locked input does not exist: {path}")
    record: dict[str, object] = {
        "kind": kind,
        "relative_path": _relative(path),
        "absolute_path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        record["rows"] = int(parquet.metadata.num_rows)
        record["columns"] = parquet.schema.names
    return record


def _write_json_create_once(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except FileExistsError as exc:
        raise ContractError(f"create-once artifact already exists: {path}") from exc


def _contract() -> dict[str, object]:
    payload = _read_json(CONTRACT_PATH)
    if payload.get("status") != "LOCKED_BEFORE_FORMAL_BOOTSTRAP":
        raise ContractError("analysis contract is not locked before formal bootstrap")
    return payload


def build_input_manifest() -> dict[str, object]:
    if INPUT_MANIFEST_PATH.exists():
        return verify_input_manifest()
    contract = _contract()
    methods = list(contract["baseline_methods"])
    if methods != list(BASELINE_REGISTRIES):
        raise ContractError("contract baseline method order disagrees with the explicit registry map")
    sources: list[dict[str, object]] = [_file_record(CONTRACT_PATH, kind="analysis_contract")]
    fixed_paths = {
        "ehr_oof": REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/02_oof_hr_km_inputs/ehr_age_d07_003_e55_oof.parquet",
        "ehr_registry": REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/00_prediction_registry/ehr_age_d07_003_e55.json",
        "ehr_evaluation_summary": REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/01_evaluation/parity_audit/evaluation_summary.json",
        "ehr_cox_cells": REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/01_evaluation/parity_audit/cox_fold_organ_cells.csv",
        "participant_masks": REVISION_DIR / "01_data_rebuild/02_outcomes_60m/participant_followup_and_evaluation_masks.parquet",
        "organ_outcomes": REVISION_DIR / "01_data_rebuild/02_outcomes_60m/organ_outcomes_60m.parquet",
        "fold_roles": REVISION_DIR / "01_data_rebuild/03_folds/outer_fold_roles.parquet",
        "canonical_evaluator": REVISION_DIR / "05_downstream/common/crossfit_metrics.py",
    }
    for label, path in fixed_paths.items():
        sources.append(_file_record(path, kind=label))

    registry_records: dict[str, object] = {}
    ehr_registry = fixed_paths["ehr_registry"]
    ehr_entries = resolve_registry_prediction_files(
        ehr_registry,
        expected_method=str(contract["ehr_method"]),
        verify_hash=True,
    )
    registry_records[str(contract["ehr_method"])] = {
        "registry_relative_path": _relative(ehr_registry),
        "registry_sha256": sha256_file(ehr_registry),
        "prediction_files": [
            {**{key: value for key, value in entry.items() if key != "path"}, "relative_path": _relative(Path(entry["path"]))}
            for entry in ehr_entries
        ],
    }
    for method, registry in BASELINE_REGISTRIES.items():
        sources.extend(
            [
                _file_record(registry, kind=f"{method}:prediction_registry"),
                _file_record(REVISION_DIR / f"05_downstream/02_baseline_evaluation/{method}/evaluation_summary.json", kind=f"{method}:evaluation_summary"),
                _file_record(REVISION_DIR / f"05_downstream/02_baseline_evaluation/{method}/linear_correction_parameters.csv", kind=f"{method}:age_correction"),
                _file_record(REVISION_DIR / f"05_downstream/02_baseline_evaluation/{method}/cox_fold_organ_cells.csv", kind=f"{method}:cox_cells"),
            ]
        )
        entries = resolve_registry_prediction_files(registry, expected_method=method, verify_hash=True)
        registry_records[method] = {
            "registry_relative_path": _relative(registry),
            "registry_sha256": sha256_file(registry),
            "prediction_files": [
                {**{key: value for key, value in entry.items() if key != "path"}, "relative_path": _relative(Path(entry["path"]))}
                for entry in entries
            ],
        }
    payload: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.paired_uncertainty.input_manifest.v1",
        "status": "ALL_INPUT_HASHES_VERIFIED",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_count": len(sources),
        "sources": sources,
        "prediction_registries": registry_records,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    _write_json_create_once(INPUT_MANIFEST_PATH, payload)
    return payload


def verify_input_manifest() -> dict[str, object]:
    manifest = _read_json(INPUT_MANIFEST_PATH)
    if manifest.get("status") != "ALL_INPUT_HASHES_VERIFIED":
        raise ContractError("input manifest is not verified")
    if manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ContractError("analysis contract hash changed after input lock")
    for source in manifest.get("sources", []):
        path = REPO_ROOT / str(source["relative_path"])
        observed = sha256_file(path)
        if observed != source["sha256"]:
            raise ContractError(f"locked source hash changed: {path}")
    for registry in manifest.get("prediction_registries", {}).values():
        registry_path = REPO_ROOT / str(registry["registry_relative_path"])
        if sha256_file(registry_path) != registry["registry_sha256"]:
            raise ContractError(f"locked prediction registry hash changed: {registry_path}")
        for entry in registry["prediction_files"]:
            prediction_path = REPO_ROOT / str(entry["relative_path"])
            if sha256_file(prediction_path) != entry["sha256"]:
                raise ContractError(f"locked prediction file hash changed: {prediction_path}")
    return manifest


def _ehr_nuisance_root(fold: int) -> Path:
    run_root = REVISION_DIR / f"03_ehr_age/04_fixed_replication/01_runs/outer_fold_{fold}"
    if fold == 0:
        return run_root / "reuse_d07_003_e55/sealed_nuisance"
    return run_root / "attempts/primary_e55_v1/evaluation/sealed_nuisance"


def lock_risk_inputs() -> dict[str, object]:
    if RISK_INPUT_MANIFEST_PATH.exists():
        return verify_risk_input_manifest()
    verify_input_manifest()
    sources = [
        _file_record(CONTRACT_PATH, kind="analysis_contract"),
        _file_record(INPUT_MANIFEST_PATH, kind="base_input_manifest"),
        _file_record(SCRIPT_DIR / "paired_uncertainty.py", kind="statistical_engine"),
        _file_record(Path(__file__), kind="execution_entrypoint"),
    ]
    nuisance_manifests: list[dict[str, object]] = []
    for fold in range(5):
        root = _ehr_nuisance_root(fold)
        manifest_path = root / "COX_NUISANCE_MANIFEST.json"
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "ALL_14_MODELS_SEALED_BEFORE_OUTER_READ":
            raise ContractError(f"EHR Cox nuisance manifest is not sealed: fold={fold}")
        if manifest.get("outer_fold") != fold or manifest.get("model_count") != 14:
            raise ContractError(f"EHR Cox nuisance manifest fold/model count mismatch: fold={fold}")
        sources.append(_file_record(manifest_path, kind=f"ehr_fold{fold}:nuisance_manifest"))
        sources.append(_file_record(Path(str(manifest["age_correction_path"])), kind=f"ehr_fold{fold}:age_correction"))
        locked_models: list[dict[str, object]] = []
        for model in manifest["models"]:
            model_path = Path(str(model["path"])).resolve()
            observed = sha256_file(model_path)
            if observed != model["sha256"]:
                raise ContractError(f"EHR sealed nuisance model hash mismatch: {model_path}")
            sources.append(_file_record(model_path, kind=f"ehr_fold{fold}:cox_{model['organ_system']}_{model['model_role']}"))
            locked_models.append(
                {
                    "organ_system": model["organ_system"],
                    "model_role": model["model_role"],
                    "relative_path": _relative(model_path),
                    "sha256": observed,
                }
            )
        nuisance_manifests.append(
            {
                "outer_fold": fold,
                "manifest_relative_path": _relative(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "models": locked_models,
            }
        )
    try:
        import lifelines
        import numba
    except ImportError as exc:
        raise ContractError(f"formal risk environment is incomplete: {exc}") from exc
    payload: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.risk_input_manifest.v1",
        "status": "SEALED_NUISANCE_AND_CODE_HASHES_VERIFIED",
        "base_input_manifest_sha256": sha256_file(INPUT_MANIFEST_PATH),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "sources": sources,
        "ehr_nuisance_manifests": nuisance_manifests,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "lifelines": lifelines.__version__,
            "numba": numba.__version__,
        },
    }
    _write_json_create_once(RISK_INPUT_MANIFEST_PATH, payload)
    return payload


def verify_risk_input_manifest() -> dict[str, object]:
    payload = _read_json(RISK_INPUT_MANIFEST_PATH)
    if payload.get("status") != "SEALED_NUISANCE_AND_CODE_HASHES_VERIFIED":
        raise ContractError("risk input manifest is not sealed")
    if payload.get("base_input_manifest_sha256") != sha256_file(INPUT_MANIFEST_PATH):
        raise ContractError("base input manifest changed after risk lock")
    if payload.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ContractError("analysis contract changed after risk lock")
    for source in payload.get("sources", []):
        path = REPO_ROOT / str(source["relative_path"])
        if sha256_file(path) != source["sha256"]:
            raise ContractError(f"risk-locked source changed: {path}")
    return payload


def _participant_id_hash(ids: pd.Series) -> str:
    digest = hashlib.sha256()
    for participant_id in ids.astype(str):
        digest.update(participant_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_risk_reference() -> dict[str, object]:
    outputs = [
        RISK_REFERENCE_DIR / "participant_fold_index.parquet",
        RISK_REFERENCE_DIR / "ehr_base_full_fixed_scores.parquet",
        RISK_REFERENCE_DIR / "EHR_POINT_PARITY.csv",
        RISK_REFERENCE_DIR / "REFERENCE_SCORE_MANIFEST.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise ContractError(f"risk reference outputs are create-once and already exist: {existing}")
    verify_risk_input_manifest()
    started = time.perf_counter()
    oof_path = REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/02_oof_hr_km_inputs/ehr_age_d07_003_e55_oof.parquet"
    oof = pd.read_parquet(
        oof_path,
        columns=[
            "participant_id",
            "organ_system",
            "outer_fold",
            "actual_age",
            "sex",
            "corrected_age_gap",
            "at_risk",
            "duration_months",
            "event",
            "cox_nuisance_manifest_sha256",
        ],
    )
    oof["participant_id"] = oof["participant_id"].astype(str).str.strip()
    participants = oof.loc[:, ["participant_id", "outer_fold"]].drop_duplicates()
    if participants["participant_id"].duplicated().any() or len(participants) != 456773:
        raise ContractError("OOF participant-fold index is not one-to-one with 456773 participants")
    participants = participants.sort_values(["outer_fold", "participant_id"], kind="mergesort").reset_index(drop=True)
    participants["participant_index"] = np.arange(len(participants), dtype=np.int32)
    risk = oof.loc[oof["at_risk"].eq(1)].copy()
    risk = risk.merge(participants, on=["participant_id", "outer_fold"], how="left", validate="many_to_one")
    risk = risk.sort_values(["outer_fold", "organ_system", "participant_id"], kind="mergesort").reset_index(drop=True)
    if risk.duplicated(["participant_id", "organ_system"]).any():
        raise ContractError("EHR risk OOF has duplicate participant-organ rows")
    published = pd.read_csv(
        REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/01_evaluation/parity_audit/cox_fold_organ_cells.csv"
    )
    parity_rows: list[dict[str, object]] = []
    from lifelines.utils import concordance_index

    risk["base_predicted_event_time"] = np.nan
    risk["ehr_predicted_event_time"] = np.nan
    for fold in range(5):
        root = _ehr_nuisance_root(fold)
        expected_manifest_sha = sha256_file(root / "COX_NUISANCE_MANIFEST.json")
        fold_manifest_values = set(risk.loc[risk["outer_fold"].eq(fold), "cox_nuisance_manifest_sha256"].astype(str))
        if fold_manifest_values != {expected_manifest_sha}:
            raise ContractError(f"OOF nuisance manifest hash does not match sealed fold {fold}")
        for organ in ORGANS:
            mask = risk["outer_fold"].eq(fold) & risk["organ_system"].eq(organ)
            cell = risk.loc[mask, ["actual_age", "sex", "corrected_age_gap", "duration_months", "event"]]
            base_score = apply_sealed_cox_model(cell, root / f"cox_nuisance/{organ}__base.json")
            full_score = apply_sealed_cox_model(cell, root / f"cox_nuisance/{organ}__full.json")
            risk.loc[mask, "base_predicted_event_time"] = base_score
            risk.loc[mask, "ehr_predicted_event_time"] = full_score
            observed_base = float(concordance_index(cell["duration_months"], base_score, cell["event"]))
            observed_full = float(concordance_index(cell["duration_months"], full_score, cell["event"]))
            locked = published.loc[published["fold"].eq(fold) & published["organ_system"].eq(organ)].iloc[0]
            parity_rows.append(
                {
                    "fold": fold,
                    "organ_system": organ,
                    "observed_c_base": observed_base,
                    "locked_c_base": float(locked["c_base"]),
                    "base_difference": observed_base - float(locked["c_base"]),
                    "observed_c_plus": observed_full,
                    "locked_c_plus": float(locked["c_plus"]),
                    "plus_difference": observed_full - float(locked["c_plus"]),
                    "pass_atol_1e_12": abs(observed_base - float(locked["c_base"])) <= 1e-12
                    and abs(observed_full - float(locked["c_plus"])) <= 1e-12,
                }
            )
    if risk[["base_predicted_event_time", "ehr_predicted_event_time"]].isna().any(axis=None):
        raise ContractError("risk reference scoring left missing held-out scores")
    if not all(bool(row["pass_atol_1e_12"]) for row in parity_rows):
        raise ContractError("sealed EHR risk scores do not reproduce canonical Cox cells")
    RISK_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    participants.to_parquet(outputs[0], index=False)
    score_columns = [
        "participant_id",
        "participant_index",
        "outer_fold",
        "organ_system",
        "duration_months",
        "event",
        "base_predicted_event_time",
        "ehr_predicted_event_time",
    ]
    risk.loc[:, score_columns].to_parquet(outputs[1], index=False)
    pd.DataFrame(parity_rows).to_csv(outputs[2], index=False)
    manifest: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.risk_reference_scores.v1",
        "status": "SEALED_EHR_SCORES_POINT_PARITY_PASS",
        "risk_input_manifest_sha256": sha256_file(RISK_INPUT_MANIFEST_PATH),
        "participant_count": int(len(participants)),
        "risk_rows": int(len(risk)),
        "participant_ids_sha256": _participant_id_hash(participants["participant_id"]),
        "runtime_seconds": time.perf_counter() - started,
        "outputs": [
            {"relative_path": _relative(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
            for path in outputs[:3]
        ],
    }
    _write_json_create_once(outputs[3], manifest)
    return manifest


def _load_participant_age_sex() -> pd.DataFrame:
    path = REVISION_DIR / "01_data_rebuild/02_outcomes_60m/participant_followup_and_evaluation_masks.parquet"
    participants = pd.read_parquet(path, columns=["participant_id", "chronological_age", "sex"])
    participants["participant_id"] = participants["participant_id"].astype(str).str.strip()
    if participants["participant_id"].duplicated().any():
        raise ContractError("participant masks has duplicate participant_id rows")
    sex_text = participants["sex"].astype(str).str.strip().str.lower()
    participants["sex"] = sex_text.map(
        {"female": 0.0, "f": 0.0, "0": 0.0, "male": 1.0, "m": 1.0, "1": 1.0}
    )
    if participants["sex"].isna().any():
        raise ContractError("participant masks contains unsupported sex values")
    participants = participants.rename(columns={"chronological_age": "authoritative_age"})
    return participants


def _load_prediction_role(method: str, fold: int, role: str) -> tuple[pd.DataFrame, dict[str, object]]:
    entries = resolve_registry_prediction_files(BASELINE_REGISTRIES[method], expected_method=method, verify_hash=True)
    matches = [entry for entry in entries if entry["outer_fold"] == fold and entry["split_role"] == role]
    if len(matches) != 1:
        raise ContractError(f"registry did not resolve one {method} fold={fold} role={role} file")
    entry = matches[0]
    frame = pd.read_parquet(
        Path(entry["path"]),
        columns=["participant_id", "organ_system", "actual_age", "predicted_age", "outer_fold", "split_role"],
    )
    frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
    frame["organ_system"] = frame["organ_system"].astype(str)
    if not frame["split_role"].astype(str).eq(role).all() or not frame["outer_fold"].eq(fold).all():
        raise ContractError(f"prediction role file content mismatch: {method} fold={fold} role={role}")
    if len(frame) != entry["rows"] or frame.duplicated(["participant_id", "organ_system"]).any():
        raise ContractError(f"prediction role row count/key mismatch: {method} fold={fold} role={role}")
    return frame.drop(columns="split_role"), entry


def _enrich_baseline_risk_role(
    predictions: pd.DataFrame,
    *,
    participants: pd.DataFrame,
    risk_outcomes: pd.DataFrame,
    correction_parameters: pd.DataFrame,
) -> pd.DataFrame:
    work = predictions.merge(participants, on="participant_id", how="left", validate="many_to_one")
    if work[["authoritative_age", "sex"]].isna().any(axis=None):
        raise ContractError("baseline predictions failed participant age/sex join")
    age_difference = np.abs(
        work["actual_age"].to_numpy(dtype=np.float64)
        - work["authoritative_age"].to_numpy(dtype=np.float64)
    )
    if (age_difference > 1e-5).any():
        raise ContractError("baseline exported age disagrees with authoritative age beyond 1e-5 years")
    work["actual_age"] = work["authoritative_age"].to_numpy(dtype=np.float64)
    work = work.drop(columns="authoritative_age")
    work = apply_frozen_linear_correction(work, correction_parameters)
    work = work.merge(
        risk_outcomes,
        on=["participant_id", "organ_system"],
        how="inner",
        validate="one_to_one",
    )
    return work


def prepare_risk_baseline(method: str) -> dict[str, object]:
    contract = _contract()
    if method not in contract["baseline_methods"]:
        raise ContractError(f"unknown baseline method: {method}")
    method_dir = RISK_BASELINE_DIR / method
    score_path = method_dir / "heldout_fixed_scores.parquet"
    parity_path = method_dir / "POINT_PARITY.csv"
    manifest_path = method_dir / "BASELINE_SCORE_MANIFEST.json"
    if method_dir.exists():
        raise ContractError(f"baseline fixed-score directory is create-once and already exists: {method_dir}")
    verify_risk_input_manifest()
    reference_manifest = _read_json(RISK_REFERENCE_DIR / "REFERENCE_SCORE_MANIFEST.json")
    if reference_manifest.get("status") != "SEALED_EHR_SCORES_POINT_PARITY_PASS":
        raise ContractError("risk reference scores are not complete")
    started = time.perf_counter()
    participants = _load_participant_age_sex()
    outcomes_path = REVISION_DIR / "01_data_rebuild/02_outcomes_60m/organ_outcomes_60m.parquet"
    outcomes = pd.read_parquet(
        outcomes_path,
        columns=["participant_id", "organ_system", "at_risk", "duration_months", "event"],
    )
    outcomes["participant_id"] = outcomes["participant_id"].astype(str).str.strip()
    outcomes["organ_system"] = outcomes["organ_system"].astype(str)
    risk_outcomes = outcomes.loc[outcomes["at_risk"].eq(1), ["participant_id", "organ_system", "duration_months", "event"]].rename(
        columns={"duration_months": "duration"}
    )
    correction_path = REVISION_DIR / f"05_downstream/02_baseline_evaluation/{method}/linear_correction_parameters.csv"
    correction = pd.read_csv(correction_path)
    published_path = REVISION_DIR / f"05_downstream/02_baseline_evaluation/{method}/cox_fold_organ_cells.csv"
    published = pd.read_csv(published_path)
    model_payloads: list[tuple[Path, dict[str, object]]] = []
    score_parts: list[pd.DataFrame] = []
    parity_rows: list[dict[str, object]] = []
    from lifelines.utils import concordance_index

    for fold in range(5):
        validation_predictions, validation_entry = _load_prediction_role(method, fold, "validation")
        test_predictions, test_entry = _load_prediction_role(method, fold, "test")
        validation = _enrich_baseline_risk_role(
            validation_predictions,
            participants=participants,
            risk_outcomes=risk_outcomes,
            correction_parameters=correction,
        )
        test = _enrich_baseline_risk_role(
            test_predictions,
            participants=participants,
            risk_outcomes=risk_outcomes,
            correction_parameters=correction,
        )
        if set(validation["participant_id"]).intersection(test["participant_id"]):
            raise ContractError(f"baseline calibration/test participant overlap: method={method}, fold={fold}")
        for organ in ORGANS:
            calibration_cell = validation.loc[validation["organ_system"].eq(organ)].copy()
            test_cell = test.loc[test["organ_system"].eq(organ)].copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                payload = fit_frozen_cox_nuisance(
                    calibration_cell,
                    covariates=("actual_age", "sex", "corrected_age_gap"),
                    penalizer=1e-4,
                )
            payload.update(
                {
                    "method": method,
                    "outer_fold": fold,
                    "organ_system": organ,
                    "calibration_prediction_sha256": validation_entry["sha256"],
                    "test_prediction_sha256": test_entry["sha256"],
                    "published_correction_sha256": sha256_file(correction_path),
                    "risk_input_manifest_sha256": sha256_file(RISK_INPUT_MANIFEST_PATH),
                    "calibration_participant_ids_sha256": _participant_id_hash(
                        calibration_cell["participant_id"].sort_values(kind="mergesort")
                    ),
                }
            )
            predicted_event_time = apply_cox_payload(test_cell, payload)
            observed = float(concordance_index(test_cell["duration"], predicted_event_time, test_cell["event"]))
            locked = published.loc[published["fold"].eq(fold) & published["organ_system"].eq(organ)].iloc[0]
            difference = observed - float(locked["c_plus"])
            parity_rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "organ_system": organ,
                    "n_calibration": int(len(calibration_cell)),
                    "n_test": int(len(test_cell)),
                    "calibration_events": int(calibration_cell["event"].sum()),
                    "test_events": int(test_cell["event"].sum()),
                    "observed_c_plus": observed,
                    "locked_c_plus": float(locked["c_plus"]),
                    "difference": difference,
                    "pass_atol_1e_12": abs(difference) <= 1e-12,
                }
            )
            model_path = method_dir / f"models/fold{fold}/{organ}__full.json"
            model_payloads.append((model_path, payload))
            score_parts.append(
                pd.DataFrame(
                    {
                        "participant_id": test_cell["participant_id"].to_numpy(),
                        "outer_fold": fold,
                        "organ_system": organ,
                        "predicted_event_time": predicted_event_time,
                    }
                )
            )
    if not all(bool(row["pass_atol_1e_12"]) for row in parity_rows):
        failed = [row for row in parity_rows if not row["pass_atol_1e_12"]]
        raise ContractError(f"baseline fixed scores failed point parity: {failed[:3]}")
    scores = pd.concat(score_parts, ignore_index=True).sort_values(
        ["outer_fold", "organ_system", "participant_id"], kind="mergesort"
    ).reset_index(drop=True)
    reference = pd.read_parquet(
        RISK_REFERENCE_DIR / "ehr_base_full_fixed_scores.parquet",
        columns=["participant_id", "outer_fold", "organ_system"],
    ).sort_values(["outer_fold", "organ_system", "participant_id"], kind="mergesort").reset_index(drop=True)
    keys = ["participant_id", "outer_fold", "organ_system"]
    if len(scores) != len(reference) or not scores[keys].equals(reference[keys]):
        raise ContractError(f"baseline held-out risk keys do not exactly match EHR reference: {method}")
    method_dir.mkdir(parents=True, exist_ok=False)
    for model_path, payload in model_payloads:
        _write_json_create_once(model_path, payload)
    scores.to_parquet(score_path, index=False)
    pd.DataFrame(parity_rows).to_csv(parity_path, index=False)
    manifest: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.baseline_fixed_scores.v1",
        "status": "CALIBRATION_FIT_ONCE_AND_HELDOUT_POINT_PARITY_PASS",
        "method": method,
        "risk_input_manifest_sha256": sha256_file(RISK_INPUT_MANIFEST_PATH),
        "reference_score_manifest_sha256": sha256_file(RISK_REFERENCE_DIR / "REFERENCE_SCORE_MANIFEST.json"),
        "model_count": len(model_payloads),
        "risk_rows": int(len(scores)),
        "runtime_seconds": time.perf_counter() - started,
        "conditional_inference": "nuisance coefficients fixed after one calibration-role fit; no outer-test or bootstrap refitting",
        "outputs": [
            {"relative_path": _relative(score_path), "sha256": sha256_file(score_path), "bytes": int(score_path.stat().st_size)},
            {"relative_path": _relative(parity_path), "sha256": sha256_file(parity_path), "bytes": int(parity_path.stat().st_size)},
        ],
        "models": [
            {"relative_path": _relative(path), "sha256": sha256_file(path)} for path, _ in model_payloads
        ],
    }
    _write_json_create_once(manifest_path, manifest)
    return manifest


def _load_aligned_fixed_scores(method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_manifest = _read_json(RISK_REFERENCE_DIR / "REFERENCE_SCORE_MANIFEST.json")
    baseline_manifest = _read_json(RISK_BASELINE_DIR / method / "BASELINE_SCORE_MANIFEST.json")
    if reference_manifest.get("status") != "SEALED_EHR_SCORES_POINT_PARITY_PASS":
        raise ContractError("risk reference fixed-score cache is not complete")
    if baseline_manifest.get("status") != "CALIBRATION_FIT_ONCE_AND_HELDOUT_POINT_PARITY_PASS":
        raise ContractError(f"baseline fixed-score cache is not complete: {method}")
    participants = pd.read_parquet(RISK_REFERENCE_DIR / "participant_fold_index.parquet").sort_values(
        "participant_index", kind="mergesort"
    ).reset_index(drop=True)
    if not participants["participant_index"].eq(np.arange(len(participants))).all():
        raise ContractError("participant index is not dense and ordered")
    reference = pd.read_parquet(RISK_REFERENCE_DIR / "ehr_base_full_fixed_scores.parquet")
    baseline = pd.read_parquet(RISK_BASELINE_DIR / method / "heldout_fixed_scores.parquet")
    keys = ["participant_id", "outer_fold", "organ_system"]
    reference = reference.sort_values(keys, kind="mergesort").reset_index(drop=True)
    baseline = baseline.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(reference) != len(baseline) or not reference[keys].equals(baseline[keys]):
        raise ContractError("fixed-score EHR/baseline keys are not exactly aligned")
    reference[f"{method}_predicted_event_time"] = baseline["predicted_event_time"].to_numpy(dtype=np.float64)
    return participants, reference


def compute_risk_replicates(method: str, replicate_ids: Sequence[int]) -> pd.DataFrame:
    contract = _contract()
    if method not in contract["baseline_methods"]:
        raise ContractError(f"unknown baseline method: {method}")
    participants, scores = _load_aligned_fixed_scores(method)
    ids = [int(item) for item in replicate_ids]
    weights = build_stratified_draw_weights(
        participants[["participant_id", "outer_fold"]],
        replicate_ids=ids,
        seed=int(contract["bootstrap"]["master_seed"]),
    )
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for organ in ORGANS:
            cell = scores.loc[scores["outer_fold"].eq(fold) & scores["organ_system"].eq(organ)]
            predictions = cell[["ehr_predicted_event_time", f"{method}_predicted_event_time"]].to_numpy(dtype=np.float64)
            c_values = weighted_harrell_c_batch(
                cell["duration_months"].to_numpy(dtype=np.float64),
                cell["event"].to_numpy(dtype=np.int8),
                predictions,
                cell["participant_index"].to_numpy(dtype=np.int32),
                weights,
            )
            for row_index, replicate in enumerate(ids):
                rows.append(
                    {
                        "replicate": replicate,
                        "baseline_method": method,
                        "fold": fold,
                        "organ_system": organ,
                        "ehr_c_plus": float(c_values[row_index, 0]),
                        "baseline_c_plus": float(c_values[row_index, 1]),
                        "ehr_minus_baseline_delta_c": float(c_values[row_index, 0] - c_values[row_index, 1]),
                    }
                )
    return pd.DataFrame(rows)


def benchmark_risk(method: str, *, n_resamples: int) -> dict[str, object]:
    if n_resamples < 1:
        raise ContractError("risk benchmark requires at least one replicate")
    benchmark_dir = RISK_BOOTSTRAP_DIR / f"00_benchmarks/{method}_b{n_resamples}"
    output = benchmark_dir / "BENCHMARK.json"
    if benchmark_dir.exists():
        raise ContractError(f"risk benchmark is create-once and already exists: {benchmark_dir}")
    started = time.perf_counter()
    replicates = compute_risk_replicates(method, range(n_resamples))
    runtime = time.perf_counter() - started
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    replicate_path = benchmark_dir / "benchmark_cells.parquet"
    replicates.to_parquet(replicate_path, index=False)
    payload: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.risk_bootstrap_benchmark.v1",
        "status": "BENCHMARK_COMPLETE",
        "method": method,
        "n_resamples": n_resamples,
        "runtime_seconds": runtime,
        "seconds_per_replicate": runtime / n_resamples,
        "projected_seconds_b1000": runtime / n_resamples * 1000.0,
        "risk_rows": int(len(pd.read_parquet(RISK_REFERENCE_DIR / "ehr_base_full_fixed_scores.parquet", columns=["participant_id"]))),
        "fixed_score_conditioning": "conditional on calibration-fitted sealed nuisance; no nuisance refit uncertainty",
        "replicate_output": {"relative_path": _relative(replicate_path), "sha256": sha256_file(replicate_path)},
    }
    _write_json_create_once(output, payload)
    return payload


def _formal_risk_batch_paths(method: str, start: int, stop: int) -> tuple[Path, Path, Path]:
    directory = RISK_BOOTSTRAP_DIR / f"01_formal_batches/{method}/batch_{start:04d}_{stop:04d}"
    return directory, directory / "fold_organ_cells.parquet", directory / "BATCH_MANIFEST.json"


def _verify_formal_risk_batch(method: str, start: int, stop: int) -> dict[str, object]:
    _, cells_path, manifest_path = _formal_risk_batch_paths(method, start, stop)
    payload = _read_json(manifest_path)
    expected = {
        "status": "FORMAL_FIXED_SCORE_BATCH_COMPLETE",
        "method": method,
        "replicate_start_inclusive": start,
        "replicate_stop_exclusive": stop,
        "risk_input_manifest_sha256": sha256_file(RISK_INPUT_MANIFEST_PATH),
        "statistical_engine_sha256": sha256_file(SCRIPT_DIR / "paired_uncertainty.py"),
        "execution_entrypoint_sha256": sha256_file(Path(__file__)),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(f"formal risk batch manifest mismatch: {manifest_path}, key={key}")
    if sha256_file(cells_path) != payload.get("cells_sha256"):
        raise ContractError(f"formal risk batch cell hash mismatch: {cells_path}")
    return payload


def _write_formal_risk_batch(method: str, start: int, stop: int) -> dict[str, object]:
    directory, cells_path, manifest_path = _formal_risk_batch_paths(method, start, stop)
    if directory.exists():
        return _verify_formal_risk_batch(method, start, stop)
    started = time.perf_counter()
    cells = compute_risk_replicates(method, range(start, stop))
    expected_rows = (stop - start) * 5 * len(ORGANS)
    if len(cells) != expected_rows or cells["replicate"].nunique() != stop - start:
        raise ContractError("formal risk batch row/replicate count mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    cells.to_parquet(cells_path, index=False)
    baseline_manifest_path = RISK_BASELINE_DIR / method / "BASELINE_SCORE_MANIFEST.json"
    payload: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.risk_formal_batch.v1",
        "status": "FORMAL_FIXED_SCORE_BATCH_COMPLETE",
        "method": method,
        "replicate_start_inclusive": start,
        "replicate_stop_exclusive": stop,
        "seed": int(_contract()["bootstrap"]["master_seed"]),
        "risk_input_manifest_sha256": sha256_file(RISK_INPUT_MANIFEST_PATH),
        "reference_score_manifest_sha256": sha256_file(RISK_REFERENCE_DIR / "REFERENCE_SCORE_MANIFEST.json"),
        "baseline_score_manifest_sha256": sha256_file(baseline_manifest_path),
        "statistical_engine_sha256": sha256_file(SCRIPT_DIR / "paired_uncertainty.py"),
        "execution_entrypoint_sha256": sha256_file(Path(__file__)),
        "runtime_seconds": time.perf_counter() - started,
        "cell_rows": int(len(cells)),
        "cells_relative_path": _relative(cells_path),
        "cells_sha256": sha256_file(cells_path),
        "inference_conditioning": "outer-test participant bootstrap conditional on calibration-fitted sealed nuisance and fixed held-out scores",
    }
    _write_json_create_once(manifest_path, payload)
    return payload


def _risk_point_estimates(method: str) -> pd.DataFrame:
    ehr = pd.read_csv(
        REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/01_evaluation/parity_audit/cox_fold_organ_cells.csv"
    )
    baseline = pd.read_csv(REVISION_DIR / f"05_downstream/02_baseline_evaluation/{method}/cox_fold_organ_cells.csv")
    keys = ["fold", "organ_system"]
    merged = ehr[keys + ["delta_c"]].merge(
        baseline[keys + ["delta_c"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_ehr", "_baseline"),
        indicator=True,
    )
    if len(merged) != 35 or not merged["_merge"].eq("both").all():
        raise ContractError(f"published risk cells are not aligned: {method}")
    merged["directional_improvement"] = merged["delta_c_ehr"] - merged["delta_c_baseline"]
    organ = merged.groupby("organ_system", as_index=False).agg(
        directional_improvement=("directional_improvement", "mean"),
        ehr_delta_c=("delta_c_ehr", "mean"),
        baseline_delta_c=("delta_c_baseline", "mean"),
        fold_count=("fold", "nunique"),
    )
    if not organ["fold_count"].eq(5).all():
        raise ContractError("published organ risk point estimates do not contain five folds")
    organ["level"] = "organ"
    organ["baseline_method"] = method
    macro = pd.DataFrame(
        [
            {
                "organ_system": "__macro__",
                "directional_improvement": float(organ["directional_improvement"].mean()),
                "ehr_delta_c": float(organ["ehr_delta_c"].mean()),
                "baseline_delta_c": float(organ["baseline_delta_c"].mean()),
                "fold_count": 5,
                "level": "macro",
                "baseline_method": method,
            }
        ]
    )
    return pd.concat([organ, macro], ignore_index=True)


def _summarize_one_risk_method(
    method: str,
    aggregate_replicates: pd.DataFrame,
    *,
    confidence_level: float,
) -> pd.DataFrame:
    point = _risk_point_estimates(method)
    alpha = 1.0 - confidence_level
    rows: list[dict[str, object]] = []
    for point_row in point.to_dict("records"):
        values = aggregate_replicates.loc[
            aggregate_replicates["level"].eq(point_row["level"])
            & aggregate_replicates["organ_system"].eq(point_row["organ_system"]),
            "ehr_minus_baseline_delta_c",
        ].to_numpy(dtype=np.float64)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ContractError(f"invalid formal risk replicates: {method}, {point_row['organ_system']}")
        lower_tail = (int(np.sum(values <= 0.0)) + 1.0) / (len(values) + 1.0)
        upper_tail = (int(np.sum(values >= 0.0)) + 1.0) / (len(values) + 1.0)
        rows.append(
            {
                **point_row,
                "ci_lower_percentile": float(np.quantile(values, alpha / 2.0)),
                "ci_upper_percentile": float(np.quantile(values, 1.0 - alpha / 2.0)),
                "bootstrap_probability_improvement_gt_0": float(np.mean(values > 0.0)),
                "p_value_two_sided_bootstrap": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
                "valid_resamples": int(len(values)),
                "fdr_family": "risk_macro_vs_six_baselines"
                if point_row["level"] == "macro"
                else "risk_organ_vs_six_baselines",
                "q_value_bh": np.nan,
                "fdr_status": "PENDING_ALL_SIX_METHOD_RESULTS",
            }
        )
    return pd.DataFrame(rows)


def finalize_risk_method(method: str, ranges: Sequence[tuple[int, int]]) -> dict[str, object]:
    output_dir = RISK_BOOTSTRAP_DIR / f"02_formal_results/{method}"
    if output_dir.exists():
        receipt = _read_json(output_dir / "FORMAL_COMPLETE.json")
        if receipt.get("status") != "FORMAL_B1000_FIXED_SCORE_COMPLETE":
            raise ContractError(f"existing formal risk output is incomplete: {output_dir}")
        return receipt
    target = int(_contract()["bootstrap"]["risk_target_resamples"])
    coverage: list[int] = []
    parts: list[pd.DataFrame] = []
    batch_receipts: list[dict[str, object]] = []
    for start, stop in ranges:
        receipt = _verify_formal_risk_batch(method, start, stop)
        coverage.extend(range(start, stop))
        _, cells_path, _ = _formal_risk_batch_paths(method, start, stop)
        parts.append(pd.read_parquet(cells_path))
        batch_receipts.append(
            {
                "start": start,
                "stop": stop,
                "manifest_sha256": sha256_file(_formal_risk_batch_paths(method, start, stop)[2]),
            }
        )
    if coverage != list(range(target)):
        raise ContractError("explicit formal risk batch ranges do not cover 0..B-1 exactly in order")
    cells = pd.concat(parts, ignore_index=True)
    if cells.duplicated(["replicate", "fold", "organ_system"]).any():
        raise ContractError("formal risk cells contain duplicate replicate-fold-organ rows")
    organ = cells.groupby(["replicate", "organ_system"], as_index=False).agg(
        ehr_minus_baseline_delta_c=("ehr_minus_baseline_delta_c", "mean"),
        fold_count=("fold", "nunique"),
    )
    if not organ["fold_count"].eq(5).all():
        raise ContractError("formal risk organ replicates do not contain five folds")
    organ["level"] = "organ"
    macro = organ.groupby("replicate", as_index=False).agg(
        ehr_minus_baseline_delta_c=("ehr_minus_baseline_delta_c", "mean"),
        organ_count=("organ_system", "nunique"),
    )
    if not macro["organ_count"].eq(len(ORGANS)).all():
        raise ContractError("formal risk macro replicates do not contain seven organs")
    macro["level"] = "macro"
    macro["organ_system"] = "__macro__"
    aggregate = pd.concat(
        [
            organ[["replicate", "level", "organ_system", "ehr_minus_baseline_delta_c"]],
            macro[["replicate", "level", "organ_system", "ehr_minus_baseline_delta_c"]],
        ],
        ignore_index=True,
    )
    summary = _summarize_one_risk_method(
        method,
        aggregate,
        confidence_level=float(_contract()["bootstrap"]["confidence_level"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cells_path = output_dir / "fold_organ_bootstrap_cells.parquet"
    aggregate_path = output_dir / "organ_macro_bootstrap_replicates.parquet"
    point_path = output_dir / "risk_point_estimates.csv"
    summary_path = output_dir / "risk_paired_summary_fdr_pending.csv"
    cells.to_parquet(cells_path, index=False)
    aggregate.to_parquet(aggregate_path, index=False)
    _risk_point_estimates(method).to_csv(point_path, index=False)
    summary.to_csv(summary_path, index=False)
    receipt: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.risk_formal_complete.v1",
        "status": "FORMAL_B1000_FIXED_SCORE_COMPLETE",
        "method": method,
        "n_resamples": target,
        "seed": int(_contract()["bootstrap"]["master_seed"]),
        "risk_input_manifest_sha256": sha256_file(RISK_INPUT_MANIFEST_PATH),
        "statistical_engine_sha256": sha256_file(SCRIPT_DIR / "paired_uncertainty.py"),
        "execution_entrypoint_sha256": sha256_file(Path(__file__)),
        "conditional_inference": "test-participant bootstrap conditional on sealed/frozen calibration nuisance; nuisance-refit uncertainty is not included",
        "fdr_status": "PENDING_ALL_SIX_METHOD_RESULTS",
        "explicit_batches": batch_receipts,
        "outputs": [
            {"relative_path": _relative(path), "sha256": sha256_file(path)}
            for path in (cells_path, aggregate_path, point_path, summary_path)
        ],
    }
    _write_json_create_once(output_dir / "FORMAL_COMPLETE.json", receipt)
    return receipt


def run_risk_formal(method: str, *, batch_size: int) -> dict[str, object]:
    if batch_size < 1:
        raise ContractError("formal risk batch_size must be positive")
    verify_risk_input_manifest()
    target = int(_contract()["bootstrap"]["risk_target_resamples"])
    ranges = [(start, min(start + batch_size, target)) for start in range(0, target, batch_size)]
    for start, stop in ranges:
        _write_formal_risk_batch(method, start, stop)
    return finalize_risk_method(method, ranges)


def finalize_risk_fdr() -> dict[str, object]:
    output_dir = RISK_BOOTSTRAP_DIR / "03_all_six_fdr"
    if output_dir.exists():
        receipt = _read_json(output_dir / "FDR_COMPLETE.json")
        if receipt.get("status") != "ALL_SIX_RISK_FDR_COMPLETE":
            raise ContractError("existing all-six risk FDR directory is incomplete")
        return receipt
    methods = list(_contract()["baseline_methods"])
    parts: list[pd.DataFrame] = []
    method_receipts: list[dict[str, str]] = []
    for method in methods:
        result_dir = RISK_BOOTSTRAP_DIR / f"02_formal_results/{method}"
        receipt_path = result_dir / "FORMAL_COMPLETE.json"
        receipt = _read_json(receipt_path)
        if receipt.get("status") != "FORMAL_B1000_FIXED_SCORE_COMPLETE":
            raise ContractError(f"formal risk result is incomplete: {method}")
        parts.append(pd.read_csv(result_dir / "risk_paired_summary_fdr_pending.csv"))
        method_receipts.append({"method": method, "receipt_sha256": sha256_file(receipt_path)})
    combined = pd.concat(parts, ignore_index=True).drop(columns=["q_value_bh", "fdr_status"])
    combined = apply_bh_fdr(combined, p_col="p_value_two_sided_bootstrap", family_col="fdr_family")
    family_sizes = combined.groupby("fdr_family").size().to_dict()
    if family_sizes != {"risk_macro_vs_six_baselines": 6, "risk_organ_vs_six_baselines": 42}:
        raise ContractError(f"risk FDR family sizes mismatch: {family_sizes}")
    combined["fdr_status"] = "FINAL_ALL_SIX_METHOD_FAMILY"
    output_dir.mkdir(parents=True, exist_ok=False)
    table_path = output_dir / "risk_paired_summary_with_bh_fdr.csv"
    combined.to_csv(table_path, index=False)
    receipt: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.risk_fdr_complete.v1",
        "status": "ALL_SIX_RISK_FDR_COMPLETE",
        "family_sizes": family_sizes,
        "method_receipts": method_receipts,
        "table_relative_path": _relative(table_path),
        "table_sha256": sha256_file(table_path),
    }
    _write_json_create_once(output_dir / "FDR_COMPLETE.json", receipt)
    return receipt


def _load_baseline_test_predictions(method: str, strict_ids: frozenset[str]) -> pd.DataFrame:
    entries = resolve_registry_prediction_files(BASELINE_REGISTRIES[method], expected_method=method, verify_hash=True)
    parts: list[pd.DataFrame] = []
    for entry in entries:
        if entry["split_role"] != "test":
            continue
        frame = pd.read_parquet(
            Path(entry["path"]),
            columns=["participant_id", "organ_system", "actual_age", "predicted_age", "outer_fold", "split_role"],
        )
        if not frame["split_role"].astype(str).eq("test").all():
            raise ContractError(f"{method} test registry file contains a non-test row")
        if not pd.to_numeric(frame["outer_fold"], errors="coerce").eq(int(entry["outer_fold"])).all():
            raise ContractError(f"{method} test registry file contains the wrong outer fold")
        frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
        parts.append(frame.loc[frame["participant_id"].isin(strict_ids)].drop(columns="split_role"))
    return pd.concat(parts, ignore_index=True)


def _published_strict_metrics(method: str) -> dict[str, float]:
    if method == "ehr_age_d07_003_e55":
        path = REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/01_evaluation/parity_audit/evaluation_summary.json"
    else:
        path = REVISION_DIR / f"05_downstream/02_baseline_evaluation/{method}/evaluation_summary.json"
    payload = _read_json(path)
    values = payload["age_metrics"]["strict_healthy_reference_60m"]
    return {metric: float(values[metric]) for metric in ("mae", "pearson", "r2")}


def run_age_bootstrap(*, n_resamples: int | None, batch_size: int) -> dict[str, object]:
    outputs = [
        AGE_DIR / "strict_age_participant_sufficient_statistics.parquet",
        AGE_DIR / "age_point_estimates.csv",
        AGE_DIR / "age_bootstrap_replicates.parquet",
        AGE_DIR / "age_paired_summary.csv",
        AGE_DIR / "AGE_POINT_PARITY.json",
        AGE_DIR / "AGE_RUN_MANIFEST.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise ContractError(f"formal age outputs are create-once and already exist: {existing}")
    verify_input_manifest()
    contract = _contract()
    bootstrap_contract = contract["bootstrap"]
    resamples = int(n_resamples or bootstrap_contract["age_resamples"])
    if resamples < 2:
        raise ContractError("age bootstrap requires at least two resamples")
    seed = int(bootstrap_contract["master_seed"])
    confidence_level = float(bootstrap_contract["confidence_level"])
    ehr_method = str(contract["ehr_method"])
    baseline_methods = tuple(str(item) for item in contract["baseline_methods"])

    started = time.perf_counter()
    oof_path = REVISION_DIR / "03_ehr_age/04_fixed_replication/02_published_fivefold/02_oof_hr_km_inputs/ehr_age_d07_003_e55_oof.parquet"
    ehr_oof = pd.read_parquet(
        oof_path,
        columns=[
            "participant_id",
            "organ_system",
            "outer_fold",
            "actual_age",
            "strict_healthy_reference_60m",
            "predicted_age_raw",
        ],
    )
    strict_ids = frozenset(
        ehr_oof.loc[ehr_oof["strict_healthy_reference_60m"].eq(1), "participant_id"].astype(str).str.strip()
    )
    baselines = {method: _load_baseline_test_predictions(method, strict_ids) for method in baseline_methods}
    age_frame = build_age_analysis_frame(ehr_oof, baselines, ehr_method=ehr_method)
    del ehr_oof, baselines
    stats = participant_age_sufficient_statistics(
        age_frame,
        prediction_columns=(ehr_method, *baseline_methods),
    )
    del age_frame
    point = paired_age_point_estimates(stats, ehr_method=ehr_method, baseline_methods=baseline_methods)

    parity_rows: list[dict[str, object]] = []
    published_ehr = _published_strict_metrics(ehr_method)
    for method in baseline_methods:
        published_baseline = _published_strict_metrics(method)
        for metric in ("mae", "pearson", "r2"):
            row = point.loc[point["baseline_method"].eq(method) & point["metric"].eq(metric)].iloc[0]
            ehr_difference = float(row["ehr_value"] - published_ehr[metric])
            baseline_difference = float(row["baseline_value"] - published_baseline[metric])
            parity_rows.append(
                {
                    "baseline_method": method,
                    "metric": metric,
                    "ehr_observed": float(row["ehr_value"]),
                    "ehr_published": published_ehr[metric],
                    "ehr_difference": ehr_difference,
                    "baseline_observed": float(row["baseline_value"]),
                    "baseline_published": published_baseline[metric],
                    "baseline_difference": baseline_difference,
                    "pass_atol_1e_10": abs(ehr_difference) <= 1e-10 and abs(baseline_difference) <= 1e-10,
                }
            )
    if not all(bool(row["pass_atol_1e_10"]) for row in parity_rows):
        raise ContractError("age point estimates do not reproduce the locked canonical evaluations")

    replicates = bootstrap_age_from_stats(
        stats,
        ehr_method=ehr_method,
        baseline_methods=baseline_methods,
        replicate_ids=range(resamples),
        seed=seed,
        batch_size=batch_size,
    )
    summary = summarize_paired_replicates(
        replicates,
        point_estimates=point,
        group_columns=("baseline_method", "metric"),
        family="age_all_metrics_vs_six_baselines",
        confidence_level=confidence_level,
    )
    summary = summary.merge(
        point.drop(columns="directional_improvement"),
        on=["baseline_method", "metric"],
        how="left",
        validate="one_to_one",
    )

    AGE_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(outputs[0], index=False)
    point.to_csv(outputs[1], index=False)
    replicates.to_parquet(outputs[2], index=False)
    summary.to_csv(outputs[3], index=False)
    parity_payload: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.age_point_parity.v1",
        "status": "PASS",
        "absolute_tolerance": 1e-10,
        "comparisons": parity_rows,
    }
    _write_json_create_once(outputs[4], parity_payload)
    runtime_seconds = time.perf_counter() - started
    manifest: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.age_bootstrap_run.v1",
        "status": "COMPLETE",
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_PATH),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "seed": seed,
        "n_resamples": resamples,
        "confidence_level": confidence_level,
        "participant_clusters": int(len(stats)),
        "strict_participant_organ_rows": int(stats["n_rows"].sum()),
        "outer_fold_participant_counts": {
            str(int(fold)): int(count) for fold, count in stats.groupby("outer_fold").size().items()
        },
        "runtime_seconds": runtime_seconds,
        "outputs": [
            {"relative_path": _relative(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
            for path in outputs[:5]
        ],
    }
    _write_json_create_once(outputs[5], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock-inputs", help="verify and hash every frozen input")
    age = subparsers.add_parser("age", help="run strict-age paired participant bootstrap")
    age.add_argument("--n-resamples", type=int)
    age.add_argument("--batch-size", type=int, default=25)
    subparsers.add_parser("lock-risk-inputs", help="lock sealed nuisance models and risk code")
    subparsers.add_parser("prepare-risk-reference", help="score sealed EHR base/full models on held-out rows")
    baseline = subparsers.add_parser(
        "prepare-risk-baseline",
        help="fit one baseline nuisance once on calibration and seal held-out scores",
    )
    baseline.add_argument("--method", required=True, choices=list(BASELINE_REGISTRIES))
    benchmark = subparsers.add_parser(
        "benchmark-risk",
        help="benchmark fixed-score participant bootstrap in an isolated non-formal directory",
    )
    benchmark.add_argument("--method", required=True, choices=list(BASELINE_REGISTRIES))
    benchmark.add_argument("--n-resamples", type=int, default=2)
    formal = subparsers.add_parser(
        "risk-formal",
        help="run/resume create-once formal B=1000 fixed-score batches for one method",
    )
    formal.add_argument("--method", required=True, choices=list(BASELINE_REGISTRIES))
    formal.add_argument("--batch-size", type=int, default=20)
    subparsers.add_parser(
        "finalize-risk-fdr",
        help="apply locked BH families after all six formal risk results exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lock-inputs":
        payload = build_input_manifest()
    elif args.command == "age":
        payload = run_age_bootstrap(
            n_resamples=args.n_resamples,
            batch_size=args.batch_size,
        )
    elif args.command == "lock-risk-inputs":
        payload = lock_risk_inputs()
    elif args.command == "prepare-risk-reference":
        payload = prepare_risk_reference()
    elif args.command == "prepare-risk-baseline":
        payload = prepare_risk_baseline(args.method)
    elif args.command == "benchmark-risk":
        payload = benchmark_risk(args.method, n_resamples=args.n_resamples)
    elif args.command == "risk-formal":
        payload = run_risk_formal(args.method, batch_size=args.batch_size)
    elif args.command == "finalize-risk-fdr":
        payload = finalize_risk_fdr()
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
