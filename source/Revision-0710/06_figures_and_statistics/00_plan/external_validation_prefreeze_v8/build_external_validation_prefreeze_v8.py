from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "artifacts_v8"
SMOKE_DIR = HERE / "smoke"
FORMAL_TARGET = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/04_external_validation/source_data_v1"
)
FORMAL_AUTHORIZATION = HERE / "EXTERNAL_VALIDATION_FORMAL_AUTHORIZATION_V8.json"
FORMAL_LOCK = HERE / ".EXTERNAL_VALIDATION_FORMAL_V8_ONCE.lock"
FORMAL_CONSUMED_AUTHORIZATION = HERE / "EXTERNAL_VALIDATION_FORMAL_AUTHORIZATION_V8.CONSUMED.json"
MIMIC_ROOT = Path(r"F:\Data\mimic-iv-3.1")
MIMIC_SHA256SUMS = MIMIC_ROOT / "SHA256SUMS.txt"
NHANES_ROOT = Path(r"F:\Data\NHANES\NHANES")
CLIP_WEIGHT = Path(
    "C:/Users/Mickey/.cache/huggingface/hub/models--timm--"
    "vit_base_patch16_clip_224.openai/snapshots/"
    "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_model.safetensors"
)

ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
P_RELEASE = "P_2017_MARCH2020_PREPANDEMIC"
L_RELEASE = "L_AUG2021_AUG2023"
RELEASES = (P_RELEASE, L_RELEASE)
FORMAL_TAG = "source_data_v1"
FORMAL_BOOTSTRAP_REPLICATES = 200
FORMAL_BOOTSTRAP_SEED = 7_102_026
HASH_FOLD_SALT = "revision0710-external-v3"
STATISTICS_PYTHON = Path(r"D:\Anaconda\python.exe")
INFERENCE_PYTHON = Path(r"C:\Users\Mickey\.conda\envs\Torch_25\python.exe")
STATISTICS_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "statsmodels",
)
INFERENCE_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "open-clip-torch",
)
MIMIC_PRIMARY_VARIANT = "mimic_primary_hospital_icd_only_explicit_unit_phenotype"
MIMIC_BARE_OMR_SENSITIVITY = "mimic_sensitivity_hospital_icd_only_bare_omr_unit_assumptions"
MIMIC_RX_SENSITIVITY = "mimic_sensitivity_hospital_icd_plus_inpatient_prescription_proxy"
NHANES_PRIMARY_VARIANT = "nhanes_primary_phenotype_only_label_blind"
INPUT_VARIANTS = (
    MIMIC_PRIMARY_VARIANT,
    MIMIC_BARE_OMR_SENSITIVITY,
    MIMIC_RX_SENSITIVITY,
    NHANES_PRIMARY_VARIANT,
)
AGE_POPULATIONS = (
    "mimic_primary_common_support_40_73_integer_year",
    "mimic_sensitivity_adult_nontopcoded_integer_year",
    "mimic_sensitivity_common_support_age_minus_0_5_year",
    "mimic_sensitivity_common_support_age_plus_0_5_year",
    "nhanes_primary_examined_40_73_exact",
    "mimic_risk_adult_nontopcoded_integer_year",
)
DROP_REASONS = (
    "none",
    "padding",
    "invalid_or_missing_timestamp",
    "at_or_after_landmark",
    "unsupported_code_system",
    "invalid_or_missing_code",
    "invalid_or_unsupported_code",
    "invalid_or_missing_text",
    "duplicate_after_exact_stable_sort",
    "outside_variant_l32_crop",
    "source_forbidden_for_input_variant",
    "unmapped_inpatient_prescription",
    "label_or_endpoint_field_detected",
)
MIMIC_FLOW_STEPS = (
    "raw_subjects",
    "has_demographics",
    "sex_binary_m_or_f",
    "anchor_age_not_91_topcoded",
    "adult_age_at_time_valid_discharge",
    "alive_at_time_valid_adult_discharge",
    "has_two_valid_adult_surviving_discharge_admissions",
    "second_valid_adult_surviving_discharge_selected",
    "has_subject_specific_observation_end",
    "has_valid_6_or_12_month_endpoint_window",
    "no_preindex_icd9_for_primary_endpoint",
    "model_input_cache_eligible",
)
NHANES_LOCAL_CALIBRATION_TERMS = (
    "intercept",
    "predicted_age_raw_mean5",
    "sex_male",
    "not_applicable_metric_row",
)
ESTIMABILITY_STATUSES = (
    "estimable",
    "zero_target_events",
    "zero_competing_deaths",
    "single_outcome_class",
    "singular_design",
    "insufficient_complete_cases",
    "not_estimable_by_design",
)

INFERENCE_ENVIRONMENT = {
    "python_executable": str(INFERENCE_PYTHON),
    "required_packages": list(INFERENCE_PACKAGES),
    "device": "cpu",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "deterministic_algorithms": True,
    "worker_protocol": "hash_bound_subprocess_with_create_once_stage_receipt",
}
STATISTICS_ENVIRONMENT = {
    "python_executable": str(STATISTICS_PYTHON),
    "required_packages": list(STATISTICS_PACKAGES),
    "cox_implementation": "statsmodels.duration.hazard_regression.PHReg",
    "worker_protocol": "hash_bound_subprocess_with_create_once_stage_receipt",
}

EXPECTED_WORKSPACE_HASHES = {
    "source_paths_config": "7e4e76a9630e47909df7957301a7fe201d5f32ca365e4e81862b9b000d689ec5",
    "cache_config": "c804d07be9430aee98b70e692601b97c3234fb76ca92288ab30d634e276484c0",
    "compact_cache_core": "4687592d19acb730fc2e6c9aa95d019e179d18351f6cb0347c93e51939287fe7",
    "build_compact_cache": "9bcb4746dc787df4d1b8105cfcf3838699c9a25dd0ade891755752f7bbdbb6a6",
    "phenotype_feature_columns": "dd5515d3df836faf51b4449507d89aab38dee0a56dabfe99b6b23e9b1c8605a4",
    "event_code_vocab": "3d2d61b6e0a34261a6d5681e87fc9d539f62376c838a21c04c6cc4c4c47a025c",
    "event_text_vocab": "53b352dd9ebead5d5a78c3e8bf440485341110845d9265cf0620549f70196310",
    "embedding_tables": "d711b36404898e3d488316c7e6eb82668beb6ce60f8c23b1c5e2d85017788d31",
    "embedding_manifest": "5abb2d27bc5b16307578a16a8db6b800b6675a921a1340f87cbf0c3e925d89c6",
    "fold_transform_manifest": "d78f0076071bea925384647bd2c2b72fd6f1139cebac43d967f242eb48211892",
    "ehr_age_model": "9c282a7d534ac49d4ec3dd3e6c6167dc266fa691b1b1af34c1dab547245bd153",
    "private_retain_age": "76f477299dc40af92031f8b76136e25cdda54392d0c02e45bd4f637b1250115b",
    "replication_contract": "05b8c00123cc2c6accb9ec6bd1a33aec999f0789513be4bc9a15c1f68b866f81",
    "prediction_registry": "c11b9d2b81017444d8f10154429f10aa98493c0f75004b2aec770170eb0b639b",
    "correction_parameters": "23af3185a859b6acb90dec9078419a8a67b598001882345ea9678497bbc52cda",
    "five_fold_receipts": "4c7bd87f0dd0e669df2a80d007ae44cd61455df691d97e6bf892cde43a010000",
    "clip_weight": "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f",
}

CHECKPOINT_HASHES = {
    0: "4900c9d332d1a7c56281a465ad9b02d37d69a3a4ed76400cbe98a24efa0a5c08",
    1: "c11c6205db0b2009ef0a5716339cf3686d3fa6884357a1054cf65ff4ba6fc110",
    2: "931fefec0d3077fec6de675f076e3ecc1ce5d5d2dd3fa63e6061d72be5513044",
    3: "461cea401137cf8824a2cf27bf115982c4aff582a17de14ee5928dc8db458b25",
    4: "643712e20d95597f4b9b49c97e0ae47a57396038b642195ae6ce5a04329de54c",
}
CORRECTION_HASHES = {
    0: "e9171acc3a9f25687640c84776a1986d76b984ac14e7bf22f81674e7d63f1c34",
    1: "1f373cb6f9927570d8b2750016c57de26757ffb981806b419d3cd66366b87423",
    2: "5eef990122f44da3d63e756515d32c83b6b98028eaca944eb58fe78f7118f69b",
    3: "66bddfba3e9f3c589dabfb3774d299b202cf151c4dfe8b31be3d9bb349296650",
    4: "b34c0e40f9cb298a40160f84d5724ba29a8de977ca34589698da1f3a034fabcc",
}
RECEIPT_HASHES = {
    0: "979b1246c02c240da1ebf6f422c7b94ba295400f017866d0c5ed37a84f861e27",
    1: "89228f37bdac846e29847e414a278c8417cfd6e45f029e04e74119fb9ed6a281",
    2: "2c9b7f93b681904420b52a0605117d6b2b205c3d6fb21bc82b4b277a824d09bb",
    3: "d1ecc1370f15574d602572d7fe2bca8f6f92a825355ebd0ec9fa2399ea946eb5",
    4: "4587095f46bf241821036b3feb4d74ae6e84c4265581786a147c2c59d9d33364",
}
TRANSFORM_HASHES = {
    0: "98b8ba07bfd248a584efd62726f4d6f373d74143a5362faabed3b991f5ab43cf",
    1: "30467f63014cea4d24af835f4b602bbf14f3c6c9d26b4a2d38c5d019ad04f03b",
    2: "5a6cc638c1ed0bb943b4278bdf62c5cdc0590c51239dc35115ddbe58cf547a16",
    3: "40a2fba5ca510c49c198bdef183b2cc0738da70e79d0980d606982d22038a116",
    4: "28b44283e543f161f4b0df31f2aa76985722eb1e47b477bd35ee183aa635425c",
}

V7_PREFREEZE_MANIFEST_SHA256 = "f29138079ab1a28eda07fa5fa32264f635c6503d8e8f28c452937e954ada164a"
V7_CONTRACT_SHA256 = "3ff8ef20701ce09fd9cfc39e4eb182fb473f439291d92b2999d7ce7e318f716b"
V7_RUNTIME_SHA256 = "73f4996b73be55916b9811dba1c50a65c6708866f62baf42e1be6a5b1f563244"
V7_SOURCE_REGISTRY_SHA256 = "9faec15419d1a34d15ee40767d02095167b59209b5170faf602f8f6329a3bd76"
V7_SCIENTIFIC_NO_GO_SHA256 = "4c85c203ea81f58bb2dbc36da324ba20dc3829d5847b167afa8200a9c8b44d07"
V8_ATTEMPT_01_SUPPORT_NO_GO_SHA256 = "94c2168ee4d0b7862767c2aee0d7acdd1bb29299ce31836617b18f80b0ce3e7f"
V6_FROZEN_CORRECTED_CORE_HOURS = 11.662225266512021
V6_FRESH_CORRECTED_CORE_HOURS = 12.347672947476752
PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS = 4
CONDITIONAL_BOOTSTRAP_REPLICATES_PER_MODEL = 200
OPERATIONAL_RESERVATION_MIN_HOURS = 36.0
OPERATIONAL_RESERVATION_SAFER_HOURS = 48.0
RESOURCE_RESERVATION_COMPONENTS = [
    "label-blind cache construction",
    "CLIP vocabulary extension encoding and model restore",
    "prediction sealing and disk IO",
    "NHANES survey and age-metric aggregation",
    "stage05 and stage06 four-model conditional B=200 bootstrap statistics",
    "complete shared-core PH diagnostics for all target/death model covariates",
    "publication copy, rehash, and receipt validation",
]


def build_resource_budget() -> dict[str, Any]:
    return {
        "schema_version": "revision0710.external_validation.resource_budget.v8",
        "status": "V8_FOUR_MODEL_B200_COMPLETE_PH_ACCOUNTING_36_TO_48_HOUR_RESERVATION",
        "scientific_method_changed_from_v7": True,
        "predecessor_evidence": {
            "v7_prefreeze_manifest_sha256": V7_PREFREEZE_MANIFEST_SHA256,
            "v7_contract_sha256": V7_CONTRACT_SHA256,
            "v7_runtime_sha256": V7_RUNTIME_SHA256,
            "v7_source_registry_sha256": V7_SOURCE_REGISTRY_SHA256,
            "v7_scientific_no_go_sha256": V7_SCIENTIFIC_NO_GO_SHA256,
            "v8_attempt_01_support_no_go_sha256": V8_ATTEMPT_01_SUPPORT_NO_GO_SHA256,
        },
        "accounting": {
            "heldout_analysis_count": 210,
            "predictive_models_per_heldout_analysis": PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS,
            "cause_specific_cox_fits_per_heldout_analysis": 8,
            "shared_core_ph_fits_per_heldout_analysis": 8,
            "conditional_bootstrap_calls_per_heldout_analysis": PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS,
            "conditional_bootstrap_replicates_per_model": CONDITIONAL_BOOTSTRAP_REPLICATES_PER_MODEL,
            "full_association_analysis_count": 42,
            "full_association_primary_fits_per_analysis": 1,
            "full_association_shared_core_ph_fits_per_analysis": 1,
            "formal_recompute_passes": 2,
            "total_cause_specific_predictive_fits": 3360,
            "total_heldout_shared_core_ph_fits": 3360,
            "total_conditional_b200_calls": 1680,
            "total_full_association_primary_fits": 84,
            "total_full_association_shared_core_ph_fits": 84,
            "v6_frozen_corrected_core_hours": V6_FROZEN_CORRECTED_CORE_HOURS,
            "v6_fresh_corrected_core_hours": V6_FRESH_CORRECTED_CORE_HOURS,
        },
        "operational_reservation": {
            "minimum_hours": OPERATIONAL_RESERVATION_MIN_HOURS,
            "safer_hours": OPERATIONAL_RESERVATION_SAFER_HOURS,
            "not_a_wall_time_upper_bound": True,
            "covers_or_reserves": RESOURCE_RESERVATION_COMPONENTS,
            "monitoring": "retain resumable stage receipts and monitor RAM, disk, and elapsed time",
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_closed_object
    )
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalize_unicode_text(value: object) -> str:
    if value is None:
        raise ValueError("text value cannot be None")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError("text value cannot be missing/NaN")
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split())


def canonicalize_external_code(raw_code: object, code_system: str) -> str:
    canonical = normalize_unicode_text(raw_code).upper().replace(".", "")
    if not canonical or any(character.isspace() for character in canonical):
        raise ValueError("external code must be non-empty and contain no internal whitespace")
    if code_system == "ICD10":
        if not canonical[0].isalpha() or not canonical[1:].isalnum():
            raise ValueError("invalid ICD10 code")
        return canonical
    if code_system == "ICD9":
        if not canonical.isalnum():
            raise ValueError("invalid ICD9 code")
        return f"ICD9\x1f{canonical}"
    raise ValueError(f"unknown code system: {code_system}")


def canonicalize_external_description(
    raw_description: object, *, code_system: str, canonical_code: str
) -> str:
    try:
        description = normalize_unicode_text(raw_description)
    except ValueError:
        description = ""
    if description:
        return description
    bare = canonical_code.split("\x1f", 1)[-1]
    return f"{code_system} code {bare}"


def external_text_key(role: str, description: str) -> str:
    if role not in {"icd_description", "inpatient_prescription_proxy"}:
        raise ValueError(f"invalid text role: {role}")
    clean = normalize_unicode_text(description)
    if not clean:
        raise ValueError("text key description must be non-empty")
    return f"{role}\x1f{clean}"


def deterministic_hash_fold(identifier: object) -> int:
    payload = f"{HASH_FOLD_SALT}|{normalize_unicode_text(identifier)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 5


def corrected_gap(
    raw_gap: np.ndarray | float,
    chronological_age: np.ndarray | float,
    intercept: float,
    slope: float,
) -> np.ndarray:
    return np.asarray(raw_gap, dtype=np.float64) - (
        float(intercept)
        + float(slope) * np.asarray(chronological_age, dtype=np.float64)
    )


def encode_mimic_sex(raw_sex: object) -> int:
    if raw_sex == "M":
        return 1
    if raw_sex == "F":
        return 0
    raise ValueError("MIMIC primary sex must be exactly M or F")


def encode_nhanes_sex(riagendr: object) -> int:
    try:
        value = float(riagendr)
    except (TypeError, ValueError) as exc:
        raise ValueError("NHANES RIAGENDR must be numeric 1 or 2") from exc
    if not np.isfinite(value):
        raise ValueError("NHANES RIAGENDR cannot be missing/nonfinite")
    if value == 1.0:
        return 1
    if value == 2.0:
        return 0
    raise ValueError("NHANES primary RIAGENDR must be exactly 1 or 2")


def mimic_integer_age(anchor_age: object, anchor_year: object, index_time: object) -> int:
    try:
        age = int(anchor_age)
        year = int(anchor_year)
    except (TypeError, ValueError) as exc:
        raise ValueError("MIMIC anchor_age and anchor_year must be integers") from exc
    if float(anchor_age) != age or float(anchor_year) != year:
        raise ValueError("MIMIC anchor_age and anchor_year must have integer resolution")
    if age == 91:
        raise ValueError("MIMIC anchor_age=91 is top-coded and excluded")
    timestamp = pd.Timestamp(index_time)
    if pd.isna(timestamp):
        raise ValueError("MIMIC index time is missing")
    result = age + (int(timestamp.year) - year)
    if result < 18:
        raise ValueError("MIMIC index age is not adult")
    return result


def mimic_age_population_mask(age: np.ndarray, population: str) -> np.ndarray:
    values = np.asarray(age, dtype=np.float64)
    if population == "mimic_primary_common_support_40_73_integer_year":
        return np.isfinite(values) & (values >= 40) & (values <= 73) & (values == np.floor(values))
    if population == "mimic_sensitivity_adult_nontopcoded_integer_year":
        return np.isfinite(values) & (values >= 18) & (values <= 90) & (values == np.floor(values))
    if population in {
        "mimic_sensitivity_common_support_age_minus_0_5_year",
        "mimic_sensitivity_common_support_age_plus_0_5_year",
    }:
        return np.isfinite(values) & (values >= 40) & (values <= 73) & (values == np.floor(values))
    raise ValueError(f"invalid MIMIC age population: {population}")


def checkpoint_paths() -> dict[int, Path]:
    base = ROOT / "Revision-0710/03_ehr_age"
    return {
        0: base
        / "03_runs/fold0/open_development/candidates/003_private_retain/"
        "epoch_checkpoints/trainable_checkpoint_epoch_0055.pt",
        **{
            fold: base
            / "04_fixed_replication/01_runs"
            / f"outer_fold_{fold}/attempts/primary_e55_v1/fixed_e55_checkpoint.pt"
            for fold in range(1, 5)
        },
    }


def correction_paths() -> dict[int, Path]:
    base = ROOT / "Revision-0710/03_ehr_age/04_fixed_replication/01_runs"
    return {
        0: base
        / "outer_fold_0/reuse_d07_003_e55/sealed_nuisance/age_correction.json",
        **{
            fold: base
            / f"outer_fold_{fold}/attempts/primary_e55_v1/evaluation/"
            "sealed_nuisance/age_correction.json"
            for fold in range(1, 5)
        },
    }


def receipt_paths() -> dict[int, Path]:
    base = ROOT / "Revision-0710/03_ehr_age/04_fixed_replication/01_runs"
    return {
        0: base / "outer_fold_0/reuse_d07_003_e55/FOLD_COMPLETE_RECEIPT.json",
        **{
            fold: base
            / f"outer_fold_{fold}/attempts/primary_e55_v1/score/"
            "FOLD_COMPLETE_RECEIPT.json"
            for fold in range(1, 5)
        },
    }


def transform_paths() -> dict[int, Path]:
    base = ROOT / "Revision-0710/01_data_rebuild/04_cache_v1/fold_transforms"
    return {fold: base / f"outer_fold_{fold}.npz" for fold in range(5)}


def workspace_artifact_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "artifact_id": "source_paths_config",
            "path": ROOT / "Revision-0710/01_data_rebuild/00_config/source_paths.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["source_paths_config"],
            "role": "frozen UKB source configuration",
        },
        {
            "artifact_id": "cache_config",
            "path": ROOT / "Revision-0710/01_data_rebuild/04_cache_v1/cache_config.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["cache_config"],
            "role": "compact cache configuration",
        },
        {
            "artifact_id": "compact_cache_core",
            "path": ROOT / "Revision-0710/01_data_rebuild/scripts/compact_cache_core.py",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["compact_cache_core"],
            "role": "authoritative timeline selection and packing algorithm",
        },
        {
            "artifact_id": "build_compact_cache",
            "path": ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/scripts/build_compact_cache.py",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["build_compact_cache"],
            "role": "authoritative compact cache builder",
        },
        {
            "artifact_id": "phenotype_feature_columns",
            "path": ROOT
            / "Revision-0710/01_data_rebuild/01_index_and_events/phenotype_feature_columns.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["phenotype_feature_columns"],
            "role": "frozen 78-slot feature order",
        },
        {
            "artifact_id": "event_code_vocab",
            "path": ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_code_vocab.parquet",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["event_code_vocab"],
            "role": "immutable UKB event code vocabulary",
        },
        {
            "artifact_id": "event_text_vocab",
            "path": ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_text_vocab.parquet",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["event_text_vocab"],
            "role": "immutable UKB namespaced text vocabulary",
        },
        {
            "artifact_id": "embedding_tables",
            "path": ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/embedding_tables.pt",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["embedding_tables"],
            "role": "immutable UKB code/text embedding tables",
        },
        {
            "artifact_id": "embedding_manifest",
            "path": ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/embedding_manifest.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["embedding_manifest"],
            "role": "embedding model and weight lineage",
        },
        {
            "artifact_id": "fold_transform_manifest",
            "path": ROOT
            / "Revision-0710/01_data_rebuild/04_cache_v1/fold_transform_manifest.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["fold_transform_manifest"],
            "role": "five outer-fold phenotype transform index",
        },
        {
            "artifact_id": "ehr_age_model",
            "path": ROOT / "Revision-0710/03_ehr_age/01_model/ehr_age_model.py",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["ehr_age_model"],
            "role": "EHR-AGE model code",
        },
        {
            "artifact_id": "private_retain_age",
            "path": ROOT / "Revision-0710/03_ehr_age/01_model/private_retain_age.py",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["private_retain_age"],
            "role": "selected private RETAIN module code",
        },
        {
            "artifact_id": "replication_contract",
            "path": ROOT
            / "Revision-0710/03_ehr_age/00_contract/"
            "ehr_age_d07_003_e55_fixed_replication_v1.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["replication_contract"],
            "role": "fixed five-fold model contract",
        },
        {
            "artifact_id": "prediction_registry",
            "path": ROOT
            / "Revision-0710/03_ehr_age/04_fixed_replication/02_published_fivefold/"
            "00_prediction_registry/ehr_age_d07_003_e55.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["prediction_registry"],
            "role": "published five-checkpoint prediction registry",
        },
        {
            "artifact_id": "correction_parameters",
            "path": ROOT
            / "Revision-0710/03_ehr_age/04_fixed_replication/02_published_fivefold/"
            "01_evaluation/parity_audit/linear_correction_parameters.csv",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["correction_parameters"],
            "role": "published 5x7 correction parameter table",
        },
        {
            "artifact_id": "five_fold_receipts",
            "path": ROOT
            / "Revision-0710/03_ehr_age/04_fixed_replication/02_published_fivefold/"
            "03_receipts/FIVE_FOLD_RECEIPTS.json",
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["five_fold_receipts"],
            "role": "fold completion receipt index",
        },
        {
            "artifact_id": "clip_weight",
            "path": CLIP_WEIGHT,
            "expected_sha256": EXPECTED_WORKSPACE_HASHES["clip_weight"],
            "role": "OpenCLIP ViT-B-16 OpenAI safetensors weight",
        },
        {
            "artifact_id": "shared_survival_analysis_core",
            "path": ROOT
            / "Revision-0710/06_figures_and_statistics/03_reviewer_statistics/"
            "04_cox_diagnostics_and_competing_risk/survival_analysis_core.py",
            "expected_sha256": "6c4dd7eef4ea690d0d88a330f117d8d0b45274d788e10862ab665e245e435a1b",
            "role": "byte-bound Cox fit and scaled Schoenfeld PH diagnostic core",
        },
    ]
    for fold in range(5):
        specs.extend(
            [
                {
                    "artifact_id": f"checkpoint_fold_{fold}",
                    "path": checkpoint_paths()[fold],
                    "expected_sha256": CHECKPOINT_HASHES[fold],
                    "role": "frozen E55 trainable checkpoint",
                    "outer_fold": fold,
                },
                {
                    "artifact_id": f"correction_fold_{fold}",
                    "path": correction_paths()[fold],
                    "expected_sha256": CORRECTION_HASHES[fold],
                    "role": "sealed UKB validation-fold age-gap correction",
                    "outer_fold": fold,
                },
                {
                    "artifact_id": f"completion_receipt_fold_{fold}",
                    "path": receipt_paths()[fold],
                    "expected_sha256": RECEIPT_HASHES[fold],
                    "role": "fixed E55 fold completion receipt",
                    "outer_fold": fold,
                },
                {
                    "artifact_id": f"phenotype_transform_fold_{fold}",
                    "path": transform_paths()[fold],
                    "expected_sha256": TRANSFORM_HASHES[fold],
                    "role": "frozen outer-fold phenotype transform",
                    "outer_fold": fold,
                },
            ]
        )
    return specs


def external_source_specs() -> list[dict[str, Any]]:
    mimic = [
        ("mimic_patients", "hosp/patients.csv.gz", "cohort_neutral"),
        ("mimic_admissions", "hosp/admissions.csv.gz", "cohort_neutral"),
        ("mimic_diagnoses", "hosp/diagnoses_icd.csv.gz", "model_and_endpoint"),
        ("mimic_diagnosis_dictionary", "hosp/d_icd_diagnoses.csv.gz", "mapping"),
        ("mimic_omr", "hosp/omr.csv.gz", "model_input"),
        ("mimic_labevents", "hosp/labevents.csv.gz", "model_input"),
        ("mimic_lab_dictionary", "hosp/d_labitems.csv.gz", "mapping"),
        (
            "mimic_prescriptions",
            "hosp/prescriptions.csv.gz",
            "nonprimary_medication_proxy_sensitivity_only",
        ),
    ]
    records = [
        {
            "source_id": source_id,
            "cohort": "MIMIC_IV_3_1",
            "release_id": "MIMIC_IV_3_1",
            "role": role,
            "path": MIMIC_ROOT / relative,
            "format": "csv.gz",
        }
        for source_id, relative, role in mimic
    ]
    p_files = [
        ("nhanes_p_demo", "Demographics/p_demo.xpt", "cohort_neutral"),
        ("nhanes_p_bmx", "Examination/p_bmx.xpt", "model_input"),
        ("nhanes_p_bpxo", "Examination/p_bpxo.xpt", "model_input"),
        ("nhanes_p_cbc", "Laboratory/p_cbc.xpt", "model_input"),
        ("nhanes_p_alb_cr", "Laboratory/p_alb_cr.xpt", "model_input"),
        ("nhanes_p_biopro", "Laboratory/p_biopro.xpt", "model_input"),
        ("nhanes_p_tchol", "Laboratory/p_tchol.xpt", "model_input"),
        ("nhanes_p_hdl", "Laboratory/p_hdl.xpt", "model_input"),
        ("nhanes_p_hscrp", "Laboratory/p_hscrp.xpt", "model_input"),
        ("nhanes_p_ghb", "Laboratory/p_ghb.xpt", "model_input"),
        ("nhanes_p_glu", "Laboratory/p_glu.xpt", "model_input"),
        ("nhanes_p_trigly", "Laboratory/p_trigly.xpt", "model_input"),
        ("nhanes_p_bpq", "Questionnaire/p_bpq.xpt", "label_only"),
        ("nhanes_p_mcq", "Questionnaire/p_mcq.xpt", "label_only"),
        ("nhanes_p_diq", "Questionnaire/p_diq.xpt", "label_only"),
        ("nhanes_p_kiq", "Questionnaire/p_kiq_u.xpt", "label_only"),
    ]
    l_files = [
        ("nhanes_l_demo", "Demographics/DEMO_L.xpt", "cohort_neutral"),
        ("nhanes_l_bmx", "Examination/BMX_L.xpt", "model_input"),
        ("nhanes_l_bpxo", "Examination/BPXO_L.xpt", "model_input"),
        ("nhanes_l_cbc", "Laboratory/CBC_L.xpt", "model_input"),
        ("nhanes_l_alb_cr", "Laboratory/ALB_CR_L.xpt", "model_input"),
        ("nhanes_l_biopro", "Laboratory/BIOPRO_L.xpt", "model_input"),
        ("nhanes_l_tchol", "Laboratory/TCHOL_L.xpt", "model_input"),
        ("nhanes_l_hdl", "Laboratory/HDL_L.xpt", "model_input"),
        ("nhanes_l_hscrp", "Laboratory/HSCRP_L.xpt", "model_input"),
        ("nhanes_l_ghb", "Laboratory/GHB_L.xpt", "model_input"),
        ("nhanes_l_glu", "Laboratory/GLU_L.xpt", "model_input"),
        ("nhanes_l_trigly", "Laboratory/TRIGLY_L.xpt", "model_input"),
        ("nhanes_l_tst", "Laboratory/TST_L.xpt", "model_input"),
        ("nhanes_l_vid", "Laboratory/VID_L.xpt", "model_input"),
        ("nhanes_l_bpq", "Questionnaire/BPQ_L.xpt", "label_only"),
        ("nhanes_l_mcq", "Questionnaire/MCQ_L.xpt", "label_only"),
        ("nhanes_l_diq", "Questionnaire/DIQ_L.xpt", "label_only"),
        ("nhanes_l_kiq", "Questionnaire/KIQ_U_L.xpt", "label_only"),
    ]
    for release_id, release_dir, specs in (
        (P_RELEASE, "2019-2020", p_files),
        (L_RELEASE, "2021-2023", l_files),
    ):
        records.extend(
            {
                "source_id": source_id,
                "cohort": "NHANES",
                "release_id": release_id,
                "role": role,
                "path": NHANES_ROOT / release_dir / relative,
                "format": "xpt",
            }
            for source_id, relative, role in specs
        )
    return records


def _gzip_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
        return next(csv.reader([handle.readline().rstrip("\r\n")]))


def read_mimic_official_checksums() -> dict[str, str]:
    if not MIMIC_SHA256SUMS.is_file():
        raise FileNotFoundError(MIMIC_SHA256SUMS)
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        MIMIC_SHA256SUMS.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise AssertionError(f"invalid MIMIC checksum line {line_number}")
        digest, relative_path = parts
        relative_path = relative_path.lstrip("*").replace("\\", "/")
        if relative_path in checksums:
            raise AssertionError(f"duplicate MIMIC checksum path: {relative_path}")
        checksums[relative_path] = digest.lower()
    return checksums


def build_mimic_omr_value_branch_audit() -> dict[str, Any]:
    names = ("Height", "Height (Inches)", "Weight", "Weight (Lbs)")
    counts = {
        name: {
            "rows": 0,
            "numeric": 0,
            "inch_branch_40_90": 0,
            "cm_branch_120_230": 0,
            "weight_pounds_branch_50_1000": 0,
        }
        for name in names
    }
    for chunk in pd.read_csv(
        MIMIC_ROOT / "hosp/omr.csv.gz",
        compression="gzip",
        usecols=["result_name", "result_value"],
        chunksize=500_000,
    ):
        selected = chunk.loc[chunk["result_name"].isin(names)].copy()
        selected["numeric_value"] = pd.to_numeric(
            selected["result_value"], errors="coerce"
        )
        for name, group in selected.groupby("result_name"):
            values = group["numeric_value"]
            counts[name]["rows"] += int(len(group))
            counts[name]["numeric"] += int(values.notna().sum())
            if name.startswith("Height"):
                counts[name]["inch_branch_40_90"] += int(
                    values.between(40, 90, inclusive="both").sum()
                )
                counts[name]["cm_branch_120_230"] += int(
                    values.between(120, 230, inclusive="both").sum()
                )
            else:
                counts[name]["weight_pounds_branch_50_1000"] += int(
                    values.between(50, 1000, inclusive="both").sum()
                )
    expected = {
        "Height": {"rows": 2365, "numeric": 2360, "inch_branch_40_90": 2354, "cm_branch_120_230": 1, "weight_pounds_branch_50_1000": 0},
        "Height (Inches)": {"rows": 814964, "numeric": 814963, "inch_branch_40_90": 814332, "cm_branch_120_230": 98, "weight_pounds_branch_50_1000": 0},
        "Weight": {"rows": 15108, "numeric": 15108, "inch_branch_40_90": 0, "cm_branch_120_230": 0, "weight_pounds_branch_50_1000": 15088},
        "Weight (Lbs)": {"rows": 2145353, "numeric": 2145353, "inch_branch_40_90": 0, "cm_branch_120_230": 0, "weight_pounds_branch_50_1000": 2144599},
    }
    if counts != expected:
        raise AssertionError(f"MIMIC OMR value-branch audit mismatch: {counts}")
    return {
        "status": "PREFREEZE_OMR_VALUE_BRANCH_AUDIT_PASS",
        "source_sha256": sha256_file(MIMIC_ROOT / "hosp/omr.csv.gz"),
        "counts_exact_order": [
            {"result_name": name, **counts[name]} for name in names
        ],
        "primary_height": "Height (Inches), 40..90 inches, multiply 2.54; all other explicit-name values rejected",
        "primary_weight": "Weight (Lbs), 50..1000 pounds, multiply 0.45359237; all other explicit-name values rejected",
        "bare_height_role": "named sensitivity only with non-overlapping 40..90 inch and 120..230 cm branches",
        "bare_weight_role": "named assume-pounds sensitivity only; never primary",
    }


def build_source_registry() -> dict[str, Any]:
    official_mimic_checksums = read_mimic_official_checksums()
    external_records: list[dict[str, Any]] = []
    nhanes_primary_counts: dict[str, int] = {}
    for spec in external_source_specs():
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha256 = sha256_file(path)
        if spec["cohort"] == "MIMIC_IV_3_1":
            relative_path = path.resolve().relative_to(MIMIC_ROOT.resolve()).as_posix()
            if path.resolve().parent != (MIMIC_ROOT / "hosp").resolve():
                raise AssertionError(f"MIMIC source is outside the official hosp directory: {path}")
            if not path.name.lower().endswith(".csv.gz"):
                raise AssertionError(f"MIMIC source is not an official csv.gz: {path}")
            if relative_path not in official_mimic_checksums:
                raise AssertionError(f"MIMIC source is absent from SHA256SUMS.txt: {relative_path}")
            if observed_sha256 != official_mimic_checksums[relative_path]:
                raise AssertionError(f"MIMIC source differs from SHA256SUMS.txt: {relative_path}")
        if spec["format"] == "xpt":
            frame = pd.read_sas(path, format="xport", encoding="latin1")
            columns = [str(value) for value in frame.columns]
            row_count: int | None = int(len(frame))
            if spec["source_id"] in {"nhanes_p_demo", "nhanes_l_demo"}:
                required = {"RIDSTATR", "RIDAGEYR"}
                if not required.issubset(frame.columns):
                    raise AssertionError(f"NHANES DEMO columns missing: {path}")
                age = pd.to_numeric(frame["RIDAGEYR"], errors="coerce")
                status = pd.to_numeric(frame["RIDSTATR"], errors="coerce")
                eligible = status.eq(2) & age.between(40, 73, inclusive="both") & age.eq(np.floor(age))
                nhanes_primary_counts[spec["release_id"]] = int(eligible.sum())
        else:
            columns = _gzip_header(path)
            row_count = None
        external_records.append(
            {
                "source_id": spec["source_id"],
                "cohort": spec["cohort"],
                "release_id": spec["release_id"],
                "role": spec["role"],
                "format": spec["format"],
                "absolute_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": observed_sha256,
                "row_count": row_count,
                "columns": columns,
                "columns_sha256": sha256_json(columns),
                "restricted_raw_data_copied_to_workspace": False,
            }
        )
    source_ids = [record["source_id"] for record in external_records]
    if len(source_ids) != len(set(source_ids)):
        raise AssertionError("external source IDs are not unique")
    nhanes_paths = [
        record["absolute_path"]
        for record in external_records
        if record["cohort"] == "NHANES"
    ]
    if any("DEMO_J" in path.upper() or "_J.XPT" in path.upper() for path in nhanes_paths):
        raise AssertionError("DEMO_J/J_* is forbidden from the V8 NHANES registry")
    expected_primary_counts = {P_RELEASE: 4916, L_RELEASE: 3720}
    if nhanes_primary_counts != expected_primary_counts:
        raise AssertionError(
            f"NHANES RIDSTATR=2, age 40-73 sanity mismatch: {nhanes_primary_counts}"
        )

    locked_records: list[dict[str, Any]] = []
    for spec in workspace_artifact_specs():
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != spec["expected_sha256"]:
            raise AssertionError(
                f"locked artifact hash mismatch for {spec['artifact_id']}: {observed}"
            )
        relative_path = repo_relative(path) if path.resolve().is_relative_to(ROOT.resolve()) else None
        record = {
            "artifact_id": spec["artifact_id"],
            "role": spec["role"],
            "path_kind": "repo_relative" if relative_path is not None else "absolute_external",
            "path": relative_path if relative_path is not None else str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
        if "outer_fold" in spec:
            record["outer_fold"] = int(spec["outer_fold"])
        locked_records.append(record)

    correction_keys: list[tuple[int, str]] = []
    correction_values: dict[tuple[int, str], dict[str, Any]] = {}
    for fold, path in correction_paths().items():
        payload = read_json(path)
        if payload.get("status") != "FITTED_AND_SEALED_BEFORE_OUTER_READ":
            raise AssertionError(f"fold {fold} correction is not sealed")
        if int(payload.get("outer_fold", -1)) != fold:
            raise AssertionError(f"fold {fold} correction outer-fold mismatch")
        if int(payload.get("calibration_fold", -1)) == fold:
            raise AssertionError(f"fold {fold} correction was calibrated on the outer fold")
        if payload.get("fixed_e55_checkpoint_sha256") != CHECKPOINT_HASHES[fold]:
            raise AssertionError(f"fold {fold} correction/checkpoint binding mismatch")
        if payload.get("replication_contract_sha256") != EXPECTED_WORKSPACE_HASHES["replication_contract"]:
            raise AssertionError(f"fold {fold} correction/contract binding mismatch")
        for parameter in payload.get("parameters", []):
            key = (fold, str(parameter["organ_system"]))
            correction_keys.append(key)
            correction_values[key] = parameter
        receipt = read_json(receipt_paths()[fold])
        if receipt.get("status") != "COMPLETE_FIXED_E55_OUTER_SCORE":
            raise AssertionError(f"fold {fold} completion receipt is not complete")
        if int(receipt.get("outer_fold", -1)) != fold:
            raise AssertionError(f"fold {fold} completion receipt outer-fold mismatch")
        if receipt.get("replication_contract_sha256") != EXPECTED_WORKSPACE_HASHES["replication_contract"]:
            raise AssertionError(f"fold {fold} receipt/contract binding mismatch")
        if receipt.get("fixed_e55_checkpoint_sha256") != CHECKPOINT_HASHES[fold]:
            raise AssertionError(f"fold {fold} receipt/checkpoint binding mismatch")
        if receipt.get("age_correction_sha256") != CORRECTION_HASHES[fold]:
            raise AssertionError(f"fold {fold} receipt/correction binding mismatch")
    if sorted(correction_keys) != sorted((fold, organ) for fold in range(5) for organ in ORGANS):
        raise AssertionError("sealed corrections do not contain the exact 5x7 keys")
    published_correction_path = ROOT / (
        "Revision-0710/03_ehr_age/04_fixed_replication/02_published_fivefold/"
        "01_evaluation/parity_audit/linear_correction_parameters.csv"
    )
    published = pd.read_csv(published_correction_path)
    if len(published) != 35 or published["method"].ne("ehr_age_d07_003_e55").any():
        raise AssertionError("published correction CSV schema/cardinality mismatch")
    published_keys = list(
        zip(published["outer_fold"].astype(int), published["organ_system"].astype(str))
    )
    if published_keys != [(fold, organ) for fold in range(5) for organ in ORGANS]:
        raise AssertionError("published correction CSV 5x7 key order mismatch")
    correction_differences: list[float] = []
    for row in published.itertuples(index=False):
        parameter = correction_values[(int(row.outer_fold), str(row.organ_system))]
        correction_differences.extend(
            [
                abs(float(row.intercept) - float(parameter["intercept"])),
                abs(float(row.age_slope) - float(parameter["slope"])),
            ]
        )
        if int(row.n_reference_rows) != int(parameter["fit_population_count"]):
            raise AssertionError("published correction CSV reference count mismatch")
    correction_max_difference = max(correction_differences, default=0.0)
    if correction_max_difference > 5e-15:
        raise AssertionError("published correction CSV differs from sealed JSON above tolerance")

    transform_manifest_path = ROOT / "Revision-0710/01_data_rebuild/04_cache_v1/fold_transform_manifest.json"
    transform_manifest = json.loads(transform_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(transform_manifest, list) or len(transform_manifest) != 5:
        raise AssertionError("fold transform manifest cardinality mismatch")
    for fold, record in enumerate(transform_manifest):
        if (
            int(record.get("outer_fold", -1)) != fold
            or int(record.get("validation_fold", -1)) != (fold + 1) % 5
            or record.get("sha256") != TRANSFORM_HASHES[fold]
            or Path(str(record.get("path", "")).replace("\\", "/")).as_posix()
            != f"fold_transforms/outer_fold_{fold}.npz"
        ):
            raise AssertionError(f"fold {fold} transform manifest lineage mismatch")

    return {
        "schema_version": "revision0710.external_validation.source_registry.v8",
        "status": "PREFREEZE_READ_ONLY_SOURCE_REGISTRY_PASS",
        "created_date": "2026-07-11",
        "restricted_raw_data_copied_to_workspace": False,
        "external_roots": {
            "mimic_iv_3_1": str(MIMIC_ROOT.resolve()),
            "mimic_official_sha256sums": {
                "absolute_path": str(MIMIC_SHA256SUMS.resolve()),
                "bytes": MIMIC_SHA256SUMS.stat().st_size,
                "sha256": sha256_file(MIMIC_SHA256SUMS),
            },
            "nhanes": str(NHANES_ROOT.resolve()),
        },
        "nhanes_release_policy": {
            "allowed_releases_exact_order": list(RELEASES),
            "p_release_definition": "official 2017-March 2020 pre-pandemic P_* release",
            "l_release_definition": "official August 2021-August 2023 *_L release",
            "cdc_p_demo_documentation": "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2017/DataFiles/P_DEMO.htm",
            "DEMO_J_or_J_star_may_coexist_with_P_star": False,
            "cross_release_pooled_survey_weight_authorized": False,
        },
        "external_sources": external_records,
        "mimic_omr_value_branch_audit": build_mimic_omr_value_branch_audit(),
        "nhanes_primary_age_sample_sanity": {
            "eligibility": "RIDSTATR == 2 and integer RIDAGEYR between 40 and 73 inclusive",
            "release_counts_exact_order": [
                {"release_id": release, "count": nhanes_primary_counts[release]}
                for release in RELEASES
            ],
            "total": sum(nhanes_primary_counts.values()),
            "expected_total": 8636,
            "matches_expected": True,
        },
        "locked_artifacts": locked_records,
        "correction_key_count": len(correction_keys),
        "correction_csv_json_parameter_tolerance": 5e-15,
        "correction_csv_json_max_abs_difference": correction_max_difference,
        "fold_lineage_crosscheck": True,
        "checkpoint_count": 5,
        "correction_file_count": 5,
        "completion_receipt_count": 5,
        "phenotype_transform_count": 5,
    }


TARGET_UNITS = (
    "kg",
    "cm",
    "cm",
    "cm",
    "beats/min",
    "L",
    "L",
    "L/min",
    "mmHg",
    "mmHg",
    "kg/m2",
    "kg",
    "10^9/L",
    "10^12/L",
    "g/dL",
    "%",
    "fL",
    "pg",
    "g/dL",
    "%",
    "10^9/L",
    "%",
    "fL",
    "%",
    "10^9/L",
    "10^9/L",
    "10^9/L",
    "10^9/L",
    "10^9/L",
    "10^9/L",
    "%",
    "%",
    "%",
    "%",
    "%",
    "%",
    "%",
    "10^12/L",
    "fL",
    "fL",
    "%",
    "%",
    "10^9/L",
    "umol/L",
    "mmol/L",
    "mmol/L",
    "g/L",
    "U/L",
    "U/L",
    "g/L",
    "g/L",
    "U/L",
    "umol/L",
    "mmol/L",
    "mmol/L",
    "mmol/L",
    "umol/L",
    "mg/L",
    "mg/L",
    "U/L",
    "mmol/L",
    "mmol/mol",
    "mmol/L",
    "nmol/L",
    "mmol/L",
    "nmol/L",
    "mmol/L",
    "nmol/L",
    "umol/L",
    "nmol/L",
    "g/L",
    "mmol/L",
    "umol/L",
    "nmol/L",
    "ratio",
    "ratio",
    "g/cm2",
    "cm",
)


def feature_names() -> list[str]:
    path = ROOT / (
        "Revision-0710/01_data_rebuild/01_index_and_events/"
        "phenotype_feature_columns.json"
    )
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != 78 or len(set(values)) != 78:
        raise AssertionError("frozen phenotype feature order is not 78 unique strings")
    return [str(value) for value in values]


def _nhanes_release_prefix(release_id: str) -> str:
    if release_id == P_RELEASE:
        return "nhanes_p"
    if release_id == L_RELEASE:
        return "nhanes_l"
    raise ValueError(release_id)


def _nhanes_feature_definition(
    release_id: str, slot: int
) -> dict[str, Any] | None:
    prefix = _nhanes_release_prefix(release_id)
    common: dict[int, tuple[str, tuple[str, ...], str, str, str]] = {
        1: ("bmx", ("BMXWAIST",), "direct", "BMXWAIST", "cm"),
        2: ("bmx", ("BMXHIP",), "direct", "BMXHIP", "cm"),
        3: ("bmx", ("BMXHT",), "direct", "BMXHT", "cm"),
        4: ("bpxo", ("BPXOPLS1",), "direct", "BPXOPLS1", "beats/min"),
        8: ("bpxo", ("BPXODI1",), "direct", "BPXODI1", "mmHg"),
        9: ("bpxo", ("BPXOSY1",), "direct", "BPXOSY1", "mmHg"),
        10: ("bmx", ("BMXBMI",), "direct", "BMXBMI", "kg/m2"),
        11: ("bmx", ("BMXWT",), "direct", "BMXWT", "kg"),
        12: ("cbc", ("LBXWBCSI",), "direct", "LBXWBCSI", "10^9/L"),
        13: ("cbc", ("LBXRBCSI",), "direct", "LBXRBCSI", "10^12/L"),
        14: ("cbc", ("LBXHGB",), "direct", "LBXHGB", "g/dL"),
        15: ("cbc", ("LBXHCT",), "direct", "LBXHCT", "%"),
        16: ("cbc", ("LBXMCVSI",), "direct", "LBXMCVSI", "fL"),
        17: ("cbc", ("LBXMCHSI",), "direct", "LBXMCHSI", "pg"),
        18: ("cbc", ("LBXMC",), "direct", "LBXMC", "g/dL"),
        19: ("cbc", ("LBXRDW",), "direct", "LBXRDW", "%"),
        20: ("cbc", ("LBXPLTSI",), "direct", "LBXPLTSI", "10^9/L"),
        21: (
            "cbc",
            ("LBXPLTSI", "LBXMPSI"),
            "derived",
            "LBXPLTSI * LBXMPSI / 10000.0",
            "%",
        ),
        22: ("cbc", ("LBXMPSI",), "direct", "LBXMPSI", "fL"),
        24: ("cbc", ("LBDLYMNO",), "direct", "LBDLYMNO", "10^9/L"),
        25: ("cbc", ("LBDMONO",), "direct", "LBDMONO", "10^9/L"),
        26: ("cbc", ("LBDNENO",), "direct", "LBDNENO", "10^9/L"),
        27: ("cbc", ("LBDEONO",), "direct", "LBDEONO", "10^9/L"),
        28: ("cbc", ("LBDBANO",), "direct", "LBDBANO", "10^9/L"),
        30: ("cbc", ("LBXLYPCT",), "direct", "LBXLYPCT", "%"),
        31: ("cbc", ("LBXMOPCT",), "direct", "LBXMOPCT", "%"),
        32: ("cbc", ("LBXNEPCT",), "direct", "LBXNEPCT", "%"),
        33: ("cbc", ("LBXEOPCT",), "direct", "LBXEOPCT", "%"),
        34: ("cbc", ("LBXBAPCT",), "direct", "LBXBAPCT", "%"),
        35: ("cbc", ("LBXNRBC",), "direct", "LBXNRBC", "%"),
        43: (
            "alb_cr",
            ("URXUCR",),
            "direct_converted",
            "URXUCR * 88.4",
            "mg/dL",
        ),
        46: ("biopro", ("LBDSALSI",), "direct", "LBDSALSI", "g/L"),
        47: ("biopro", ("LBXSAPSI",), "direct", "LBXSAPSI", "U/L"),
        48: ("biopro", ("LBXSATSI",), "direct", "LBXSATSI", "U/L"),
        51: ("biopro", ("LBXSASSI",), "direct", "LBXSASSI", "U/L"),
        53: ("biopro", ("LBDSBUSI",), "direct", "LBDSBUSI", "mmol/L"),
        54: ("biopro", ("LBDSCASI",), "direct", "LBDSCASI", "mmol/L"),
        55: ("tchol", ("LBDTCSI",), "direct", "LBDTCSI", "mmol/L"),
        56: ("biopro", ("LBDSCRSI",), "direct", "LBDSCRSI", "umol/L"),
        57: ("hscrp", ("LBXHSCRP",), "direct", "LBXHSCRP", "mg/L"),
        59: ("biopro", ("LBXSGTSI",), "direct", "LBXSGTSI", "U/L"),
        60: ("glu", ("LBDGLUSI",), "direct", "LBDGLUSI", "mmol/L"),
        61: (
            "ghb",
            ("LBXGH",),
            "direct_converted",
            "(LBXGH - 2.15) * 10.929",
            "%",
        ),
        62: ("hdl", ("LBDHDDSI",), "direct", "LBDHDDSI", "mmol/L"),
        64: ("trigly", ("LBDLDLSI",), "direct", "LBDLDLSI", "mmol/L"),
        66: ("biopro", ("LBDSPHSI",), "direct", "LBDSPHSI", "mmol/L"),
        68: ("biopro", ("LBDSTBSI",), "direct", "LBDSTBSI", "umol/L"),
        70: ("biopro", ("LBDSTPSI",), "direct", "LBDSTPSI", "g/L"),
        71: ("trigly", ("LBDTRSI",), "direct", "LBDTRSI", "mmol/L"),
        72: ("biopro", ("LBDSUASI",), "direct", "LBDSUASI", "umol/L"),
        75: (
            "bmx",
            ("BMXWAIST", "BMXHIP"),
            "derived",
            "BMXWAIST / BMXHIP if BMXHIP > 0 else missing",
            "ratio",
        ),
    }
    if release_id == L_RELEASE:
        common.update(
            {
                67: ("tst", ("LBXSHBG",), "direct", "LBXSHBG", "nmol/L"),
                69: ("tst", ("LBDTSTSI",), "direct", "LBDTSTSI", "nmol/L"),
                73: ("vid", ("LBXVIDMS",), "direct", "LBXVIDMS", "nmol/L"),
            }
        )
    definition = common.get(slot)
    if definition is None:
        return None
    suffix, variables, status, formula, source_unit = definition
    return {
        "map_status": status,
        "source_file_id": f"{prefix}_{suffix}",
        "source_variables": list(variables),
        "source_unit": source_unit,
        "formula": formula,
    }


def build_nhanes_feature_map(registry: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {record["source_id"]: record for record in registry["external_sources"]}
    rows: list[dict[str, Any]] = []
    no_equivalent = {76}
    for release_id in RELEASES:
        for slot, (name, target_unit) in enumerate(zip(feature_names(), TARGET_UNITS)):
            definition = _nhanes_feature_definition(release_id, slot)
            if definition is None:
                status = "no_defensible_equivalent" if slot in no_equivalent else "not_measured"
                row = {
                    "release_id": release_id,
                    "slot_index": slot,
                    "feature_name": name,
                    "map_status": status,
                    "source_file_id": "",
                    "source_variables": "",
                    "source_unit": "not_applicable",
                    "target_unit": target_unit,
                    "formula": "missing",
                    "null_semantics": "phenotype_raw_zero=0 and phenotype_missing_mask=true",
                    "source_columns_verified": True,
                    "all_78_slots_adjudicated": True,
                    "adjudication_note": (
                        "selected release lacks a defensible measurement"
                        if status == "not_measured"
                        else "available alternatives are not equivalent to the frozen UKB feature"
                    ),
                }
            else:
                source = sources[definition["source_file_id"]]
                missing = sorted(set(definition["source_variables"]) - set(source["columns"]))
                if missing:
                    raise AssertionError(
                        f"NHANES {release_id} slot {slot} source columns missing: {missing}"
                    )
                row = {
                    "release_id": release_id,
                    "slot_index": slot,
                    "feature_name": name,
                    "map_status": definition["map_status"],
                    "source_file_id": definition["source_file_id"],
                    "source_variables": "|".join(definition["source_variables"]),
                    "source_unit": definition["source_unit"],
                    "target_unit": target_unit,
                    "formula": definition["formula"],
                    "null_semantics": "source missing/refused/don't-know/invalid -> raw_zero=0 and missing_mask=true",
                    "source_columns_verified": True,
                    "all_78_slots_adjudicated": True,
                    "adjudication_note": "release-specific direct or deterministic mapping frozen before inference",
                }
            rows.append(row)
    if len(rows) != 156:
        raise AssertionError("NHANES map must contain 78 rows for each of two releases")
    for release_id in RELEASES:
        slots = [row["slot_index"] for row in rows if row["release_id"] == release_id]
        if slots != list(range(78)):
            raise AssertionError(f"NHANES {release_id} slot order mismatch")
    return rows


def _mimic_mapping_definitions() -> dict[int, dict[str, Any]]:
    definitions: dict[int, dict[str, Any]] = {}

    def omr(
        slot: int,
        selector: str,
        source_unit: str,
        formula: str,
        value_rule: str,
        *,
        sensitivity_selector: str = "not_applicable",
        sensitivity_unit: str = "not_applicable",
        sensitivity_formula: str = "not_applicable",
        sensitivity_value_rule: str = "not_applicable",
        quality: str = "stable",
    ) -> None:
        definitions[slot] = {
            "map_status": quality,
            "source_file_id": "mimic_omr",
            "primary_source_selector": selector,
            "primary_source_unit": source_unit,
            "primary_formula": formula,
            "primary_value_rule": value_rule,
            "sensitivity_source_selector": sensitivity_selector,
            "sensitivity_source_unit": sensitivity_unit,
            "sensitivity_formula": sensitivity_formula,
            "sensitivity_value_rule": sensitivity_value_rule,
        }

    def lab(
        slot: int,
        selector: str,
        source_unit: str,
        formula: str,
        value_rule: str = "finite numeric value; negative values rejected",
        quality: str = "stable",
    ) -> None:
        definitions[slot] = {
            "map_status": quality,
            "source_file_id": "mimic_labevents",
            "primary_source_selector": selector,
            "primary_source_unit": source_unit,
            "primary_formula": formula,
            "primary_value_rule": value_rule,
            "sensitivity_source_selector": "not_applicable",
            "sensitivity_source_unit": "not_applicable",
            "sensitivity_formula": "not_applicable",
            "sensitivity_value_rule": "not_applicable",
        }

    omr(
        3,
        "result_name == 'Height (Inches)'",
        "inches",
        "value * 2.54",
        "40 <= value <= 90 inches",
        sensitivity_selector="result_name == 'Height'",
        sensitivity_unit="mixed-by-value branch: inches or cm",
        sensitivity_formula="if 40<=value<=90 then value*2.54; elif 120<=value<=230 then value; else missing",
        sensitivity_value_rule="non-overlapping 40..90 inch and 120..230 cm branches only",
    )
    omr(
        8,
        "result_name in {Blood Pressure,Blood Pressure Sitting,Blood Pressure Lying,Blood Pressure Standing,Blood Pressure Standing (1 min),Blood Pressure Standing (3 mins)}",
        "slash-delimited systolic/diastolic mmHg",
        "parse float after the single '/' as diastolic mmHg",
        "exactly one slash; 20<=diastolic<=200; 40<=systolic<=300; systolic>diastolic",
    )
    omr(
        9,
        "result_name in {Blood Pressure,Blood Pressure Sitting,Blood Pressure Lying,Blood Pressure Standing,Blood Pressure Standing (1 min),Blood Pressure Standing (3 mins)}",
        "slash-delimited systolic/diastolic mmHg",
        "parse float before the single '/' as systolic mmHg",
        "exactly one slash; 20<=diastolic<=200; 40<=systolic<=300; systolic>diastolic",
    )
    omr(
        10,
        "result_name in {BMI,BMI (kg/m2)}",
        "kg/m2",
        "value",
        "10 <= value <= 100 kg/m2",
    )
    omr(
        11,
        "result_name == 'Weight (Lbs)'",
        "pounds",
        "value * 0.45359237",
        "50 <= value <= 1000 pounds",
        sensitivity_selector="result_name == 'Weight'",
        sensitivity_unit="pounds by explicitly named legacy assumption",
        sensitivity_formula="value * 0.45359237",
        sensitivity_value_rule="50<=value<=1000; sensitivity only because bare Weight has no bound unit",
    )

    lab(12, "51301[valueuom=K/uL]|51755[valueuom=K/uL]|51756[valueuom=K/uL]", "K/uL", "value; K/uL numerically equals 10^9/L")
    lab(13, "51279[valueuom=m/uL]", "m/uL", "value; million/uL numerically equals 10^12/L")
    lab(14, "50811[valueuom=g/dL]|51222[valueuom=g/dL]|51640[valueuom=g/dL]", "g/dL", "value")
    lab(15, "51221[valueuom=%]|51638[valueuom=%]|51639[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(16, "51250[valueuom=fL]|51691[valueuom=fL]", "fL", "value")
    lab(17, "51248[valueuom=pg]|53186[valueuom=pg]", "pg", "value")
    lab(18, "51249[valueuom=g/dL]|51249[valueuom=%]|53185[valueuom=g/dL]", "g/dL or percent w/v", "value; percent w/v is numerically g/dL", "finite; percent branch 0<=value<=100; g/dL branch requires value>=0")
    lab(19, "51277[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(20, "51265[valueuom=K/uL]|53189[valueuom=K/uL]", "K/uL", "value; K/uL numerically equals 10^9/L")
    lab(24, "51133[valueuom=K/uL]|52769[valueuom=#/uL]|53157[valueuom=K/uL]", "item-specific K/uL or #/uL", "51133:value; 52769:value/1000; 53157:value")
    lab(25, "52074[valueuom=K/uL]", "K/uL", "value; K/uL numerically equals 10^9/L")
    lab(26, "52075[valueuom=K/uL]|53159[valueuom=K/uL]", "K/uL", "value; K/uL numerically equals 10^9/L")
    lab(27, "52073[valueuom=K/uL]", "K/uL", "value; K/uL numerically equals 10^9/L")
    lab(28, "52069[valueuom=K/uL]", "K/uL", "value; K/uL numerically equals 10^9/L")
    lab(30, "51244[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(31, "51254[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(32, "51256[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(33, "51200[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(34, "51146[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(35, "51257[valueuom=%]", "%", "value", "finite; 0<=value<=100")
    lab(36, "51283[valueuom=%]|51284[valueuom=%]", "%", "value", "finite; 0<=value<=100", "weak")
    lab(37, "51282[valueuom=m/uL]", "m/uL", "value; million/uL numerically equals target 10^12/L", quality="weak")
    lab(43, "51082[valueuom=mg/dL]", "mg/dL", "value * 88.4", quality="weak")
    lab(44, "51097[valueuom=mEq/L]", "mEq/L", "value; monovalent potassium mEq/L equals mmol/L", quality="weak")
    lab(45, "51100[valueuom=mEq/L]", "mEq/L", "value; monovalent sodium mEq/L equals mmol/L", quality="weak")
    lab(46, "50862[valueuom=g/dL]", "g/dL", "value * 10")
    lab(47, "50863[valueuom=IU/L]|53086[valueuom=IU/L]", "IU/L", "value; IU/L treated as U/L")
    lab(48, "50861[valueuom=IU/L]|53084[valueuom=IU/L]", "IU/L", "value; IU/L treated as U/L")
    lab(51, "50878[valueuom=IU/L]", "IU/L", "value; IU/L treated as U/L")
    lab(52, "50883[valueuom=mg/dL]", "mg/dL", "value * 17.104")
    lab(53, "51006[valueuom=mg/dL]|52647[valueuom=mg/dL]", "mg/dL BUN", "value * 0.357")
    lab(54, "50893[valueuom=mg/dL]", "mg/dL", "value * 0.2495")
    lab(55, "50907[valueuom=mg/dL]", "mg/dL", "value * 0.02586")
    lab(56, "50912[valueuom=mg/dL]|52546[valueuom=mg/dL]", "mg/dL", "value * 88.4")
    lab(57, "50889[valueuom=mg/L]", "mg/L", "value")
    lab(59, "50927[valueuom=IU/L]|53093[valueuom=IU/L]", "IU/L", "value; IU/L treated as U/L")
    lab(60, "50931[valueuom=mg/dL]|52569[valueuom=mg/dL]", "mg/dL", "value * 0.0555")
    lab(61, "50852[valueuom=%]", "%", "(value - 2.15) * 10.929", "finite; 2<=value<=25 percent")
    lab(62, "50904[valueuom=mg/dL]", "mg/dL", "value * 0.02586")
    lab(64, "50906[valueuom=mg/dL]", "mg/dL", "value * 0.02586")
    lab(66, "50970[valueuom=mg/dL]", "mg/dL", "value * 0.3229")
    lab(67, "50982[valueuom=nmol/L]", "nmol/L", "value", quality="weak")
    lab(68, "50885[valueuom=mg/dL]|53089[valueuom=mg/dL]", "mg/dL", "value * 17.104")
    lab(69, "50988[valueuom=ng/dL]", "ng/dL", "value * 0.0347", quality="weak")
    lab(70, "50976[valueuom=g/dL]", "g/dL", "value * 10")
    lab(71, "51000[valueuom=mg/dL]", "mg/dL", "value * 0.01129")
    lab(72, "51007[valueuom=mg/dL]", "mg/dL", "value * 59.48")
    lab(73, "50853[valueuom=ng/mL]", "ng/mL", "value * 2.496")
    return definitions


def build_mimic_feature_map(registry: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {record["source_id"]: record for record in registry["external_sources"]}
    lab_dictionary_path = Path(sources["mimic_lab_dictionary"]["absolute_path"])
    lab_dictionary = pd.read_csv(lab_dictionary_path, compression="gzip")
    observed_itemids = set(lab_dictionary["itemid"].astype(int).tolist())
    omr_path = Path(sources["mimic_omr"]["absolute_path"])
    omr_names = set(
        pd.read_csv(omr_path, compression="gzip", usecols=["result_name"])[
            "result_name"
        ]
        .astype("string")
        .dropna()
        .tolist()
    )
    definitions = _mimic_mapping_definitions()
    rows: list[dict[str, Any]] = []
    for slot, (name, target_unit) in enumerate(zip(feature_names(), TARGET_UNITS)):
        definition = definitions.get(slot)
        if definition is None:
            row = {
                "slot_index": slot,
                "feature_name": name,
                "map_status": "not_measured",
                "primary_input_variant": MIMIC_PRIMARY_VARIANT,
                "source_file_id": "",
                "primary_source_selector": "not_applicable",
                "primary_source_unit": "not_applicable",
                "primary_formula": "missing",
                "primary_value_rule": "not_applicable",
                "sensitivity_input_variant": MIMIC_BARE_OMR_SENSITIVITY,
                "sensitivity_source_selector": "not_applicable",
                "sensitivity_source_unit": "not_applicable",
                "sensitivity_formula": "not_applicable",
                "sensitivity_value_rule": "not_applicable",
                "target_unit": target_unit,
                "lookback_days": 365,
                "aggregation": "not_applicable",
                "tie_break": "not_applicable",
                "pre_index_rule": "not_applicable_unmeasured_slot",
                "selector_unit_formula_verified": True,
                "all_78_slots_adjudicated": True,
                "adjudication_note": "MIMIC-IV v3.1 lacks a defensible UKB-equivalent measurement",
            }
        else:
            if definition["source_file_id"] == "mimic_labevents":
                itemids = {
                    int(value.split("[", 1)[0])
                    for value in definition["primary_source_selector"].split("|")
                }
                missing = sorted(itemids - observed_itemids)
                if missing:
                    raise AssertionError(f"MIMIC slot {slot} itemids missing: {missing}")
            elif definition["source_file_id"] == "mimic_omr":
                selector = definition["primary_source_selector"]
                candidates = {name for name in omr_names if name in selector}
                if not candidates:
                    raise AssertionError(
                        f"MIMIC slot {slot} has no observed OMR selector"
                    )
            row = {
                "slot_index": slot,
                "feature_name": name,
                "map_status": definition["map_status"],
                "primary_input_variant": MIMIC_PRIMARY_VARIANT,
                "source_file_id": definition["source_file_id"],
                "primary_source_selector": definition["primary_source_selector"],
                "primary_source_unit": definition["primary_source_unit"],
                "primary_formula": definition["primary_formula"],
                "primary_value_rule": definition["primary_value_rule"],
                "sensitivity_input_variant": MIMIC_BARE_OMR_SENSITIVITY,
                "sensitivity_source_selector": definition["sensitivity_source_selector"],
                "sensitivity_source_unit": definition["sensitivity_source_unit"],
                "sensitivity_formula": definition["sensitivity_formula"],
                "sensitivity_value_rule": definition["sensitivity_value_rule"],
                "target_unit": target_unit,
                "lookback_days": 365,
                "aggregation": "latest finite valid value",
                "tie_break": (
                    "charttime, storetime, labevent_id"
                    if definition["source_file_id"] == "mimic_labevents"
                    else "chartdate, seq_num"
                ),
                "pre_index_rule": (
                    "charttime <= discharge landmark AND storetime <= discharge landmark; both nonmissing"
                    if definition["source_file_id"] == "mimic_labevents"
                    else "chartdate < discharge calendar date AND chartdate >= discharge calendar date minus 365 days"
                ),
                "selector_unit_formula_verified": True,
                "all_78_slots_adjudicated": True,
                "adjudication_note": "source selector, observed unit branch, formula, value rule, and optional sensitivity frozen before inference",
            }
        rows.append(row)
    if len(rows) != 78 or [row["slot_index"] for row in rows] != list(range(78)):
        raise AssertionError("MIMIC map must contain the ordered 78 slots")
    return rows


def build_nhanes_label_map(registry: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {record["source_id"]: record for record in registry["external_sources"]}
    definitions = {
        "cardiovascular": {
            "p": ("nhanes_p_bpq|nhanes_p_mcq", "BPQ020|MCQ160B|MCQ160C|MCQ160D|MCQ160E|MCQ160F"),
            "l": ("nhanes_l_bpq|nhanes_l_mcq", "BPQ020|MCQ160B|MCQ160C|MCQ160D|MCQ160E|MCQ160F"),
        },
        "hepatic": {
            "p": ("nhanes_p_mcq", "MCQ160L"),
            "l": ("nhanes_l_mcq", "MCQ160L"),
        },
        "metabolic": {
            "p": ("nhanes_p_diq", "DIQ010"),
            "l": ("nhanes_l_diq", "DIQ010"),
        },
        "musculoskeletal": {
            "p": ("nhanes_p_mcq", "MCQ160A"),
            "l": ("nhanes_l_mcq", "MCQ160A"),
        },
        "pulmonary": {
            "p": ("nhanes_p_mcq", "MCQ010"),
            "l": ("nhanes_l_mcq", "MCQ010"),
        },
        "renal": {
            "p": ("nhanes_p_kiq", "KIQ022"),
            "l": ("nhanes_l_kiq", "KIQ022"),
        },
    }
    rows: list[dict[str, Any]] = []
    for release_id, short in ((P_RELEASE, "p"), (L_RELEASE, "l")):
        for organ in ORGANS:
            if organ == "immune":
                rows.append(
                    {
                        "release_id": release_id,
                        "organ_system": organ,
                        "estimability": "not_estimable",
                        "source_file_ids": "",
                        "source_variables": "",
                        "positive_rule": "not_applicable",
                        "negative_rule": "not_applicable",
                        "missing_rule": "all rows",
                        "formula": "fixed not_estimable; no defensible questionnaire equivalent",
                        "label_role": "outcome_only",
                        "input_eligible": False,
                        "source_columns_verified": True,
                    }
                )
                continue
            source_ids, variables = definitions[organ][short]
            source_id_list = source_ids.split("|")
            variable_list = variables.split("|")
            available = set().union(*(set(sources[source_id]["columns"]) for source_id in source_id_list))
            missing = sorted(set(variable_list) - available)
            if missing:
                raise AssertionError(
                    f"NHANES label variables missing for {release_id}/{organ}: {missing}"
                )
            rows.append(
                {
                    "release_id": release_id,
                    "organ_system": organ,
                    "estimability": "estimable",
                    "source_file_ids": source_ids,
                    "source_variables": variables,
                    "positive_rule": "any source variable == 1",
                    "negative_rule": "all source variables == 2",
                    "missing_rule": "otherwise missing, including 7/9/refused/don't-know",
                    "formula": "positive=1; negative=0; incomplete/discordant=missing",
                    "label_role": "outcome_only",
                    "input_eligible": False,
                    "source_columns_verified": True,
                }
            )
    if len(rows) != 14:
        raise AssertionError("NHANES label map must have 7 organs x 2 releases")
    if sum(row["estimability"] == "estimable" for row in rows) != 12:
        raise AssertionError("NHANES label map must have six estimable organs per release")
    return rows


def build_mimic_icd10_map() -> list[dict[str, Any]]:
    ranges = {
        "cardiovascular": [("I00", "I99"), ("C38", "C38")],
        "pulmonary": [
            ("J09", "J22"),
            ("J40", "J47"),
            ("J60", "J70"),
            ("J80", "J99"),
            ("C30", "C34"),
        ],
        "musculoskeletal": [("M00", "M94"), ("C40", "C41"), ("C49", "C49")],
        "immune": [("D80", "D89"), ("C81", "C86"), ("C88", "C88"), ("C96", "C96"), ("C37", "C37")],
        "renal": [("N00", "N29"), ("C64", "C68")],
        "hepatic": [("K70", "K77"), ("C22", "C22")],
        "metabolic": [("E08", "E13"), ("E65", "E77"), ("M80", "M85")],
    }
    rows: list[dict[str, Any]] = []
    order = 0
    for organ in ORGANS:
        for start, end in ranges[organ]:
            rows.append(
                {
                    "rule_order": order,
                    "organ_system": organ,
                    "code_system": "ICD10",
                    "match_type": "category3_inclusive_range",
                    "category_start": start,
                    "category_end": end,
                    "normalization": "NFKC; trim; uppercase; remove decimal; category3=first 3 bytes",
                    "prevalent_window": "all observed history through index admission",
                    "incident_event_window": "strictly after discharge landmark",
                    "multi_organ_match_allowed": True,
                    "icd9_endpoint_authorized": False,
                }
            )
            order += 1
    return rows


def _column(
    name: str,
    dtype: str,
    nullable: bool,
    is_key: bool,
    unit: str,
    formula: str,
    *,
    enum: list[Any] | None = None,
    tolerance: str = "exact",
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": dtype,
        "nullable": nullable,
        "is_key": is_key,
        "grain": "assigned_by_table",
        "cardinality": "assigned_by_table",
        "enum": enum,
        "unit": unit,
        "formula": formula,
        "tolerance": tolerance,
    }


def _table(
    table_id: str,
    relative_path: str,
    grain: str,
    key: list[str],
    cardinality: str,
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    for column in columns:
        column["grain"] = grain
        column["cardinality"] = cardinality
    if [column["name"] for column in columns if column["is_key"]] != key:
        raise AssertionError(f"table {table_id} key flags do not match declared key")
    return {
        "table_id": table_id,
        "relative_path": relative_path,
        "grain": grain,
        "key": key,
        "cardinality": cardinality,
        "sort_order": key,
        "columns": columns,
    }


def build_output_schemas() -> dict[str, Any]:
    organ_enum = list(ORGANS)
    cohort_enum = ["MIMIC_IV_3_1", "NHANES"]
    release_enum = [P_RELEASE, L_RELEASE]
    input_variant_enum = list(INPUT_VARIANTS)
    mimic_variant_enum = [MIMIC_PRIMARY_VARIANT, MIMIC_BARE_OMR_SENSITIVITY, MIMIC_RX_SENSITIVITY]
    age_metric_enum = ["mae", "rmse", "pearson", "spearman", "r2", "calibration_intercept", "calibration_slope"]
    estimability_enum = list(ESTIMABILITY_STATUSES)
    tables: list[dict[str, Any]] = []
    tables.append(
        _table(
            "per_checkpoint_predictions",
            "shared/04_inference/per_checkpoint_predictions.parquet",
            "one external participant-organ-checkpoint row",
            ["cohort", "release_id", "participant_id", "input_variant", "organ_system", "checkpoint_outer_fold"],
            "sum over exact cohort-input variants of N_model_eligible * 7 organs * 5 checkpoints",
            [
                _column("cohort", "string", False, True, "category", "source cohort", enum=cohort_enum),
                _column("release_id", "string", True, True, "category", "NHANES release; null for MIMIC", enum=release_enum),
                _column("participant_id", "string", False, True, "identifier", "source identifier; never displayed"),
                _column("landmark_policy", "string", False, False, "category", "exact external landmark", enum=["mimic_second_valid_adult_discharge", "nhanes_mec_exam"]),
                _column("input_variant", "string", False, True, "category", "closed prespecified input variant", enum=input_variant_enum),
                _column("organ_system", "string", False, True, "category", "frozen organ query", enum=organ_enum),
                _column("checkpoint_outer_fold", "int8", False, True, "fold", "frozen outer fold", enum=list(range(5))),
                _column("sex_male", "int8", False, False, "binary", "MIMIC M=1/F=0; NHANES RIAGENDR 1=1/2=0", enum=[0, 1]),
                _column("chronological_age", "float64", False, False, "years", "MIMIC integer-year age; NHANES exact RIDAGEYR", tolerance="exact"),
                _column("predicted_age_raw", "float64", False, False, "years", "base_age_raw + risk_age_offset", tolerance="1e-12"),
                _column("base_age_raw", "float64", False, False, "years", "checkpoint base-age output", tolerance="1e-12"),
                _column("risk_age_offset", "float64", False, False, "years", "checkpoint organ-specific offset", tolerance="1e-12"),
                _column("risk_score", "float64", False, False, "model score", "checkpoint organ risk score", tolerance="1e-12"),
                _column("raw_age_gap", "float64", False, False, "years", "predicted_age_raw - chronological_age", tolerance="1e-12"),
                _column("correction_intercept", "float64", False, False, "years", "sealed UKB fold-organ intercept", tolerance="1e-15"),
                _column("correction_slope", "float64", False, False, "years/year", "sealed UKB fold-organ slope", tolerance="1e-15"),
                _column("corrected_age_gap", "float64", False, False, "years", "raw_age_gap - (correction_intercept + correction_slope * chronological_age)", tolerance="1e-12"),
                _column("checkpoint_sha256", "string", False, False, "sha256", "exact checkpoint bytes"),
                _column("external_cache_manifest_sha256", "string", False, False, "sha256", "exact no-label cache manifest"),
            ],
        )
    )
    tables.append(
        _table(
            "ensemble_predictions",
            "shared/04_inference/ensemble_predictions.parquet",
            "one external participant-organ row",
            ["cohort", "release_id", "participant_id", "input_variant", "organ_system"],
            "sum over exact cohort-input variants of N_model_eligible * 7 organs",
            [
                _column("cohort", "string", False, True, "category", "source cohort", enum=cohort_enum),
                _column("release_id", "string", True, True, "category", "NHANES release; null for MIMIC", enum=release_enum),
                _column("participant_id", "string", False, True, "identifier", "source identifier; never displayed"),
                _column("input_variant", "string", False, True, "category", "closed prespecified input variant", enum=input_variant_enum),
                _column("organ_system", "string", False, True, "category", "frozen organ query", enum=organ_enum),
                _column("sex_male", "int8", False, False, "binary", "exact source-specific binary encoding", enum=[0, 1]),
                _column("chronological_age", "float64", False, False, "years", "MIMIC integer-year age; NHANES exact RIDAGEYR", tolerance="exact"),
                _column("predicted_age_raw_mean5", "float64", False, False, "years", "arithmetic mean of five raw checkpoint predictions", tolerance="1e-12"),
                _column("predicted_age_raw_sd5", "float64", False, False, "years", "sample SD of five raw checkpoint predictions, ddof=1", tolerance="1e-12"),
                _column("raw_age_gap_mean5", "float64", False, False, "years", "mean of five raw_age_gap values", tolerance="1e-12"),
                _column("corrected_age_gap_mean5", "float64", False, False, "years", "mean of five sealed-UKB corrected_age_gap values", tolerance="1e-12"),
                _column("risk_score_mean5", "float64", False, False, "model score", "mean of five risk scores", tolerance="1e-12"),
                _column("complete_five_checkpoint_set", "bool", False, False, "flag", "true iff folds exactly 0..4 and hashes match", enum=[True]),
            ],
        )
    )
    tables.append(
        _table(
            "external_vocabulary_coverage",
            "shared/03_cache_audit/external_vocabulary_coverage.csv",
            "one cohort-partition-input-modality-code-system-token-class-reason row",
            [
                "cohort",
                "source_partition",
                "input_variant",
                "vocabulary_kind",
                "modality",
                "code_system",
                "token_class",
                "reason",
            ],
            "exactly 60 declared cohort-partition-input-modality-code-system-token-class-reason rows",
            [
                _column("cohort", "string", False, True, "category", "source cohort", enum=cohort_enum),
                _column("source_partition", "string", False, True, "category", "MIMIC_IV_3_1 or exact NHANES release", enum=["MIMIC_IV_3_1", *release_enum]),
                _column("input_variant", "string", False, True, "category", "closed prespecified input variant", enum=input_variant_enum),
                _column("vocabulary_kind", "string", False, True, "category", "code and text vocabularies audited separately", enum=["code", "text"]),
                _column("modality", "string", False, True, "category", "event token modality", enum=["hospital_icd", "inpatient_prescription_proxy", "padding"]),
                _column("code_system", "string", False, True, "category", "canonical source code system", enum=["ICD10", "ICD9", "INPATIENT_RX", "not_applicable"]),
                _column("token_class", "string", False, True, "category", "frozen-vocab/extension/drop/zero-event class", enum=["exact_frozen_vocab", "deterministic_extension", "invalid_dropped", "zero_event"]),
                _column("reason", "string", False, True, "category", "closed canonicalization/drop reason", enum=list(DROP_REASONS)),
                _column("source_rows", "int64", False, False, "rows", "rows before deduplication and crop"),
                _column("unique_canonical_tokens", "int64", False, False, "tokens", "unique canonical UTF-8 token byte strings"),
                _column("participants", "int64", False, False, "participants", "unique participants represented by row class"),
                _column("pre_crop_active_events", "int64", False, False, "events", "active events after strict pre-index/dedup and before L32 crop"),
                _column("post_crop_active_events", "int64", False, False, "events", "active events retained in L32"),
                _column("original_vocab_rows", "int64", False, False, "rows", "9906 when vocabulary_kind=code; 44565 when vocabulary_kind=text"),
                _column("original_vocab_prefix_bytes", "int64", False, False, "bytes", "exact serialized original vocabulary prefix bytes"),
                _column("original_vocab_prefix_sha256", "string", False, False, "sha256", "SHA-256 of exact original vocabulary prefix bytes"),
                _column("extension_rows", "int64", False, False, "rows", "deterministically appended token rows"),
                _column("original_embedding_rows", "int64", False, False, "rows", "9906 code or 44565 text original embedding rows"),
                _column("original_embedding_prefix_bytes", "int64", False, False, "bytes", "original_embedding_rows * embedding_width * 2 little-endian bytes"),
                _column("original_embedding_prefix_sha256", "string", False, False, "sha256", "SHA-256 of exact original little-endian float16 embedding prefix bytes"),
                _column("extension_manifest_sha256", "string", False, False, "sha256", "exact create-once extension manifest"),
                _column("original_embedding_prefix_bytes_unchanged", "bool", False, False, "flag", "byte comparison against frozen UKB table", enum=[True]),
                _column("label_or_endpoint_fields_read", "bool", False, False, "flag", "must remain false in token generation", enum=[False]),
            ],
        )
    )
    tables.append(
        _table(
            "zero_shot_age_metrics",
            "shared/05_age_metrics/zero_shot_age_metrics.csv",
            "one cohort-population-organ-estimator-metric row",
            ["cohort", "release_id", "input_variant", "population", "organ_system", "estimator", "metric"],
            "closed cohort-release-input-population combinations * 9 organ summaries * 6 estimators * 7 metrics",
            [
                _column("cohort", "string", False, True, "category", "source cohort", enum=cohort_enum),
                _column("release_id", "string", False, True, "category", "MIMIC_IV_3_1 or exact non-overlapping NHANES release", enum=["MIMIC_IV_3_1", *release_enum]),
                _column("input_variant", "string", False, True, "category", "closed prespecified input variant", enum=input_variant_enum),
                _column("population", "string", False, True, "category", "prespecified age population", enum=list(AGE_POPULATIONS)),
                _column("organ_system", "string", False, True, "category", "organ, pooled_rows, or macro_organ_mean", enum=organ_enum + ["pooled_rows", "macro_organ_mean"]),
                _column("estimator", "string", False, True, "category", "fold0..fold4 or mean5", enum=["fold0", "fold1", "fold2", "fold3", "fold4", "mean5"]),
                _column("metric", "string", False, True, "category", "age-transfer metric", enum=age_metric_enum),
                _column("n_participants", "int64", False, False, "participants", "unique participant denominator"),
                _column("n_rows", "int64", False, False, "rows", "participant-organ denominator"),
                _column("estimability_status", "string", False, False, "category", "closed zero-event/degenerate status", enum=estimability_enum),
                _column("estimate", "float64", True, False, "metric-specific", "null iff estimability_status is not estimable", tolerance="1e-12"),
                _column("ci_lower", "float64", True, False, "metric-specific", "null because zero-shot age tables report point estimates only", tolerance="1e-12"),
                _column("ci_upper", "float64", True, False, "metric-specific", "null because zero-shot age tables report point estimates only", tolerance="1e-12"),
                _column("uncertainty_scope", "string", False, False, "category", "age-transfer uncertainty policy", enum=["point_estimate_only_no_resampling"]),
                _column("bootstrap_replicates", "int32", False, False, "replicates", "no resampling for age-transfer rows", enum=[0]),
                _column("bootstrap_seed", "int64", False, False, "seed", "reserved global seed recorded for lineage only", enum=[FORMAL_BOOTSTRAP_SEED]),
                _column("no_external_fit", "bool", False, False, "flag", "all rows use frozen raw checkpoint outputs without external fitting", enum=[True]),
                _column("raw_zero_shot_primary", "bool", False, False, "flag", "true only for mean5 and the prespecified primary cohort/release population; fold and age-quantization rows are sensitivity", enum=[False, True]),
            ],
        )
    )
    tables.append(
        _table(
            "mimic_cohort_flow",
            "01_mimic/02_cohort_and_endpoints/cohort_flow.csv",
            "one ordered MIMIC eligibility step",
            ["input_variant", "step_order"],
            "3 MIMIC input variants * exactly 12 prespecified flow steps",
            [
                _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                _column("step_order", "int16", False, True, "order", "one-based prespecified step index", enum=list(range(1, 13))),
                _column("step_name", "string", False, False, "category", "prespecified eligibility rule", enum=list(MIMIC_FLOW_STEPS)),
                _column("n_subjects", "int64", False, False, "subjects", "unique subject count after step"),
                _column("excluded_at_step", "int64", False, False, "subjects", "prior n minus current n"),
                _column("definition", "string", False, False, "text", "mechanical rule from contract"),
            ],
        )
    )
    tables.append(
        _table(
            "mimic_feature_coverage",
            "01_mimic/01_mapping/feature_coverage.csv",
            "one frozen MIMIC phenotype slot row",
            ["input_variant", "slot_index"],
            "3 MIMIC input variants * exactly 78 ordered frozen phenotype slots",
            [
                _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                _column("slot_index", "int16", False, True, "slot", "frozen slot 0..77", enum=list(range(78))),
                _column("feature_name", "string", False, False, "text", "frozen UKB feature name"),
                _column("map_status", "string", False, False, "category", "prefrozen MIMIC map quality", enum=["stable", "weak", "not_measured"]),
                _column("observed_n", "int64", False, False, "participants", "nonmissing mapped values inside the 365-day window"),
                _column("eligible_n", "int64", False, False, "participants", "MIMIC primary model-input denominator"),
                _column("observed_fraction", "float64", False, False, "fraction", "observed_n / eligible_n", tolerance="1e-15"),
                _column("source_file_sha256", "string", True, False, "sha256", "null only for not_measured slots"),
                _column("source_selector", "string", True, False, "selector", "exact variant-specific OMR result_name/value branch or itemid/valueuom branch; null only when unavailable"),
                _column("source_unit", "string", False, False, "unit", "exact variant-specific source unit branch"),
                _column("target_unit", "string", False, False, "unit", "frozen UKB slot unit"),
                _column("formula", "string", False, False, "formula", "exact variant-specific deterministic conversion or missing"),
                _column("value_rule", "string", False, False, "rule", "exact frozen finite/value-domain rule"),
                _column("all_78_slots_adjudicated", "bool", False, False, "flag", "mechanical map audit", enum=[True]),
            ],
        )
    )
    endpoint_columns = [
        _column("subject_id", "string", False, True, "identifier", "MIMIC subject"),
        _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
        _column("age_population", "string", False, True, "category", "risk cohort age population", enum=["mimic_risk_adult_nontopcoded_integer_year"]),
        _column("organ_system", "string", False, True, "category", "frozen ICD10 organ map", enum=organ_enum),
        _column("horizon_months", "int16", False, True, "months", "fixed primary horizon", enum=[6, 12]),
        _column("horizon_days", "float64", False, False, "days", "6 months=182.5 days; 12 months=365 days", enum=[182.5, 365.0], tolerance="exact"),
        _column("index_time", "datetime64[ns]", False, False, "UTC-naive MIMIC shifted time", "second valid adult hospital discharge"),
        _column("chronological_age", "int16", False, False, "integer years", "anchor_age + (index calendar year - anchor_year); anchor_age 91 rejected"),
        _column("sex_male", "int8", False, False, "binary", "M=1 and F=0; all other values rejected", enum=[0, 1]),
        _column("event_time_days", "float64", True, False, "days", "(first post-index target admission time-index_time)/86400", tolerance="1e-9"),
        _column("death_time_days", "float64", True, False, "days", "(death date at 00:00-index_time)/86400", tolerance="1e-9"),
        _column("subject_observation_end_days", "float64", False, False, "days", "maximum valid in-system admissions/lab/OMR/prescription timestamp minus index", tolerance="1e-9"),
        _column("death_ascertainment_cap_days", "float64", False, False, "days", "min(patients.dod, last valid hospital discharge plus 365 days) minus index", tolerance="1e-9"),
        _column("observed_time_days", "float64", False, False, "days", "min(target, death, subject_observation_end, death_ascertainment_cap, horizon_days)", tolerance="1e-9"),
        _column("observed_at_horizon", "bool", False, False, "flag", "target/death occurred by horizon or event-free follow-up reached horizon"),
        _column("event_indicator", "int8", False, False, "flag", "1 iff target_time strictly precedes death/censor/horizon", enum=[0, 1]),
        _column("competing_death_indicator", "int8", False, False, "flag", "1 iff death_time <= target_time and precedes other censoring", enum=[0, 1]),
        _column("censor_reason", "string", False, False, "category", "first terminal mechanism", enum=["target", "death", "subject_observation_end", "death_ascertainment_cap", "horizon"]),
        _column("has_any_preindex_icd9", "bool", False, False, "flag", "any ICD9 diagnosis through index admission", enum=[False]),
        _column("endpoint_code_system", "string", False, False, "category", "primary endpoint system", enum=["ICD10"]),
        _column("hash_fold", "int8", False, False, "fold", "SHA256 revision0710-external-v3|subject_id first8 big-endian mod5", enum=list(range(5))),
        _column("estimability_status", "string", False, False, "category", "row eligibility/status", enum=estimability_enum),
    ]
    tables.append(
        _table(
            "mimic_incident_endpoints",
            "01_mimic/06_incident_risk/incident_endpoints.parquet",
            "one ICD10-target-free MIMIC subject-organ-horizon row",
            ["subject_id", "input_variant", "age_population", "organ_system", "horizon_months"],
            "N incident-eligible subject-input-organ rows * exactly 2 horizons",
            endpoint_columns,
        )
    )
    recurrent_columns = [dict(column) for column in endpoint_columns]
    tables.append(
        _table(
            "mimic_recurrent_endpoints",
            "01_mimic/07_recurrent_sensitivity/recurrent_endpoints.parquet",
            "one baseline-target-positive MIMIC subject-organ-horizon sensitivity row",
            ["subject_id", "input_variant", "age_population", "organ_system", "horizon_months"],
            "N recurrent-eligible subject-input-organ rows * exactly 2 horizons",
            recurrent_columns,
        )
    )
    tables.append(
        _table(
            "mimic_predictive_metrics",
            "01_mimic/06_incident_risk/predictive_metrics.csv",
            "one organ-horizon-heldout-fold-model-metric row",
            ["input_variant", "organ_system", "horizon_months", "hash_fold", "model", "metric"],
            "3 MIMIC input variants * 7 organs * 2 horizons * 5 folds * 2 models * 6 metrics",
            [
                _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                _column("horizon_months", "int16", False, True, "months", "only prespecified primary horizons", enum=[6, 12]),
                _column("analysis_role", "string", False, False, "category", "horizon role", enum=["primary_in_system_incident_prediction"]),
                _column("hash_fold", "int8", False, True, "fold", "held-out external fold", enum=list(range(5))),
                _column("model", "string", False, True, "category", "base or augmented", enum=["age_sex", "age_sex_corrected_gap"]),
                _column("metric", "string", False, True, "category", "out-of-sample prediction metric", enum=["harrell_c", "delta_harrell_c", "ipcw_auc", "ipcw_brier", "ipcw_linear_calibration_intercept", "ipcw_linear_calibration_slope"]),
                _column("n_test", "int64", False, False, "subjects", "held-out test denominator"),
                _column("n_events", "int64", False, False, "events", "held-out target events"),
                _column("n_competing_deaths", "int64", False, False, "events", "held-out deaths before horizon"),
                _column("absolute_risk_estimand", "string", False, False, "estimand", "target absolute risk is the competing-risk CIF integrated from explicit post-jump Breslow target/death hazards reconstructed from complete risk sets", enum=["target_cif_from_postjump_breslow_target_and_death_cause_specific_hazards"]),
                _column("ipcw_auc_score", "string", False, False, "estimand", "IPCW AUC ranks the held-out target CIF probability, not the target cause-specific Cox linear predictor", enum=["target_cif_probability"]),
                _column("brier_normalization", "string", False, False, "estimand", "standard IPCW Brier normalization", enum=["sum_ipcw_squared_error_divided_by_full_test_n"]),
                _column("estimability_status", "string", False, False, "category", "closed zero-event/degenerate fold status", enum=estimability_enum),
                _column("estimate", "float64", True, False, "metric-specific", "null iff fold/metric is not estimable", tolerance="1e-12"),
                _column("ci_lower", "float64", True, False, "metric-specific", "2.5th conditional held-out subject-cluster bootstrap percentile; null only when bootstrap support is insufficient", tolerance="1e-12"),
                _column("ci_upper", "float64", True, False, "metric-specific", "97.5th conditional held-out subject-cluster bootstrap percentile; null only when bootstrap support is insufficient", tolerance="1e-12"),
                _column("ipcw_training_fold_only", "bool", False, False, "flag", "censor KM fit on four training folds", enum=[True]),
                _column("ipcw_outcome_definition", "string", False, False, "category", "target-by-horizon binary risk with competing death as known control", enum=["case=target_before_death_censor_horizon; control=event_free_at_horizon_or_competing_death_before_horizon; loss_or_admin_before_horizon=unobserved"]),
                _column("ipcw_censor_definition", "string", False, False, "category", "only observation/admin censor is a censor event; target/death are terminal non-censor events", enum=["G fitted on training folds; censor event=subject_observation_end_or_death_cap_before_target_death_horizon; target_and_death_censor_G_as_non-censor_terminal"]),
                _column("ipcw_weight_definition", "string", False, False, "category", "case 1/G(T-); death control 1/G(D-); horizon control 1/G(tau); censored before tau weight zero", enum=["left_limit_G_min_0.05_training_weight_cap_p99"]),
                _column("ipcw_g_min_raw", "float64", True, False, "probability", "minimum unfloored G used by training/test diagnostics", tolerance="1e-15"),
                _column("ipcw_g_floor", "float64", False, False, "probability", "frozen positivity floor", enum=[0.05], tolerance="exact"),
                _column("ipcw_weight_cap_p99_training", "float64", True, False, "weight", "training-fold p99 IPCW cap", tolerance="1e-12"),
                _column("ipcw_effective_sample_size", "float64", True, False, "subjects", "(sum weights)^2/sum(weights^2)", tolerance="1e-12"),
                _column("ipcw_n_observed_at_horizon", "int64", True, False, "subjects", "known binary status by horizon"),
                _column("ipcw_n_censored_before_horizon", "int64", True, False, "subjects", "unknown binary status due to censoring before horizon"),
                _column("ipcw_n_weights_capped", "int64", True, False, "subjects", "held-out weights above the training p99 cap"),
                _column("bootstrap_unit", "string", False, False, "resampling", "single shared participant-cluster replicate plan", enum=["one_global_subject_cluster_draw_shared_across_organs_horizons_folds_variants"]),
                _column("bootstrap_refits_target_cox_death_cox_and_censor_km", "bool", False, False, "flag", "training-fold target/death Cox and censoring KM stay fixed in every replicate", enum=[False]),
                _column("uncertainty_scope", "string", False, False, "category", "conditional bootstrap scope or explicit point-only fallback", enum=["conditional_on_fixed_training_fold_target_death_cox_and_censor_km", "point_estimate_only_bootstrap_insufficient"]),
                _column("bootstrap_replicates", "int32", False, False, "replicates", "one shared global subject plan", enum=[FORMAL_BOOTSTRAP_REPLICATES]),
                _column("bootstrap_seed", "int64", False, False, "seed", "fixed global subject-cluster seed", enum=[FORMAL_BOOTSTRAP_SEED]),
                _column("bootstrap_namespace", "string", False, False, "seed", "shared PCG64 SeedSequence namespace"),
            ],
        )
    )
    tables.append(
        _table(
            "mimic_full_cohort_associations",
            "01_mimic/06_incident_risk/full_cohort_associations.csv",
            "one organ-horizon association row",
            ["input_variant", "organ_system", "horizon_months", "exposure"],
            "3 MIMIC input variants * 7 organs * 2 horizons * 1 exposure",
            [
                _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                _column("horizon_months", "int16", False, True, "months", "fixed primary horizon", enum=[6, 12]),
                _column("analysis_population", "string", False, False, "category", "all hash folds in the organ-specific incident-eligible cohort", enum=["full_nonheldout_incident_eligible_organ_specific_cohort"]),
                _column("n", "int64", False, False, "subjects", "unique incident-eligible subjects entering this organ-horizon model"),
                _column("baseline_exclusion_rule", "string", False, False, "rule", "baseline disease and ICD9 exclusion before association", enum=["exclude_baseline_target_icd10_and_any_preindex_icd9_then_use_all_remaining_hash_folds"]),
                _column("exposure", "string", False, True, "category", "continuous corrected gap", enum=["corrected_age_gap_mean5_per_incident_eligible_analysis_population_sd"]),
                _column("covariates", "string", False, False, "formula", "exact adjustment set", enum=["chronological_age_integer_year + sex_male + corrected_age_gap_mean5_per_incident_eligible_analysis_population_sd"]),
                _column("ties", "string", False, False, "method", "Cox partial-likelihood tie method", enum=["efron"]),
                _column("standard_error", "string", False, False, "method", "Huber-White robust sandwich", enum=["robust_sandwich_subject"]),
                _column("estimability_status", "string", False, False, "category", "closed zero-event/singular status", enum=estimability_enum),
                _column("hazard_ratio", "float64", True, False, "ratio", "exp(Cox coefficient); null iff not estimable", tolerance="1e-12"),
                _column("ci_lower", "float64", True, False, "ratio", "robust Wald 95% lower; null iff not estimable", tolerance="1e-12"),
                _column("ci_upper", "float64", True, False, "ratio", "robust Wald 95% upper; null iff not estimable", tolerance="1e-12"),
                _column("p_value", "float64", True, False, "probability", "two-sided robust Wald p; null iff not estimable", tolerance="1e-12"),
                _column("bh_q_value", "float64", True, False, "probability", "Benjamini-Hochberg across seven organs within input_variant and horizon", tolerance="1e-12"),
                _column("ph_test_p", "float64", True, False, "probability", "global Schoenfeld test for corrected-gap term; null iff not estimable", tolerance="1e-12"),
                _column("ph_time_transform", "string", False, False, "transform", "time transform used for the PH diagnostic", enum=["log_observed_event_time"]),
                _column("ph_residual_covariance", "string", False, False, "covariance", "exact scaled-Schoenfeld covariance operation", enum=["scaled_schoenfeld_equals_raw_schoenfeld_times_PHReg_cov_params"]),
                _column("ph_n_events", "int64", False, False, "events", "target-event residual rows entering the diagnostic"),
                _column("interpretation", "string", False, False, "category", "association only", enum=["nonheldout_incident_eligible_association_not_prediction_not_causal"]),
            ],
        )
    )
    tables.append(
        _table(
            "mimic_incident_competing_risk_curves",
            "01_mimic/06_incident_risk/competing_risk_curves.csv",
            "one incident organ-horizon-curve-event-time row",
            ["input_variant", "organ_system", "horizon_months", "curve", "time_days"],
            "all unique event times through each input-organ-horizon for three declared curves",
            [
                _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                _column("horizon_months", "int16", False, True, "months", "only 6/12 month horizons", enum=[6, 12]),
                _column("analysis_role", "string", False, False, "category", "curve role", enum=["aj_primary", "km_descriptive_sensitivity"]),
                _column("curve", "string", False, True, "category", "declared competing-risk or cause-specific KM estimand", enum=["aalen_johansen_target_cif", "aalen_johansen_death_cif", "kaplan_meier_target_free_survival_cause_specific"]),
                _column("time_days", "float64", False, True, "days", "sorted unique terminal time from zero through the horizon", tolerance="1e-9"),
                _column("n_at_risk", "int64", False, False, "subjects", "risk set immediately before time_days"),
                _column("target_events", "int64", False, False, "events", "target events at time_days"),
                _column("death_events", "int64", False, False, "events", "competing deaths at time_days; death wins exact target ties"),
                _column("censored", "int64", False, False, "subjects", "non-event censorings at time_days"),
                _column("estimability_status", "string", False, False, "category", "closed zero-event curve status", enum=estimability_enum),
                _column("estimate", "float64", True, False, "probability", "exact declared AJ or KM recursion; null iff not estimable", tolerance="1e-12"),
                _column("ci_lower", "float64", True, False, "probability", "null because the full time grid is descriptive and has no simultaneous band", tolerance="1e-12"),
                _column("ci_upper", "float64", True, False, "probability", "null because the full time grid is descriptive and has no simultaneous band", tolerance="1e-12"),
                _column("uncertainty_scope", "string", False, False, "category", "curve uncertainty policy", enum=["point_curve_only_no_simultaneous_band"]),
                _column("bootstrap_replicates", "int32", False, False, "replicates", "no curve resampling", enum=[0]),
                _column("bootstrap_seed", "int64", False, False, "seed", "reserved global seed recorded for lineage only", enum=[FORMAL_BOOTSTRAP_SEED]),
                _column("interpretation", "string", False, False, "category", "hospital-recorded incident risk only", enum=["hospital_recorded_incident_risk_not_population_incidence_not_causal"]),
            ],
        )
    )
    tables.append(
        _table(
            "mimic_recurrent_sensitivity_metrics",
            "01_mimic/07_recurrent_sensitivity/recurrent_sensitivity_metrics.csv",
            "one recurrent organ-horizon-metric row",
            ["input_variant", "organ_system", "horizon_months", "metric"],
            "3 MIMIC input variants * 7 organs * 2 horizons * declared recurrent metrics",
            [
                _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                _column("horizon_months", "int16", False, True, "months", "fixed sensitivity horizon", enum=[6, 12]),
                _column("metric", "string", False, True, "category", "recurrent sensitivity metric", enum=["cause_specific_hazard_ratio", "target_cumulative_incidence", "death_cumulative_incidence"]),
                _column("n", "int64", False, False, "subjects", "baseline target-positive recurrent-risk denominator"),
                _column("target_events", "int64", False, False, "events", "post-discharge recurrent target events"),
                _column("death_events", "int64", False, False, "events", "competing deaths"),
                _column("estimability_status", "string", False, False, "category", "closed zero-event/singular status", enum=estimability_enum),
                _column("estimate", "float64", True, False, "metric-specific", "separately estimated recurrent sensitivity or null iff not estimable", tolerance="1e-12"),
                _column("ci_lower", "float64", True, False, "metric-specific", "robust Wald lower for the Cox HR; null for point-only CIF rows", tolerance="1e-12"),
                _column("ci_upper", "float64", True, False, "metric-specific", "robust Wald upper for the Cox HR; null for point-only CIF rows", tolerance="1e-12"),
                _column("uncertainty_scope", "string", False, False, "category", "metric-specific uncertainty policy", enum=["robust_wald_95_percent_interval", "point_estimate_only_no_resampling"]),
                _column("physically_separate_from_incident", "bool", False, False, "flag", "must remain separate", enum=[True]),
                _column("interpretation", "string", False, False, "category", "recurrent sensitivity only", enum=["recurrent_hospital_recorded_sensitivity_not_incident_prediction_not_causal"]),
            ],
        )
    )
    tables.append(
        _table(
            "nhanes_feature_coverage",
            "02_nhanes/01_mapping/feature_coverage.csv",
            "one release-slot row",
            ["release_id", "input_variant", "slot_index"],
            "2 releases * 78 slots = 156",
            [
                _column("release_id", "string", False, True, "category", "official non-overlapping release", enum=release_enum),
                _column("input_variant", "string", False, True, "category", "phenotype-only label-blind primary", enum=[NHANES_PRIMARY_VARIANT]),
                _column("slot_index", "int16", False, True, "slot", "frozen slot 0..77", enum=list(range(78))),
                _column("feature_name", "string", False, False, "text", "frozen UKB feature name"),
                _column("map_status", "string", False, False, "category", "release-specific map status", enum=["direct", "direct_converted", "derived", "not_measured", "no_defensible_equivalent"]),
                _column("observed_n", "int64", False, False, "participants", "nonmissing mapped values"),
                _column("eligible_n", "int64", False, False, "participants", "release-specific primary input denominator"),
                _column("observed_fraction", "float64", False, False, "fraction", "observed_n/eligible_n", tolerance="1e-15"),
                _column("source_file_sha256", "string", True, False, "sha256", "null only when map_status is unavailable"),
                _column("all_78_slots_adjudicated", "bool", False, False, "flag", "mechanical map audit", enum=[True]),
            ],
        )
    )
    tables.append(
        _table(
            "nhanes_questionnaire_associations",
            "02_nhanes/07_cross_sectional_disease/questionnaire_associations.csv",
            "one release-organ association row",
            ["release_id", "input_variant", "organ_system", "analysis", "metric"],
            "2 releases * 6 estimable organs * declared analyses; immune emitted as not_estimable status rows",
            [
                _column("release_id", "string", False, True, "category", "release-specific analysis only", enum=release_enum),
                _column("input_variant", "string", False, True, "category", "phenotype-only label-blind primary predictions", enum=[NHANES_PRIMARY_VARIANT]),
                _column("organ_system", "string", False, True, "category", "organ label", enum=organ_enum),
                _column("analysis", "string", False, True, "category", "release-specific survey-weighted questionnaire association", enum=["survey_weighted_logistic_questionnaire_association"]),
                _column("metric", "string", False, True, "category", "reported association measure", enum=["odds_ratio_per_1sd_corrected_gap"]),
                _column("estimability_status", "string", False, False, "category", "immune fixed not estimable plus closed degeneracy states", enum=estimability_enum),
                _column("n", "int64", False, False, "participants", "complete label/prediction denominator"),
                _column("cases", "int64", True, False, "participants", "questionnaire positive count"),
                _column("weight_variable", "string", False, False, "column", "native release MEC weight", enum=["WTMECPRP", "WTMEC2YR"]),
                _column("strata_variable", "string", False, False, "column", "NHANES design strata", enum=["SDMVSTRA"]),
                _column("psu_variable", "string", False, False, "column", "NHANES design PSU", enum=["SDMVPSU"]),
                _column("covariates", "string", False, False, "formula", "survey-weighted exposure SD and age centering inside the 40-73 domain", enum=["intercept + corrected_age_gap_mean5_per_release_survey_weighted_sd + chronological_age_survey_weighted_centered_per10y + sex_male"]),
                _column("inference", "string", False, False, "method", "full-release design Taylor linearization for a 40-73 domain", enum=["survey_domain_pseudolikelihood_logit_taylor_psu_within_strata_lonely_psu_centered_t_design_df"]),
                _column("survey_domain", "string", False, False, "domain", "age 40-73 is a survey domain rather than a pre-analysis design filter", enum=["MEC_examined_integer_age_40_73_with_full_release_design_retained"]),
                _column("full_design_n", "int64", False, False, "participants", "complete positive-weight examined release design rows before domain restriction"),
                _column("full_design_strata", "int64", False, False, "strata", "unique design strata in the full release design"),
                _column("full_design_psus", "int64", False, False, "PSUs", "unique strata-PSU pairs in the full release design"),
                _column("design_df", "int64", False, False, "degrees of freedom", "full_design_psus - full_design_strata"),
                _column("reference_distribution", "string", False, False, "distribution", "CI and Wald p-value reference", enum=["t_design_df"]),
                _column("estimate", "float64", True, False, "odds ratio", "exp(corrected-gap coefficient); null iff not estimable", tolerance="1e-12"),
                _column("ci_lower", "float64", True, False, "odds ratio", "design-based Wald 95% lower or null", tolerance="1e-12"),
                _column("ci_upper", "float64", True, False, "odds ratio", "design-based Wald 95% upper or null", tolerance="1e-12"),
                _column("p_value", "float64", True, False, "probability", "two-sided design-based Wald p or null", tolerance="1e-12"),
                _column("bh_q_value", "float64", True, False, "probability", "Benjamini-Hochberg across six estimable organs within release", tolerance="1e-12"),
                _column("input_label_overlap_count", "int64", False, False, "rows", "must equal zero", enum=[0]),
                _column("interpretation", "string", False, False, "category", "cross-sectional association only", enum=["association_not_incident_prediction_not_causal"]),
            ],
        )
    )
    tables.append(
        _table(
            "nhanes_local_recalibration_sensitivity",
            "02_nhanes/06_local_recalibration_sensitivity/metrics.csv",
            "one release-input-organ-heldout-fold coefficient-or-metric row",
            ["release_id", "input_variant", "organ_system", "hash_fold", "record_type", "term", "metric"],
            "2 releases * 1 input * 7 organs * 5 folds * (3 ordered coefficients + 7 held-out metrics)",
            [
                _column("release_id", "string", False, True, "category", "fit and score within one release", enum=release_enum),
                _column("input_variant", "string", False, True, "category", "phenotype-only label-blind predictions", enum=[NHANES_PRIMARY_VARIANT]),
                _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                _column("hash_fold", "int8", False, True, "fold", "held-out calibration fold", enum=list(range(5))),
                _column("record_type", "string", False, True, "category", "coefficient or held-out metric row", enum=["coefficient", "heldout_metric"]),
                _column("term", "string", False, True, "category", "coefficient application order or not-applicable marker", enum=list(NHANES_LOCAL_CALIBRATION_TERMS)),
                _column("metric", "string", False, True, "category", "held-out metric or not-applicable marker", enum=[*age_metric_enum, "not_applicable_coefficient_row"]),
                _column("regression_formula", "string", False, False, "formula", "OLS on four folds only", enum=["chronological_age = beta0 + beta1*predicted_age_raw_mean5 + beta2*sex_male"]),
                _column("coefficient_application_order", "string", False, False, "order", "fixed coefficient multiplication order", enum=["intercept,predicted_age_raw_mean5,sex_male"]),
                _column("fit_fold_rule", "string", False, False, "rule", "fit folds exclude heldout hash_fold", enum=["fit_on_other_four_release_specific_participant_hash_folds"]),
                _column("fit_n", "int64", False, False, "participants", "complete primary-age training participants"),
                _column("test_n", "int64", False, False, "participants", "complete primary-age held-out participants"),
                _column("estimability_status", "string", False, False, "category", "closed singular/insufficient status", enum=estimability_enum),
                _column("estimate", "float64", True, False, "coefficient-or-metric", "coefficient or held-out metric; null iff not estimable", tolerance="1e-12"),
                _column("zero_shot_primary", "bool", False, False, "flag", "always false", enum=[False]),
                _column("label", "string", False, False, "category", "required display label", enum=["NHANES-local cross-fitted calibration sensitivity; not zero-shot"]),
            ],
        )
    )
    tables.append(
        _table(
            "nhanes_release_survey_sensitivity",
            "02_nhanes/05_age_metrics/release_survey_sensitivity.csv",
            "one release-organ-metric weighted sensitivity row",
            ["release_id", "input_variant", "organ_system", "metric"],
            "2 releases * 7 organs * declared weighted metrics",
            [
                _column("release_id", "string", False, True, "category", "release-specific survey design", enum=release_enum),
                _column("input_variant", "string", False, True, "category", "phenotype-only label-blind predictions", enum=[NHANES_PRIMARY_VARIANT]),
                _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                _column("metric", "string", False, True, "category", "survey-weighted sensitivity metric", enum=age_metric_enum),
                _column("weight_variable", "string", False, False, "column", "WTMECPRP for P; WTMEC2YR for L", enum=["WTMECPRP", "WTMEC2YR"]),
                _column("estimability_status", "string", False, False, "category", "closed release/organ/metric status", enum=estimability_enum),
                _column("estimate", "float64", True, False, "metric-specific", "native release-weighted estimate or null iff not estimable", tolerance="1e-12"),
                _column("cross_release_pooled_weight", "bool", False, False, "flag", "pooled weights prohibited", enum=[False]),
                _column("sensitivity_only", "bool", False, False, "flag", "not primary", enum=[True]),
                _column("between_release_test_performed", "bool", False, False, "flag", "false unless a separately frozen heterogeneity test is implemented", enum=[False]),
                _column("interpretation", "string", False, False, "claim", "side-by-side release sensitivity only", enum=["release_specific_descriptive_sensitivity_no_heterogeneity_test"]),
            ],
        )
    )
    tables.extend(
        [
            _table(
                "mimic_density_diagnostics",
                "01_mimic/08_density_sensitivity/density_diagnostics.csv",
                "one MIMIC variant-organ-density diagnostic row",
                [
                    "input_variant",
                    "organ_system",
                    "analysis",
                    "density_measure",
                    "density_stratum",
                    "target",
                    "metric",
                ],
                "exactly 693 rows: 3 variants * 7 organs * (4 density measures * 3 targets * 2 correlations + 3 fixed strata * 3 age-error metrics)",
                [
                    _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                    _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                    _column("analysis", "string", False, True, "category", "density diagnostic family", enum=["density_correlation", "fixed_density_stratum_age_error"]),
                    _column("density_measure", "string", False, True, "category", "participant-level label-blind density measure", enum=["pre_crop_event_count", "post_crop_event_count", "history_span_days", "observed_phenotype_slots"]),
                    _column("density_stratum", "string", False, True, "category", "absolute training-independent pre-crop count stratum", enum=["all", "low_0_7", "medium_8_31", "high_32_plus"]),
                    _column("target", "string", False, True, "category", "age, gap, absolute error, or stratum error target", enum=["chronological_age", "corrected_age_gap_mean5", "absolute_age_error", "raw_mean5_age_error"]),
                    _column("metric", "string", False, True, "category", "correlation or age-error metric", enum=["pearson", "spearman", "mae", "rmse", "mean_error"]),
                    _column("n", "int64", False, False, "participants", "participant denominator"),
                    _column("estimate", "float64", True, False, "metric-specific", "estimate or null under explicit non-estimability", tolerance="1e-12"),
                    _column("estimability_status", "string", False, False, "category", "closed estimability status", enum=estimability_enum),
                    _column("fixed_training_independent_strata", "bool", False, False, "flag", "strata never use external outcome, age, gap, or fold quantiles", enum=[True]),
                ],
            ),
            _table(
                "mimic_density_adjusted_risk_sensitivity",
                "01_mimic/08_density_sensitivity/density_adjusted_risk_sensitivity.csv",
                "one MIMIC organ-horizon-heldout-fold-density-model-metric row",
                ["input_variant", "organ_system", "horizon_months", "hash_fold", "model", "metric"],
                "exactly 2520 rows: 3 variants * 7 organs * 2 horizons * 5 folds * 2 density models * 6 metrics",
                [
                    _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                    _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                    _column("horizon_months", "int16", False, True, "months", "fixed supported horizon", enum=[6, 12]),
                    _column("hash_fold", "int8", False, True, "fold", "held-out external hash fold", enum=list(range(5))),
                    _column("analysis_role", "string", False, False, "category", "separate sensitivity role", enum=["density_adjusted_heldout_sensitivity"]),
                    _column("n_test", "int64", False, False, "participants", "held-out denominator"),
                    _column("n_events", "int64", False, False, "events", "held-out target events"),
                    _column("n_competing_deaths", "int64", False, False, "events", "held-out competing deaths"),
                    _column("density_covariates", "string", False, False, "formula", "three prespecified label-blind density covariates", enum=["log1p_pre_crop_event_count_z+log1p_history_span_days_z+observed_phenotype_fraction_z"]),
                    _column("density_standardization", "string", False, False, "rule", "density scaling uses training fold only", enum=["training_fold_mean_sd_only"]),
                    _column("fixed_training_independent_strata", "bool", False, False, "flag", "descriptive strata remain absolute and training independent", enum=[True]),
                    _column("bootstrap_replicates", "int32", False, False, "replicates", "conditional subject-cluster replicates per model", enum=[FORMAL_BOOTSTRAP_REPLICATES]),
                    _column("bootstrap_seed", "int64", False, False, "seed", "global external subject-cluster seed", enum=[FORMAL_BOOTSTRAP_SEED]),
                    _column("uncertainty_scope", "string", False, False, "category", "conditional interval scope", enum=["conditional_on_fixed_training_fold_target_death_cox_and_censor_km", "point_estimate_only_bootstrap_insufficient"]),
                    _column("clinical_utility_claim_allowed", "bool", False, False, "flag", "relative statistical increment is not clinical utility", enum=[False]),
                    _column("comparison_reference_model", "string", False, False, "category", "delta metrics use the density-adjusted base", enum=["age_sex_density"]),
                    _column("log1p_pre_crop_event_count_train_mean", "float64", True, False, "log-count", "training-fold mean used for density standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("log1p_pre_crop_event_count_train_sd", "float64", True, False, "log-count", "training-fold sample SD used for density standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("log1p_history_span_days_train_mean", "float64", True, False, "log-days", "training-fold mean used for density standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("log1p_history_span_days_train_sd", "float64", True, False, "log-days", "training-fold sample SD used for density standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("observed_phenotype_fraction_train_mean", "float64", True, False, "fraction", "training-fold mean used for density standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("observed_phenotype_fraction_train_sd", "float64", True, False, "fraction", "training-fold sample SD used for density standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("corrected_age_gap_train_mean", "float64", True, False, "years", "training-fold mean used for gap standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("corrected_age_gap_train_sd", "float64", True, False, "years", "training-fold sample SD used for gap standardization; null when the fold is not estimable", tolerance="1e-12"),
                    _column("model", "string", False, True, "category", "density base or density plus sealed gap", enum=["age_sex_density", "age_sex_density_corrected_gap"]),
                    _column("metric", "string", False, True, "category", "held-out risk metric", enum=["harrell_c", "delta_harrell_c", "ipcw_auc", "ipcw_brier", "ipcw_linear_calibration_intercept", "ipcw_linear_calibration_slope"]),
                    _column("estimability_status", "string", False, False, "category", "closed estimability status", enum=estimability_enum),
                    _column("estimate", "float64", True, False, "metric-specific", "held-out estimate or explicit null", tolerance="1e-12"),
                    _column("ci_lower", "float64", True, False, "metric-specific", "conditional B=200 lower or null", tolerance="1e-12"),
                    _column("ci_upper", "float64", True, False, "metric-specific", "conditional B=200 upper or null", tolerance="1e-12"),
                ],
            ),
            _table(
                "mimic_cox_ph_diagnostics",
                "01_mimic/09_cox_diagnostics/mimic_cox_ph_diagnostics.csv",
                "one role-variant-organ-horizon-fold-model-cause-covariate-transform PH row",
                ["analysis_role", "input_variant", "organ_system", "horizon_months", "hash_fold", "model", "cause", "covariate", "transform"],
                "exactly 17136 rows: 336 full-association plus 16800 held-out rows; every covariate plus GLOBAL and both transforms; failures retained",
                [
                    _column("survival_core_relative_path", "string", False, False, "path", "byte-bound shared implementation", enum=["Revision-0710/06_figures_and_statistics/03_reviewer_statistics/04_cox_diagnostics_and_competing_risk/survival_analysis_core.py"]),
                    _column("survival_core_sha256", "string", False, False, "sha256", "exact shared implementation bytes", enum=["6c4dd7eef4ea690d0d88a330f117d8d0b45274d788e10862ab665e245e435a1b"]),
                    _column("ties", "string", False, False, "method", "shared-core Cox tie method", enum=["breslow"]),
                    _column("prediction_fit_backend", "string", False, False, "backend", "actual target/death risk prediction backend", enum=["statsmodels_PHReg_efron"]),
                    _column("diagnostic_fit_backend", "string", False, False, "backend", "separate PH diagnostic-only refit", enum=["diagnostic_only_byte_bound_shared_core_breslow_refit"]),
                    _column("diagnostic_beta_or_risk_reused_for_prediction", "bool", False, False, "flag", "diagnostic refit cannot alter prediction", enum=[False]),
                    _column("diagnostic_duration_rule", "string", False, False, "rule", "uniform day shift supports day-0 death and preserves ordering/ties", enum=["observed_time_days_plus_1_for_all_rows_preserves_order_and_ties"]),
                    _column("analysis_role", "string", False, True, "category", "association or held-out training-model diagnostic", enum=["full_nonheldout_incident_eligible_association", "heldout_training_model_diagnostic"]),
                    _column("input_variant", "string", False, True, "category", "closed MIMIC input variant", enum=mimic_variant_enum),
                    _column("organ_system", "string", False, True, "category", "frozen organ", enum=organ_enum),
                    _column("horizon_months", "int16", False, True, "months", "fixed supported horizon", enum=[6, 12]),
                    _column("hash_fold", "int8", False, True, "fold", "-1 for full association; 0..4 held-out fold", enum=[-1, 0, 1, 2, 3, 4]),
                    _column("model", "string", False, True, "category", "complete base/augmented/density universe", enum=["age_sex", "age_sex_corrected_gap", "age_sex_density", "age_sex_density_corrected_gap"]),
                    _column("cause", "string", False, True, "category", "target or competing death cause-specific model", enum=["target", "death"]),
                    _column("n_input", "int64", False, False, "participants", "rows presented to the diagnostic refit"),
                    _column("n_complete", "int64", False, False, "participants", "rows finite across duration, cause, and all model covariates"),
                    _column("n_events", "int64", False, False, "events", "cause-specific event count retained on success and failure"),
                    _column("covariate", "string", False, True, "category", "every model covariate plus joint GLOBAL test", enum=["age_per10", "sex_male", "corrected_gap_z", "log1p_pre_crop_event_count_z", "log1p_history_span_days_z", "observed_phenotype_fraction_z", "GLOBAL"]),
                    _column("transform", "string", False, True, "category", "bound core rank and log1p-day transforms after uniform +1 day diagnostic shift", enum=["rank_lifelines_cumsum_event", "log1p_days"]),
                    _column("estimability_status", "string", False, False, "category", "explicit fit status; no silent drop", enum=estimability_enum),
                    _column("fit_converged", "bool", False, False, "flag", "shared-core convergence status", enum=[False, True]),
                    _column("failure_code", "string", False, False, "text", "none or preserved failure reason"),
                    _column("chi_square", "float64", True, False, "statistic", "per-covariate or full-covariance GLOBAL score statistic", tolerance="1e-12"),
                    _column("df", "int16", True, False, "degrees of freedom", "1 per estimable covariate; covariance rank for estimable GLOBAL; null on failure"),
                    _column("p_value", "float64", True, False, "probability", "chi-square tail probability or explicit null", tolerance="1e-12"),
                ],
            ),
            _table(
                "nhanes_release_slot_qc",
                "02_nhanes/01_mapping/release_slot_qc.csv",
                "one NHANES release-phenotype-slot QC row",
                ["release_id", "slot_index"],
                "2 releases * exactly 78 slots = 156",
                [
                    _column("release_id", "string", False, True, "category", "official non-overlapping release", enum=release_enum),
                    _column("slot_index", "int16", False, True, "slot", "frozen slot 0..77", enum=list(range(78))),
                    _column("feature_name", "string", False, False, "text", "frozen UKB feature name"),
                    _column("target_unit", "string", False, False, "unit", "frozen target unit"),
                    _column("n_eligible", "int64", False, False, "participants", "release primary age-transfer denominator"),
                    _column("n_observed", "int64", False, False, "participants", "finite or nonfinite nonmissing mapped values"),
                    _column("n_nonfinite_observed", "int64", False, False, "values", "observed NaN/inf count"),
                    _column("n_outside_fixed_range", "int64", False, False, "values", "finite values outside prefrozen unit range"),
                    _column("fixed_range_min", "float64", False, False, "target-unit", "prefrozen physical lower bound", tolerance="exact"),
                    _column("fixed_range_max", "float64", False, False, "target-unit", "prefrozen physical upper bound", tolerance="exact"),
                    _column("observed_min", "float64", True, False, "target-unit", "finite observed minimum; null if not measured", tolerance="1e-12"),
                    _column("q01", "float64", True, False, "target-unit", "descriptive 1st percentile", tolerance="1e-12"),
                    _column("q25", "float64", True, False, "target-unit", "descriptive 25th percentile", tolerance="1e-12"),
                    _column("q50", "float64", True, False, "target-unit", "descriptive median", tolerance="1e-12"),
                    _column("q75", "float64", True, False, "target-unit", "descriptive 75th percentile", tolerance="1e-12"),
                    _column("q99", "float64", True, False, "target-unit", "descriptive 99th percentile", tolerance="1e-12"),
                    _column("observed_max", "float64", True, False, "target-unit", "finite observed maximum; null if not measured", tolerance="1e-12"),
                    _column("qc_status", "string", False, False, "category", "pass, unmeasured, or fail-before-inference", enum=["pass", "not_measured", "fail_abort_before_inference"]),
                    _column("execution_rule", "string", False, False, "rule", "fixed fail-closed execution rule", enum=["nonfinite_or_outside_fixed_unit_range_aborts_before_inference"]),
                    _column("trimming_or_winsorization", "string", False, False, "rule", "external-data-driven clipping is forbidden", enum=["prohibited"]),
                    _column("quantiles_used_for_decision", "bool", False, False, "flag", "quantiles are descriptive only", enum=[False]),
                ],
            ),
        ]
    )
    ordered_table_ids = [
        "per_checkpoint_predictions",
        "ensemble_predictions",
        "external_vocabulary_coverage",
        "zero_shot_age_metrics",
        "mimic_cohort_flow",
        "mimic_feature_coverage",
        "mimic_incident_endpoints",
        "mimic_recurrent_endpoints",
        "mimic_predictive_metrics",
        "mimic_full_cohort_associations",
        "mimic_incident_competing_risk_curves",
        "mimic_recurrent_sensitivity_metrics",
        "mimic_density_diagnostics",
        "mimic_density_adjusted_risk_sensitivity",
        "mimic_cox_ph_diagnostics",
        "nhanes_feature_coverage",
        "nhanes_release_slot_qc",
        "nhanes_questionnaire_associations",
        "nhanes_local_recalibration_sensitivity",
        "nhanes_release_survey_sensitivity",
    ]
    by_id = {table["table_id"]: table for table in tables}
    if set(by_id) != set(ordered_table_ids):
        raise AssertionError("V8 output table universe is not exactly the closed 20-table set")
    tables = [by_id[table_id] for table_id in ordered_table_ids]
    expected_column_keys = {
        "name",
        "dtype",
        "nullable",
        "is_key",
        "grain",
        "cardinality",
        "enum",
        "unit",
        "formula",
        "tolerance",
    }
    for table in tables:
        if len(table["columns"]) != len({column["name"] for column in table["columns"]}):
            raise AssertionError(f"duplicate output column in {table['table_id']}")
        for column in table["columns"]:
            if set(column) != expected_column_keys:
                raise AssertionError(f"column schema is not closed in {table['table_id']}")
    return {
        "schema_version": "revision0710.external_validation.output_schemas.v8",
        "status": "PREFREEZE_EXACT_OUTPUT_SCHEMAS_PASS",
        "table_count": len(tables),
        "column_schema_keys_exact_order": [
            "name",
            "dtype",
            "nullable",
            "is_key",
            "grain",
            "cardinality",
            "enum",
            "unit",
            "formula",
            "tolerance",
        ],
        "tables": tables,
    }


def build_formal_authorization_schema() -> dict[str, Any]:
    return {
        "schema_version": "revision0710.external_validation.formal_authorization_schema.v8",
        "status": "PREFREEZE_SCHEMA_ONLY_NO_AUTHORIZATION_GRANTED",
        "duplicate_json_keys": "reject",
        "additional_properties": False,
        "required_top_level_keys_exact_order": [
            "schema_version",
            "status",
            "scope",
            "contract",
            "code_artifacts",
            "prefreeze_artifacts",
            "locked_source_registry",
            "environments",
            "independent_review_receipt",
            "review_freeze",
            "formal_publication",
        ],
        "exact_constants": {
            "schema_version": "revision0710.external_validation.formal_authorization.v8",
            "status": "APPROVED_ONCE_FOR_EXTERNAL_VALIDATION_SOURCE_DATA_V1",
            "scope": "one create-once MIMIC-IV and NHANES formal external-validation source bundle",
            "formal_tag": FORMAL_TAG,
            "formal_target_relative_path": repo_relative(FORMAL_TARGET),
            "formal_target_resolved": str(FORMAL_TARGET.resolve()),
            "target_must_be_absent_before_authorization": True,
            "fine_tuning": False,
            "distillation": False,
            "formal_bootstrap_replicates": FORMAL_BOOTSTRAP_REPLICATES,
            "formal_bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
            "formal_mode": "FORMAL_ONCE_CREATE_ONLY",
            "formal_executor_name": "formal_executor",
        },
        "record_schemas": {
            "contract": ["relative_path", "bytes", "sha256"],
            "code_artifact": ["name", "relative_path", "bytes", "sha256"],
            "prefreeze_artifact": ["name", "relative_path", "bytes", "sha256"],
            "locked_source_registry": ["relative_path", "bytes", "sha256"],
            "environment": [
                "name",
                "environment_lock_sha256",
                "python_executable",
                "python_executable_sha256",
            ],
            "independent_review_receipt": [
                "relative_path",
                "bytes",
                "sha256",
                "status",
                "reviewed_contract_sha256",
                "reviewed_code_bundle_sha256",
            ],
            "review_freeze": [
                "independent_review_status",
                "contract_bytes_frozen",
                "all_authorized_hashes_exact",
            ],
            "formal_publication": [
                "mode",
                "executor_name",
                "tag",
                "target_relative_path",
                "target_resolved",
                "create_once",
                "single_atomic_bundle",
                "fine_tuning",
                "distillation",
                "bootstrap_replicates",
                "bootstrap_seed",
                "authorization_single_use",
            ],
        },
        "rejection_policy": [
            "reject extra, missing, duplicate, or reordered keys and records",
            "reject any path, byte count, SHA-256, environment, tag, target, seed, or replicate mismatch",
            "reject if formal target or a sibling staging target already exists",
            "reject if source registry, maps, output schemas, builder, executor, verifier, or tests changed after authorization",
            "reject mode other than FORMAL_ONCE_CREATE_ONLY or authorization reused after any target/staging creation",
            "reject authorization without independent_review_status=PASS and contract_bytes_frozen=true",
            "reject absent, self-authored, stale, or hash-mismatched independent GO review receipt",
            "reject any authorization that permits fine-tuning, distillation, external age fitting in the zero-shot path, or ICD9 endpoints",
        ],
    }


def _artifact_record(path: Path, *, name: str, root: Path = ROOT) -> dict[str, Any]:
    return {
        "name": name,
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _probe_locked_environment(
    *, executable: Path, role: str, packages: tuple[str, ...]
) -> dict[str, Any]:
    if not executable.is_file():
        raise FileNotFoundError(f"required V8 {role} interpreter is absent: {executable}")
    command = [
        str(executable),
        str(HERE / "v8_environment_probe.py"),
        "--role",
        role,
        "--packages",
        ",".join(packages),
    ]
    completed = subprocess.run(
        command,
        cwd=HERE,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty V8 {role} environment probe")
    value = json.loads(lines[-1])
    if value.get("role") != role:
        raise RuntimeError(f"V8 {role} environment probe role mismatch")
    return value


def build_environment_lock() -> dict[str, Any]:
    return {
        "schema_version": "revision0710.external_validation.environment_lock.v8",
        "status": "TWO_HASH_BOUND_SUBPROCESS_ENVIRONMENTS_NO_IMPLICIT_SWITCH",
        "statistics": _probe_locked_environment(
            executable=STATISTICS_PYTHON,
            role="statistics",
            packages=STATISTICS_PACKAGES,
        ),
        "inference": _probe_locked_environment(
            executable=INFERENCE_PYTHON,
            role="inference",
            packages=INFERENCE_PACKAGES,
        ),
        "phase_assignment": {
            "01_cache": "statistics",
            "02_inference": "inference",
            "03_prediction_seal": "statistics",
            "04_outcomes": "statistics",
            "05_tables": "statistics",
            "06_independent_recompute": "statistics",
            "07_publish": "statistics",
        },
        "stage_receipt": "each phase binds environment-lock SHA, exact inputs, and every output byte; completed phases are rehashed and resumed",
    }


def build_contract(stage: Path, registry: dict[str, Any]) -> dict[str, Any]:
    support_paths = {
        "source_registry": stage / "SOURCE_REGISTRY_V8.json",
        "mimic_feature_map": stage / "MIMIC_78_SLOT_MAP_V8.csv",
        "nhanes_feature_map": stage / "NHANES_78_SLOT_RELEASE_MAP_V8.csv",
        "nhanes_label_map": stage / "NHANES_QUESTIONNAIRE_LABEL_MAP_V8.csv",
        "mimic_icd10_map": stage / "MIMIC_ICD10_ORGAN_MAP_V8.csv",
        "output_schemas": stage / "OUTPUT_SCHEMAS_V8.json",
        "formal_authorization_schema": stage / "FORMAL_AUTHORIZATION_SCHEMA_V8.json",
        "environment_lock": stage / "ENVIRONMENT_LOCK_V8.json",
    }
    support = [
        {
            "name": name,
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in support_paths.items()
    ]
    code_artifacts = [
        _artifact_record(Path(__file__).resolve(), name="builder"),
        _artifact_record(
            HERE / "verify_external_validation_prefreeze_v8.py", name="independent_verifier"
        ),
        _artifact_record(
            HERE / "tests/test_external_validation_prefreeze_v8.py", name="tests"
        ),
        _artifact_record(
            HERE / "execute_external_validation_v8.py", name="formal_executor"
        ),
        _artifact_record(
            HERE / "execute_external_validation_v8_runtime.py",
            name="formal_executor_runtime",
        ),
        _artifact_record(HERE / "v8_execution_guard.py", name="execution_guard"),
        _artifact_record(HERE / "v8_environment_probe.py", name="environment_probe"),
        _artifact_record(
            HERE / "v8_scientific_extensions.py", name="scientific_extensions"
        ),
        _artifact_record(
            HERE / "V7_SCIENTIFIC_REVIEWER_NO_GO.json",
            name="v7_scientific_reviewer_no_go",
        ),
        _artifact_record(
            HERE / "V7_READ_ONLY_TREE_SNAPSHOT.json",
            name="v7_read_only_tree_snapshot",
        ),
        _artifact_record(
            HERE / "FIG13_FIG14_PANEL_SCHEMA_V8.json",
            name="fig13_fig14_panel_schema",
        ),
        _artifact_record(
            HERE / "FIG13_FIG14_CAPTION_AND_REVIEWER_RESPONSE_V8.md",
            name="fig13_fig14_caption_and_reviewer_response",
        ),
        _artifact_record(
            HERE
            / "00_superseded/v8_prefreeze_attempt_01_support_audit_no_go/"
            "SUPPORT_AUDIT_NO_GO_V8_ATTEMPT_01.json",
            name="v8_attempt_01_support_audit_no_go",
        ),
    ]
    return {
        "schema_version": "revision0710.external_validation.contract.v8",
        "status": "DRAFT_PREFREEZE_V8_NOT_FROZEN_NOT_AUTHORIZED",
        "created_date": "2026-07-11",
        "purpose": "Validity-first locked external-cohort reanalysis of the frozen five-checkpoint D07-003/E55 model in MIMIC-IV v3.1 and two non-overlapping NHANES releases; zero-shot means checkpoint parameters are not updated from external data and does not mean untouched, prospective, or independent confirmatory validation.",
        "success_definition": {
            "validity_first": True,
            "minimum_external_performance_threshold": None,
            "post_hoc_external_optimization_allowed": False,
            "weak_or_negative_transport_results_must_be_reported": True,
        },
        "resource_budget": build_resource_budget(),
        "reviewer_editor_alignment": {
            "Reviewer1_Major6": "external validation is interpreted through calibration, domain shift, differing disease ascertainment, and explicitly sensitivity-only recalibration; weak transport remains reportable",
            "Reviewer2_Comment2": "the exact sequence alignment and external code, text, phenotype, unit, and missingness mappings are published and mechanically verified before outcome read",
            "Editor_original_supported_scope": "report results accurately, remove overstated conclusions, explain cohort and data-source limitations, and predeclare sequence alignment, outlier handling, cross-validation, model selection, bias correction, Cox assumptions, calibration, and competing death",
            "post_V7_scientific_review": "V8 closes prior-result exposure, participant-level density confounding, complete per-covariate and GLOBAL PH diagnostics, conditional-utility language, and NHANES release-slot QC gaps recorded in the byte-bound V7 scientific NO-GO receipt",
        },
        "prohibited_actions": [
            "fine-tune or update any learned model parameter on MIMIC or NHANES",
            "distill from any model, age label, disease label, endpoint, or external metric",
            "fit any external correction in the raw zero-shot or sealed-UKB-corrected primary path",
            "select checkpoints, mappings, landmarks, feature cleaning, horizons, endpoints, or calibration after external performance is viewed",
            "use any NHANES questionnaire label file in the model-input process",
            "combine DEMO_J/J_* with P_* in one NHANES analysis",
            "invent cross-release pooled NHANES survey weights",
            "combine MIMIC incident and recurrent endpoints in one table, curve, metric, or claim",
            "use ICD9 for any MIMIC endpoint without a separately audited future authorization",
            "write any formal cache, prediction, metric, HR, KM, CIF, or figure before a frozen contract and exact authorization exist",
            "describe this locked external-cohort reanalysis as untouched, prospective, or independent confirmatory validation",
            "describe relative age+sex statistical increments as clinical utility, deployment readiness, or deployable performance",
            "use external-data-driven trimming, winsorization, quantile clipping, or outlier thresholds for NHANES",
        ],
        "bindings": {
            "code_artifacts": code_artifacts,
            "prefreeze_artifacts": support,
            "source_registry_sha256": sha256_file(support_paths["source_registry"]),
            "source_registry_external_records": len(registry["external_sources"]),
            "source_registry_locked_artifacts": len(registry["locked_artifacts"]),
            "checkpoint_count": 5,
            "correction_file_count": 5,
            "completion_receipt_count": 5,
            "phenotype_transform_count": 5,
            "correction_keys_exact": 35,
            "inference_environment": INFERENCE_ENVIRONMENT,
            "statistics_environment": STATISTICS_ENVIRONMENT,
            "environment_lock_sha256": sha256_file(support_paths["environment_lock"]),
            "v7_scientific_reviewer_no_go_sha256": sha256_file(
                HERE / "V7_SCIENTIFIC_REVIEWER_NO_GO.json"
            ),
            "v7_tree_snapshot_sha256": sha256_file(
                HERE / "V7_READ_ONLY_TREE_SNAPSHOT.json"
            ),
            "shared_survival_core_relative_path": "Revision-0710/06_figures_and_statistics/03_reviewer_statistics/04_cox_diagnostics_and_competing_risk/survival_analysis_core.py",
            "shared_survival_core_sha256": "6c4dd7eef4ea690d0d88a330f117d8d0b45274d788e10862ab665e245e435a1b",
        },
        "sealed_age_correction": {
            "key": ["checkpoint_outer_fold", "organ_system"],
            "cardinality": "5 * 7 = 35 exact unique keys",
            "formula": "corrected_age_gap = raw_age_gap - (intercept + slope * chronological_age)",
            "formula_tolerance": 1e-12,
            "csv_json_parameter_tolerance": 5e-15,
            "external_fit_performed": False,
            "per_checkpoint_correction_then_mean5": True,
        },
        "compact_cache_contract": {
            "compact_cache_core_sha256": EXPECTED_WORKSPACE_HASHES["compact_cache_core"],
            "build_compact_cache_sha256": EXPECTED_WORKSPACE_HASHES["build_compact_cache"],
            "cache_config_sha256": EXPECTED_WORKSPACE_HASHES["cache_config"],
            "sequence_length": 32,
            "selection": {
                "primary": {
                    "input_variant": MIMIC_PRIMARY_VARIANT,
                    "allowed_sources": ["hospital_icd"],
                    "crop": "latest 32 after exact stable sort and dedup",
                },
                "bare_omr_sensitivity": {
                    "input_variant": MIMIC_BARE_OMR_SENSITIVITY,
                    "allowed_sources": ["hospital_icd"],
                    "crop": "latest 32 after exact stable sort and dedup",
                },
                "inpatient_prescription_proxy_sensitivity": {
                    "input_variant": MIMIC_RX_SENSITIVITY,
                    "allowed_sources": [
                        "hospital_icd",
                        "inpatient_prescription_proxy",
                    ],
                    "minimum_latest_hospital_icd": 8,
                    "minimum_latest_inpatient_prescription_proxy": 8,
                    "remaining_global_latest": 16,
                    "short_modality_unused_quota_reallocated_to_global_latest": True,
                },
            },
            "strict_pre_index": "active event_time < landmark and relative_days is integer <= -1",
            "relative_day_formula": "floor((event_time - landmark_time) / 86400 seconds)",
            "sort_order": [
                "participant_id",
                "event_time",
                "event_type_id",
                "canonical_code_utf8_bytes",
                "canonical_description_utf8_bytes",
                "source_row_number",
            ],
            "sort_algorithm": "stable mergesort",
            "dedup_key": [
                "participant_id",
                "event_time",
                "canonical_code",
                "canonical_description",
                "source",
            ],
            "dedup_keep": "first after exact stable sort",
            "event_type_ids": {
                "padding": 0,
                "hospital_icd": 1,
                "inpatient_prescription_proxy": 2,
            },
            "text_key_bytes": "ASCII semantic role + byte 0x1F + normalized UTF-8 description",
            "padding": "valid_mask=false and code_id=text_id=relative_days=event_type_id=0",
            "inpatient_prescription_proxy_active_code_id": 0,
            "inpatient_prescription_proxy_active_text_id": "strictly positive",
            "nhanes_sequence": "32 padding slots for phenotype_only_label_blind primary input",
        },
        "vocabulary_and_embedding_extension": {
            "original_code_vocab_sha256": EXPECTED_WORKSPACE_HASHES["event_code_vocab"],
            "original_text_vocab_sha256": EXPECTED_WORKSPACE_HASHES["event_text_vocab"],
            "original_embedding_table_sha256": EXPECTED_WORKSPACE_HASHES["embedding_tables"],
            "original_rows_immutable": True,
            "icd10": "NFKC, trim Unicode whitespace, uppercase, remove decimal, exact bare UKB code match first",
            "icd9_namespace_text": "ICD9\\u001F{canonical_code_without_decimal}",
            "icd9_namespace_bytes_hex_prefix": "494344391f",
            "description_normalization": "NFKC then collapse every Unicode whitespace run to one ASCII space and trim",
            "empty_description_fallback": "{code_system} code {canonical_code_without_namespace}",
            "text_role_separator_byte_hex": "1f",
            "unseen_token_order": "unique canonical UTF-8 byte strings sorted lexicographically",
            "clip_model": "ViT-B-16",
            "clip_pretrained": "openai",
            "clip_weight_sha256": EXPECTED_WORKSPACE_HASHES["clip_weight"],
            "open_clip_version": "3.3.0",
            "torch_version": "2.5.1",
            "device": "cpu",
            "encode_dtype": "float32",
            "stored_extension_dtype": "little-endian float16",
            "encode_batch_size": 4,
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "deterministic_algorithms": True,
            "original_row_bytes_policy": "copy original little-endian float16 row bytes as the exact prefix; append extension row bytes; never re-encode original rows",
            "cpu_probe_requirement": "encode the same fixed texts twice in one process and require identical SHA-256 after one float32-to-float16 cast",
            "required_coverage_output_table": "shared/03_cache_audit/external_vocabulary_coverage.csv",
            "required_coverage_dimensions": [
                "source rows by modality and code system",
                "unique canonical tokens",
                "exact frozen-vocab matches",
                "deterministic extension tokens",
                "invalid or dropped tokens with closed reason",
                "participants with zero events",
                "pre-crop and post-crop active events",
                "original embedding prefix byte hash",
            ],
            "coverage_token_classes_exact": [
                "exact_frozen_vocab",
                "deterministic_extension",
                "invalid_dropped",
                "zero_event",
            ],
            "coverage_closed_cardinality": 60,
            "coverage_code_systems_separate": [
                "ICD9",
                "ICD10",
                "INPATIENT_RX",
                "not_applicable",
            ],
            "learned_projection_or_parameter_change": False,
        },
        "shared_prediction": {
            "fine_tuning": False,
            "distillation": False,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_selection_from_external_metrics": False,
            "per_checkpoint_predictions_required": True,
            "ensemble": "arithmetic mean of the exact same participant-organ rows from folds 0..4",
            "checkpoint_age_anchor": {
                "fold0": {
                    "key": "age_anchor_train_folds_2_to_4_only",
                    "value": 57.03986740112305
                },
                "fold1": {"key": "age_anchor_train_role_only", "value": 57.03695297241211},
                "fold2": {"key": "age_anchor_train_role_only", "value": 57.03803253173828},
                "fold3": {"key": "age_anchor_train_role_only", "value": 57.04105758666992},
                "fold4": {"key": "age_anchor_train_role_only", "value": 57.04116439819336}
            },
            "raw_zero_shot_primary": True,
            "sealed_ukb_corrected_gap_primary_for_downstream": True,
            "local_external_calibration_zero_shot": False,
            "zero_shot_definition": "checkpoint_parameters_never_updated_using_external_data_only",
            "untouched_validation_claim_allowed": False,
            "prospective_validation_claim_allowed": False,
            "independent_confirmatory_claim_allowed": False,
        },
        "mimic_iv_3_1": {
            "role": "hospitalized-EHR transport and hospital-recorded post-discharge risk",
            "landmark": "discharge time of the second chronologically sorted admission after time-valid, adult-at-discharge, and survived-discharge filtering",
            "admission_selection_order": "filter finite ordered times, compute integer age and require age>=18, require deathtime absent or after discharge, then stable sort and take ordinal 2",
            "admission_sort": "admittime ascending, dischtime ascending, numeric hadm_id ascending",
            "valid_admission": "finite admittime/dischtime, dischtime > admittime, age at discharge >=18, and admissions.deathtime absent or strictly after discharge",
            "chronological_age": "integer anchor_age + (index discharge calendar year - integer anchor_year); reject anchor_age=91",
            "age_populations": {
                "age_metric_primary": "integer 40 through 73 inclusive common support",
                "age_metric_adult_sensitivity": "integer age >=18, anchor_age !=91",
                "age_quantization_sensitivities": [
                    "common-support chronological age minus 0.5 year",
                    "common-support chronological age plus 0.5 year",
                ],
                "risk_primary": "all adult non-topcoded integer-year participants",
            },
            "model_event_cutoff": "strictly before discharge landmark",
            "preseal_diagnosis_materialization": "PyArrow record-batch stream; predicate hadm_id against completed-admission allowlist with dischtime strictly before landmark; write preindex-only Parquet before any pandas conversion or canonicalization; raw rows converted to pandas and materialized post-index rows must both equal zero",
            "phenotype_cutoff": "labs require charttime and storetime at or before discharge; date-only OMR must precede the discharge calendar date; both use the frozen 365-day window",
            "feature_coverage_output": "01_mimic/01_mapping/feature_coverage.csv with all 78 slots, observed n/fraction, source hash/selector/unit/formula",
            "participant_density": {
                "construction_phase": "label_blind_cache_before_prediction_and_outcome_read",
                "measures": ["pre_crop_event_count", "post_crop_event_count", "history_span_days", "observed_phenotype_slots"],
                "fixed_strata": "pre_crop_event_count low=0-7, medium=8-31, high>=32",
                "strata_training_or_outcome_dependent": False,
                "age_error_and_gap_correlations_output": "01_mimic/08_density_sensitivity/density_diagnostics.csv",
                "heldout_risk_sensitivity": "age+sex+density versus age+sex+density+corrected_gap; density scaling fit on four training folds only",
                "interpretation": "relative statistical increment only; no clinical utility or deployability claim",
            },
            "cox_ph_diagnostics": {
                "shared_core_relative_path": "Revision-0710/06_figures_and_statistics/03_reviewer_statistics/04_cox_diagnostics_and_competing_risk/survival_analysis_core.py",
                "shared_core_sha256": "6c4dd7eef4ea690d0d88a330f117d8d0b45274d788e10862ab665e245e435a1b",
                "coverage": "full association age/sex/gap and every held-out target/death base/augmented/density covariate",
                "grain": "role x variant x organ x horizon x fold x model x cause x covariate x transform",
                "duration_rule": "diagnostic duration = observed_time_days + 1 day for every row, preserving order and ties while supporting day-0 death; log transform is therefore log1p(original days)",
                "transforms": ["rank_lifelines_cumsum_event", "log1p_days"],
                "global_test": "joint scaled-Schoenfeld score vector with complete covariance; never Fisher or minimum-p aggregation",
                "failed_fit_rows_retained": True,
                "silent_drop_allowed": False,
            },
            "diagnosis_time_proxy": "associated completed admission dischtime is the availability/event timestamp; preseal model events require dischtime < landmark, baseline endpoint history uses dischtime <= landmark, and post-index outcomes require dischtime > landmark; diagnosis seq_num and hadm_id break ties",
            "subject_observation_end": "maximum valid admissions.dischtime, labevents.charttime, omr.chartdate, and prescriptions stop/start time, bounded by the frozen death ascertainment cap",
            "phenotype_data_availability": {
                "labevents": "require nonmissing charttime and storetime, both <= discharge landmark; latest tie order charttime, storetime, labevent_id",
                "omr": "date-only OMR requires chartdate strictly before the discharge calendar date and within the prior 365 calendar days; same-calendar-date OMR is excluded",
            },
            "administrative_censor": "minimum of prespecified 6/12-month horizon and subject-specific observation end",
            "death_ascertainment_cap": "minimum of index_time plus the integer calendar-day offset from index date to patients.dod, and last valid hospital discharge plus 365 days; cap is also an administrative censor boundary",
            "month_days": "6 months=182.5 days; 12 months=365.0 days",
            "horizons": {"authorized_months_exact": [6, 12], "authorized_days_exact": [182.5, 365.0]},
            "death_time": "date-only patients.dod: dates before index calendar date are inconsistent/excluded; the same calendar date is a conservative day-0 competing death after survived-discharge eligibility; later dates use integer calendar-day offset",
            "postseal_only_fields": [
                "patients.dod",
                "postindex diagnoses",
                "subject observation end",
                "min(index_time + integer calendar-day DOD offset, last valid discharge plus 365 days) follow-up cap",
                "NHANES questionnaire labels",
            ],
            "target_death_tie": "death wins when death_time <= target_time",
            "incident_primary": {
                "endpoint_code_system": "ICD10 only",
                "exclude_any_preindex_icd9": True,
                "target_free_definition": "no mapped target-organ ICD10 from any completed admission with dischtime <= index discharge",
                "event": "first mapped target-organ ICD10 availability at a completed admission dischtime strictly after index discharge",
                "icd9_endpoint_authorized": False,
            },
            "recurrent_sensitivity": {
                "physically_separate": True,
                "population": "mapped target-organ ICD10 available from a completed admission with dischtime <= index discharge and no completed pre-index/index ICD9",
                "event": "next mapped target-organ ICD10 availability at a completed admission dischtime strictly after index discharge",
                "may_combine_with_incident": False,
            },
            "competing_death": {
                "primary_cause_specific": True,
                "aalen_johansen_target_and_death_cif": True,
                "target_or_death_composite_sensitivity": False,
                "curve_output": "01_mimic/06_incident_risk/competing_risk_curves.csv with AJ target/death CIF and separately labelled cause-specific KM only",
                "fine_gray": "unauthorized unless separately audited",
            },
            "external_hash_fold": "first 8 SHA256 bytes of 'revision0710-external-v3|{subject_id}', big-endian, modulo 5",
            "ipcw": {
                "censoring_km_fit": "four training hash folds only",
                "evaluation": "held-out fifth fold only",
                "outcome_definition": "case=target before death/censor/horizon; control=event-free at horizon or competing death before horizon; loss/admin censor before horizon is unobserved",
                "censor_definition": "censor event=subject observation end or death ascertainment cap before target/death/horizon; target and death are non-censor terminal events for G",
                "left_limit": True,
                "minimum_G": 0.05,
                "weight_cap": "99th percentile of training-fold weights",
                "same_horizon_training_nuisance_only": True,
                "absolute_risk": "reconstruct post-jump Breslow baseline hazard increments from complete target and competing-death PHReg risk sets, including first and final event-time jumps, then integrate them into target CIF; use target Cox linear predictor for cause-specific Harrell C and target CIF probability for IPCW AUC, Brier, and linear calibration",
                "brier_normalization": "sum IPCW squared error divided by full held-out test n",
                "diagnostics": "unfloored G minimum, G floor, training p99 cap, capped count, effective sample size, observed and censored counts",
            },
            "full_cohort_hr": {
                "analysis_population": "full non-held-out organ-specific incident-eligible cohort after excluding baseline target ICD10 and any completed pre-index ICD9; not the full 95,683-subject MIMIC input cohort",
                "exposure": "corrected_age_gap_mean5 standardized by the incident-eligible organ-specific analysis-population SD",
                "covariates": "integer chronological age + sex_male + standardized corrected age gap",
                "ties": "Efron",
                "standard_error": "subject-level robust sandwich",
                "multiplicity": "Benjamini-Hochberg across seven organs within each input variant and horizon",
                "proportional_hazards_test": "scaled Schoenfeld residual test for exposure",
            },
            "bootstrap": "one global PCG64 SeedSequence subject-cluster multiplicity plan with B=200 shared across all organs, horizons, variants, and fixed hash folds; resample held-out subjects while keeping training-fold target Cox, death Cox, and censoring KM fixed",
            "claim_limits": [
                "hospital-recorded endpoints, not complete population incidence",
                "only 6- and 12-month horizons are authorized under the available death ascertainment",
                "non-held-out incident-eligible organ-specific HR is association, held-out delta C/IPCW is prediction, neither is causal",
            ],
        },
        "nhanes": {
            "role": "cross-sectional phenotype-domain transfer and disease association, not longitudinal incident prediction",
            "allowed_releases_exact_order": list(RELEASES),
            "forbid_DEMO_J_or_J_star_with_P_star": True,
            "p_release": "official P_* 2017-March 2020 pre-pandemic release, including the 2017-2018 sample",
            "l_release": "official *_L August 2021-August 2023 release",
            "primary_input": "phenotype_only_label_blind",
            "questionnaire_files_read_by_input_builder": [],
            "event_sequence": "all 32 slots padding",
            "primary_age_population": "RIDSTATR == 2 and RIDAGEYR integer 40 through 73 inclusive",
            "primary_age_integer_guard": "fractional RIDAGEYR in the examined 40-73 candidate range is a hard error",
            "raw_registry_primary_count_sanity": {
                P_RELEASE: 4916,
                L_RELEASE: 3720,
                "total": 8636,
            },
            "age_80_top_coded": "exclude from MAE, RMSE, R2, calibration intercept/slope, and every exact-error calculation; inclusion is an exact error",
            "feature_map": "78 slots for each release; unavailable slots remain raw_zero=0/missing_mask=true",
            "all_78_slots_adjudicated_only_after_mechanical_audit": True,
            "release_slot_qc": {
                "grain": "release x all 78 phenotype slots",
                "required_fields": "finite count, prefrozen physical-unit range count, min, q01, q25, median, q75, q99, max",
                "decision_rule": "any observed nonfinite or value outside a prefrozen unit range aborts before inference",
                "external_data_driven_trimming_or_winsorization": False,
                "quantiles_descriptive_only": True,
            },
            "questionnaire_labels": "exploratory Panel E; six estimable organs per release; immune fixed not_estimable; outcome-only physical process",
            "input_label_physical_isolation": True,
            "survey_weight_sensitivity": {
                P_RELEASE: "WTMECPRP within P release only",
                L_RELEASE: "WTMEC2YR within L release only",
                "cross_release_pooled_weight": "forbidden",
            },
            "local_crossfit_calibration": {
                "regression": "chronological_age = beta0 + beta1*predicted_age_raw_mean5 + beta2*sex_male",
                "folds": "for held-out hash fold k, fit on the other four folds within the same release and organ",
                "coefficient_order": "intercept,predicted_age_raw_mean5,sex_male",
                "application": "apply the three training-fold coefficients once to held-out predicted_age_raw_mean5 and sex_male",
                "role": "sensitivity only; never zero-shot and never enters primary predictions or corrected gaps",
            },
            "questionnaire_association": {
                "model": "release-specific survey-weighted logistic regression",
                "exposure": "corrected_age_gap_mean5 standardized within release and organ",
                "covariates": "intercept + standardized corrected gap + chronological age centered per 10 years + sex_male",
                "design": "full release-specific examined MEC design retained; integer age 40-73 enters as a survey domain; MEC weight, SDMVSTRA strata, SDMVPSU PSU",
                "inference": "survey-domain pseudolikelihood Taylor linearization with PSU within strata, centered lonely PSU, and t/Wald reference using design df = PSU minus strata",
                "multiplicity": "Benjamini-Hochberg across six estimable organs within each release",
            },
            "claim_limit": "locked release-specific age-transfer reanalysis and exploratory cross-sectional association only; not untouched confirmation, incident prediction, clinical utility, deployability, longitudinal robustness, causal effect, or network conservation",
        },
        "statistical_algorithms": {
            "aalen_johansen": "at each unique event time update all-cause survival S(t)=S(t-)*(1-d_all/n); add CIF_k(t)=CIF_k(t-)+S(t-)*d_k/n; death is cause 2 and wins exact ties",
            "harrell_c": "comparable pairs require earlier observed target event; ties in risk contribute 0.5",
            "phreg_baseline_hazard": "use fitted Efron PHReg coefficients but reconstruct Breslow dH0(t)=d(t)/sum_{i:Ti>=t}exp(xi beta) directly from the complete risk set; cumulative risk at a horizon is the sum of all jumps at t<=horizon, including first and final event-time jumps",
            "ipcw_auc_brier": "target CIF from explicit post-jump Breslow target/death cause-specific hazard jumps; IPCW AUC ranks target CIF probability rather than target cause-specific linear predictor; case=target before terminal boundary; control=event-free at horizon or competing death before horizon; subject observation/admin censor before horizon is unobserved; fit censoring KM on four training folds, evaluate left-limit G on held-out fold, clamp G at 0.05, cap at training-weight p99; Brier divides by full test n",
            "bootstrap": "one global PCG64 SeedSequence participant-cluster multiplicity plan retained across organs, horizons, variants, and folds; predictive intervals are conditional on fixed training-fold target/death Cox and censor KM",
            "uncertainty": "predictive rows use linear 2.5/97.5 percentiles over B=200 conditional held-out subject-cluster replicates; zero-shot age metrics, full time-grid curves, and recurrent CIFs are explicitly point-only; Cox HRs use robust Wald intervals",
        },
        "output_schema": {
            "relative_path": "OUTPUT_SCHEMAS_V8.json",
            "sha256": sha256_file(support_paths["output_schemas"]),
            "all_columns_bind_dtype_nullability_key_grain_cardinality_enum_unit_formula_tolerance": True,
            "formal_receipt": "one create-once manifest with path, bytes, SHA256, rows, columns, and formula/tolerance schema SHA256 for each of all 20 tables",
        },
        "negative_tests_required": [
            "duplicate JSON key",
            "extra key or record",
            "missing key or record",
            "wrong JSON type including bool-versus-int",
            "reordered ordered record",
            "wrong source or artifact SHA-256",
            "wrong relative or absolute path",
            "formal target or staging target already exists",
            "DEMO_J/J_* combined with P_*",
            "NHANES questionnaire source enters input registry",
            "NHANES age 80 enters exact-error population",
            "NHANES primary L32 contains any active event",
            "MIMIC primary incident row has any pre-index ICD9",
            "MIMIC ICD9 endpoint requested",
            "MIMIC incident and recurrent rows share one physical table",
            "event at or after landmark",
            "timeline tie/dedup/order, primary latest-32, or inpatient-prescription sensitivity 8+8+16 crop mismatch",
            "padding nonzero or active inpatient-prescription-proxy event_code_id nonzero",
            "text role or 0x1F separator mismatch",
            "Unicode/whitespace/code namespace/fallback mismatch",
            "correction key count other than 5x7 or external fit flag true",
            "corrected gap formula error above 1e-12",
            "CLIP model/version/weight/device/dtype/batch/thread/determinism mismatch",
            "original UKB embedding prefix bytes change",
            "checkpoint count/hash or transform count/hash mismatch",
            "MIMIC map selector/unit/formula mutation including lymphocyte and reticulocyte conversions",
            "lab charttime before landmark but storetime after landmark",
            "date-only OMR on the discharge calendar date",
            "overlapping admission diagnosis classified before completed dischtime",
            "same-calendar-date DOD silently dropped instead of day-0 competing death",
            "non-midnight index with later date-only DOD shifted by a fractional day or censored at the death cap",
            "PHReg pre-jump cumulative hazard used as post-jump or first/final Breslow event jump omitted",
            "IPCW AUC ranks cause-specific Cox linear predictor instead of target CIF probability",
            "NHANES release identity or cohort-specific input variant substituted in a sealed prediction universe",
            "runtime ICD NFKC/internal-whitespace mismatch",
            "one output key silently deleted from an upstream-bound universe",
            "completed publish bundle table/manifest/lineage bytes tampered before resume or jointly self-consistent bundle copies diverge from stage05/stage06 successful-attempt originals",
            "output schema dtype/cardinality/enum mutation",
            "formal authorization field/path/hash/environment/mode/executor/target/seed/key-order/record-schema/rejection-policy mismatch",
            "None or NaN accepted by a canonicalizer",
        ],
        "formal_gate": {
            "contract_frozen": False,
            "formal_authorization_exists": FORMAL_AUTHORIZATION.exists(),
            "formal_lock_exists": FORMAL_LOCK.exists(),
            "formal_consumed_authorization_exists": FORMAL_CONSUMED_AUTHORIZATION.exists(),
            "formal_target_relative_path": repo_relative(FORMAL_TARGET),
            "formal_target_resolved": str(FORMAL_TARGET.resolve()),
            "formal_target_exists": FORMAL_TARGET.exists(),
            "formal_execution_authorized": False,
            "formal_B200_conditional_execution_performed": False,
            "formal_cache_inference_metrics_hr_km_cif_written": False,
            "authorization_schema_relative_path": "FORMAL_AUTHORIZATION_SCHEMA_V8.json",
            "authorization_schema_sha256": sha256_file(
                support_paths["formal_authorization_schema"]
            ),
        },
    }


def validate_formal_targets_absent(
    formal_target: Path = FORMAL_TARGET,
    authorization: Path = FORMAL_AUTHORIZATION,
) -> None:
    if authorization.exists():
        raise RuntimeError(f"formal authorization must remain absent: {authorization}")
    if formal_target.exists():
        raise RuntimeError(f"formal target must remain absent: {formal_target}")
    lock = authorization.parent / ".EXTERNAL_VALIDATION_FORMAL_V8_ONCE.lock"
    consumed = authorization.with_name(f"{authorization.stem}.CONSUMED.json")
    if lock.exists() or consumed.exists():
        raise RuntimeError(
            f"formal lock/consumed authorization must remain absent: {lock}, {consumed}"
        )
    if formal_target.parent.exists():
        staging = list(formal_target.parent.glob(f".{formal_target.name}.staging.*"))
        if staging:
            raise RuntimeError(f"formal staging target already exists: {staging}")


def validate_nhanes_release_paths(paths: Iterable[Path]) -> None:
    names = [path.name.upper() for path in paths]
    has_p = any(name.startswith("P_") for name in names)
    has_j = any(name == "DEMO_J.XPT" or name.endswith("_J.XPT") for name in names)
    if has_p and has_j:
        raise ValueError("DEMO_J/J_* cannot coexist with P_* in V8 NHANES")
    if has_j:
        raise ValueError("DEMO_J/J_* is not an authorized V8 NHANES release")


def nhanes_exact_age_mask(age: np.ndarray, ridstatr: np.ndarray) -> np.ndarray:
    values = np.asarray(age, dtype=np.float64)
    interview_exam_status = np.asarray(ridstatr, dtype=np.float64)
    if values.shape != interview_exam_status.shape:
        raise ValueError("NHANES age and RIDSTATR shapes differ")
    if np.any(values == 80):
        raise ValueError("NHANES age 80 is top-coded and forbidden in exact-error input")
    return (
        np.isfinite(values)
        & np.isfinite(interview_exam_status)
        & (interview_exam_status == 2)
        & (values >= 40)
        & (values <= 73)
        & (values == np.floor(values))
    )


def mimic_months_to_days(months: int) -> float:
    if months not in {6, 12}:
        raise ValueError("V8 authorizes only 6 and 12 month horizons")
    return {6: 182.5, 12: 365.0}[months]


def adjudicate_target_death(
    *,
    target_time_days: float | None,
    death_time_days: float | None,
    subject_observation_end_days: float,
    death_ascertainment_cap_days: float,
    horizon_months: int,
) -> dict[str, Any]:
    horizon_days = mimic_months_to_days(horizon_months)
    target = np.inf if target_time_days is None else float(target_time_days)
    death = np.inf if death_time_days is None else float(death_time_days)
    observation_end = float(subject_observation_end_days)
    death_cap = float(death_ascertainment_cap_days)
    if observation_end < 0 or death_cap < 0:
        raise ValueError("observation end/death cap cannot precede the landmark")
    observed = min(target, death, observation_end, death_cap, horizon_days)
    if (
        death <= target
        and death <= observation_end
        and death <= death_cap
        and death <= horizon_days
    ):
        reason = "death"
        event = 0
        competing = 1
    elif (
        target < death
        and target <= observation_end
        and target <= death_cap
        and target <= horizon_days
    ):
        reason = "target"
        event = 1
        competing = 0
    elif observation_end <= death_cap and observation_end < horizon_days:
        reason = "subject_observation_end"
        event = 0
        competing = 0
    elif death_cap < horizon_days:
        reason = "death_ascertainment_cap"
        event = 0
        competing = 0
    else:
        reason = "horizon"
        event = 0
        competing = 0
    return {
        "observed_time_days": observed,
        "event_indicator": event,
        "competing_death_indicator": competing,
        "censor_reason": reason,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_independent_verifier(artifact_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(HERE / "verify_external_validation_prefreeze_v8.py"),
        "--artifact-dir",
        str(artifact_dir),
        "--repo-root",
        str(ROOT),
        "--rehash-external",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "independent verifier failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    payload = json.loads(completed.stdout)
    if payload.get("status") != "PREFREEZE_V8_INDEPENDENT_VERIFY_PASS":
        raise RuntimeError(f"independent verifier returned unexpected payload: {payload}")
    return payload


def prepare_assets() -> dict[str, Any]:
    validate_formal_targets_absent()
    if ARTIFACT_DIR.exists():
        raise FileExistsError(f"create-once prefreeze artifact directory exists: {ARTIFACT_DIR}")
    stage = HERE / f".{ARTIFACT_DIR.name}.staging.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        registry = build_source_registry()
        write_json(stage / "SOURCE_REGISTRY_V8.json", registry)
        _write_csv(stage / "MIMIC_78_SLOT_MAP_V8.csv", build_mimic_feature_map(registry))
        _write_csv(
            stage / "NHANES_78_SLOT_RELEASE_MAP_V8.csv",
            build_nhanes_feature_map(registry),
        )
        _write_csv(
            stage / "NHANES_QUESTIONNAIRE_LABEL_MAP_V8.csv",
            build_nhanes_label_map(registry),
        )
        _write_csv(stage / "MIMIC_ICD10_ORGAN_MAP_V8.csv", build_mimic_icd10_map())
        write_json(stage / "OUTPUT_SCHEMAS_V8.json", build_output_schemas())
        write_json(stage / "ENVIRONMENT_LOCK_V8.json", build_environment_lock())
        write_json(
            stage / "FORMAL_AUTHORIZATION_SCHEMA_V8.json",
            build_formal_authorization_schema(),
        )
        contract = build_contract(stage, registry)
        write_json(stage / "EXTERNAL_VALIDATION_CONTRACT_V8.json", contract)
        independent = _run_independent_verifier(stage)
        if ARTIFACT_DIR.exists():
            raise FileExistsError(ARTIFACT_DIR)
        os.rename(stage, ARTIFACT_DIR)
        final = _run_independent_verifier(ARTIFACT_DIR)
        return {
            "status": "PREFREEZE_V8_ASSETS_CREATED_SINGLE_ATOMIC_RENAME",
            "artifact_dir": str(ARTIFACT_DIR.resolve()),
            "contract_sha256": sha256_file(
                ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
            ),
            "stage_independent": independent,
            "final_independent": final,
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def import_compact_cache_core() -> Any:
    core_dir = ROOT / "Revision-0710/01_data_rebuild/scripts"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    import compact_cache_core

    return compact_cache_core


def _load_formal_executor_for_probe() -> Any:
    path = HERE / "execute_external_validation_v8.py"
    spec = importlib.util.spec_from_file_location("external_validation_v8_probe_executor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V8 formal executor primitives")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_timeline_probe() -> dict[str, Any]:
    executor = _load_formal_executor_for_probe()
    landmark = pd.Timestamp("2020-01-01")
    rows: list[dict[str, Any]] = []
    for source_index, (source, event_type_id) in enumerate(
        (("hospital_icd", 1), ("inpatient_prescription_proxy", 2))
    ):
        for index in range(40):
            relative = -(80 - 2 * index - source_index)
            rows.append(
                {
                    "participant_id": "synthetic-1",
                    "event_time": landmark + pd.Timedelta(days=relative),
                    "landmark_time": landmark,
                    "source": source,
                    "event_type_id": event_type_id,
                    "canonical_code": f"I{index:02d}" if event_type_id == 1 else "",
                    "canonical_description": f"{source} synthetic {index}",
                    "source_row_number": source_index * 100 + index,
                }
            )
    frame = pd.DataFrame(rows)
    primary = executor.crop_variant_timeline(frame.loc[frame["source"].eq("hospital_icd")], MIMIC_PRIMARY_VARIANT)
    sensitivity = executor.crop_variant_timeline(frame, MIMIC_RX_SENSITIVITY)
    primary_counts = primary["source"].value_counts().to_dict()
    sensitivity_counts = sensitivity["source"].value_counts().to_dict()
    if len(primary) != 32 or primary_counts != {"hospital_icd": 32}:
        raise AssertionError("primary crop is not hospital-ICD-only latest 32")
    if (
        len(sensitivity) != 32
        or sensitivity_counts.get("hospital_icd", 0) != 16
        or sensitivity_counts.get("inpatient_prescription_proxy", 0) != 16
    ):
        raise AssertionError("prescription-proxy sensitivity is not exact 8+8+16")
    aligned = executor.left_align_l32(
        primary, ["synthetic-1", "synthetic-padding"]
    )
    if not aligned["event_valid_mask"][0].all() or aligned["event_valid_mask"][1].any():
        raise AssertionError("modality-deleted primary sequence is not left-aligned L32")
    nhanes_padding = {
        "event_code_id": np.zeros((2, 32), dtype=np.int64),
        "event_text_id": np.zeros((2, 32), dtype=np.int64),
        "event_relative_days": np.zeros((2, 32), dtype=np.int32),
        "event_type_id": np.zeros((2, 32), dtype=np.int8),
        "event_valid_mask": np.zeros((2, 32), dtype=bool),
    }
    executor.validate_nhanes_padding(nhanes_padding)
    active = frame.iloc[[0]].copy()
    active["event_time"] = landmark
    try:
        executor.crop_variant_timeline(active, MIMIC_PRIMARY_VARIANT)
    except ValueError as exc:
        active_rejected = "at or after landmark" in str(exc)
    else:
        active_rejected = False
    if not active_rejected:
        raise AssertionError("active landmark event was not rejected")
    digest = hashlib.sha256()
    for selected in (primary, sensitivity):
        semantic = selected[
            [
                "participant_id",
                "event_time",
                "source",
                "event_type_id",
                "canonical_code",
                "canonical_description",
                "source_row_number",
            ]
        ].to_csv(index=False, lineterminator="\n")
        digest.update(semantic.encode("utf-8"))
    for key in sorted(aligned):
        digest.update(key.encode("ascii"))
        digest.update(np.ascontiguousarray(aligned[key]).tobytes())
    return {
        "status": "SYNTHETIC_V8_TIMELINE_PROBE_PASS",
        "primary_selected_rows": len(primary),
        "primary_hospital_icd_selected": int(primary_counts["hospital_icd"]),
        "sensitivity_selected_rows": len(sensitivity),
        "sensitivity_hospital_icd_selected": int(sensitivity_counts["hospital_icd"]),
        "sensitivity_inpatient_prescription_proxy_selected": int(
            sensitivity_counts["inpatient_prescription_proxy"]
        ),
        "timeline_sha256": digest.hexdigest(),
        "strictly_pre_index": bool(
            (primary["event_time"] < primary["landmark_time"]).all()
            and (sensitivity["event_time"] < sensitivity["landmark_time"]).all()
        ),
        "primary_left_aligned_l32": True,
        "nhanes_padding_only": True,
        "event_at_landmark_rejected": True,
    }


def write_tiny_smoke(tag: str) -> dict[str, Any]:
    validate_formal_targets_absent()
    if not tag.startswith("tiny_") or any(character in tag for character in "\\/:"):
        raise ValueError("tiny smoke tag must be one path component beginning tiny_")
    _run_independent_verifier(ARTIFACT_DIR)
    destination = SMOKE_DIR / tag
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{tag}.staging.{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        probe = synthetic_timeline_probe()
        write_json(stage / "SYNTHETIC_TIMELINE_PROBE.json", probe)
        receipt = {
            "schema_version": "revision0710.external_validation.tiny_smoke_receipt.v8",
            "status": "TINY_NO_FORMAL_SMOKE_PASS",
            "tag": tag,
            "contract_sha256": sha256_file(
                ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
            ),
            "source_registry_sha256": sha256_file(
                ARTIFACT_DIR / "SOURCE_REGISTRY_V8.json"
            ),
            "synthetic_probe_sha256": sha256_file(stage / "SYNTHETIC_TIMELINE_PROBE.json"),
            "formal_cache_inference_metrics_created": False,
            "formal_target_absent": True,
        }
        write_json(stage / "RECEIPT.json", receipt)
        os.rename(stage, destination)
        return {
            "status": "TINY_NO_FORMAL_SMOKE_CREATED",
            "destination": str(destination.resolve()),
            "receipt_sha256": sha256_file(destination / "RECEIPT.json"),
            "probe": probe,
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def cpu_clip_probe() -> dict[str, Any]:
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError("CPU CLIP probe requires the locked Torch_25 environment") from exc
    from v8_execution_guard import validate_current_environment

    environment_lock_path = ARTIFACT_DIR / "ENVIRONMENT_LOCK_V8.json"
    environment_lock = read_json(environment_lock_path)
    locked_record = environment_lock["inference"]
    validated_record = validate_current_environment(
        locked_record, expected_role="inference"
    )
    package_versions = {
        row["name"]: row["version"] for row in validated_record["packages"]
    }
    observed = {
        "python": validated_record["python_version"],
        "numpy": package_versions["numpy"],
        "pandas": package_versions["pandas"],
        "torch": package_versions["torch"],
        "open_clip": package_versions["open-clip-torch"],
        "environment_lock_sha256": sha256_file(environment_lock_path),
    }
    if sha256_file(CLIP_WEIGHT) != EXPECTED_WORKSPACE_HASHES["clip_weight"]:
        raise RuntimeError("CPU CLIP weight hash mismatch")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(CLIP_WEIGHT)
    )
    model = model.cpu().float().eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    texts = [
        "ICD9 code 25000",
        "Essential hypertension",
        "inpatient prescription proxy synthetic",
        "Unicode whitespace normalized",
    ]
    tokens = tokenizer(texts)

    def encode_hash() -> str:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), 4):
                encoded = model.encode_text(tokens[start : start + 4].cpu()).float().cpu()
                chunks.append(encoded.numpy().astype("<f2", copy=False))
        payload = np.ascontiguousarray(np.concatenate(chunks, axis=0)).tobytes()
        return hashlib.sha256(payload).hexdigest()

    first = encode_hash()
    second = encode_hash()
    if first != second:
        raise AssertionError("CPU float32 encode then fp16 probe is not repeatable")
    return {
        "schema_version": "revision0710.external_validation.cpu_clip_probe.v8",
        "status": "CPU_FLOAT32_ENCODE_FP16_REPEAT_HASH_PASS",
        "environment": observed,
        "device": "cpu",
        "model": "ViT-B-16",
        "pretrained": "openai",
        "weight_sha256": EXPECTED_WORKSPACE_HASHES["clip_weight"],
        "batch_size": 4,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "encode_dtype": "float32",
        "stored_dtype": "little-endian float16",
        "texts": len(texts),
        "first_sha256": first,
        "second_sha256": second,
        "repeat_equal": True,
    }


def write_cpu_clip_probe(tag: str) -> dict[str, Any]:
    validate_formal_targets_absent()
    if not tag.startswith("cpu_probe_") or any(character in tag for character in "\\/:"):
        raise ValueError("CPU probe tag must begin cpu_probe_")
    destination = SMOKE_DIR / tag
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    probe = cpu_clip_probe()
    stage = destination.parent / f".{tag}.staging.{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        write_json(stage / "CPU_CLIP_PROBE.json", probe)
        os.rename(stage, destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "status": probe["status"],
        "destination": str(destination.resolve()),
        "probe_sha256": sha256_file(destination / "CPU_CLIP_PROBE.json"),
        "embedding_sha256": probe["first_sha256"],
    }


def validate_only() -> dict[str, Any]:
    validate_formal_targets_absent()
    independent = _run_independent_verifier(ARTIFACT_DIR)
    return {
        "status": "EXTERNAL_VALIDATION_V8_VALIDATE_ONLY_PASS_NO_FORMAL_WRITE",
        "artifact_dir": str(ARTIFACT_DIR.resolve()),
        "contract_sha256": sha256_file(
            ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
        ),
        "independent": independent,
        "formal_target_absent": True,
        "formal_authorization_absent": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-assets", action="store_true")
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--tiny-smoke", action="store_true")
    action.add_argument("--cpu-clip-probe", action="store_true")
    parser.add_argument("--tag", default="tiny_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_assets:
        result = prepare_assets()
    elif args.validate_only:
        result = validate_only()
    elif args.tiny_smoke:
        result = write_tiny_smoke(args.tag)
    else:
        result = write_cpu_clip_probe(args.tag)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
