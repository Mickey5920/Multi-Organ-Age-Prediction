from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
FORMAL_BOOTSTRAP_REPLICATES = 200
FORMAL_BOOTSTRAP_SEED = 7_102_026
HASH_FOLD_SALT = "revision0710-external-v3"
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
ESTIMABILITY_STATUSES = (
    "estimable",
    "zero_target_events",
    "zero_competing_deaths",
    "single_outcome_class",
    "singular_design",
    "insufficient_complete_cases",
    "not_estimable_by_design",
)
CONTRACT_TOP_LEVEL_KEYS = [
    "schema_version",
    "status",
    "created_date",
    "purpose",
    "success_definition",
    "resource_budget",
    "reviewer_editor_alignment",
    "prohibited_actions",
    "bindings",
    "sealed_age_correction",
    "compact_cache_contract",
    "vocabulary_and_embedding_extension",
    "shared_prediction",
    "mimic_iv_3_1",
    "nhanes",
    "statistical_algorithms",
    "output_schema",
    "negative_tests_required",
    "formal_gate",
]
ARTIFACT_FILENAMES = {
    "source_registry": "SOURCE_REGISTRY_V8.json",
    "mimic_feature_map": "MIMIC_78_SLOT_MAP_V8.csv",
    "nhanes_feature_map": "NHANES_78_SLOT_RELEASE_MAP_V8.csv",
    "nhanes_label_map": "NHANES_QUESTIONNAIRE_LABEL_MAP_V8.csv",
    "mimic_icd10_map": "MIMIC_ICD10_ORGAN_MAP_V8.csv",
    "output_schemas": "OUTPUT_SCHEMAS_V8.json",
    "formal_authorization_schema": "FORMAL_AUTHORIZATION_SCHEMA_V8.json",
    "environment_lock": "ENVIRONMENT_LOCK_V8.json",
}
NHANES_MAP_COLUMNS = [
    "release_id",
    "slot_index",
    "feature_name",
    "map_status",
    "source_file_id",
    "source_variables",
    "source_unit",
    "target_unit",
    "formula",
    "null_semantics",
    "source_columns_verified",
    "all_78_slots_adjudicated",
    "adjudication_note",
]
MIMIC_MAP_COLUMNS = [
    "slot_index",
    "feature_name",
    "map_status",
    "primary_input_variant",
    "source_file_id",
    "primary_source_selector",
    "primary_source_unit",
    "primary_formula",
    "primary_value_rule",
    "sensitivity_input_variant",
    "sensitivity_source_selector",
    "sensitivity_source_unit",
    "sensitivity_formula",
    "sensitivity_value_rule",
    "target_unit",
    "lookback_days",
    "aggregation",
    "tie_break",
    "pre_index_rule",
    "selector_unit_formula_verified",
    "all_78_slots_adjudicated",
    "adjudication_note",
]
LABEL_MAP_COLUMNS = [
    "release_id",
    "organ_system",
    "estimability",
    "source_file_ids",
    "source_variables",
    "positive_rule",
    "negative_rule",
    "missing_rule",
    "formula",
    "label_role",
    "input_eligible",
    "source_columns_verified",
]
ICD10_MAP_COLUMNS = [
    "rule_order",
    "organ_system",
    "code_system",
    "match_type",
    "category_start",
    "category_end",
    "normalization",
    "prevalent_window",
    "incident_event_window",
    "multi_organ_match_allowed",
    "icd9_endpoint_authorized",
]
COLUMN_SCHEMA_KEYS = {
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
MIMIC_MAP_SEMANTIC_COLUMNS = MIMIC_MAP_COLUMNS
EXPECTED_MIMIC_MAP_SEMANTIC_SHA256 = "5e0185e1343ee38f08888123fa5dfa5398fd4a500a742a744e2941920208943d"
EXPECTED_OUTPUT_SCHEMA_SEMANTIC_SHA256 = "87bd23349d393748dca345c35fb70aff6e5000ed1fbd6d8bb790cb5f7972fb85"
MIMIC_EXPECTED_SOURCES = [
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
MIMIC_ALLOWED_RELATIVE_PATHS = {record[1] for record in MIMIC_EXPECTED_SOURCES}
NHANES_EXPECTED_SOURCES = [
    ("nhanes_p_demo", P_RELEASE, "Demographics/p_demo.xpt", "cohort_neutral"),
    ("nhanes_p_bmx", P_RELEASE, "Examination/p_bmx.xpt", "model_input"),
    ("nhanes_p_bpxo", P_RELEASE, "Examination/p_bpxo.xpt", "model_input"),
    ("nhanes_p_cbc", P_RELEASE, "Laboratory/p_cbc.xpt", "model_input"),
    ("nhanes_p_alb_cr", P_RELEASE, "Laboratory/p_alb_cr.xpt", "model_input"),
    ("nhanes_p_biopro", P_RELEASE, "Laboratory/p_biopro.xpt", "model_input"),
    ("nhanes_p_tchol", P_RELEASE, "Laboratory/p_tchol.xpt", "model_input"),
    ("nhanes_p_hdl", P_RELEASE, "Laboratory/p_hdl.xpt", "model_input"),
    ("nhanes_p_hscrp", P_RELEASE, "Laboratory/p_hscrp.xpt", "model_input"),
    ("nhanes_p_ghb", P_RELEASE, "Laboratory/p_ghb.xpt", "model_input"),
    ("nhanes_p_glu", P_RELEASE, "Laboratory/p_glu.xpt", "model_input"),
    ("nhanes_p_trigly", P_RELEASE, "Laboratory/p_trigly.xpt", "model_input"),
    ("nhanes_p_bpq", P_RELEASE, "Questionnaire/p_bpq.xpt", "label_only"),
    ("nhanes_p_mcq", P_RELEASE, "Questionnaire/p_mcq.xpt", "label_only"),
    ("nhanes_p_diq", P_RELEASE, "Questionnaire/p_diq.xpt", "label_only"),
    ("nhanes_p_kiq", P_RELEASE, "Questionnaire/p_kiq_u.xpt", "label_only"),
    ("nhanes_l_demo", L_RELEASE, "Demographics/DEMO_L.xpt", "cohort_neutral"),
    ("nhanes_l_bmx", L_RELEASE, "Examination/BMX_L.xpt", "model_input"),
    ("nhanes_l_bpxo", L_RELEASE, "Examination/BPXO_L.xpt", "model_input"),
    ("nhanes_l_cbc", L_RELEASE, "Laboratory/CBC_L.xpt", "model_input"),
    ("nhanes_l_alb_cr", L_RELEASE, "Laboratory/ALB_CR_L.xpt", "model_input"),
    ("nhanes_l_biopro", L_RELEASE, "Laboratory/BIOPRO_L.xpt", "model_input"),
    ("nhanes_l_tchol", L_RELEASE, "Laboratory/TCHOL_L.xpt", "model_input"),
    ("nhanes_l_hdl", L_RELEASE, "Laboratory/HDL_L.xpt", "model_input"),
    ("nhanes_l_hscrp", L_RELEASE, "Laboratory/HSCRP_L.xpt", "model_input"),
    ("nhanes_l_ghb", L_RELEASE, "Laboratory/GHB_L.xpt", "model_input"),
    ("nhanes_l_glu", L_RELEASE, "Laboratory/GLU_L.xpt", "model_input"),
    ("nhanes_l_trigly", L_RELEASE, "Laboratory/TRIGLY_L.xpt", "model_input"),
    ("nhanes_l_tst", L_RELEASE, "Laboratory/TST_L.xpt", "model_input"),
    ("nhanes_l_vid", L_RELEASE, "Laboratory/VID_L.xpt", "model_input"),
    ("nhanes_l_bpq", L_RELEASE, "Questionnaire/BPQ_L.xpt", "label_only"),
    ("nhanes_l_mcq", L_RELEASE, "Questionnaire/MCQ_L.xpt", "label_only"),
    ("nhanes_l_diq", L_RELEASE, "Questionnaire/DIQ_L.xpt", "label_only"),
    ("nhanes_l_kiq", L_RELEASE, "Questionnaire/KIQ_U_L.xpt", "label_only"),
]
OUTPUT_TABLE_PATHS = {
    "per_checkpoint_predictions": "shared/04_inference/per_checkpoint_predictions.parquet",
    "ensemble_predictions": "shared/04_inference/ensemble_predictions.parquet",
    "external_vocabulary_coverage": "shared/03_cache_audit/external_vocabulary_coverage.csv",
    "zero_shot_age_metrics": "shared/05_age_metrics/zero_shot_age_metrics.csv",
    "mimic_cohort_flow": "01_mimic/02_cohort_and_endpoints/cohort_flow.csv",
    "mimic_feature_coverage": "01_mimic/01_mapping/feature_coverage.csv",
    "mimic_incident_endpoints": "01_mimic/06_incident_risk/incident_endpoints.parquet",
    "mimic_recurrent_endpoints": "01_mimic/07_recurrent_sensitivity/recurrent_endpoints.parquet",
    "mimic_predictive_metrics": "01_mimic/06_incident_risk/predictive_metrics.csv",
    "mimic_full_cohort_associations": "01_mimic/06_incident_risk/full_cohort_associations.csv",
    "mimic_incident_competing_risk_curves": "01_mimic/06_incident_risk/competing_risk_curves.csv",
    "mimic_recurrent_sensitivity_metrics": "01_mimic/07_recurrent_sensitivity/recurrent_sensitivity_metrics.csv",
    "mimic_density_diagnostics": "01_mimic/08_density_sensitivity/density_diagnostics.csv",
    "mimic_density_adjusted_risk_sensitivity": "01_mimic/08_density_sensitivity/density_adjusted_risk_sensitivity.csv",
    "mimic_cox_ph_diagnostics": "01_mimic/09_cox_diagnostics/mimic_cox_ph_diagnostics.csv",
    "nhanes_feature_coverage": "02_nhanes/01_mapping/feature_coverage.csv",
    "nhanes_release_slot_qc": "02_nhanes/01_mapping/release_slot_qc.csv",
    "nhanes_questionnaire_associations": "02_nhanes/07_cross_sectional_disease/questionnaire_associations.csv",
    "nhanes_local_recalibration_sensitivity": "02_nhanes/06_local_recalibration_sensitivity/metrics.csv",
    "nhanes_release_survey_sensitivity": "02_nhanes/05_age_metrics/release_survey_sensitivity.csv",
}

V7_PREFREEZE_MANIFEST_SHA256 = "f29138079ab1a28eda07fa5fa32264f635c6503d8e8f28c452937e954ada164a"
V7_CONTRACT_SHA256 = "3ff8ef20701ce09fd9cfc39e4eb182fb473f439291d92b2999d7ce7e318f716b"
V7_RUNTIME_SHA256 = "73f4996b73be55916b9811dba1c50a65c6708866f62baf42e1be6a5b1f563244"
V7_SOURCE_REGISTRY_SHA256 = "9faec15419d1a34d15ee40767d02095167b59209b5170faf602f8f6329a3bd76"
V7_SCIENTIFIC_NO_GO_SHA256 = "4c85c203ea81f58bb2dbc36da324ba20dc3829d5847b167afa8200a9c8b44d07"
V8_ATTEMPT_01_SUPPORT_NO_GO_SHA256 = "94c2168ee4d0b7862767c2aee0d7acdd1bb29299ce31836617b18f80b0ce3e7f"
RESOURCE_RESERVATION_COMPONENTS = [
    "label-blind cache construction",
    "CLIP vocabulary extension encoding and model restore",
    "prediction sealing and disk IO",
    "NHANES survey and age-metric aggregation",
    "stage05 and stage06 four-model conditional B=200 bootstrap statistics",
    "complete shared-core PH diagnostics for all target/death model covariates",
    "publication copy, rehash, and receipt validation",
]


def expected_resource_budget() -> dict[str, Any]:
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
            "predictive_models_per_heldout_analysis": 4,
            "cause_specific_cox_fits_per_heldout_analysis": 8,
            "shared_core_ph_fits_per_heldout_analysis": 8,
            "conditional_bootstrap_calls_per_heldout_analysis": 4,
            "conditional_bootstrap_replicates_per_model": 200,
            "full_association_analysis_count": 42,
            "full_association_primary_fits_per_analysis": 1,
            "full_association_shared_core_ph_fits_per_analysis": 1,
            "formal_recompute_passes": 2,
            "total_cause_specific_predictive_fits": 3360,
            "total_heldout_shared_core_ph_fits": 3360,
            "total_conditional_b200_calls": 1680,
            "total_full_association_primary_fits": 84,
            "total_full_association_shared_core_ph_fits": 84,
            "v6_frozen_corrected_core_hours": 11.662225266512021,
            "v6_fresh_corrected_core_hours": 12.347672947476752,
        },
        "operational_reservation": {
            "minimum_hours": 36.0,
            "safer_hours": 48.0,
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


def _first_mismatch(observed: Any, expected: Any, path: str = "$") -> str | None:
    if type(observed) is not type(expected):
        return f"{path} type {type(observed).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        if missing:
            return f"{path} missing keys {missing}"
        if extra:
            return f"{path} extra keys {extra}"
        if list(observed) != list(expected):
            return f"{path} key order {list(observed)!r} != {list(expected)!r}"
        for key in expected:
            mismatch = _first_mismatch(observed[key], expected[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(observed) != len(expected):
            return f"{path} length {len(observed)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(observed, expected)):
            mismatch = _first_mismatch(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if observed != expected:
        return f"{path} value {observed!r} != {expected!r}"
    return None


def require_exact_payload(label: str, observed: Any, expected: Any) -> None:
    mismatch = _first_mismatch(observed, expected)
    if mismatch is not None:
        raise AssertionError(f"{label} closed exact mismatch: {mismatch}")


def normalize_unicode_text(value: object) -> str:
    if value is None:
        raise ValueError("text value cannot be None")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError("text value cannot be missing/NaN")
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def canonicalize_external_code(raw_code: object, code_system: str) -> str:
    canonical = normalize_unicode_text(raw_code).upper().replace(".", "")
    if not canonical or any(character.isspace() for character in canonical):
        raise ValueError("external code must be non-empty and contain no whitespace")
    if code_system == "ICD10":
        if not canonical[0].isalpha() or not canonical[1:].isalnum():
            raise ValueError("invalid ICD10 code")
        return canonical
    if code_system == "ICD9":
        if not canonical.isalnum():
            raise ValueError("invalid ICD9 code")
        return f"ICD9\x1f{canonical}"
    raise ValueError(code_system)


def external_text_key(role: str, description: object) -> str:
    if role not in {"icd_description", "inpatient_prescription_proxy"}:
        raise ValueError(role)
    clean = normalize_unicode_text(description)
    if not clean:
        raise ValueError("description cannot be empty")
    return f"{role}\x1f{clean}"


def deterministic_hash_fold(identifier: object) -> int:
    payload = f"{HASH_FOLD_SALT}|{normalize_unicode_text(identifier)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 5


def corrected_gap(raw_gap: Any, age: Any, intercept: float, slope: float) -> np.ndarray:
    return np.asarray(raw_gap, dtype=np.float64) - (
        float(intercept) + float(slope) * np.asarray(age, dtype=np.float64)
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


def validate_formal_target_absent(target: Path, authorization: Path) -> None:
    if target.exists() or authorization.exists():
        raise AssertionError("formal target/authorization must be absent")
    lock = authorization.parent / ".EXTERNAL_VALIDATION_FORMAL_V8_ONCE.lock"
    consumed = authorization.with_name(f"{authorization.stem}.CONSUMED.json")
    if lock.exists() or consumed.exists():
        raise AssertionError("formal lock/consumed authorization must be absent")
    if target.parent.exists() and list(target.parent.glob(f".{target.name}.staging.*")):
        raise AssertionError("formal staging target must be absent")


def independently_compute_mimic_omr_audit(omr_path: Path) -> dict[str, Any]:
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
        omr_path,
        compression="gzip",
        usecols=["result_name", "result_value"],
        chunksize=500_000,
    ):
        selected = chunk.loc[chunk["result_name"].isin(names)].copy()
        selected["numeric_value"] = pd.to_numeric(selected["result_value"], errors="coerce")
        for name, group in selected.groupby("result_name"):
            values = group["numeric_value"]
            counts[name]["rows"] += int(len(group))
            counts[name]["numeric"] += int(values.notna().sum())
            if name.startswith("Height"):
                counts[name]["inch_branch_40_90"] += int(values.between(40, 90, inclusive="both").sum())
                counts[name]["cm_branch_120_230"] += int(values.between(120, 230, inclusive="both").sum())
            else:
                counts[name]["weight_pounds_branch_50_1000"] += int(values.between(50, 1000, inclusive="both").sum())
    return {
        "status": "PREFREEZE_OMR_VALUE_BRANCH_AUDIT_PASS",
        "source_sha256": sha256_file(omr_path),
        "counts_exact_order": [{"result_name": name, **counts[name]} for name in names],
        "primary_height": "Height (Inches), 40..90 inches, multiply 2.54; all other explicit-name values rejected",
        "primary_weight": "Weight (Lbs), 50..1000 pounds, multiply 0.45359237; all other explicit-name values rejected",
        "bare_height_role": "named sensitivity only with non-overlapping 40..90 inch and 120..230 cm branches",
        "bare_weight_role": "named assume-pounds sensitivity only; never primary",
    }


def validate_registry(
    registry: dict[str, Any], repo_root: Path, *, rehash_external: bool
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "created_date",
        "restricted_raw_data_copied_to_workspace",
        "external_roots",
        "nhanes_release_policy",
        "external_sources",
        "mimic_omr_value_branch_audit",
        "nhanes_primary_age_sample_sanity",
        "locked_artifacts",
        "correction_key_count",
        "correction_csv_json_parameter_tolerance",
        "correction_csv_json_max_abs_difference",
        "fold_lineage_crosscheck",
        "checkpoint_count",
        "correction_file_count",
        "completion_receipt_count",
        "phenotype_transform_count",
    }
    if set(registry) != expected_top:
        raise AssertionError("source registry top-level closed schema mismatch")
    if registry["status"] != "PREFREEZE_READ_ONLY_SOURCE_REGISTRY_PASS":
        raise AssertionError("source registry is not PASS")
    if registry["restricted_raw_data_copied_to_workspace"] is not False:
        raise AssertionError("source registry reports copied restricted raw data")
    roots = registry["external_roots"]
    if set(roots) != {"mimic_iv_3_1", "mimic_official_sha256sums", "nhanes"}:
        raise AssertionError("external root schema mismatch")
    mimic_root = Path(roots["mimic_iv_3_1"]).resolve()
    checksum_record = roots["mimic_official_sha256sums"]
    if set(checksum_record) != {"absolute_path", "bytes", "sha256"}:
        raise AssertionError("MIMIC official checksum record schema mismatch")
    checksum_path = Path(checksum_record["absolute_path"])
    if checksum_path.resolve() != (mimic_root / "SHA256SUMS.txt").resolve():
        raise AssertionError("MIMIC checksum manifest path mismatch")
    if (
        not checksum_path.is_file()
        or checksum_path.stat().st_size != int(checksum_record["bytes"])
        or sha256_file(checksum_path) != checksum_record["sha256"]
    ):
        raise AssertionError("MIMIC checksum manifest byte/hash mismatch")
    official_checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise AssertionError(f"invalid MIMIC checksum line {line_number}")
        digest, relative_path = parts
        relative_path = relative_path.lstrip("*").replace("\\", "/")
        if relative_path in official_checksums:
            raise AssertionError(f"duplicate MIMIC checksum path: {relative_path}")
        official_checksums[relative_path] = digest.lower()
    policy = registry["nhanes_release_policy"]
    if policy["allowed_releases_exact_order"] != list(RELEASES):
        raise AssertionError("NHANES release order mismatch")
    if policy["DEMO_J_or_J_star_may_coexist_with_P_star"] is not False:
        raise AssertionError("NHANES J/P overlap is permitted")
    sources = registry["external_sources"]
    source_ids = [record["source_id"] for record in sources]
    if len(source_ids) != len(set(source_ids)):
        raise AssertionError("external source IDs duplicate")
    expected_source_ids = [record[0] for record in MIMIC_EXPECTED_SOURCES] + [
        record[0] for record in NHANES_EXPECTED_SOURCES
    ]
    if source_ids != expected_source_ids:
        raise AssertionError("external source set/order mismatch")
    source_paths = [record["absolute_path"] for record in sources]
    if len(source_paths) != len(set(source_paths)):
        raise AssertionError("external source paths duplicate")
    for record in sources:
        if set(record) != {
            "source_id",
            "cohort",
            "release_id",
            "role",
            "format",
            "absolute_path",
            "bytes",
            "sha256",
            "row_count",
            "columns",
            "columns_sha256",
            "restricted_raw_data_copied_to_workspace",
        }:
            raise AssertionError(f"external source record schema mismatch: {record.get('source_id')}")
        path = Path(record["absolute_path"])
        upper = path.name.upper()
        if upper == "DEMO_J.XPT" or upper.endswith("_J.XPT"):
            raise AssertionError("J release source present")
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"external source existence/byte mismatch: {path}")
        if rehash_external and sha256_file(path) != record["sha256"]:
            raise AssertionError(f"external source hash mismatch: {path}")
        column_hash = hashlib.sha256(
            json.dumps(
                record["columns"], sort_keys=True, ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest()
        if column_hash != record["columns_sha256"]:
            raise AssertionError(f"external source column hash mismatch: {path}")
        if record["restricted_raw_data_copied_to_workspace"] is not False:
            raise AssertionError("raw source copy flag mismatch")
    mimic = [record for record in sources if record["cohort"] == "MIMIC_IV_3_1"]
    mimic_relative_paths: set[str] = set()
    for record, expected in zip(mimic, MIMIC_EXPECTED_SOURCES):
        expected_id, expected_relative_path, expected_role = expected
        if (
            record["source_id"] != expected_id
            or record["release_id"] != "MIMIC_IV_3_1"
            or record["role"] != expected_role
            or record["format"] != "csv.gz"
        ):
            raise AssertionError("MIMIC source identity/release/role/format mismatch")
        path = Path(record["absolute_path"]).resolve()
        try:
            relative_path = path.relative_to(mimic_root).as_posix()
        except ValueError as exc:
            raise AssertionError("MIMIC source is outside the official release root") from exc
        if path.parent != (mimic_root / "hosp").resolve():
            raise AssertionError("nested or derived MIMIC source is forbidden")
        if not path.name.lower().endswith(".csv.gz"):
            raise AssertionError("MIMIC source is not an official csv.gz")
        if official_checksums.get(relative_path) != record["sha256"]:
            raise AssertionError("MIMIC source is not bound to the official SHA256SUMS.txt")
        if relative_path != expected_relative_path:
            raise AssertionError("MIMIC source relative path/order mismatch")
        mimic_relative_paths.add(relative_path)
    if mimic_relative_paths != MIMIC_ALLOWED_RELATIVE_PATHS:
        raise AssertionError("MIMIC official source set mismatch")
    omr_record = next(record for record in mimic if record["source_id"] == "mimic_omr")
    require_exact_payload(
        "MIMIC OMR value branch audit",
        registry["mimic_omr_value_branch_audit"],
        independently_compute_mimic_omr_audit(Path(omr_record["absolute_path"])),
    )
    nhanes = [record for record in sources if record["cohort"] == "NHANES"]
    nhanes_root = Path(roots["nhanes"]).resolve()
    for record, expected in zip(nhanes, NHANES_EXPECTED_SOURCES):
        expected_id, expected_release, expected_relative_path, expected_role = expected
        release_dir = "2019-2020" if expected_release == P_RELEASE else "2021-2023"
        release_root = (nhanes_root / release_dir).resolve()
        path = Path(record["absolute_path"]).resolve()
        try:
            relative_path = path.relative_to(release_root).as_posix()
        except ValueError as exc:
            raise AssertionError("NHANES source is outside its exact release directory") from exc
        if (
            record["source_id"] != expected_id
            or record["release_id"] != expected_release
            or record["role"] != expected_role
            or record["format"] != "xpt"
            or relative_path != expected_relative_path
        ):
            raise AssertionError("NHANES source identity/release/role/format/path mismatch")
    if sorted(set(record["release_id"] for record in nhanes)) != sorted(RELEASES):
        raise AssertionError("NHANES registry release set mismatch")
    label_paths = {record["source_id"] for record in nhanes if record["role"] == "label_only"}
    input_paths = {record["source_id"] for record in nhanes if record["role"] == "model_input"}
    if label_paths.intersection(input_paths):
        raise AssertionError("NHANES input/label source overlap")
    sanity = registry["nhanes_primary_age_sample_sanity"]
    expected_sanity = {
        "eligibility": "RIDSTATR == 2 and integer RIDAGEYR between 40 and 73 inclusive",
        "release_counts_exact_order": [
            {"release_id": P_RELEASE, "count": 4916},
            {"release_id": L_RELEASE, "count": 3720},
        ],
        "total": 8636,
        "expected_total": 8636,
        "matches_expected": True,
    }
    require_exact_payload("NHANES primary age sample sanity", sanity, expected_sanity)
    demo_by_release = {
        record["release_id"]: record
        for record in nhanes
        if record["source_id"] in {"nhanes_p_demo", "nhanes_l_demo"}
    }
    if list(demo_by_release) != list(RELEASES):
        raise AssertionError("NHANES DEMO release order/set mismatch")
    for expected in sanity["release_counts_exact_order"]:
        record = demo_by_release[expected["release_id"]]
        frame = pd.read_sas(record["absolute_path"], format="xport", encoding="latin1")
        age = pd.to_numeric(frame["RIDAGEYR"], errors="coerce")
        status = pd.to_numeric(frame["RIDSTATR"], errors="coerce")
        eligible = status.eq(2) & age.between(40, 73, inclusive="both") & age.eq(np.floor(age))
        if int(eligible.sum()) != expected["count"]:
            raise AssertionError("NHANES raw DEMO primary age sample count mismatch")

    locked = registry["locked_artifacts"]
    locked_ids = [record["artifact_id"] for record in locked]
    if len(locked_ids) != len(set(locked_ids)):
        raise AssertionError("locked artifact IDs duplicate")
    for record in locked:
        expected_keys = {"artifact_id", "role", "path_kind", "path", "bytes", "sha256"}
        if "outer_fold" in record:
            expected_keys.add("outer_fold")
        if set(record) != expected_keys:
            raise AssertionError(f"locked artifact record schema mismatch: {record.get('artifact_id')}")
        path = (
            repo_root / record["path"]
            if record["path_kind"] == "repo_relative"
            else Path(record["path"])
        )
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"locked artifact existence/byte mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"locked artifact hash mismatch: {path}")
    for prefix, expected_count in (
        ("checkpoint_fold_", 5),
        ("correction_fold_", 5),
        ("completion_receipt_fold_", 5),
        ("phenotype_transform_fold_", 5),
    ):
        records = [record for record in locked if record["artifact_id"].startswith(prefix)]
        if len(records) != expected_count or sorted(record["outer_fold"] for record in records) != list(range(5)):
            raise AssertionError(f"locked fold artifact set mismatch: {prefix}")
    if registry["correction_key_count"] != 35:
        raise AssertionError("correction key count is not 35")
    if registry["correction_csv_json_parameter_tolerance"] != 5e-15:
        raise AssertionError("correction CSV/JSON tolerance mismatch")
    if not isinstance(registry["correction_csv_json_max_abs_difference"], float):
        raise AssertionError("correction CSV/JSON maximum difference type mismatch")
    if registry["correction_csv_json_max_abs_difference"] > 5e-15:
        raise AssertionError("correction CSV/JSON maximum difference exceeds tolerance")
    if registry["fold_lineage_crosscheck"] is not True:
        raise AssertionError("fold lineage crosscheck is false")

    locked_by_id = {record["artifact_id"]: record for record in locked}

    def locked_path(artifact_id: str) -> Path:
        record = locked_by_id[artifact_id]
        return (
            repo_root / record["path"]
            if record["path_kind"] == "repo_relative"
            else Path(record["path"])
        )

    correction_values: dict[tuple[int, str], dict[str, Any]] = {}
    expected_checkpoint_hashes: dict[int, str] = {}
    for fold in range(5):
        expected_checkpoint_hashes[fold] = locked_by_id[f"checkpoint_fold_{fold}"]["sha256"]
        correction = read_json(locked_path(f"correction_fold_{fold}"))
        receipt = read_json(locked_path(f"completion_receipt_fold_{fold}"))
        if (
            correction.get("status") != "FITTED_AND_SEALED_BEFORE_OUTER_READ"
            or int(correction.get("outer_fold", -1)) != fold
            or int(correction.get("calibration_fold", -1)) == fold
            or correction.get("fixed_e55_checkpoint_sha256") != expected_checkpoint_hashes[fold]
            or correction.get("replication_contract_sha256") != locked_by_id["replication_contract"]["sha256"]
        ):
            raise AssertionError(f"sealed correction lineage mismatch: fold {fold}")
        if (
            receipt.get("status") != "COMPLETE_FIXED_E55_OUTER_SCORE"
            or int(receipt.get("outer_fold", -1)) != fold
            or receipt.get("fixed_e55_checkpoint_sha256") != expected_checkpoint_hashes[fold]
            or receipt.get("replication_contract_sha256") != locked_by_id["replication_contract"]["sha256"]
            or receipt.get("age_correction_sha256") != locked_by_id[f"correction_fold_{fold}"]["sha256"]
        ):
            raise AssertionError(f"completion receipt lineage mismatch: fold {fold}")
        parameters = correction.get("parameters")
        if not isinstance(parameters, list):
            raise AssertionError(f"correction parameters missing: fold {fold}")
        for parameter in parameters:
            key = (fold, str(parameter["organ_system"]))
            if key in correction_values:
                raise AssertionError("duplicate correction key")
            correction_values[key] = parameter
    expected_correction_keys = [(fold, organ) for fold in range(5) for organ in ORGANS]
    if list(correction_values) != expected_correction_keys:
        raise AssertionError("sealed correction 5x7 key order mismatch")
    published = pd.read_csv(locked_path("correction_parameters"))
    published_keys = list(zip(published["outer_fold"].astype(int), published["organ_system"].astype(str)))
    if len(published) != 35 or published_keys != expected_correction_keys:
        raise AssertionError("published correction CSV 5x7 key/order mismatch")
    differences: list[float] = []
    for row in published.itertuples(index=False):
        parameter = correction_values[(int(row.outer_fold), str(row.organ_system))]
        differences.extend(
            [
                abs(float(row.intercept) - float(parameter["intercept"])),
                abs(float(row.age_slope) - float(parameter["slope"])),
            ]
        )
        if int(row.n_reference_rows) != int(parameter["fit_population_count"]):
            raise AssertionError("published correction reference count mismatch")
    observed_max_difference = max(differences, default=0.0)
    if observed_max_difference != registry["correction_csv_json_max_abs_difference"]:
        raise AssertionError("reported correction CSV/JSON maximum difference mismatch")

    transform_manifest = json.loads(locked_path("fold_transform_manifest").read_text(encoding="utf-8"))
    if not isinstance(transform_manifest, list) or len(transform_manifest) != 5:
        raise AssertionError("fold transform manifest cardinality mismatch")
    for fold, record in enumerate(transform_manifest):
        normalized_path = Path(str(record.get("path", "")).replace("\\", "/")).as_posix()
        if (
            int(record.get("outer_fold", -1)) != fold
            or int(record.get("validation_fold", -1)) != (fold + 1) % 5
            or record.get("sha256") != locked_by_id[f"phenotype_transform_fold_{fold}"]["sha256"]
            or normalized_path != f"fold_transforms/outer_fold_{fold}.npz"
        ):
            raise AssertionError(f"fold transform manifest lineage mismatch: fold {fold}")
    return {
        "external_sources": len(sources),
        "locked_artifacts": len(locked),
        "nhanes_input_sources": len(input_paths),
        "nhanes_label_sources": len(label_paths),
    }


def _bool_series_true(series: pd.Series) -> bool:
    return series.astype(str).str.lower().eq("true").all()


def validate_maps(artifact_dir: Path, registry: dict[str, Any]) -> dict[str, Any]:
    sources = {record["source_id"]: record for record in registry["external_sources"]}
    nhanes = pd.read_csv(
        artifact_dir / ARTIFACT_FILENAMES["nhanes_feature_map"], keep_default_na=False
    )
    if list(nhanes.columns) != NHANES_MAP_COLUMNS or len(nhanes) != 156:
        raise AssertionError("NHANES feature map schema/cardinality mismatch")
    for release in RELEASES:
        subset = nhanes.loc[nhanes["release_id"].eq(release)]
        if subset["slot_index"].astype(int).tolist() != list(range(78)):
            raise AssertionError(f"NHANES map slot order mismatch: {release}")
    if not _bool_series_true(nhanes["all_78_slots_adjudicated"]):
        raise AssertionError("NHANES all78 gate is false")
    if not _bool_series_true(nhanes["source_columns_verified"]):
        raise AssertionError("NHANES source column verification flag is false")
    mapped = nhanes.loc[nhanes["source_file_id"].ne("")]
    for row in mapped.itertuples(index=False):
        source = sources[row.source_file_id]
        if source["role"] != "model_input":
            raise AssertionError("NHANES feature map references a non-input source")
        variables = set(str(row.source_variables).split("|"))
        if not variables.issubset(set(source["columns"])):
            raise AssertionError("NHANES feature map variable missing from source")

    mimic = pd.read_csv(
        artifact_dir / ARTIFACT_FILENAMES["mimic_feature_map"], keep_default_na=False
    )
    if list(mimic.columns) != MIMIC_MAP_COLUMNS or len(mimic) != 78:
        raise AssertionError("MIMIC feature map schema/cardinality mismatch")
    if mimic["slot_index"].astype(int).tolist() != list(range(78)):
        raise AssertionError("MIMIC feature map slot order mismatch")
    if not _bool_series_true(mimic["all_78_slots_adjudicated"]):
        raise AssertionError("MIMIC all78 gate is false")
    if not _bool_series_true(mimic["selector_unit_formula_verified"]):
        raise AssertionError("MIMIC selector/unit/formula verification flag is false")
    if not mimic["primary_input_variant"].eq(MIMIC_PRIMARY_VARIANT).all():
        raise AssertionError("MIMIC primary input variant mismatch")
    if not mimic["sensitivity_input_variant"].eq(MIMIC_BARE_OMR_SENSITIVITY).all():
        raise AssertionError("MIMIC map sensitivity variant mismatch")
    if not set(mimic["map_status"]).issubset({"stable", "weak", "not_measured"}):
        raise AssertionError("MIMIC map status outside closed enum")
    availability_rules = {
        "": "not_applicable_unmeasured_slot",
        "mimic_labevents": "charttime <= discharge landmark AND storetime <= discharge landmark; both nonmissing",
        "mimic_omr": "chartdate < discharge calendar date AND chartdate >= discharge calendar date minus 365 days",
    }
    for source_id, expected_rule in availability_rules.items():
        if not mimic.loc[mimic["source_file_id"].eq(source_id), "pre_index_rule"].eq(
            expected_rule
        ).all():
            raise AssertionError(f"MIMIC data-availability rule mismatch: {source_id}")
    semantic_payload = "\n".join(
        "\x1f".join(str(row[column]) for column in MIMIC_MAP_SEMANTIC_COLUMNS)
        for _, row in mimic.iterrows()
    ).encode("utf-8")
    semantic_sha256 = hashlib.sha256(semantic_payload).hexdigest()
    if semantic_sha256 != EXPECTED_MIMIC_MAP_SEMANTIC_SHA256:
        raise AssertionError("MIMIC per-slot selector/unit/formula semantic hash mismatch")
    by_slot = mimic.set_index("slot_index")
    if by_slot.loc[3, "primary_source_selector"] != "result_name == 'Height (Inches)'":
        raise AssertionError("MIMIC height primary selector is not explicit-unit only")
    if by_slot.loc[3, "primary_formula"] != "value * 2.54":
        raise AssertionError("MIMIC height primary conversion mismatch")
    if by_slot.loc[11, "primary_source_selector"] != "result_name == 'Weight (Lbs)'":
        raise AssertionError("MIMIC weight primary selector is not explicit-unit only")
    if by_slot.loc[11, "primary_formula"] != "value * 0.45359237":
        raise AssertionError("MIMIC weight primary conversion mismatch")
    if "52769:value/1000" not in by_slot.loc[24, "primary_formula"]:
        raise AssertionError("MIMIC absolute lymphocyte #/uL conversion mismatch")
    if by_slot.loc[37, "primary_formula"] != "value; million/uL numerically equals target 10^12/L":
        raise AssertionError("MIMIC absolute reticulocyte conversion mismatch")

    labels = pd.read_csv(
        artifact_dir / ARTIFACT_FILENAMES["nhanes_label_map"], keep_default_na=False
    )
    if list(labels.columns) != LABEL_MAP_COLUMNS or len(labels) != 14:
        raise AssertionError("NHANES label map schema/cardinality mismatch")
    if int(labels["estimability"].eq("estimable").sum()) != 12:
        raise AssertionError("NHANES must have six estimable labels per release")
    immune = labels.loc[labels["organ_system"].eq("immune")]
    if len(immune) != 2 or not immune["estimability"].eq("not_estimable").all():
        raise AssertionError("NHANES immune label must be fixed not_estimable")
    if not labels["label_role"].eq("outcome_only").all():
        raise AssertionError("NHANES label role mismatch")
    if not labels["input_eligible"].astype(str).str.lower().eq("false").all():
        raise AssertionError("NHANES label is input-eligible")
    for row in labels.loc[labels["estimability"].eq("estimable")].itertuples(index=False):
        for source_id in row.source_file_ids.split("|"):
            if sources[source_id]["role"] != "label_only":
                raise AssertionError("NHANES label map references non-label source")

    icd10 = pd.read_csv(
        artifact_dir / ARTIFACT_FILENAMES["mimic_icd10_map"], keep_default_na=False
    )
    if list(icd10.columns) != ICD10_MAP_COLUMNS:
        raise AssertionError("MIMIC ICD10 map schema mismatch")
    if icd10["rule_order"].astype(int).tolist() != list(range(len(icd10))):
        raise AssertionError("MIMIC ICD10 rule order mismatch")
    if not icd10["code_system"].eq("ICD10").all():
        raise AssertionError("MIMIC endpoint map includes a non-ICD10 system")
    if not icd10["icd9_endpoint_authorized"].astype(str).str.lower().eq("false").all():
        raise AssertionError("MIMIC ICD9 endpoint is authorized")
    if set(icd10["organ_system"]) != set(ORGANS):
        raise AssertionError("MIMIC ICD10 organ set mismatch")
    return {
        "nhanes_feature_rows": len(nhanes),
        "nhanes_mapped_rows": len(mapped),
        "mimic_feature_rows": len(mimic),
        "mimic_mapped_rows": int(mimic["source_file_id"].ne("").sum()),
        "nhanes_estimable_label_rows": 12,
        "mimic_icd10_rules": len(icd10),
    }


def validate_output_schemas(payload: dict[str, Any]) -> dict[str, Any]:
    semantic = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    if hashlib.sha256(semantic).hexdigest() != EXPECTED_OUTPUT_SCHEMA_SEMANTIC_SHA256:
        raise AssertionError("output schema semantic hash mismatch")
    if set(payload) != {"schema_version", "status", "table_count", "column_schema_keys_exact_order", "tables"}:
        raise AssertionError("output schema top-level keys mismatch")
    if payload["status"] != "PREFREEZE_EXACT_OUTPUT_SCHEMAS_PASS":
        raise AssertionError("output schemas are not PASS")
    tables = payload["tables"]
    if payload["table_count"] != len(tables):
        raise AssertionError("output schema table count mismatch")
    table_ids = [table["table_id"] for table in tables]
    if len(table_ids) != len(set(table_ids)):
        raise AssertionError("output table IDs duplicate")
    if table_ids != list(OUTPUT_TABLE_PATHS):
        raise AssertionError("output table set/order mismatch")
    required_table_keys = {
        "table_id",
        "relative_path",
        "grain",
        "key",
        "cardinality",
        "sort_order",
        "columns",
    }
    for table in tables:
        if set(table) != required_table_keys:
            raise AssertionError(f"output table closed schema mismatch: {table.get('table_id')}")
        names = [column["name"] for column in table["columns"]]
        if len(names) != len(set(names)):
            raise AssertionError(f"output columns duplicate: {table['table_id']}")
        if [column["name"] for column in table["columns"] if column["is_key"]] != table["key"]:
            raise AssertionError(f"output key flags mismatch: {table['table_id']}")
        if table["sort_order"] != table["key"]:
            raise AssertionError(f"output sort order mismatch: {table['table_id']}")
        if table["relative_path"] != OUTPUT_TABLE_PATHS[table["table_id"]]:
            raise AssertionError(f"output table path mismatch: {table['table_id']}")
        for column in table["columns"]:
            if set(column) != COLUMN_SCHEMA_KEYS:
                raise AssertionError(f"output column closed schema mismatch: {table['table_id']}")
            if column["grain"] != table["grain"] or column["cardinality"] != table["cardinality"]:
                raise AssertionError(f"output column grain/cardinality mismatch: {table['table_id']}")
            if not column["dtype"] or not column["unit"] or not column["formula"] or not column["tolerance"]:
                raise AssertionError(f"output column field empty: {table['table_id']}/{column['name']}")
    paths = {table["table_id"]: table["relative_path"] for table in tables}
    if paths["mimic_incident_endpoints"] == paths["mimic_recurrent_endpoints"]:
        raise AssertionError("incident and recurrent endpoints share a physical table")
    by_id = {table["table_id"]: table for table in tables}
    for table_id in (
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
        "nhanes_questionnaire_associations",
        "nhanes_local_recalibration_sensitivity",
        "nhanes_release_survey_sensitivity",
    ):
        if "input_variant" not in by_id[table_id]["key"]:
            raise AssertionError(f"input_variant is absent from key: {table_id}")
    vocab_columns = {column["name"]: column for column in by_id["external_vocabulary_coverage"]["columns"]}
    if vocab_columns["vocabulary_kind"]["enum"] != ["code", "text"]:
        raise AssertionError("vocabulary_kind code/text enum mismatch")
    for table_id in (
        "mimic_incident_endpoints",
        "mimic_recurrent_endpoints",
        "mimic_predictive_metrics",
        "mimic_full_cohort_associations",
        "mimic_incident_competing_risk_curves",
        "mimic_recurrent_sensitivity_metrics",
        "mimic_density_adjusted_risk_sensitivity",
        "mimic_cox_ph_diagnostics",
    ):
        horizon = next(
            column for column in by_id[table_id]["columns"] if column["name"] == "horizon_months"
        )
        if horizon["enum"] != [6, 12]:
            raise AssertionError(f"unauthorized horizon in output schema: {table_id}")
    local = {column["name"]: column for column in by_id["nhanes_local_recalibration_sensitivity"]["columns"]}
    if local["regression_formula"]["enum"] != [
        "chronological_age = beta0 + beta1*predicted_age_raw_mean5 + beta2*sex_male"
    ]:
        raise AssertionError("NHANES local calibration formula mismatch")
    if local["coefficient_application_order"]["enum"] != [
        "intercept,predicted_age_raw_mean5,sex_male"
    ]:
        raise AssertionError("NHANES local coefficient order mismatch")
    zero = {
        column["name"]: column for column in by_id["zero_shot_age_metrics"]["columns"]
    }
    if (
        zero["uncertainty_scope"]["enum"]
        != ["point_estimate_only_no_resampling"]
        or zero["bootstrap_replicates"]["enum"] != [0]
    ):
        raise AssertionError("zero-shot age uncertainty scope mismatch")
    predictive = {
        column["name"]: column for column in by_id["mimic_predictive_metrics"]["columns"]
    }
    if predictive["bootstrap_refits_target_cox_death_cox_and_censor_km"][
        "enum"
    ] != [False] or predictive["bootstrap_replicates"]["enum"] != [200]:
        raise AssertionError("predictive conditional bootstrap schema mismatch")
    if predictive["uncertainty_scope"]["enum"] != [
        "conditional_on_fixed_training_fold_target_death_cox_and_censor_km",
        "point_estimate_only_bootstrap_insufficient",
    ]:
        raise AssertionError("predictive uncertainty scope mismatch")
    if predictive["metric"]["enum"][-2:] != [
        "ipcw_linear_calibration_intercept",
        "ipcw_linear_calibration_slope",
    ]:
        raise AssertionError("linear probability calibration is not explicitly named")
    association = {
        column["name"]: column
        for column in by_id["mimic_full_cohort_associations"]["columns"]
    }
    if association["analysis_population"]["enum"] != [
        "full_nonheldout_incident_eligible_organ_specific_cohort"
    ] or "n" not in association:
        raise AssertionError("MIMIC association population/n schema mismatch")
    curves = {
        column["name"]: column
        for column in by_id["mimic_incident_competing_risk_curves"]["columns"]
    }
    if curves["bootstrap_replicates"]["enum"] != [0] or curves[
        "uncertainty_scope"
    ]["enum"] != ["point_curve_only_no_simultaneous_band"]:
        raise AssertionError("MIMIC curve point-only scope mismatch")
    density_risk = {
        column["name"]: column
        for column in by_id["mimic_density_adjusted_risk_sensitivity"]["columns"]
    }
    if (
        density_risk["model"]["enum"]
        != ["age_sex_density", "age_sex_density_corrected_gap"]
        or density_risk["clinical_utility_claim_allowed"]["enum"] != [False]
    ):
        raise AssertionError("density-adjusted risk sensitivity schema mismatch")
    ph = {
        column["name"]: column
        for column in by_id["mimic_cox_ph_diagnostics"]["columns"]
    }
    if (
        ph["survival_core_sha256"]["enum"]
        != ["6c4dd7eef4ea690d0d88a330f117d8d0b45274d788e10862ab665e245e435a1b"]
        or "GLOBAL" not in ph["covariate"]["enum"]
        or ph["transform"]["enum"] != ["rank_lifelines_cumsum_event", "log1p_days"]
    ):
        raise AssertionError("complete PH diagnostic schema mismatch")
    nhanes_qc = {
        column["name"]: column
        for column in by_id["nhanes_release_slot_qc"]["columns"]
    }
    if (
        nhanes_qc["trimming_or_winsorization"]["enum"] != ["prohibited"]
        or nhanes_qc["quantiles_used_for_decision"]["enum"] != [False]
    ):
        raise AssertionError("NHANES fixed release-slot QC policy mismatch")
    for table in tables:
        for column in table["columns"]:
            if column["name"] in {
                "estimate",
                "ci_lower",
                "ci_upper",
                "hazard_ratio",
                "p_value",
                "bh_q_value",
                "ph_test_p",
            } and column["dtype"] == "float64" and not column["nullable"]:
                raise AssertionError(
                    f"degenerate estimate is not nullable: {table['table_id']}/{column['name']}"
                )
    return {"tables": len(tables), "columns": sum(len(table["columns"]) for table in tables)}


def validate_authorization_schema(
    payload: dict[str, Any], repo_root: Path | None = None
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "duplicate_json_keys",
        "additional_properties",
        "required_top_level_keys_exact_order",
        "exact_constants",
        "record_schemas",
        "rejection_policy",
    }
    if set(payload) != expected_keys:
        raise AssertionError("formal authorization schema top-level mismatch")
    if payload["status"] != "PREFREEZE_SCHEMA_ONLY_NO_AUTHORIZATION_GRANTED":
        raise AssertionError("formal authorization schema status mismatch")
    if payload["additional_properties"] is not False or payload["duplicate_json_keys"] != "reject":
        raise AssertionError("formal authorization schema is not closed")
    if payload["required_top_level_keys_exact_order"] != [
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
    ]:
        raise AssertionError("formal authorization top-level key order mismatch")
    constants = payload["exact_constants"]
    expected_constants = {
        "schema_version": "revision0710.external_validation.formal_authorization.v8",
        "status": "APPROVED_ONCE_FOR_EXTERNAL_VALIDATION_SOURCE_DATA_V1",
        "scope": "one create-once MIMIC-IV and NHANES formal external-validation source bundle",
        "formal_tag": "source_data_v1",
        "formal_target_relative_path": "Revision-0710/06_figures_and_statistics/04_external_validation/source_data_v1",
        "formal_target_resolved": (
            str(
                (
                    repo_root
                    / "Revision-0710/06_figures_and_statistics/04_external_validation/source_data_v1"
                ).resolve()
            )
            if repo_root is not None
            else constants.get("formal_target_resolved")
        ),
        "target_must_be_absent_before_authorization": True,
        "fine_tuning": False,
        "distillation": False,
        "formal_bootstrap_replicates": FORMAL_BOOTSTRAP_REPLICATES,
        "formal_bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
        "formal_mode": "FORMAL_ONCE_CREATE_ONLY",
        "formal_executor_name": "formal_executor",
    }
    require_exact_payload("formal authorization exact constants", constants, expected_constants)
    expected_record_schemas = {
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
    }
    require_exact_payload(
        "formal authorization record schemas",
        payload["record_schemas"],
        expected_record_schemas,
    )
    expected_rejection_policy = [
        "reject extra, missing, duplicate, or reordered keys and records",
        "reject any path, byte count, SHA-256, environment, tag, target, seed, or replicate mismatch",
        "reject if formal target or a sibling staging target already exists",
        "reject if source registry, maps, output schemas, builder, executor, verifier, or tests changed after authorization",
        "reject mode other than FORMAL_ONCE_CREATE_ONLY or authorization reused after any target/staging creation",
        "reject authorization without independent_review_status=PASS and contract_bytes_frozen=true",
        "reject absent, self-authored, stale, or hash-mismatched independent GO review receipt",
        "reject any authorization that permits fine-tuning, distillation, external age fitting in the zero-shot path, or ICD9 endpoints",
    ]
    if payload["rejection_policy"] != expected_rejection_policy:
        raise AssertionError("formal authorization rejection policy mismatch")


def expected_formal_authorization(
    *, artifact_dir: Path, repo_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    contract_path = artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
    code = contract["bindings"]["code_artifacts"]
    support = []
    for record in contract["bindings"]["prefreeze_artifacts"]:
        path = artifact_dir / record["path"]
        support.append(
            {
                "name": record["name"],
                "relative_path": path.resolve().relative_to(repo_root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    source_path = artifact_dir / "SOURCE_REGISTRY_V8.json"
    formal_target = repo_root / contract["formal_gate"]["formal_target_relative_path"]
    environment_lock_path = artifact_dir / "ENVIRONMENT_LOCK_V8.json"
    environment_lock = read_json(environment_lock_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)
    code_bundle_sha256 = hashlib.sha256(
        json.dumps(
            code,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    review_path = artifact_dir.parent / "INDEPENDENT_GO_REVIEW_V8.json"
    if not review_path.is_file():
        raise PermissionError("independent V8 GO review receipt is absent")
    review = read_json(review_path)
    expected_review = {
        "schema_version": "revision0710.external_validation.independent_go_review.v8",
        "status": "INDEPENDENT_REVIEW_GO",
        "review_provenance": "independent_agent_terminal_review",
        "reviewed_contract_sha256": sha256_file(contract_path),
        "reviewed_code_bundle_sha256": code_bundle_sha256,
        "p0_count": 0,
        "p1_count": 0,
    }
    require_exact_payload("independent GO review receipt", review, expected_review)
    return {
        "schema_version": "revision0710.external_validation.formal_authorization.v8",
        "status": "APPROVED_ONCE_FOR_EXTERNAL_VALIDATION_SOURCE_DATA_V1",
        "scope": "one create-once MIMIC-IV and NHANES formal external-validation source bundle",
        "contract": {
            "relative_path": contract_path.resolve().relative_to(repo_root.resolve()).as_posix(),
            "bytes": contract_path.stat().st_size,
            "sha256": sha256_file(contract_path),
        },
        "code_artifacts": code,
        "prefreeze_artifacts": support,
        "locked_source_registry": {
            "relative_path": source_path.resolve().relative_to(repo_root.resolve()).as_posix(),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        },
        "environments": [
            {
                "name": "inference",
                "environment_lock_sha256": environment_lock_sha256,
                "python_executable": environment_lock["inference"]["python_executable"],
                "python_executable_sha256": environment_lock["inference"]["python_executable_sha256"],
            },
            {
                "name": "statistics",
                "environment_lock_sha256": environment_lock_sha256,
                "python_executable": environment_lock["statistics"]["python_executable"],
                "python_executable_sha256": environment_lock["statistics"]["python_executable_sha256"],
            },
        ],
        "independent_review_receipt": {
            "relative_path": review_path.resolve().relative_to(repo_root.resolve()).as_posix(),
            "bytes": review_path.stat().st_size,
            "sha256": sha256_file(review_path),
            "status": review["status"],
            "reviewed_contract_sha256": review["reviewed_contract_sha256"],
            "reviewed_code_bundle_sha256": review["reviewed_code_bundle_sha256"],
        },
        "review_freeze": {
            "independent_review_status": "PASS",
            "contract_bytes_frozen": True,
            "all_authorized_hashes_exact": True,
        },
        "formal_publication": {
            "mode": "FORMAL_ONCE_CREATE_ONLY",
            "executor_name": "formal_executor",
            "tag": "source_data_v1",
            "target_relative_path": formal_target.resolve().relative_to(repo_root.resolve()).as_posix(),
            "target_resolved": str(formal_target.resolve()),
            "create_once": True,
            "single_atomic_bundle": True,
            "fine_tuning": False,
            "distillation": False,
            "bootstrap_replicates": FORMAL_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
            "authorization_single_use": True,
        },
    }


def validate_formal_authorization_payload(
    payload: dict[str, Any], *, artifact_dir: Path, repo_root: Path, contract: dict[str, Any]
) -> None:
    require_exact_payload(
        "formal authorization",
        payload,
        expected_formal_authorization(
            artifact_dir=artifact_dir, repo_root=repo_root, contract=contract
        ),
    )


def validate_contract_semantics(
    contract: dict[str, Any],
    artifact_dir: Path,
    repo_root: Path,
    *,
    formal_runtime_state: bool = False,
) -> None:
    if list(contract) != CONTRACT_TOP_LEVEL_KEYS:
        raise AssertionError("contract top-level exact order/keys mismatch")
    if contract["status"] != "DRAFT_PREFREEZE_V8_NOT_FROZEN_NOT_AUTHORIZED":
        raise AssertionError("contract is unexpectedly frozen or authorized")
    if contract["success_definition"]["minimum_external_performance_threshold"] is not None:
        raise AssertionError("contract includes an external performance threshold")
    require_exact_payload(
        "V8 resource budget",
        contract["resource_budget"],
        expected_resource_budget(),
    )
    if contract["sealed_age_correction"]["external_fit_performed"] is not False:
        raise AssertionError("contract permits external correction fit")
    if contract["sealed_age_correction"]["cardinality"] != "5 * 7 = 35 exact unique keys":
        raise AssertionError("contract correction key cardinality mismatch")
    if contract["shared_prediction"].get("checkpoint_age_anchor") != {
        "fold0": {
            "key": "age_anchor_train_folds_2_to_4_only",
            "value": 57.03986740112305,
        },
        "fold1": {
            "key": "age_anchor_train_role_only",
            "value": 57.03695297241211,
        },
        "fold2": {
            "key": "age_anchor_train_role_only",
            "value": 57.03803253173828,
        },
        "fold3": {
            "key": "age_anchor_train_role_only",
            "value": 57.04105758666992,
        },
        "fold4": {
            "key": "age_anchor_train_role_only",
            "value": 57.04116439819336,
        },
    }:
        raise AssertionError("fixed checkpoint age-anchor mapping mismatch")
    if contract["compact_cache_contract"]["sequence_length"] != 32:
        raise AssertionError("contract sequence length mismatch")
    selection = contract["compact_cache_contract"]["selection"]
    if selection["primary"] != {
        "input_variant": MIMIC_PRIMARY_VARIANT,
        "allowed_sources": ["hospital_icd"],
        "crop": "latest 32 after exact stable sort and dedup",
    }:
        raise AssertionError("contract primary ICD-only L32 mismatch")
    if selection["bare_omr_sensitivity"] != {
        "input_variant": MIMIC_BARE_OMR_SENSITIVITY,
        "allowed_sources": ["hospital_icd"],
        "crop": "latest 32 after exact stable sort and dedup",
    }:
        raise AssertionError("contract bare-OMR sensitivity L32 mismatch")
    prescription = selection["inpatient_prescription_proxy_sensitivity"]
    if (
        prescription["minimum_latest_hospital_icd"] != 8
        or prescription["minimum_latest_inpatient_prescription_proxy"] != 8
        or prescription["remaining_global_latest"] != 16
        or prescription["allowed_sources"]
        != ["hospital_icd", "inpatient_prescription_proxy"]
    ):
        raise AssertionError("contract inpatient-prescription sensitivity 8+8+16 mismatch")
    alignment = contract["reviewer_editor_alignment"]
    if list(alignment) != [
        "Reviewer1_Major6",
        "Reviewer2_Comment2",
        "Editor_original_supported_scope",
        "post_V7_scientific_review",
    ]:
        raise AssertionError("reviewer/editor traceability labels mismatch")
    if "calibration" not in alignment["Reviewer1_Major6"] or "domain shift" not in alignment["Reviewer1_Major6"] or "disease ascertainment" not in alignment["Reviewer1_Major6"]:
        raise AssertionError("Reviewer 1 Major 6 content mismatch")
    if "sequence alignment" not in alignment["Reviewer2_Comment2"] or "mappings" not in alignment["Reviewer2_Comment2"]:
        raise AssertionError("Reviewer 2 Comment 2 content mismatch")
    extension = contract["vocabulary_and_embedding_extension"]
    exact_extension = {
        "open_clip_version": "3.3.0",
        "torch_version": "2.5.1",
        "device": "cpu",
        "encode_dtype": "float32",
        "stored_extension_dtype": "little-endian float16",
        "encode_batch_size": 4,
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "deterministic_algorithms": True,
    }
    for key, expected in exact_extension.items():
        if extension[key] != expected:
            raise AssertionError(f"contract extension mismatch: {key}")
    if extension["required_coverage_output_table"] != OUTPUT_TABLE_PATHS["external_vocabulary_coverage"]:
        raise AssertionError("contract external vocabulary coverage output mismatch")
    if extension.get("coverage_token_classes_exact") != [
        "exact_frozen_vocab",
        "deterministic_extension",
        "invalid_dropped",
        "zero_event",
    ] or extension.get("coverage_closed_cardinality") != 60:
        raise AssertionError("contract external vocabulary token classes/cardinality mismatch")
    if extension.get("coverage_code_systems_separate") != [
        "ICD9",
        "ICD10",
        "INPATIENT_RX",
        "not_applicable",
    ]:
        raise AssertionError("contract external vocabulary code-system separation mismatch")
    mimic = contract["mimic_iv_3_1"]
    if mimic["horizons"] != {
        "authorized_months_exact": [6, 12],
        "authorized_days_exact": [182.5, 365.0],
    }:
        raise AssertionError("MIMIC horizon roles mismatch")
    if mimic["chronological_age"] != "integer anchor_age + (index discharge calendar year - integer anchor_year); reject anchor_age=91":
        raise AssertionError("MIMIC integer chronological age mismatch")
    if mimic.get("admission_selection_order") != (
        "filter finite ordered times, compute integer age and require age>=18, "
        "require deathtime absent or after discharge, then stable sort and take ordinal 2"
    ):
        raise AssertionError("MIMIC admission filter-before-rank order mismatch")
    if mimic.get("preseal_diagnosis_materialization") != (
        "PyArrow record-batch stream; predicate hadm_id against completed-admission allowlist "
        "with dischtime strictly before landmark; write preindex-only Parquet before any pandas "
        "conversion or canonicalization; raw rows converted to pandas and materialized post-index "
        "rows must both equal zero"
    ):
        raise AssertionError("MIMIC preseal diagnosis materialization boundary mismatch")
    if mimic.get("death_ascertainment_cap") != (
        "minimum of index_time plus the integer calendar-day offset from index date to "
        "patients.dod, and last valid hospital discharge plus 365 days; cap is also an "
        "administrative censor boundary"
    ):
        raise AssertionError("MIMIC follow-up cap mismatch")
    if mimic["age_populations"] != {
        "age_metric_primary": "integer 40 through 73 inclusive common support",
        "age_metric_adult_sensitivity": "integer age >=18, anchor_age !=91",
        "age_quantization_sensitivities": [
            "common-support chronological age minus 0.5 year",
            "common-support chronological age plus 0.5 year",
        ],
        "risk_primary": "all adult non-topcoded integer-year participants",
    }:
        raise AssertionError("MIMIC age population/sensitivity mismatch")
    if mimic["incident_primary"]["exclude_any_preindex_icd9"] is not True:
        raise AssertionError("MIMIC incident primary does not exclude pre-index ICD9")
    if mimic["incident_primary"]["icd9_endpoint_authorized"] is not False:
        raise AssertionError("MIMIC ICD9 endpoint is authorized")
    if OUTPUT_TABLE_PATHS["mimic_feature_coverage"] not in mimic["feature_coverage_output"]:
        raise AssertionError("MIMIC feature coverage output mismatch")
    if OUTPUT_TABLE_PATHS["mimic_incident_competing_risk_curves"] not in mimic["competing_death"]["curve_output"]:
        raise AssertionError("MIMIC competing-risk curve output mismatch")
    if mimic["external_hash_fold"] != "first 8 SHA256 bytes of 'revision0710-external-v3|{subject_id}', big-endian, modulo 5":
        raise AssertionError("MIMIC hash-fold salt mismatch")
    if mimic["postseal_only_fields"] != [
        "patients.dod",
        "postindex diagnoses",
        "subject observation end",
        "min(index_time + integer calendar-day DOD offset, last valid discharge plus 365 days) follow-up cap",
        "NHANES questionnaire labels",
    ]:
        raise AssertionError("postseal-only outcome field boundary mismatch")
    expected_statistical_algorithms = {
        "aalen_johansen": "at each unique event time update all-cause survival S(t)=S(t-)*(1-d_all/n); add CIF_k(t)=CIF_k(t-)+S(t-)*d_k/n; death is cause 2 and wins exact ties",
        "harrell_c": "comparable pairs require earlier observed target event; ties in risk contribute 0.5",
        "phreg_baseline_hazard": "use fitted Efron PHReg coefficients but reconstruct Breslow dH0(t)=d(t)/sum_{i:Ti>=t}exp(xi beta) directly from the complete risk set; cumulative risk at a horizon is the sum of all jumps at t<=horizon, including first and final event-time jumps",
        "ipcw_auc_brier": "target CIF from explicit post-jump Breslow target/death cause-specific hazard jumps; IPCW AUC ranks target CIF probability rather than target cause-specific linear predictor; case=target before terminal boundary; control=event-free at horizon or competing death before horizon; subject observation/admin censor before horizon is unobserved; fit censoring KM on four training folds, evaluate left-limit G on held-out fold, clamp G at 0.05, cap at training-weight p99; Brier divides by full test n",
        "bootstrap": "one global PCG64 SeedSequence participant-cluster multiplicity plan retained across organs, horizons, variants, and folds; predictive intervals are conditional on fixed training-fold target/death Cox and censor KM",
        "uncertainty": "predictive rows use linear 2.5/97.5 percentiles over B=200 conditional held-out subject-cluster replicates; zero-shot age metrics, full time-grid curves, and recurrent CIFs are explicitly point-only; Cox HRs use robust Wald intervals",
    }
    if contract.get("statistical_algorithms") != expected_statistical_algorithms:
        raise AssertionError("statistical algorithm contract mismatch")
    if mimic["ipcw"]["outcome_definition"] != "case=target before death/censor/horizon; control=event-free at horizon or competing death before horizon; loss/admin censor before horizon is unobserved":
        raise AssertionError("IPCW outcome definition mismatch")
    if mimic["ipcw"]["censor_definition"] != "censor event=subject observation end or death ascertainment cap before target/death/horizon; target and death are non-censor terminal events for G":
        raise AssertionError("IPCW censor definition mismatch")
    if mimic.get("phenotype_data_availability") != {
        "labevents": "require nonmissing charttime and storetime, both <= discharge landmark; latest tie order charttime, storetime, labevent_id",
        "omr": "date-only OMR requires chartdate strictly before the discharge calendar date and within the prior 365 calendar days; same-calendar-date OMR is excluded",
    }:
        raise AssertionError("MIMIC phenotype data-availability boundary mismatch")
    if mimic.get("bootstrap") != (
        "one global PCG64 SeedSequence subject-cluster multiplicity plan with B=200 shared "
        "across all organs, horizons, variants, and fixed hash folds; resample held-out "
        "subjects while keeping training-fold target Cox, death Cox, and censoring KM fixed"
    ):
        raise AssertionError("conditional predictive bootstrap contract mismatch")
    if mimic.get("full_cohort_hr", {}).get("analysis_population") != (
        "full non-held-out organ-specific incident-eligible cohort after excluding baseline "
        "target ICD10 and any completed pre-index ICD9; not the full 95,683-subject MIMIC "
        "input cohort"
    ):
        raise AssertionError("MIMIC association population is mislabeled")
    if contract["nhanes"]["allowed_releases_exact_order"] != list(RELEASES):
        raise AssertionError("NHANES contract release mismatch")
    if contract["nhanes"]["forbid_DEMO_J_or_J_star_with_P_star"] is not True:
        raise AssertionError("NHANES contract permits J/P overlap")
    if contract["nhanes"]["primary_age_population"] != "RIDSTATR == 2 and RIDAGEYR integer 40 through 73 inclusive":
        raise AssertionError("NHANES primary age eligibility mismatch")
    if contract["nhanes"].get("primary_age_integer_guard") != (
        "fractional RIDAGEYR in the examined 40-73 candidate range is a hard error"
    ):
        raise AssertionError("NHANES primary age integer guard mismatch")
    if contract["nhanes"]["raw_registry_primary_count_sanity"] != {
        P_RELEASE: 4916,
        L_RELEASE: 3720,
        "total": 8636,
    }:
        raise AssertionError("NHANES primary age count sanity mismatch")
    if contract["nhanes"]["questionnaire_files_read_by_input_builder"] != []:
        raise AssertionError("NHANES input builder can read questionnaire files")
    if contract["nhanes"].get("release_slot_qc") != {
        "grain": "release x all 78 phenotype slots",
        "required_fields": "finite count, prefrozen physical-unit range count, min, q01, q25, median, q75, q99, max",
        "decision_rule": "any observed nonfinite or value outside a prefrozen unit range aborts before inference",
        "external_data_driven_trimming_or_winsorization": False,
        "quantiles_descriptive_only": True,
    }:
        raise AssertionError("NHANES fixed release-slot QC contract mismatch")
    if contract["nhanes"]["local_crossfit_calibration"] != {
        "regression": "chronological_age = beta0 + beta1*predicted_age_raw_mean5 + beta2*sex_male",
        "folds": "for held-out hash fold k, fit on the other four folds within the same release and organ",
        "coefficient_order": "intercept,predicted_age_raw_mean5,sex_male",
        "application": "apply the three training-fold coefficients once to held-out predicted_age_raw_mean5 and sex_male",
        "role": "sensitivity only; never zero-shot and never enters primary predictions or corrected gaps",
    }:
        raise AssertionError("NHANES local calibration contract mismatch")
    if contract["nhanes"]["questionnaire_association"]["design"] != (
        "full release-specific examined MEC design retained; integer age 40-73 enters as a survey "
        "domain; MEC weight, SDMVSTRA strata, SDMVPSU PSU"
    ):
        raise AssertionError("NHANES questionnaire survey design mismatch")
    gate = contract["formal_gate"]
    if gate["contract_frozen"] is not False or gate["formal_execution_authorized"] is not False:
        raise AssertionError("formal gate is open")
    target = repo_root / gate["formal_target_relative_path"]
    authorization = artifact_dir.parent / "EXTERNAL_VALIDATION_FORMAL_AUTHORIZATION_V8.json"
    lock = authorization.parent / ".EXTERNAL_VALIDATION_FORMAL_V8_ONCE.lock"
    consumed = authorization.with_name(
        f"{authorization.stem}.CONSUMED.json"
    )
    if not formal_runtime_state:
        if gate["formal_target_exists"] is not False or gate["formal_authorization_exists"] is not False:
            raise AssertionError("formal target/authorization reported present")
        if gate.get("formal_lock_exists") is not False or gate.get("formal_consumed_authorization_exists") is not False:
            raise AssertionError("formal lock/consumed authorization reported present")
        validate_formal_target_absent(target, authorization)
    else:
        if target.exists() or authorization.exists():
            raise AssertionError("authorized runtime target/original authorization state invalid")
        if not lock.is_file() or not consumed.is_file():
            raise AssertionError("authorized runtime requires exact lock and consumed authorization")
        if target.parent.exists() and list(
            target.parent.glob(f".{target.name}.staging.*")
        ):
            raise AssertionError("authorized runtime legacy staging target exists")
    if gate["formal_target_resolved"] != str(target.resolve()):
        raise AssertionError("formal resolved target mismatch")
    support = contract["bindings"]["prefreeze_artifacts"]
    if [record["name"] for record in support] != list(ARTIFACT_FILENAMES):
        raise AssertionError("contract prefreeze artifact order mismatch")
    for record in support:
        path = artifact_dir / record["path"]
        if record["path"] != ARTIFACT_FILENAMES[record["name"]]:
            raise AssertionError("contract prefreeze artifact path mismatch")
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"contract prefreeze artifact binding mismatch: {path}")
    for record in contract["bindings"]["code_artifacts"]:
        path = repo_root / record["relative_path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"contract code artifact binding mismatch: {path}")
    if [record["name"] for record in contract["bindings"]["code_artifacts"]] != [
        "builder",
        "independent_verifier",
        "tests",
        "formal_executor",
        "formal_executor_runtime",
        "execution_guard",
        "environment_probe",
        "scientific_extensions",
        "v7_scientific_reviewer_no_go",
        "v7_read_only_tree_snapshot",
        "fig13_fig14_panel_schema",
        "fig13_fig14_caption_and_reviewer_response",
        "v8_attempt_01_support_audit_no_go",
    ]:
        raise AssertionError("contract code artifact set/order mismatch")
    if contract["output_schema"]["sha256"] != sha256_file(artifact_dir / "OUTPUT_SCHEMAS_V8.json"):
        raise AssertionError("contract output schema hash mismatch")


def verify_artifacts(
    artifact_dir: Path,
    repo_root: Path,
    *,
    rehash_external: bool,
    formal_runtime_state: bool = False,
) -> dict[str, Any]:
    expected_files = {
        "EXTERNAL_VALIDATION_CONTRACT_V8.json",
        *ARTIFACT_FILENAMES.values(),
    }
    actual_files = {path.name for path in artifact_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise AssertionError(
            f"prefreeze artifact file set mismatch; missing={sorted(expected_files-actual_files)}, extra={sorted(actual_files-expected_files)}"
        )
    registry = read_json(artifact_dir / "SOURCE_REGISTRY_V8.json")
    registry_result = validate_registry(
        registry, repo_root, rehash_external=rehash_external
    )
    map_result = validate_maps(artifact_dir, registry)
    schema = read_json(artifact_dir / "OUTPUT_SCHEMAS_V8.json")
    schema_result = validate_output_schemas(schema)
    authorization_schema = read_json(
        artifact_dir / "FORMAL_AUTHORIZATION_SCHEMA_V8.json"
    )
    validate_authorization_schema(authorization_schema, repo_root)
    environment_lock_path = artifact_dir / "ENVIRONMENT_LOCK_V8.json"
    environment_lock = read_json(environment_lock_path)
    if set(environment_lock) != {
        "schema_version",
        "status",
        "statistics",
        "inference",
        "phase_assignment",
        "stage_receipt",
    }:
        raise AssertionError("V8 environment lock is open or incomplete")
    if environment_lock["status"] != (
        "TWO_HASH_BOUND_SUBPROCESS_ENVIRONMENTS_NO_IMPLICIT_SWITCH"
    ):
        raise AssertionError("V8 environment lock status mismatch")
    for role in ("statistics", "inference"):
        record = environment_lock[role]
        executable = Path(record["python_executable"])
        if (
            not executable.is_file()
            or executable.stat().st_size <= 0
            or sha256_file(executable) != record["python_executable_sha256"]
        ):
            raise AssertionError(f"V8 {role} interpreter byte binding mismatch")
        for package in record["packages"]:
            metadata = Path(package["metadata_path"])
            if not metadata.is_file() or sha256_file(metadata) != package["metadata_sha256"]:
                raise AssertionError(f"V8 {role} package metadata changed")
            if package["record_path"] is not None:
                package_record = Path(package["record_path"])
                if (
                    not package_record.is_file()
                    or sha256_file(package_record) != package["record_sha256"]
                ):
                    raise AssertionError(f"V8 {role} package RECORD changed")
    contract_path = artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
    contract = read_json(contract_path)
    validate_contract_semantics(
        contract,
        artifact_dir,
        repo_root,
        formal_runtime_state=formal_runtime_state,
    )
    return {
        "status": "PREFREEZE_V8_INDEPENDENT_VERIFY_PASS",
        "artifact_dir": str(artifact_dir.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "source_registry_sha256": sha256_file(
            artifact_dir / "SOURCE_REGISTRY_V8.json"
        ),
        "registry": registry_result,
        "maps": map_result,
        "output_schemas": schema_result,
        "formal_target_absent": True,
        "formal_authorization_absent": True,
        "formal_external_outputs_created": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--rehash-external", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_artifacts(
        args.artifact_dir.resolve(),
        args.repo_root.resolve(),
        rehash_external=args.rehash_external,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
