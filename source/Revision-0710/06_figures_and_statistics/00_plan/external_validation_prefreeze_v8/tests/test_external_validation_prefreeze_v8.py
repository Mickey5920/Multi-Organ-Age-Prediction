from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]


def _load_module(name: str, filename: str):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module("external_validation_prefreeze_v8_builder", "build_external_validation_prefreeze_v8.py")
VERIFIER = _load_module("external_validation_prefreeze_v8_verifier", "verify_external_validation_prefreeze_v8.py")
EXECUTOR = _load_module("external_validation_prefreeze_v8_executor", "execute_external_validation_v8.py")
RUNTIME = _load_module("external_validation_prefreeze_v8_runtime", "execute_external_validation_v8_runtime.py")
GUARD = _load_module("external_validation_prefreeze_v8_guard", "v8_execution_guard.py")
EXTENSIONS = _load_module(
    "external_validation_prefreeze_v8_extensions", "v8_scientific_extensions.py"
)


def test_independent_verifier_does_not_import_builder() -> None:
    source = (HERE / "verify_external_validation_prefreeze_v8.py").read_text(encoding="utf-8")
    assert "import build_external_validation_prefreeze_v8" not in source
    assert "from build_external_validation_prefreeze_v8" not in source


def test_v8_resource_budget_exactly_accounts_for_density_and_complete_ph() -> None:
    budget = BUILDER.build_resource_budget()
    assert budget == VERIFIER.expected_resource_budget()
    assert budget["scientific_method_changed_from_v7"] is True
    accounting = budget["accounting"]
    assert accounting["predictive_models_per_heldout_analysis"] == 4
    assert accounting["cause_specific_cox_fits_per_heldout_analysis"] == 8
    assert accounting["shared_core_ph_fits_per_heldout_analysis"] == 8
    assert accounting["conditional_bootstrap_calls_per_heldout_analysis"] == 4
    assert accounting["conditional_bootstrap_replicates_per_model"] == 200
    assert accounting["v6_frozen_corrected_core_hours"] == pytest.approx(
        11.662225266512021, rel=0.0, abs=1e-12
    )
    assert accounting["v6_fresh_corrected_core_hours"] == pytest.approx(
        12.347672947476752, rel=0.0, abs=1e-12
    )
    reservation = budget["operational_reservation"]
    assert reservation["minimum_hours"] == 36.0
    assert reservation["safer_hours"] == 48.0
    assert reservation["not_a_wall_time_upper_bound"] is True
    assert reservation["covers_or_reserves"] == [
        "label-blind cache construction",
        "CLIP vocabulary extension encoding and model restore",
        "prediction sealing and disk IO",
        "NHANES survey and age-metric aggregation",
        "stage05 and stage06 four-model conditional B=200 bootstrap statistics",
        "complete shared-core PH diagnostics for all target/death model covariates",
        "publication copy, rehash, and receipt validation",
    ]


def test_runtime_projection_counts_b200_once_for_each_predictive_model() -> None:
    projection = RUNTIME._project_formal_statistics_core(
        conditional_b200_seconds_per_model=45.40907699993113,
        four_cause_specific_cox_seconds=3.2063678999984404,
        shared_core_ph_fit_seconds=1.25,
        aalen_johansen_seconds=0.007898299998487346,
        km_seconds=0.007220700004836544,
    )
    assert projection["heldout_analysis_count"] == 210
    assert projection["formal_recompute_passes"] == 2
    assert projection["predictive_models_per_heldout_analysis"] == 4
    assert projection["conditional_b200_calls_per_heldout_analysis"] == 4
    assert projection["conditional_bootstrap_replicates_per_model"] == 200
    assert projection[
        "conditional_b200_seconds_all_models_per_heldout_analysis"
    ] == pytest.approx(4.0 * 45.40907699993113, rel=0.0, abs=1e-12)
    assert projection["cause_specific_cox_seconds_all_four_models_per_heldout_analysis"] == pytest.approx(2.0 * 3.2063678999984404, rel=0.0, abs=1e-12)
    assert projection["shared_core_ph_seconds_all_eight_model_causes_per_heldout_analysis"] == pytest.approx(8.0 * 1.25, rel=0.0, abs=1e-12)
    source = inspect.getsource(RUNTIME._build_mimic_predictive_metrics)
    assert 'for model, risk, probability in (' in source
    assert "_conditional_bootstrap_prediction_values(" in source


@pytest.mark.parametrize("reader", [BUILDER.read_json, VERIFIER.read_json])
def test_duplicate_json_keys_are_rejected(reader, tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="duplicate JSON object key"):
        reader(path)


@pytest.mark.parametrize(
    "observed",
    [
        {"path": "a", "sha256": "0" * 64, "count": 1, "extra": False},
        {"path": "a", "sha256": "0" * 64},
        {"path": "a", "sha256": "0" * 64, "count": True},
        {"count": 1, "sha256": "0" * 64, "path": "a"},
        {"path": "wrong", "sha256": "0" * 64, "count": 1},
        {"path": "a", "sha256": "1" * 64, "count": 1},
    ],
)
def test_closed_exact_payload_rejects_extra_missing_type_order_path_and_hash(observed) -> None:
    expected = {"path": "a", "sha256": "0" * 64, "count": 1}
    with pytest.raises(AssertionError, match="closed exact mismatch"):
        VERIFIER.require_exact_payload("negative", observed, expected)


def test_external_code_namespace_unicode_whitespace_and_fallback() -> None:
    expected_icd9 = "ICD9\x1f25000"
    assert BUILDER.canonicalize_external_code(" ２５０.００ ", "ICD9") == expected_icd9
    assert VERIFIER.canonicalize_external_code(" ２５０.００ ", "ICD9") == expected_icd9
    assert expected_icd9.encode("utf-8").hex().startswith("494344391f")
    assert BUILDER.canonicalize_external_code(" i10. ", "ICD10") == "I10"
    assert VERIFIER.canonicalize_external_code(" i10. ", "ICD10") == "I10"
    description = BUILDER.canonicalize_external_description(
        "\u00a0Essential\t hypertension\n", code_system="ICD10", canonical_code="I10"
    )
    assert description == "Essential hypertension"
    assert BUILDER.canonicalize_external_description(
        "\u3000", code_system="ICD9", canonical_code=expected_icd9
    ) == "ICD9 code 25000"
    expected_text = "icd_description\x1fEssential hypertension"
    assert BUILDER.external_text_key("icd_description", description) == expected_text
    assert VERIFIER.external_text_key("icd_description", description) == expected_text
    with pytest.raises(ValueError):
        BUILDER.canonicalize_external_code("25 000", "ICD9")


def test_formal_runtime_icd_canonicalizer_matches_unicode_and_whitespace_contract() -> None:
    assert RUNTIME._canonical_icd(" ｉ１０． ", 10) == "I10"
    assert RUNTIME._canonical_icd(" ２５０．００ ", 9) == "ICD9\x1f25000"
    for invalid in ("I 10", "I\u00a010", "I\t10", "Iα10"):
        with pytest.raises(ValueError):
            RUNTIME._canonical_icd(invalid, 10)


def test_lab_storetime_and_date_only_omr_availability_boundaries() -> None:
    landmark = pd.Timestamp("2020-02-01 15:00:00")
    labs = pd.DataFrame(
        {
            "charttime": pd.to_datetime(
                ["2020-01-31 10:00", "2020-01-31 10:00", "2020-01-31 10:00"]
            ),
            "storetime": pd.to_datetime(
                ["2020-02-02 10:00", "2020-02-01 14:00", None]
            ),
            "valuenum": [1.0, 2.0, 3.0],
            "index_time": [landmark] * 3,
        }
    )
    assert RUNTIME._lab_available_preindex_mask(labs).tolist() == [False, True, False]
    omr = pd.DataFrame(
        {
            "chartdate": pd.to_datetime(
                ["2020-02-01", "2020-01-31", "2019-02-01", "2019-01-31"]
            ),
            "index_time": [landmark] * 4,
        }
    )
    assert RUNTIME._omr_available_preindex_mask(omr).tolist() == [False, True, True, False]


def test_completed_discharge_diagnosis_availability_prevents_overlap_leakage() -> None:
    cohort = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "index_time": pd.to_datetime(["2020-02-01", "2020-02-01"]),
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [101, 201],
            "dischtime": pd.to_datetime(["2020-02-03", "2020-01-25"]),
        }
    )
    diagnosis = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [101, 201],
            "seq_num": [1, 1],
            "icd_code": ["I10", "250.00"],
            "icd_version": [10, 9],
        }
    )
    mapping = pd.DataFrame(
        {
            "category_start": ["I10"],
            "category_end": ["I15"],
            "organ_system": ["cardiovascular"],
        }
    )
    excluded, baseline, first_post = RUNTIME._completed_diagnosis_endpoint_history(
        diagnosis, admissions, cohort, mapping
    )
    assert excluded == {2}
    assert (1, "cardiovascular") not in baseline
    assert first_post.loc[(1, "cardiovascular")] == pd.Timestamp("2020-02-03")


def test_date_only_dod_rule_keeps_same_day_as_day0_and_excludes_prior_date() -> None:
    index = pd.Timestamp("2020-02-01 15:00")
    same_day = RUNTIME._date_only_dod_rule(index, pd.Timestamp("2020-02-01"))
    assert same_day == {
        "status": "endpoint_eligible",
        "normalized_death": pd.Timestamp("2020-02-01"),
        "death_days": 0.0,
        "same_day_death": True,
    }
    endpoint = EXECUTOR.adjudicate_endpoint(
        target_days=1.0,
        death_days=same_day["death_days"],
        observation_end_days=10.0,
        death_cap_days=0.0,
        horizon_months=6,
    )
    assert endpoint["competing_death_indicator"] == 1
    assert endpoint["observed_time_days"] == 0.0
    prior = RUNTIME._date_only_dod_rule(index, pd.Timestamp("2020-01-31"))
    assert prior["status"] == "excluded_inconsistent_dod_before_index_date"


def test_hash_fold_and_corrected_gap_have_independent_agreement() -> None:
    identifiers = ["1", "subject-200", " fullwidth-Ａ ", 987654]
    assert [BUILDER.deterministic_hash_fold(value) for value in identifiers] == [
        VERIFIER.deterministic_hash_fold(value) for value in identifiers
    ]
    raw = np.array([1.0, -2.0, 4.0], dtype=np.float64)
    age = np.array([40.0, 55.5, 73.0], dtype=np.float64)
    expected = raw - (3.25 + (-0.125) * age)
    np.testing.assert_array_equal(BUILDER.corrected_gap(raw, age, 3.25, -0.125), expected)
    np.testing.assert_array_equal(VERIFIER.corrected_gap(raw, age, 3.25, -0.125), expected)


def test_exact_five_by_seven_sealed_correction_and_receipt_lineage() -> None:
    keys: list[tuple[int, str]] = []
    assert set(BUILDER.correction_paths()) == set(range(5))
    assert set(BUILDER.receipt_paths()) == set(range(5))
    for fold in range(5):
        correction_path = BUILDER.correction_paths()[fold]
        receipt_path = BUILDER.receipt_paths()[fold]
        assert BUILDER.sha256_file(correction_path) == BUILDER.CORRECTION_HASHES[fold]
        assert BUILDER.sha256_file(receipt_path) == BUILDER.RECEIPT_HASHES[fold]
        correction = BUILDER.read_json(correction_path)
        receipt = BUILDER.read_json(receipt_path)
        assert correction["status"] == "FITTED_AND_SEALED_BEFORE_OUTER_READ"
        assert correction["outer_fold"] == fold
        assert correction["calibration_fold"] != fold
        assert correction["fixed_e55_checkpoint_sha256"] == BUILDER.CHECKPOINT_HASHES[fold]
        assert correction["replication_contract_sha256"] == BUILDER.EXPECTED_WORKSPACE_HASHES["replication_contract"]
        assert receipt["fixed_e55_checkpoint_sha256"] == BUILDER.CHECKPOINT_HASHES[fold]
        assert receipt["replication_contract_sha256"] == BUILDER.EXPECTED_WORKSPACE_HASHES["replication_contract"]
        assert receipt["age_correction_sha256"] == BUILDER.CORRECTION_HASHES[fold]
        assert [row["organ_system"] for row in correction["parameters"]] == list(BUILDER.ORGANS)
        keys.extend((fold, row["organ_system"]) for row in correction["parameters"])
    assert keys == [(fold, organ) for fold in range(5) for organ in BUILDER.ORGANS]

    transform_manifest = json.loads(
        (ROOT / "Revision-0710/01_data_rebuild/04_cache_v1/fold_transform_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["outer_fold"] for row in transform_manifest] == list(range(5))
    assert [row["validation_fold"] for row in transform_manifest] == [1, 2, 3, 4, 0]
    assert [row["sha256"] for row in transform_manifest] == [
        BUILDER.TRANSFORM_HASHES[fold] for fold in range(5)
    ]


def test_all_five_checkpoints_and_transforms_are_bound_and_present() -> None:
    for paths, expected_hashes in (
        (BUILDER.checkpoint_paths(), BUILDER.CHECKPOINT_HASHES),
        (BUILDER.transform_paths(), BUILDER.TRANSFORM_HASHES),
    ):
        assert list(paths) == list(range(5))
        assert list(expected_hashes) == list(range(5))
        assert all(path.is_file() for path in paths.values())


def test_external_source_selection_uses_official_mimic_and_only_p_l_nhanes() -> None:
    specs = BUILDER.external_source_specs()
    assert [record["source_id"] for record in specs] == [
        record[0] for record in VERIFIER.MIMIC_EXPECTED_SOURCES
    ] + [record[0] for record in VERIFIER.NHANES_EXPECTED_SOURCES]
    mimic = [record for record in specs if record["cohort"] == "MIMIC_IV_3_1"]
    relative = {
        Path(record["path"]).resolve().relative_to(BUILDER.MIMIC_ROOT.resolve()).as_posix()
        for record in mimic
    }
    assert relative == VERIFIER.MIMIC_ALLOWED_RELATIVE_PATHS
    assert all(Path(record["path"]).parent.resolve() == (BUILDER.MIMIC_ROOT / "hosp").resolve() for record in mimic)
    checksums = BUILDER.read_mimic_official_checksums()
    assert all(path in checksums for path in relative)
    nhanes = [record for record in specs if record["cohort"] == "NHANES"]
    assert {record["release_id"] for record in nhanes} == set(BUILDER.RELEASES)
    assert not any(
        Path(record["path"]).name.upper() == "DEMO_J.XPT"
        or Path(record["path"]).name.upper().endswith("_J.XPT")
        for record in nhanes
    )
    assert {record["source_id"] for record in nhanes if record["role"] == "model_input"}.isdisjoint(
        {record["source_id"] for record in nhanes if record["role"] == "label_only"}
    )


def test_j_release_is_rejected_with_or_without_p_release() -> None:
    with pytest.raises(ValueError, match="cannot coexist"):
        BUILDER.validate_nhanes_release_paths([Path("P_DEMO.xpt"), Path("DEMO_J.xpt")])
    with pytest.raises(ValueError, match="not an authorized"):
        BUILDER.validate_nhanes_release_paths([Path("DEMO_J.xpt")])
    BUILDER.validate_nhanes_release_paths([Path("P_DEMO.xpt"), Path("DEMO_L.xpt")])


def test_nhanes_primary_age_requires_exam_status_and_rejects_top_code() -> None:
    age = np.array([39.0, 40.0, 55.0, 73.0, 74.0, 60.5])
    status = np.array([2, 2, 1, 2, 2, 2])
    assert BUILDER.nhanes_exact_age_mask(age, status).tolist() == [False, True, False, True, False, False]
    with pytest.raises(ValueError, match="top-coded"):
        BUILDER.nhanes_exact_age_mask(np.array([80.0]), np.array([2]))
    with pytest.raises(ValueError, match="shapes differ"):
        BUILDER.nhanes_exact_age_mask(np.array([40.0]), np.array([2, 2]))


@pytest.mark.parametrize("module", [BUILDER, VERIFIER, EXECUTOR])
def test_exact_sex_encodings_and_rejections(module) -> None:
    assert module.encode_mimic_sex("M") == 1
    assert module.encode_mimic_sex("F") == 0
    assert module.encode_nhanes_sex(1) == 1
    assert module.encode_nhanes_sex(2.0) == 0
    for value in ("Male", "m", "", None, np.nan, 1):
        with pytest.raises(ValueError):
            module.encode_mimic_sex(value)
    for value in (0, 3, "M", None, np.nan):
        with pytest.raises(ValueError):
            module.encode_nhanes_sex(value)


@pytest.mark.parametrize("module", [BUILDER, VERIFIER, EXECUTOR])
def test_mimic_age_is_integer_calendar_year_and_rejects_topcode(module) -> None:
    assert module.mimic_integer_age(50, 2018, "2020-12-31 23:59:59") == 52
    assert module.mimic_integer_age(50, 2018, "2020-01-01") == 52
    with pytest.raises(ValueError, match="91"):
        module.mimic_integer_age(91, 2018, "2020-01-01")
    with pytest.raises(ValueError):
        module.mimic_integer_age(17, 2020, "2020-12-31")


@pytest.mark.parametrize(
    "function,args",
    [
        (BUILDER.normalize_unicode_text, (None,)),
        (BUILDER.normalize_unicode_text, (np.nan,)),
        (VERIFIER.normalize_unicode_text, (None,)),
        (VERIFIER.normalize_unicode_text, (np.nan,)),
        (BUILDER.canonicalize_external_code, (None, "ICD10")),
        (VERIFIER.canonicalize_external_code, (np.nan, "ICD9")),
        (BUILDER.deterministic_hash_fold, (None,)),
        (EXECUTOR.deterministic_hash_fold, (np.nan,)),
    ],
)
def test_none_nan_canonicalizers_are_closed(function, args) -> None:
    with pytest.raises(ValueError):
        function(*args)


def _timeline_frame() -> pd.DataFrame:
    landmark = pd.Timestamp("2020-01-01")
    rows = []
    for source_index, (source, event_type) in enumerate(
        (("hospital_icd", 1), ("inpatient_prescription_proxy", 2))
    ):
        for index in range(40):
            rows.append(
                {
                    "participant_id": "p1",
                    "event_time": landmark - pd.Timedelta(days=80 - 2 * index - source_index),
                    "landmark_time": landmark,
                    "source": source,
                    "event_type_id": event_type,
                    "canonical_code": f"I{index:02d}" if event_type == 1 else "",
                    "canonical_description": f"{source} {index}",
                    "source_row_number": source_index * 100 + index,
                }
            )
    return pd.DataFrame(rows)


def test_crop_tie_dedup_order_primary_and_modality_deletion_left_alignment() -> None:
    frame = _timeline_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["source_row_number"] = 999
    frame = pd.concat([frame.iloc[::-1], duplicate], ignore_index=True)
    primary = EXECUTOR.crop_variant_timeline(
        frame.loc[frame["source"].eq("hospital_icd")], EXECUTOR.MIMIC_PRIMARY_VARIANT
    )
    assert len(primary) == 32
    assert set(primary["source"]) == {"hospital_icd"}
    assert primary["event_time"].is_monotonic_increasing
    sensitivity = EXECUTOR.crop_variant_timeline(frame, EXECUTOR.MIMIC_RX_SENSITIVITY)
    assert len(sensitivity) == 32
    assert sensitivity["source"].value_counts().to_dict() == {
        "hospital_icd": 16,
        "inpatient_prescription_proxy": 16,
    }
    aligned = EXECUTOR.left_align_l32(primary.head(7), ["p1", "padding"])
    assert aligned["event_valid_mask"][0, :7].all()
    assert not aligned["event_valid_mask"][0, 7:].any()
    assert not aligned["event_valid_mask"][1].any()


def test_active_landmark_event_and_nhanes_active_l32_are_rejected() -> None:
    active = _timeline_frame().iloc[[0]].copy()
    active["event_time"] = active["landmark_time"]
    with pytest.raises(ValueError, match="at or after landmark"):
        EXECUTOR.crop_variant_timeline(active, EXECUTOR.MIMIC_PRIMARY_VARIANT)
    arrays = {
        "event_code_id": np.zeros((1, 32), dtype=np.int64),
        "event_text_id": np.zeros((1, 32), dtype=np.int64),
        "event_relative_days": np.zeros((1, 32), dtype=np.int32),
        "event_type_id": np.zeros((1, 32), dtype=np.int8),
        "event_valid_mask": np.zeros((1, 32), dtype=bool),
    }
    arrays["event_valid_mask"][0, 0] = True
    with pytest.raises(ValueError, match="padding only"):
        EXECUTOR.validate_nhanes_padding(arrays)


def test_embedding_prefix_and_endpoint_negative_guards() -> None:
    original = b"\x00\x01\x02\x03"
    EXECUTOR.require_immutable_embedding_prefix(original, original + b"extension")
    with pytest.raises(ValueError, match="prefix bytes changed"):
        EXECUTOR.require_immutable_embedding_prefix(original, b"\x00\xff\x02\x03extension")
    valid = pd.DataFrame(
        {
            "has_any_preindex_icd9": [False],
            "endpoint_code_system": ["ICD10"],
            "horizon_months": [6],
            "input_variant": [EXECUTOR.MIMIC_PRIMARY_VARIANT],
        }
    )
    EXECUTOR.validate_mimic_endpoint_rows(valid)
    bad = valid.copy()
    bad["has_any_preindex_icd9"] = True
    with pytest.raises(ValueError, match="pre-index ICD9"):
        EXECUTOR.validate_mimic_endpoint_rows(bad)
    bad = valid.copy()
    bad["horizon_months"] = 24
    with pytest.raises(ValueError, match="6 or 12"):
        EXECUTOR.validate_mimic_endpoint_rows(bad)


def test_label_blind_cache_does_not_read_postindex_outcome_fields() -> None:
    cache_source = inspect.getsource(RUNTIME.build_label_blind_cache)
    assert "_update_observation_end" not in cache_source
    assert '"dod"' not in cache_source
    assert "_build_nhanes_questionnaire_labels" not in cache_source
    outcome_source = inspect.getsource(RUNTIME.build_outcomes_after_prediction_seal)
    seal_check = outcome_source.index("PREDICTIONS_SEALED_BEFORE_OUTCOME_READ")
    first_outcome_read = min(
        outcome_source.index('sources["mimic_patients"]'),
        outcome_source.index('sources["mimic_diagnoses"]'),
    )
    assert seal_check < first_outcome_read


def test_omr_primary_selector_is_exact_not_substring() -> None:
    assert RUNTIME._exact_omr_selector_names("result_name == 'Height (Inches)'") == {
        "Height (Inches)"
    }
    assert "Height" not in RUNTIME._exact_omr_selector_names(
        "result_name == 'Height (Inches)'"
    )
    names = RUNTIME._exact_omr_selector_names("result_name in {BMI,BMI (kg/m2)}")
    assert names == {"BMI", "BMI (kg/m2)"}
    with pytest.raises(ValueError, match="unsupported frozen OMR selector"):
        RUNTIME._exact_omr_selector_names("result_name contains Height")


def test_ipcw_competing_death_control_and_early_admin_censor() -> None:
    horizon = EXECUTOR.months_to_days(6)
    time = np.array([10.0, 20.0, horizon, 30.0])
    cause = np.array([1, 2, 0, 0], dtype=np.int8)
    training_time = np.array([5.0, 15.0, 25.0, horizon])
    training_censor = np.array([0, 1, 0, 0], dtype=np.int8)
    outcome, weight, observed = EXECUTOR.ipcw_binary_weights(
        time, cause, horizon, training_time, training_censor
    )
    assert outcome.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert observed.tolist() == [True, True, True, False]
    assert np.all(weight[:3] > 0)
    assert weight[3] == 0


def test_ipcw_left_limit_uses_immediately_prior_post_jump_value() -> None:
    points, post_jump = EXECUTOR.censor_km_left_limit(
        np.array([5.0, 10.0, 20.0]), np.array([1, 0, 0], dtype=np.int8)
    )
    g_left_at_10 = EXECUTOR._left_g(
        points, post_jump, np.array([10.0], dtype=np.float64)
    )[0]
    assert 1.0 / g_left_at_10 == pytest.approx(1.5)


def test_second_valid_adult_surviving_discharge_selects_hadm3(
    tmp_path: Path,
) -> None:
    patients = tmp_path / "patients.csv.gz"
    admissions = tmp_path / "admissions.csv.gz"
    pd.DataFrame(
        {
            "subject_id": [1],
            "gender": ["M"],
            "anchor_age": [50],
            "anchor_year": [2020],
        }
    ).to_csv(patients, index=False, compression="gzip")
    pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "hadm_id": [1, 2, 3],
            "admittime": ["2020-01-01", "2020-02-10", "2020-03-01"],
            "dischtime": ["2020-01-05", "2020-02-01", "2020-03-05"],
            "deathtime": [None, None, None],
        }
    ).to_csv(admissions, index=False, compression="gzip")
    cohort, valid_admissions, flow = RUNTIME._participant_cohort(
        {"mimic_patients": patients, "mimic_admissions": admissions}
    )
    assert cohort.loc[0, "hadm_id"] == 3
    assert valid_admissions["hadm_id"].tolist() == [1, 3]
    assert flow["has_two_valid_adult_surviving_discharge_admissions"] == 1


def test_followup_cap_uses_last_valid_discharge_and_dod_minimum() -> None:
    admissions = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "admittime": ["2020-01-01", "2022-05-01", "2020-02-01", "2021-04-01"],
            "dischtime": ["2020-01-05", "2022-04-01", "2020-02-05", "2021-04-05"],
        }
    )
    last = RUNTIME._last_valid_discharge(admissions)
    assert last.loc[1] == pd.Timestamp("2020-01-05")
    assert last.loc[2] == pd.Timestamp("2021-04-05")
    cap = RUNTIME._subject_followup_cap(
        pd.Series({1: pd.Timestamp("2020-12-01"), 2: pd.NaT}),
        last,
        pd.Series(
            {1: pd.Timestamp("2020-01-05"), 2: pd.Timestamp("2021-04-05")}
        ),
    )
    assert cap.loc[1] == pd.Timestamp("2020-12-01")
    assert cap.loc[2] == pd.Timestamp("2022-04-05")


@pytest.mark.parametrize("calendar_days", [1, 10])
def test_nonmidnight_date_only_dod_preserves_integer_followup_days(
    calendar_days: int,
) -> None:
    index_time = pd.Timestamp("2020-01-10 18:37:12")
    death_date = index_time.normalize() + pd.Timedelta(days=calendar_days)
    cap = RUNTIME._subject_followup_cap(
        pd.Series({1: death_date}),
        pd.Series({1: index_time + pd.Timedelta(days=30)}),
        pd.Series({1: index_time}),
    )
    cap_days = (cap.loc[1] - index_time).total_seconds() / 86400.0
    dod_rule = RUNTIME._date_only_dod_rule(index_time, death_date)
    assert cap_days == pytest.approx(float(calendar_days))
    assert dod_rule["death_days"] == pytest.approx(float(calendar_days))
    endpoint = EXECUTOR.adjudicate_endpoint(
        target_days=None,
        death_days=dod_rule["death_days"],
        observation_end_days=30.0,
        death_cap_days=cap_days,
        horizon_months=6,
    )
    assert endpoint["censor_reason"] == "death"
    assert endpoint["competing_death_indicator"] == 1


def test_preseal_diagnosis_builder_materializes_only_preindex_hadm(
    tmp_path: Path,
) -> None:
    diagnoses = tmp_path / "diagnoses.csv.gz"
    dictionary = tmp_path / "dictionary.csv.gz"
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 30],
            "seq_num": [1, 1],
            "icd_code": ["I10", "I11"],
            "icd_version": [10, 10],
        }
    ).to_csv(diagnoses, index=False, compression="gzip")
    pd.DataFrame(
        {
            "icd_code": ["I10", "I11"],
            "icd_version": [10, 10],
            "long_title": ["Hypertension", "Heart disease"],
        }
    ).to_csv(dictionary, index=False, compression="gzip")
    cohort = pd.DataFrame(
        {"subject_id": [1], "index_time": [pd.Timestamp("2020-02-01")]}
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 20, 30],
            "admittime": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-20"), pd.Timestamp("2020-03-01")],
            "dischtime": [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-05")],
        }
    )
    materialized = RUNTIME._materialize_preindex_diagnosis_allowlist(
        cohort, admissions, diagnoses, tmp_path / "preindex.parquet"
    )
    events, dropped = RUNTIME._mimic_diagnosis_events(
        cohort,
        admissions,
        {
            "mimic_diagnoses": diagnoses,
            "mimic_diagnosis_dictionary": dictionary,
        },
        Path(materialized["table"]),
    )
    assert events["canonical_code"].tolist() == ["I10"]
    assert events["code_system"].tolist() == ["ICD10"]
    assert dropped.empty
    assert materialized["raw_rows_converted_to_pandas"] == 0
    assert materialized["preindex_rows_materialized"] == 1
    source = inspect.getsource(RUNTIME._materialize_preindex_diagnosis_allowlist)
    assert "pyarrow.csv" in source
    assert "dischtime" in source and ".lt(eligible[\"landmark_time\"])" in source


def test_primary_lab_value_rules_close_percent_and_hba1c_domains() -> None:
    assert RUNTIME._value_rule_accepts(0.0, "finite; 0<=value<=100", "%")
    assert RUNTIME._value_rule_accepts(100.0, "finite; 0<=value<=100", "%")
    assert not RUNTIME._value_rule_accepts(100.1, "finite; 0<=value<=100", "%")
    assert RUNTIME._value_rule_accepts(2.0, "finite; 2<=value<=25 percent", "%")
    assert RUNTIME._value_rule_accepts(25.0, "finite; 2<=value<=25 percent", "%")
    assert not RUNTIME._value_rule_accepts(1.99, "finite; 2<=value<=25 percent", "%")
    branch_rule = "finite; percent branch 0<=value<=100; g/dL branch requires value>=0"
    assert not RUNTIME._value_rule_accepts(101.0, branch_rule, "%")
    assert RUNTIME._value_rule_accepts(101.0, branch_rule, "g/dL")


def test_centered_lonely_psu_has_nonzero_variance_contribution() -> None:
    scores = np.array([[2.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    strata = np.array([1, 2, 2])
    meat = RUNTIME._centered_psu_meat(scores, strata)
    assert np.isfinite(meat).all()
    assert meat[0, 0] > 0


def test_bh_uses_frozen_full_family_sizes() -> None:
    p = np.array([0.01, 0.02])
    np.testing.assert_allclose(
        EXECUTOR.benjamini_hochberg(p, family_size=7), [0.07, 0.07]
    )
    np.testing.assert_allclose(
        EXECUTOR.benjamini_hochberg(p, family_size=6), [0.06, 0.06]
    )
    with pytest.raises(ValueError):
        EXECUTOR.benjamini_hochberg(p, family_size=1)


def test_nhanes_primary_age_runtime_guard_rejects_fractional_year() -> None:
    with pytest.raises(AssertionError, match="fractional RIDAGEYR"):
        RUNTIME._nhanes_primary_integer_age_guard(
            pd.Series([40.0, 60.5]), pd.Series([2.0, 2.0])
        )
    mask = RUNTIME._nhanes_primary_integer_age_guard(
        pd.Series([39.0, 40.0, 73.0, 74.0]), pd.Series([2.0, 2.0, 2.0, 2.0])
    )
    assert mask.tolist() == [False, True, True, False]


def test_fast_survival_and_auc_kernels_match_bruteforce_with_ties() -> None:
    rng = np.random.default_rng(7102026)

    def brute_harrell(time, event, risk):
        concordant = comparable = 0.0
        for i in range(len(time)):
            if event[i] != 1:
                continue
            for j in range(len(time)):
                if time[i] >= time[j]:
                    continue
                comparable += 1.0
                concordant += float(risk[i] > risk[j]) + 0.5 * float(risk[i] == risk[j])
        return concordant / comparable

    def brute_auc(outcome, risk, weight):
        numerator = denominator = 0.0
        for i in range(len(outcome)):
            if outcome[i] != 1 or weight[i] <= 0:
                continue
            for j in range(len(outcome)):
                if outcome[j] != 0 or weight[j] <= 0:
                    continue
                pair = weight[i] * weight[j]
                denominator += pair
                numerator += pair * (
                    float(risk[i] > risk[j]) + 0.5 * float(risk[i] == risk[j])
                )
        return numerator / denominator

    def brute_aj(time, cause):
        survival = 1.0
        target = death = 0.0
        rows = []
        for point in np.sort(np.unique(time[cause != 0])):
            at_risk = int(np.sum(time >= point))
            d_target = int(np.sum((time == point) & (cause == 1)))
            d_death = int(np.sum((time == point) & (cause == 2)))
            prior = survival
            target += prior * d_target / at_risk
            death += prior * d_death / at_risk
            survival *= 1.0 - (d_target + d_death) / at_risk
            rows.append((point, at_risk, d_target, d_death, survival, target, death))
        return np.asarray(rows, dtype=float)

    def brute_km(time, event):
        survival = 1.0
        rows = []
        for point in np.sort(np.unique(time)):
            at_risk = int(np.sum(time >= point))
            failures = int(np.sum((time == point) & (event == 1)))
            censored = int(np.sum((time == point) & (event == 0)))
            if failures:
                survival *= 1.0 - failures / at_risk
            rows.append((point, at_risk, failures, censored, survival))
        return np.asarray(rows, dtype=float)

    for _ in range(12):
        n = 40
        duration = rng.integers(0, 8, size=n).astype(float)
        cause = rng.integers(0, 3, size=n).astype(np.int8)
        event = (cause == 1).astype(np.int8)
        if not any(
            event[i] and np.any(duration > duration[i]) for i in range(n)
        ):
            continue
        risk = rng.integers(-2, 3, size=n).astype(float)
        outcome = rng.integers(0, 2, size=n).astype(float)
        outcome[0], outcome[1] = 0.0, 1.0
        weight = rng.integers(1, 5, size=n).astype(float)
        assert EXECUTOR.harrell_c(duration, event, risk) == pytest.approx(
            brute_harrell(duration, event, risk), abs=1e-12
        )
        assert EXECUTOR.weighted_auc(outcome, risk, weight) == pytest.approx(
            brute_auc(outcome, risk, weight), abs=1e-12
        )
        observed_aj = EXECUTOR.aalen_johansen(duration, cause)
        np.testing.assert_allclose(
            observed_aj[
                [
                    "time_days",
                    "n_at_risk",
                    "target_events",
                    "death_events",
                    "survival",
                    "target_cif",
                    "death_cif",
                ]
            ].to_numpy(dtype=float),
            brute_aj(duration, cause),
            rtol=0.0,
            atol=1e-12,
        )
        observed_km = RUNTIME._km_curve(duration, event)
        np.testing.assert_allclose(
            observed_km[
                ["time_days", "n_at_risk", "events", "censored", "estimate"]
            ].to_numpy(dtype=float),
            brute_km(duration, event),
            rtol=0.0,
            atol=1e-12,
        )


def test_global_bootstrap_plan_is_single_deterministic_shared_plan() -> None:
    identifiers = np.asarray([f"S{i:03d}" for i in range(100)])
    first_ids, first = RUNTIME._global_subject_bootstrap_plan(
        global_subject_ids=identifiers, replicates=20, seed=7102026
    )
    second_ids, second = RUNTIME._global_subject_bootstrap_plan(
        global_subject_ids=identifiers[::-1], replicates=20, seed=7102026
    )
    np.testing.assert_array_equal(first_ids, second_ids)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=1), np.full(20, len(identifiers)))


def test_scaled_schoenfeld_and_fixed_bh_are_explicit_in_runtime() -> None:
    source = inspect.getsource(RUNTIME._build_mimic_associations)
    assert "residual_matrix @ covariance" in source
    assert "family_size=7" in source
    source = inspect.getsource(RUNTIME._build_nhanes_questionnaire_associations)
    assert "family_size=6" in source


def test_formal_authorization_validation_precedes_external_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json").write_text(
        "{}\n", encoding="utf-8"
    )
    events = []

    class FakeVerifier:
        @staticmethod
        def read_json(path):
            return {"formal_publication": {}, "review_freeze": {}}

        @staticmethod
        def validate_formal_authorization_payload(*args, **kwargs):
            events.append("authorization_validation")
            raise AssertionError("invalid authorization")

        @staticmethod
        def verify_artifacts(*args, **kwargs):
            events.append("external_rehash")

    monkeypatch.setattr(EXECUTOR, "AUTHORIZATION", authorization)
    monkeypatch.setattr(EXECUTOR, "FORMAL_LOCK", tmp_path / "formal.lock")
    monkeypatch.setattr(EXECUTOR, "FORMAL_TARGET", tmp_path / "formal_target")
    monkeypatch.setattr(EXECUTOR, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(EXECUTOR, "_load_verifier", lambda: FakeVerifier)
    with pytest.raises(AssertionError, match="invalid authorization"):
        EXECUTOR.require_formal_authorization()
    assert events == ["authorization_validation"]
    assert authorization.exists()
    assert not (tmp_path / "formal.lock").exists()


def test_valid_authorization_is_atomically_consumed_before_external_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text("valid\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json").write_text(
        "{}\n", encoding="utf-8"
    )
    payload = {
        "formal_publication": {
            "mode": "FORMAL_ONCE_CREATE_ONLY",
            "executor_name": "formal_executor",
        },
        "review_freeze": {
            "independent_review_status": "PASS",
            "contract_bytes_frozen": True,
            "all_authorized_hashes_exact": True,
        },
    }

    class FakeVerifier:
        @staticmethod
        def read_json(path):
            return {"bindings": {"code_artifacts": []}} if "CONTRACT" in Path(path).name else payload

        @staticmethod
        def validate_formal_authorization_payload(*args, **kwargs):
            return None

        @staticmethod
        def verify_artifacts(*args, **kwargs):
            assert not authorization.exists()
            assert kwargs["formal_runtime_state"] is True
            raise RuntimeError("synthetic external rehash stop")

    monkeypatch.setattr(EXECUTOR, "AUTHORIZATION", authorization)
    monkeypatch.setattr(EXECUTOR, "FORMAL_LOCK", tmp_path / "formal.lock")
    monkeypatch.setattr(EXECUTOR, "FORMAL_TARGET", tmp_path / "formal_target")
    monkeypatch.setattr(EXECUTOR, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(EXECUTOR, "_load_verifier", lambda: FakeVerifier)
    with pytest.raises(RuntimeError, match="synthetic external rehash stop"):
        EXECUTOR.require_formal_authorization()
    assert not authorization.exists()
    assert len(list(tmp_path.glob("authorization.CONSUMED.json"))) == 1
    assert (tmp_path / "formal.lock").exists()


def test_concurrent_formal_lock_is_never_removed_by_losing_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text("valid\n", encoding="utf-8")
    lock = tmp_path / "formal.lock"
    lock.write_text("owned by another process\n", encoding="utf-8")
    monkeypatch.setattr(EXECUTOR, "AUTHORIZATION", authorization)
    monkeypatch.setattr(EXECUTOR, "FORMAL_LOCK", lock)
    monkeypatch.setattr(EXECUTOR, "FORMAL_TARGET", tmp_path / "formal_target")
    with pytest.raises(FileExistsError):
        EXECUTOR.require_formal_authorization()
    assert lock.read_text(encoding="utf-8") == "owned by another process\n"


def test_consumed_authorization_resume_uses_runtime_state_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = tmp_path / "authorization.json"
    consumed = tmp_path / "authorization.CONSUMED.json"
    consumed.write_text("valid\n", encoding="utf-8")
    lock = tmp_path / "formal.lock"
    lock.write_text("owned\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json").write_text(
        "{}\n", encoding="utf-8"
    )
    payload = {
        "formal_publication": {
            "mode": "FORMAL_ONCE_CREATE_ONLY",
            "executor_name": "formal_executor",
        },
        "review_freeze": {
            "independent_review_status": "PASS",
            "contract_bytes_frozen": True,
            "all_authorized_hashes_exact": True,
        },
    }
    calls = []

    class FakeVerifier:
        @staticmethod
        def read_json(path):
            return {"bindings": {"code_artifacts": []}} if "CONTRACT" in Path(path).name else payload

        @staticmethod
        def validate_formal_authorization_payload(*args, **kwargs):
            return None

        @staticmethod
        def verify_artifacts(*args, **kwargs):
            calls.append(kwargs["formal_runtime_state"])
            return {"status": "PASS"}

    monkeypatch.setattr(EXECUTOR, "AUTHORIZATION", authorization)
    monkeypatch.setattr(EXECUTOR, "CONSUMED_AUTHORIZATION", consumed)
    monkeypatch.setattr(EXECUTOR, "FORMAL_LOCK", lock)
    monkeypatch.setattr(EXECUTOR, "FORMAL_TARGET", tmp_path / "formal_target")
    monkeypatch.setattr(EXECUTOR, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(EXECUTOR, "_load_verifier", lambda: FakeVerifier)
    _, _, observed_consumed, _ = EXECUTOR.require_formal_authorization()
    assert observed_consumed == consumed
    assert calls == [True]
    assert lock.read_text(encoding="utf-8") == "owned\n"


def test_resume_rejects_coexisting_original_and_consumed_authorizations_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = tmp_path / "authorization.json"
    consumed = tmp_path / "authorization.CONSUMED.json"
    lock = tmp_path / "formal.lock"
    authorization.write_text("original\n", encoding="utf-8")
    consumed.write_text("consumed\n", encoding="utf-8")
    lock.write_text("owned\n", encoding="utf-8")
    monkeypatch.setattr(EXECUTOR, "AUTHORIZATION", authorization)
    monkeypatch.setattr(EXECUTOR, "CONSUMED_AUTHORIZATION", consumed)
    monkeypatch.setattr(EXECUTOR, "FORMAL_LOCK", lock)
    monkeypatch.setattr(EXECUTOR, "FORMAL_TARGET", tmp_path / "formal_target")
    monkeypatch.setattr(
        EXECUTOR,
        "_load_verifier",
        lambda: pytest.fail("verifier must not load for ambiguous resume state"),
    )
    with pytest.raises(PermissionError, match="original and consumed authorizations coexist"):
        EXECUTOR.require_formal_authorization()
    assert authorization.is_file() and consumed.is_file() and lock.is_file()


def test_formal_table_writer_binds_all_twenty_hashes_and_formula_tolerances() -> None:
    source = inspect.getsource(RUNTIME.validate_and_write_tables)
    assert "len(table_records) != 20" in source
    assert "schema_formula_tolerance_sha256" in source
    assert "FORMAL_TABLE_MANIFEST_V8.json" in source
    source = inspect.getsource(EXECUTOR.FormalPipeline.run)
    assert "table_bundle" in source


def test_inference_and_prediction_seal_never_read_full_source_registry() -> None:
    inference_source = inspect.getsource(RUNTIME.run_five_checkpoint_inference)
    assert "SOURCE_REGISTRY_V8.json" not in inference_source
    assert 'registry.get("external_sources") != []' in inference_source
    seal_source = inspect.getsource(RUNTIME.seal_prediction_tables)
    assert "SOURCE_REGISTRY_V8.json" not in seal_source
    assert 'predictions["locked_registry"]' in seal_source
    worker_source = inspect.getsource(RUNTIME.run_worker_stage)
    assert 'set(payload) != {"cache"}' in worker_source
    cache_source = inspect.getsource(RUNTIME.build_label_blind_cache)
    assert '"external_sources": []' in cache_source
    assert 'f"correction_fold_{fold}"' in cache_source
    pipeline_source = inspect.getsource(EXECUTOR.FormalPipeline.run)
    seal_block = pipeline_source.split("sealed =", 1)[1].split("endpoints =", 1)[0]
    assert 'worker_artifact_dir=Path(cache["inference_artifact_dir"])' in seal_block
    assert "worker_artifact_dir=ARTIFACT_DIR" not in seal_block


def test_vocabulary_coverage_separates_exact_extension_invalid_zero_and_systems(
    tmp_path: Path,
) -> None:
    code_vocab = pd.read_parquet(
        ROOT / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_code_vocab.parquet"
    )
    text_vocab = pd.read_parquet(
        ROOT / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_text_vocab.parquet"
    )
    exact_code = str(code_vocab.loc[code_vocab["raw_text"].astype(str).ne(""), "raw_text"].iloc[0])
    exact_icd_text = text_vocab.loc[
        text_vocab["semantic_role"].astype(str).eq("icd_description")
    ].iloc[0]
    landmark = pd.Timestamp("2020-02-01")
    icd = pd.DataFrame(
        [
            {
                "participant_id": "1",
                "event_time": landmark - pd.Timedelta(days=2),
                "landmark_time": landmark,
                "source": "hospital_icd",
                "code_system": "ICD9",
                "event_type_id": 1,
                "canonical_code": exact_code,
                "canonical_description": str(exact_icd_text["raw_text"]),
                "source_row_number": 1,
            },
            {
                "participant_id": "1",
                "event_time": landmark - pd.Timedelta(days=1),
                "landmark_time": landmark,
                "source": "hospital_icd",
                "code_system": "ICD10",
                "event_type_id": 1,
                "canonical_code": "V4_UNSEEN_CODE",
                "canonical_description": "V8 unseen ICD description",
                "source_row_number": 2,
            },
        ]
    )
    rx = pd.DataFrame(
        [
            {
                "participant_id": "1",
                "event_time": landmark - pd.Timedelta(days=3),
                "landmark_time": landmark,
                "source": "inpatient_prescription_proxy",
                "code_system": "INPATIENT_RX",
                "event_type_id": 2,
                "canonical_code": "",
                "canonical_description": "V8 unseen inpatient prescription",
                "source_row_number": 3,
            }
        ]
    )
    dropped = pd.DataFrame(
        [
            {
                "modality": "hospital_icd",
                "code_system": "not_applicable",
                "reason": "invalid_or_unsupported_code",
                "participant_id": "1",
            },
            {
                "modality": "inpatient_prescription_proxy",
                "code_system": "INPATIENT_RX",
                "reason": "invalid_or_missing_text",
                "participant_id": "1",
            },
        ]
    )
    nhanes = pd.DataFrame(
        {
            "release_id": [BUILDER.P_RELEASE, BUILDER.L_RELEASE],
            "participant_id": ["p", "l"],
        }
    )
    icd.to_parquet(tmp_path / "mimic_icd_events_preindex.parquet", index=False)
    rx.to_parquet(
        tmp_path / "mimic_inpatient_prescription_proxy_preindex.parquet", index=False
    )
    dropped.to_parquet(tmp_path / "mimic_vocabulary_drop_audit.parquet", index=False)
    nhanes.to_parquet(tmp_path / "nhanes_participants.parquet", index=False)
    pd.DataFrame({"subject_id": [1, 2]}).to_parquet(
        tmp_path / "mimic_participants.parquet", index=False
    )
    extension_manifest = tmp_path / "VOCABULARY_EXTENSION_MANIFEST.json"
    extension_manifest.write_text("{}\n", encoding="utf-8")
    extension_sha = EXECUTOR.sha256_file(extension_manifest)
    audit = pd.DataFrame(
        [
            {
                "vocabulary_kind": kind,
                "original_vocab_rows": rows,
                "original_vocab_prefix_bytes": 1,
                "original_vocab_prefix_sha256": "0" * 64,
                "extension_rows": 1,
                "original_embedding_rows": rows,
                "original_embedding_prefix_bytes": rows * 512 * 2,
                "original_embedding_prefix_sha256": "1" * 64,
                "extension_manifest_sha256": extension_sha,
            }
            for kind, rows in (("code", 9906), ("text", 44565))
        ]
    )
    audit_path = tmp_path / "vocabulary_prefix_audit.csv"
    audit.to_csv(audit_path, index=False)
    result = RUNTIME._build_vocabulary_coverage(
        tmp_path,
        {
            "vocabulary_audit": audit_path,
            "vocabulary_extension_manifest": extension_manifest,
        },
    )
    assert len(result) == 60
    assert set(result["token_class"]) == {
        "exact_frozen_vocab",
        "deterministic_extension",
        "invalid_dropped",
        "zero_event",
    }
    assert {"ICD9", "ICD10", "INPATIENT_RX", "not_applicable"}.issubset(
        set(result["code_system"])
    )


def test_synthetic_l32_freezes_primary_and_prescription_proxy_sensitivity() -> None:
    probe = BUILDER.synthetic_timeline_probe()
    assert probe == {
        "status": "SYNTHETIC_V8_TIMELINE_PROBE_PASS",
        "primary_selected_rows": 32,
        "primary_hospital_icd_selected": 32,
        "sensitivity_selected_rows": 32,
        "sensitivity_hospital_icd_selected": 16,
        "sensitivity_inpatient_prescription_proxy_selected": 16,
        "timeline_sha256": "e32d332bfd4941c68f1c5a42cfa185f2d348722a27856ca5537cc980346e2333",
        "strictly_pre_index": True,
        "primary_left_aligned_l32": True,
        "nhanes_padding_only": True,
        "event_at_landmark_rejected": True,
    }


def test_mimic_month_conversion_and_death_wins_exact_tie() -> None:
    assert BUILDER.mimic_months_to_days(12) == pytest.approx(365.0, abs=0.0)
    assert BUILDER.mimic_months_to_days(6) == pytest.approx(182.5, abs=0.0)
    tie = BUILDER.adjudicate_target_death(
        target_time_days=10,
        death_time_days=10,
        subject_observation_end_days=30,
        death_ascertainment_cap_days=365,
        horizon_months=6,
    )
    assert tie["censor_reason"] == "death"
    assert tie["event_indicator"] == 0
    assert tie["competing_death_indicator"] == 1
    target_first = BUILDER.adjudicate_target_death(
        target_time_days=9,
        death_time_days=10,
        subject_observation_end_days=30,
        death_ascertainment_cap_days=365,
        horizon_months=6,
    )
    assert target_first["censor_reason"] == "target"
    assert target_first["event_indicator"] == 1
    with pytest.raises(ValueError, match="only 6 and 12"):
        BUILDER.mimic_months_to_days(24)


def test_output_schemas_are_closed_and_incident_recurrent_are_physical_separate() -> None:
    payload = BUILDER.build_output_schemas()
    result = VERIFIER.validate_output_schemas(payload)
    assert result["tables"] == 20
    assert payload["column_schema_keys_exact_order"] == [
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
    ]
    tables = {table["table_id"]: table for table in payload["tables"]}
    assert [table["table_id"] for table in payload["tables"]] == list(VERIFIER.OUTPUT_TABLE_PATHS)
    assert tables["mimic_incident_endpoints"]["relative_path"] != tables["mimic_recurrent_endpoints"]["relative_path"]
    for table in payload["tables"]:
        for column in table["columns"]:
            assert set(column) == VERIFIER.COLUMN_SCHEMA_KEYS


def test_environment_lock_probes_two_real_hash_bound_interpreters() -> None:
    lock = BUILDER.build_environment_lock()
    assert lock["statistics"]["python_executable"].lower().endswith(
        "anaconda\\python.exe"
    )
    assert lock["inference"]["python_executable"].lower().endswith(
        "torch_25\\python.exe"
    )
    statistics = {
        row["name"]: row["version"] for row in lock["statistics"]["packages"]
    }
    inference = {
        row["name"]: row["version"] for row in lock["inference"]["packages"]
    }
    assert statistics["statsmodels"] == "0.14.2"
    assert statistics["pyarrow"] == "14.0.2"
    assert inference["torch"] == "2.5.1"
    assert inference["open-clip-torch"] == "3.3.0"
    assert lock["phase_assignment"]["02_inference"] == "inference"
    assert lock["phase_assignment"]["05_tables"] == "statistics"


def test_composite_questionnaire_negative_requires_every_explicit_two() -> None:
    matrix = pd.DataFrame(
        [
            [2, 2, 2],
            [2, np.nan, 2],
            [2, 7, 2],
            [2, 1, np.nan],
        ]
    )
    labels = RUNTIME._questionnaire_composite_labels(matrix)
    assert labels[0] == 0
    assert np.isnan(labels[1])
    assert np.isnan(labels[2])
    assert labels[3] == 1


def test_standard_ipcw_brier_uses_full_test_n_not_weight_sum() -> None:
    observed = np.array([1.0, 0.0])
    probability = np.array([0.0, 1.0])
    weight = np.array([2.0, 0.5])
    assert EXECUTOR.weighted_brier(observed, probability, weight) == pytest.approx(
        1.25
    )
    assert EXECUTOR.weighted_brier(
        observed, probability, weight, normalization_n=4
    ) == pytest.approx(0.625)


def test_ipcw_returns_positivity_cap_ess_and_observation_diagnostics() -> None:
    outcome, weight, observed, diagnostics = EXECUTOR.ipcw_binary_weights(
        np.array([10.0, 20.0, 50.0, 100.0]),
        np.array([1, 2, 0, 0], dtype=np.int8),
        60.0,
        np.array([5.0, 15.0, 25.0, 80.0]),
        np.array([0, 1, 0, 0], dtype=np.int8),
        return_diagnostics=True,
    )
    assert len(outcome) == len(weight) == len(observed) == 4
    assert diagnostics["n_observed_at_horizon"] == 3
    assert diagnostics["n_censored_before_horizon"] == 1
    assert diagnostics["g_floor"] == 0.05
    assert 0 < diagnostics["effective_sample_size"] <= 4
    assert diagnostics["weight_cap_p99_training"] >= 1


def test_competing_death_cif_is_not_cause_specific_net_risk() -> None:
    rng = np.random.default_rng(100)
    n = 240
    training = pd.DataFrame(
        {
            "observed_time_days": rng.uniform(5, 180, size=n),
            "event_indicator": np.r_[np.ones(90), np.zeros(n - 90)],
            "competing_death_indicator": np.r_[
                np.zeros(90), np.ones(80), np.zeros(n - 170)
            ],
            "x": rng.normal(size=n),
        }
    )
    testing = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
    _, net = RUNTIME._cox_predictions(training, testing, ["x"], 180.0)
    _, cif = RUNTIME._cause_specific_cif_predictions(
        training, testing, ["x"], 180.0
    )
    assert np.all(cif >= 0) and np.all(cif <= 1)
    assert np.all(cif <= net + 1e-12)
    assert np.any(cif < net - 1e-5)


def test_phreg_breslow_reconstruction_includes_first_and_final_event_jumps() -> None:
    complete = pd.DataFrame(
        {
            "observed_time_days": [1.0, 2.0, 3.0],
            "event_indicator": [1, 0, 1],
            "x": [0.0, 0.0, 0.0],
        }
    )
    times, jumps = RUNTIME._phreg_baseline_hazard_jumps(
        complete,
        SimpleNamespace(params=np.array([0.0])),
        ["x"],
        status_column="event_indicator",
    )
    np.testing.assert_allclose(times, [1.0, 3.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(jumps, [1.0 / 3.0, 1.0], rtol=0.0, atol=1e-15)
    cif = RUNTIME._integrate_cause_specific_hazard_jumps(
        times,
        jumps,
        np.array([0.0]),
        np.empty(0),
        np.empty(0),
        np.array([0.0]),
        3.0,
    )
    assert cif[0] == pytest.approx(1.0 - np.exp(-(1.0 / 3.0 + 1.0)))


def test_competing_death_can_reverse_target_lp_rank_and_ipcw_auc_uses_cif() -> None:
    target_linear = np.array([1.0, 0.0])
    cif = RUNTIME._integrate_cause_specific_hazard_jumps(
        np.array([1.0]),
        np.array([0.2]),
        target_linear,
        np.array([0.5]),
        np.array([1.0]),
        np.array([3.0, -5.0]),
        1.0,
    )
    assert target_linear[0] > target_linear[1]
    assert cif[0] < cif[1]
    outcome = np.array([0.0, 1.0])
    weight = np.ones(2)
    assert EXECUTOR.weighted_auc(outcome, target_linear, weight) == 0.0
    assert EXECUTOR.weighted_auc(outcome, cif, weight) == 1.0
    source = inspect.getsource(RUNTIME._build_mimic_predictive_metrics)
    assert "weighted_auc(outcome, probability, weight)" in source
    bootstrap_source = inspect.getsource(
        RUNTIME._conditional_bootstrap_prediction_values
    )
    assert (
        "weighted_auc(sampled_outcome, probability[index], sampled_weight)"
        in bootstrap_source
    )


def test_stage_receipt_preserves_failed_attempt_and_resumes_completed_stage(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    calls = {"count": 0}

    def runner(root: Path):
        calls["count"] += 1
        if calls["count"] == 1:
            (root / "failed_attempt_bytes.txt").write_text("preserve\n")
            raise RuntimeError("synthetic stage failure")
        output = root / "completed.txt"
        output.write_text("complete\n")
        return [root / "failed_attempt_bytes.txt", output]

    kwargs = {
        "stage_name": "synthetic",
        "stage_root": stage,
        "input_bindings": [{"name": "input", "sha256": "0" * 64}],
        "environment_sha256": "1" * 64,
        "runner": runner,
    }
    with pytest.raises(RuntimeError, match="synthetic stage failure"):
        GUARD.run_or_resume_stage(**kwargs)
    assert (stage / "failed_attempt_bytes.txt").is_file()
    GUARD.run_or_resume_stage(**kwargs)
    assert calls["count"] == 2
    GUARD.run_or_resume_stage(**kwargs)
    assert calls["count"] == 2
    (stage / "completed.txt").write_text("mutated\n")
    with pytest.raises(GUARD.V8GuardError, match="stage output hash mismatch"):
        GUARD.run_or_resume_stage(**kwargs)


def test_formal_publish_is_attempt_atomic_and_failure_preserving() -> None:
    source = inspect.getsource(EXECUTOR.run_formal)
    assert "attempt_" in source
    assert "os.rename(attempt, bundle)" in source
    assert source.count("validate_completed_publish_bundle(") >= 2
    assert "shutil.rmtree(stage)" not in source


def test_completed_publish_bundle_deep_rehash_rejects_tampered_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    schemas = BUILDER.build_output_schemas()
    (artifact_dir / "OUTPUT_SCHEMAS_V8.json").write_text(
        json.dumps(schemas, indent=2) + "\n", encoding="utf-8"
    )
    (artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json").write_text(
        '{"synthetic": true}\n', encoding="utf-8"
    )
    (artifact_dir / "ENVIRONMENT_LOCK_V8.json").write_text(
        '{"synthetic": true}\n', encoding="utf-8"
    )
    monkeypatch.setattr(EXECUTOR, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(EXECUTOR, "FORMAL_LOCK", tmp_path / "formal.lock")
    work_stage = tmp_path / "work"
    bundle = tmp_path / "bundle_complete"
    bundle.mkdir()
    schema_by_id = {record["table_id"]: record for record in schemas["tables"]}
    table_records = []
    for table_id, relative_path in EXECUTOR.OUTPUT_TABLE_PATHS.items():
        path = bundle / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [column["name"] for column in schema_by_id[table_id]["columns"]]
        frame = pd.DataFrame([{column: "synthetic" for column in columns}])
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False, lineterminator="\n")
        schema_contract = [
            {
                "name": column["name"],
                "formula": column["formula"],
                "tolerance": column["tolerance"],
            }
            for column in schema_by_id[table_id]["columns"]
        ]
        table_records.append(
            {
                "table_id": table_id,
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": EXECUTOR.sha256_file(path),
                "rows": 1,
                "columns": columns,
                "schema_formula_tolerance_sha256": hashlib.sha256(
                    json.dumps(
                        schema_contract,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "revision0710.external_validation.formal_table_manifest.v8",
        "status": "TWENTY_TABLES_REHASHED_FORMULA_TOLERANCE_CROSSHASH_VALIDATED",
        "table_count": 20,
        "tables": table_records,
    }
    manifest_path = bundle / "FORMAL_TABLE_MANIFEST_V8.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    independent = {
        "schema_version": "revision0710.external_validation.independent_recompute.v8",
        "status": "ALL_TWENTY_TABLES_RECOMPUTED_FROM_SEALED_UPSTREAM_IN_FRESH_PROCESS",
        "fresh_process": True,
        "table_count": 20,
        "tables": [
            {
                "table_id": record["table_id"],
                "formal_table_sha256": record["sha256"],
                "rows": record["rows"],
                "recomputed_from_upstream": True,
            }
            for record in table_records
        ],
    }
    independent_path = bundle / "INDEPENDENT_TWENTY_TABLE_RECOMPUTE_V8.json"
    independent_path.write_text(
        json.dumps(independent, indent=2) + "\n", encoding="utf-8"
    )
    lineage = bundle / "stage_receipts"
    lineage.mkdir()
    environment_sha = EXECUTOR.sha256_file(artifact_dir / "ENVIRONMENT_LOCK_V8.json")
    for phase_name in (
        "01_cache",
        "02_inference",
        "03_prediction_seal",
        "04_outcomes",
        "05_tables",
        "06_independent_recompute",
    ):
        phase = work_stage / phase_name
        phase.mkdir(parents=True)
        outputs = []
        if phase_name in {"05_tables", "06_independent_recompute"}:
            attempt_name = "attempt_0001"
            attempt = phase / "attempts" / attempt_name
            attempt.mkdir(parents=True)
            if phase_name == "05_tables":
                for relative_path in EXECUTOR.OUTPUT_TABLE_PATHS.values():
                    source = bundle / relative_path
                    destination = attempt / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                shutil.copy2(
                    manifest_path, attempt / "FORMAL_TABLE_MANIFEST_V8.json"
                )
                phase_value = {
                    "table_manifest_relative_path": "FORMAL_TABLE_MANIFEST_V8.json",
                    "table_manifest_sha256": EXECUTOR.sha256_file(
                        attempt / "FORMAL_TABLE_MANIFEST_V8.json"
                    ),
                    "table_files": table_records,
                }
            else:
                original_independent = (
                    attempt / "INDEPENDENT_TWENTY_TABLE_RECOMPUTE_V8.json"
                )
                shutil.copy2(independent_path, original_independent)
                phase_value = {
                    "independent_recompute_receipt": str(
                        original_independent.resolve()
                    ),
                    "independent_recompute_receipt_sha256": EXECUTOR.sha256_file(
                        original_independent
                    ),
                    "table_count": 20,
                }
            phase_result = {
                "schema_version": "revision0710.external_validation.phase_result.v8",
                "status": "PHASE_COMPLETE_FROM_VERIFIED_ATTEMPT",
                "stage_name": phase_name,
                "successful_attempt": attempt_name,
                "failed_attempts_preserved": 0,
                "value": phase_value,
            }
            (phase / "PHASE_RESULT_V8.json").write_text(
                json.dumps(phase_result, indent=2) + "\n", encoding="utf-8"
            )
            outputs = sorted(
                [
                    {
                        "relative_path": path.relative_to(phase).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": EXECUTOR.sha256_file(path),
                    }
                    for path in phase.rglob("*")
                    if path.is_file()
                ],
                key=lambda record: record["relative_path"],
            )
        stage_receipt = {
            "schema_version": "revision0710.external_validation.stage_receipt.v8",
            "status": "STAGE_COMPLETE_CREATE_ONCE_REUSABLE",
            "stage_name": phase_name,
            "environment_lock_sha256": environment_sha,
            "input_bindings": [],
            "outputs": outputs,
        }
        original = phase / "STAGE_COMPLETION_V8.json"
        original.write_text(
            json.dumps(stage_receipt, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(original, lineage / f"{phase_name}.json")
    consumed = tmp_path / "authorization.CONSUMED.json"
    consumed.write_text("synthetic\n", encoding="utf-8")
    authorization_sha = "a" * 64
    formal_receipt = {
        "schema_version": "revision0710.external_validation.formal_execution_receipt.v8",
        "status": "FORMAL_EXECUTION_COMPLETE_CREATE_ONCE",
        "authorization_sha256": authorization_sha,
        "consumed_authorization_path": str(consumed.resolve()),
        "formal_once_lock_path": str(EXECUTOR.FORMAL_LOCK.resolve()),
        "contract_sha256": EXECUTOR.sha256_file(
            artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
        ),
        "executor_sha256": EXECUTOR.sha256_file(Path(EXECUTOR.__file__).resolve()),
        "tag": EXECUTOR.FORMAL_TAG,
        "table_count": 20,
        "table_manifest_sha256": EXECUTOR.sha256_file(manifest_path),
        "table_files": table_records,
        "independent_recompute_receipt_sha256": EXECUTOR.sha256_file(
            independent_path
        ),
    }
    (bundle / "FORMAL_EXECUTION_RECEIPT.json").write_text(
        json.dumps(formal_receipt, indent=2) + "\n", encoding="utf-8"
    )
    result = EXECUTOR.validate_completed_publish_bundle(
        bundle,
        work_stage=work_stage,
        authorization_sha256=authorization_sha,
        consumed_authorization=consumed,
    )
    assert result["status"] == "PUBLISH_BUNDLE_DEEP_REHASH_PASS"
    tampered = bundle / next(iter(EXECUTOR.OUTPUT_TABLE_PATHS.values()))
    with tampered.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(EXECUTOR.ExternalExecutionError, match="byte/hash mismatch"):
        EXECUTOR.validate_completed_publish_bundle(
            bundle,
            work_stage=work_stage,
            authorization_sha256=authorization_sha,
            consumed_authorization=consumed,
        )
    original_table = (
        work_stage
        / "05_tables"
        / "attempts"
        / "attempt_0001"
        / next(iter(EXECUTOR.OUTPUT_TABLE_PATHS.values()))
    )
    shutil.copy2(original_table, tampered)

    # Make every bundle-local table/manifest/independent/formal receipt hash
    # internally self-consistent while leaving the successful stage originals
    # unchanged. The cross-stage binding must still reject the bundle.
    table_id = next(iter(EXECUTOR.OUTPUT_TABLE_PATHS))
    relative_path = EXECUTOR.OUTPUT_TABLE_PATHS[table_id]
    coordinated_table = bundle / relative_path
    coordinated_frame = EXECUTOR._read_published_table(coordinated_table)
    coordinated_frame.iloc[0, 0] = "coordinated-tamper"
    if coordinated_table.suffix == ".parquet":
        coordinated_frame.to_parquet(coordinated_table, index=False)
    else:
        coordinated_frame.to_csv(coordinated_table, index=False, lineterminator="\n")
    table_records[0]["bytes"] = coordinated_table.stat().st_size
    table_records[0]["sha256"] = EXECUTOR.sha256_file(coordinated_table)
    manifest["tables"] = table_records
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    independent["tables"][0]["formal_table_sha256"] = table_records[0]["sha256"]
    independent_path.write_text(
        json.dumps(independent, indent=2) + "\n", encoding="utf-8"
    )
    formal_receipt["table_manifest_sha256"] = EXECUTOR.sha256_file(manifest_path)
    formal_receipt["table_files"] = table_records
    formal_receipt["independent_recompute_receipt_sha256"] = EXECUTOR.sha256_file(
        independent_path
    )
    (bundle / "FORMAL_EXECUTION_RECEIPT.json").write_text(
        json.dumps(formal_receipt, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(
        EXECUTOR.ExternalExecutionError,
        match="stage05 successful-attempt original",
    ):
        EXECUTOR.validate_completed_publish_bundle(
            bundle,
            work_stage=work_stage,
            authorization_sha256=authorization_sha,
            consumed_authorization=consumed,
        )


def test_v8_schema_closes_release_age_metrics_survey_and_schoenfeld_fields() -> None:
    schemas = {
        table["table_id"]: table for table in BUILDER.build_output_schemas()["tables"]
    }
    age_key = schemas["zero_shot_age_metrics"]["key"]
    assert age_key[:3] == ["cohort", "release_id", "input_variant"]
    survey_columns = {
        row["name"] for row in schemas["nhanes_questionnaire_associations"]["columns"]
    }
    assert {"full_design_n", "design_df", "reference_distribution"}.issubset(
        survey_columns
    )
    ph_columns = {
        row["name"] for row in schemas["mimic_full_cohort_associations"]["columns"]
    }
    assert {"analysis_population", "n", "baseline_exclusion_rule", "ph_time_transform", "ph_residual_covariance", "ph_n_events"}.issubset(
        ph_columns
    )
    predictive_columns = {
        row["name"] for row in schemas["mimic_predictive_metrics"]["columns"]
    }
    assert {
        "absolute_risk_estimand",
        "ipcw_g_min_raw",
        "ipcw_weight_cap_p99_training",
        "ipcw_effective_sample_size",
        "bootstrap_refits_target_cox_death_cox_and_censor_km",
        "uncertainty_scope",
        "bootstrap_replicates",
    }.issubset(predictive_columns)


@pytest.mark.parametrize("mutation", ["dtype", "cardinality", "enum", "key", "extra"])
def test_output_schema_malicious_mutations_are_rejected(mutation: str) -> None:
    payload = copy.deepcopy(BUILDER.build_output_schemas())
    if mutation == "dtype":
        payload["tables"][0]["columns"][0]["dtype"] = "int64"
    elif mutation == "cardinality":
        payload["tables"][0]["cardinality"] += " mutated"
    elif mutation == "enum":
        payload["tables"][6]["columns"][4]["enum"] = [6, 12, 24]
    elif mutation == "key":
        payload["tables"][0]["key"].remove("input_variant")
    else:
        payload["tables"][0]["columns"][0]["extra"] = False
    with pytest.raises(AssertionError):
        VERIFIER.validate_output_schemas(payload)


def _synthetic_prediction_universe() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    mimic_participants = pd.DataFrame(
        {"subject_id": [1], "chronological_age": [55.0], "sex_male": [1]}
    )
    nhanes_participants = pd.DataFrame(
        {
            "release_id": RUNTIME.RELEASES,
            "participant_id": ["100", "200"],
            "chronological_age": [50.0, 60.0],
            "sex_male": [0, 1],
        }
    )
    per_rows = []
    ensemble_rows = []
    for variant in RUNTIME.MIMIC_VARIANTS:
        for organ in RUNTIME.ORGANS:
            common = {
                "cohort": "MIMIC_IV_3_1",
                "release_id": np.nan,
                "participant_id": "1",
                "input_variant": variant,
                "organ_system": organ,
                "chronological_age": 55.0,
                "sex_male": 1,
            }
            ensemble_rows.append(common)
            for fold in range(5):
                per_rows.append({**common, "checkpoint_outer_fold": fold})
    for participant in nhanes_participants.itertuples(index=False):
        for organ in RUNTIME.ORGANS:
            common = {
                "cohort": "NHANES",
                "release_id": participant.release_id,
                "participant_id": participant.participant_id,
                "input_variant": EXECUTOR.NHANES_PRIMARY_VARIANT,
                "organ_system": organ,
                "chronological_age": participant.chronological_age,
                "sex_male": participant.sex_male,
            }
            ensemble_rows.append(common)
            for fold in range(5):
                per_rows.append({**common, "checkpoint_outer_fold": fold})
    return (
        pd.DataFrame(per_rows),
        pd.DataFrame(ensemble_rows),
        mimic_participants,
        nhanes_participants,
    )


def test_exact_output_key_universe_rejects_silent_row_deletion() -> None:
    expected = {
        (variant, organ)
        for variant in RUNTIME.MIMIC_VARIANTS
        for organ in RUNTIME.ORGANS
    }
    frame = pd.DataFrame(
        sorted(expected), columns=["input_variant", "organ_system"]
    )
    RUNTIME._assert_exact_key_universe(
        frame,
        columns=["input_variant", "organ_system"],
        expected=expected,
        label="synthetic",
    )
    with pytest.raises(AssertionError, match="exact key universe mismatch"):
        RUNTIME._assert_exact_key_universe(
            frame.iloc[:-1].copy(),
            columns=["input_variant", "organ_system"],
            expected=expected,
            label="synthetic",
        )
    source = inspect.getsource(RUNTIME.compute_all_twenty_tables)
    assert "_validate_upstream_output_universes(" in source

    per, ensemble, mimic, nhanes = _synthetic_prediction_universe()
    RUNTIME._validate_prediction_output_universes(
        per,
        ensemble,
        mimic_participants=mimic,
        nhanes_participants=nhanes,
    )
    with pytest.raises(AssertionError):
        RUNTIME._validate_prediction_output_universes(
            per.iloc[:-1].copy(),
            ensemble,
            mimic_participants=mimic,
            nhanes_participants=nhanes,
        )


def test_prediction_universe_rejects_release_and_cross_cohort_variant_swaps() -> None:
    per, ensemble, mimic, nhanes = _synthetic_prediction_universe()
    wrong_release = per.copy()
    nhanes_row = wrong_release.index[wrong_release["cohort"].eq("NHANES")][0]
    wrong_release.loc[nhanes_row, "release_id"] = RUNTIME.RELEASES[1]
    with pytest.raises(AssertionError):
        RUNTIME._validate_prediction_output_universes(
            wrong_release,
            ensemble,
            mimic_participants=mimic,
            nhanes_participants=nhanes,
        )

    wrong_variant = per.copy()
    mimic_row = wrong_variant.index[wrong_variant["cohort"].eq("MIMIC_IV_3_1")][0]
    wrong_variant.loc[mimic_row, "input_variant"] = EXECUTOR.NHANES_PRIMARY_VARIANT
    with pytest.raises(AssertionError, match="non-MIMIC input variant"):
        RUNTIME._validate_prediction_output_universes(
            wrong_variant,
            ensemble,
            mimic_participants=mimic,
            nhanes_participants=nhanes,
        )


@pytest.mark.parametrize("mutation", ["mode", "record_schema", "rejection_policy", "key_order"])
def test_authorization_schema_malicious_mutations_are_rejected(mutation: str) -> None:
    payload = copy.deepcopy(BUILDER.build_formal_authorization_schema())
    if mutation == "mode":
        payload["exact_constants"]["formal_mode"] = "FORMAL_REUSABLE"
    elif mutation == "record_schema":
        payload["record_schemas"]["code_artifact"] = ["name", "sha256"]
    elif mutation == "rejection_policy":
        payload["rejection_policy"].pop()
    else:
        payload["required_top_level_keys_exact_order"] = list(
            reversed(payload["required_top_level_keys_exact_order"])
        )
    with pytest.raises(AssertionError):
        VERIFIER.validate_authorization_schema(payload, ROOT)


def test_formal_cli_hard_fails_without_authorization_before_source_read() -> None:
    assert not EXECUTOR.AUTHORIZATION.exists()
    assert not EXECUTOR.FORMAL_TARGET.exists()
    completed = subprocess.run(
        [sys.executable, str(HERE / "execute_external_validation_v8.py"), "--formal"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert "formal authorization is absent" in completed.stderr


def test_formal_target_authorization_and_staging_presence_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "source_data_v1"
    authorization = tmp_path / "authorization.json"
    BUILDER.validate_formal_targets_absent(target, authorization)
    target.mkdir()
    with pytest.raises(RuntimeError, match="formal target"):
        BUILDER.validate_formal_targets_absent(target, authorization)
    target.rmdir()
    authorization.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="authorization"):
        BUILDER.validate_formal_targets_absent(target, authorization)
    authorization.unlink()
    staging = tmp_path / ".source_data_v1.staging.test"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="staging target"):
        BUILDER.validate_formal_targets_absent(target, authorization)
    staging.rmdir()
    lock = tmp_path / ".EXTERNAL_VALIDATION_FORMAL_V8_ONCE.lock"
    lock.write_text("test\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock/consumed"):
        BUILDER.validate_formal_targets_absent(target, authorization)
    lock.unlink()
    consumed = tmp_path / "authorization.CONSUMED.json"
    consumed.write_text("test\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock/consumed"):
        BUILDER.validate_formal_targets_absent(target, authorization)


def test_prefreeze_assets_and_formal_authorization_closed_mutations_when_present() -> None:
    artifact_dir = BUILDER.ARTIFACT_DIR
    if not artifact_dir.is_dir():
        pytest.skip("prefreeze assets are intentionally generated after test code is final")
    contract = VERIFIER.read_json(artifact_dir / "EXTERNAL_VALIDATION_CONTRACT_V8.json")
    if not (artifact_dir.parent / "INDEPENDENT_GO_REVIEW_V8.json").is_file():
        with pytest.raises(PermissionError, match="independent V8 GO review receipt is absent"):
            VERIFIER.expected_formal_authorization(
                artifact_dir=artifact_dir, repo_root=ROOT, contract=contract
            )
        return
    expected = VERIFIER.expected_formal_authorization(
        artifact_dir=artifact_dir, repo_root=ROOT, contract=contract
    )
    VERIFIER.validate_formal_authorization_payload(
        expected, artifact_dir=artifact_dir, repo_root=ROOT, contract=contract
    )
    mutations = []
    extra = copy.deepcopy(expected)
    extra["extra"] = None
    mutations.append(extra)
    missing = copy.deepcopy(expected)
    missing.pop("scope")
    mutations.append(missing)
    wrong_type = copy.deepcopy(expected)
    wrong_type["formal_publication"]["bootstrap_replicates"] = True
    mutations.append(wrong_type)
    mutations.append(dict(reversed(list(copy.deepcopy(expected).items()))))
    wrong_path = copy.deepcopy(expected)
    wrong_path["contract"]["relative_path"] += ".wrong"
    mutations.append(wrong_path)
    wrong_hash = copy.deepcopy(expected)
    wrong_hash["contract"]["sha256"] = "0" * 64
    mutations.append(wrong_hash)
    wrong_seed = copy.deepcopy(expected)
    wrong_seed["formal_publication"]["bootstrap_seed"] += 1
    mutations.append(wrong_seed)
    wrong_mode = copy.deepcopy(expected)
    wrong_mode["formal_publication"]["mode"] = "FORMAL_REUSABLE"
    mutations.append(wrong_mode)
    wrong_executor = copy.deepcopy(expected)
    wrong_executor["formal_publication"]["executor_name"] = "builder"
    mutations.append(wrong_executor)
    wrong_review = copy.deepcopy(expected)
    wrong_review["review_freeze"]["independent_review_status"] = "PENDING"
    mutations.append(wrong_review)
    reordered_code = copy.deepcopy(expected)
    reordered_code["code_artifacts"] = list(reversed(reordered_code["code_artifacts"]))
    mutations.append(reordered_code)
    for payload in mutations:
        with pytest.raises(AssertionError, match="closed exact mismatch"):
            VERIFIER.validate_formal_authorization_payload(
                payload, artifact_dir=artifact_dir, repo_root=ROOT, contract=contract
            )


def test_prefreeze_registry_rejects_wrong_mimic_hash_and_nested_history_when_present() -> None:
    artifact_dir = BUILDER.ARTIFACT_DIR
    if not artifact_dir.is_dir():
        pytest.skip("prefreeze assets are intentionally generated after test code is final")
    registry = VERIFIER.read_json(artifact_dir / "SOURCE_REGISTRY_V8.json")
    wrong_hash = copy.deepcopy(registry)
    first_mimic = next(
        row for row in wrong_hash["external_sources"] if row["cohort"] == "MIMIC_IV_3_1"
    )
    first_mimic["sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        VERIFIER.validate_registry(wrong_hash, ROOT, rehash_external=False)

    nested = copy.deepcopy(registry)
    first_mimic = next(
        row for row in nested["external_sources"] if row["source_id"] == "mimic_admissions"
    )
    nested_path = BUILDER.MIMIC_ROOT / "mimic-iv-3.1/hosp/admissions.csv.gz"
    if nested_path.is_file() and nested_path.stat().st_size == first_mimic["bytes"]:
        first_mimic["absolute_path"] = str(nested_path.resolve())
        with pytest.raises(AssertionError, match="nested or derived"):
            VERIFIER.validate_registry(nested, ROOT, rehash_external=False)


def test_generated_maps_retain_input_label_isolation_and_all78_when_present() -> None:
    artifact_dir = BUILDER.ARTIFACT_DIR
    if not artifact_dir.is_dir():
        pytest.skip("prefreeze assets are intentionally generated after test code is final")
    registry = VERIFIER.read_json(artifact_dir / "SOURCE_REGISTRY_V8.json")
    result = VERIFIER.validate_maps(artifact_dir, registry)
    assert result["nhanes_feature_rows"] == 156
    assert result["mimic_feature_rows"] == 78
    assert result["nhanes_estimable_label_rows"] == 12


@pytest.mark.parametrize(
    "slot,column,value",
    [
        (3, "primary_source_selector", "result_name == 'Height'"),
        (3, "primary_formula", "value"),
        (11, "primary_source_unit", "kg"),
        (24, "primary_formula", "51133:value; 52769:value; 53157:value"),
        (37, "primary_formula", "value * 0.01"),
        (37, "target_unit", "10^9/L"),
    ],
)
def test_mimic_map_selector_unit_formula_mutations_reject_when_present(
    tmp_path: Path, slot: int, column: str, value: str
) -> None:
    artifact_dir = BUILDER.ARTIFACT_DIR
    if not artifact_dir.is_dir():
        pytest.skip("prefreeze assets are intentionally generated after test code is final")
    mutated_dir = tmp_path / f"map_mutation_{slot}_{column}"
    shutil.copytree(artifact_dir, mutated_dir)
    path = mutated_dir / "MIMIC_78_SLOT_MAP_V8.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[frame["slot_index"].eq(slot), column] = value
    frame.to_csv(path, index=False, lineterminator="\n")
    registry = VERIFIER.read_json(mutated_dir / "SOURCE_REGISTRY_V8.json")
    with pytest.raises(AssertionError):
        VERIFIER.validate_maps(mutated_dir, registry)


def test_density_uses_shared_prepare_dedup_landmark_span_and_packed_mask() -> None:
    raw = _timeline_frame()
    duplicate = raw.iloc[[0]].copy()
    duplicate["source_row_number"] = 9999
    raw = pd.concat([raw.iloc[::-1], duplicate], ignore_index=True)
    icd = raw.loc[raw["source"].eq("hospital_icd")].copy()
    rx = raw.loc[raw["source"].eq("inpatient_prescription_proxy")].copy()
    density = EXTENSIONS.build_label_blind_mimic_density(
        mimic_participants=pd.DataFrame({"subject_id": ["p1"]}),
        icd_events=icd,
        prescription_events=rx,
        phenotype_missing=np.zeros((1, 78), dtype=bool),
        bare_phenotype_missing=np.zeros((1, 78), dtype=bool),
        crop_variant_timeline=EXECUTOR.crop_variant_timeline,
        prepare_variant_timeline=EXECUTOR.prepare_variant_timeline,
    )
    primary = density.loc[
        density["input_variant"].eq(EXECUTOR.MIMIC_PRIMARY_VARIANT)
    ].iloc[0]
    assert primary.pre_crop_event_count == 40
    assert primary.post_crop_event_count == 32
    assert primary.history_span_days == pytest.approx(80.0, abs=0.0)
    assert primary.observed_phenotype_slots == 78
    selected = EXECUTOR.crop_variant_timeline(icd, EXECUTOR.MIMIC_PRIMARY_VARIANT)
    code_to_id = {
        str(value): index + 1
        for index, value in enumerate(selected["canonical_code"].unique())
    }
    text_to_id = {
        ("icd_description", str(value)): index + 1
        for index, value in enumerate(selected["canonical_description"].unique())
    }
    packed = RUNTIME._pack_events(
        selected, ["p1"], code_to_id=code_to_id, text_to_id=text_to_id
    )
    assert int(packed["event_valid_mask"].sum()) == int(primary.post_crop_event_count)


def _v8_external_universe_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = [f"S{fold}" for fold in range(5)]
    density_rows = []
    ensemble_rows = []
    incident_rows = []
    for variant_index, variant in enumerate(RUNTIME.MIMIC_VARIANTS):
        for fold, subject in enumerate(subjects):
            pre_count = 2 + fold * 10 + variant_index
            density_rows.append(
                {
                    "subject_id": subject,
                    "input_variant": variant,
                    "pre_crop_event_count": pre_count,
                    "post_crop_event_count": min(pre_count, 32),
                    "history_span_days": 10.0 + 20.0 * fold + variant_index,
                    "observed_phenotype_slots": 20 + 5 * fold + variant_index,
                    "density_stratum": EXTENSIONS.fixed_density_stratum([pre_count])[0],
                    "density_stratum_rule": "pre_crop_events:low=0-7;medium=8-31;high>=32",
                    "l32_mask_policy": "left_aligned_active_events_then_inactive_padding_to_32",
                    "label_or_outcome_fields_read": False,
                }
            )
        for organ_index, organ in enumerate(RUNTIME.ORGANS):
            for fold, subject in enumerate(subjects):
                age = 45.0 + fold
                gap = float(fold - 2) + organ_index * 0.01
                ensemble_rows.append(
                    {
                        "cohort": "MIMIC_IV_3_1",
                        "release_id": None,
                        "participant_id": subject,
                        "input_variant": variant,
                        "organ_system": organ,
                        "chronological_age": age,
                        "predicted_age_raw_mean5": age + gap,
                        "corrected_age_gap_mean5": gap,
                    }
                )
                for horizon in (6, 12):
                    incident_rows.append(
                        {
                            "subject_id": subject,
                            "input_variant": variant,
                            "organ_system": organ,
                            "horizon_months": horizon,
                            "hash_fold": fold,
                            "chronological_age": age,
                            "sex_male": fold % 2,
                            "event_indicator": 0,
                            "competing_death_indicator": 0,
                            "observed_time_days": float(30 + fold),
                            "censor_reason": "horizon",
                        }
                    )
    return pd.DataFrame(incident_rows), pd.DataFrame(ensemble_rows), pd.DataFrame(density_rows)


def test_density_summary_and_risk_sensitivity_exact_closed_universes() -> None:
    incident, ensemble, density = _v8_external_universe_fixture()
    summary = EXTENSIONS.build_mimic_density_summary(density, ensemble)
    assert len(summary) == 693
    assert summary["analysis"].eq("density_correlation").sum() == 504
    assert summary["analysis"].eq("fixed_density_stratum_age_error").sum() == 189

    def forbidden(*args, **kwargs):
        pytest.fail("zero-target fixture must not enter a fitted risk kernel")

    risk = EXTENSIONS.build_density_adjusted_risk_sensitivity(
        incident=incident,
        ensemble=ensemble,
        density=density,
        bootstrap_replicates=200,
        bootstrap_seed=7102026,
        months_to_days=EXECUTOR.months_to_days,
        cause_specific_cif_predictions=forbidden,
        harrell_c=forbidden,
        ipcw_binary_weights=forbidden,
        weighted_auc=forbidden,
        weighted_brier=forbidden,
        weighted_linear_calibration=forbidden,
        global_subject_bootstrap_plan=RUNTIME._global_subject_bootstrap_plan,
        conditional_bootstrap_values=forbidden,
    )
    assert len(risk) == 2520
    assert set(risk["model"]) == {"age_sex_density", "age_sex_density_corrected_gap"}
    assert risk["comparison_reference_model"].eq("age_sex_density").all()
    assert risk["clinical_utility_claim_allowed"].eq(False).all()


def test_density_scaling_is_fit_on_training_fold_only() -> None:
    training = pd.DataFrame(
        {
            "pre_crop_event_count": [1, 3, 10, 40],
            "history_span_days": [2.0, 20.0, 200.0, 400.0],
            "observed_phenotype_slots": [10, 20, 40, 70],
            "corrected_age_gap_mean5": [-2.0, -1.0, 1.0, 2.0],
        }
    )
    testing = training.iloc[[0]].copy()
    testing.loc[:, "pre_crop_event_count"] = 10**8
    testing.loc[:, "history_span_days"] = 10**9
    testing.loc[:, "observed_phenotype_slots"] = 78
    testing.loc[:, "corrected_age_gap_mean5"] = 10**6
    _, _, first = EXTENSIONS._standardize_density(training, testing)
    testing.loc[:, "pre_crop_event_count"] = 10**12
    testing.loc[:, "corrected_age_gap_mean5"] = -(10**12)
    _, _, second = EXTENSIONS._standardize_density(training, testing)
    assert first == second


def _synthetic_ph_frame(seed: int = 17, n: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.45, size=n)
    event_time = rng.exponential(scale=np.exp(-(0.35 * x1 - 0.25 * x2)))
    censor_time = rng.exponential(scale=2.5, size=n)
    return pd.DataFrame(
        {
            "observed_time_days": np.minimum(event_time, censor_time),
            "event_indicator": (event_time <= censor_time).astype(np.int8),
            "competing_death_indicator": np.zeros(n, dtype=np.int8),
            "x1": x1,
            "x2": x2,
        }
    )


def _ph_common(cause: str = "target") -> dict[str, object]:
    return {
        "analysis_role": "heldout_training_model_diagnostic",
        "input_variant": RUNTIME.MIMIC_VARIANTS[0],
        "organ_system": RUNTIME.ORGANS[0],
        "horizon_months": 6,
        "hash_fold": 0,
        "model": "age_sex",
        "cause": cause,
        "survival_core_relative_path": EXTENSIONS.SURVIVAL_CORE_RELATIVE_PATH,
        "survival_core_sha256": EXTENSIONS.SURVIVAL_CORE_SHA256,
        "ties": "breslow",
        "prediction_fit_backend": "statsmodels_PHReg_efron",
        "diagnostic_fit_backend": "diagnostic_only_byte_bound_shared_core_breslow_refit",
        "diagnostic_beta_or_risk_reused_for_prediction": False,
        "diagnostic_duration_rule": "observed_time_days_plus_1_for_all_rows_preserves_order_and_ties",
    }


def test_ph_per_covariate_parity_with_lifelines_and_single_covariate_global() -> None:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    frame = _synthetic_ph_frame()
    core = EXTENSIONS.load_survival_core(ROOT)
    actual = pd.DataFrame(
        EXTENSIONS._fit_ph_rows(
            core=core,
            frame=frame,
            covariates=["x1", "x2"],
            event_column="event_indicator",
            common=_ph_common(),
        )
    )
    lifelines_frame = frame[["x1", "x2", "event_indicator"]].copy()
    lifelines_frame["duration"] = frame["observed_time_days"] + 1.0
    fit = CoxPHFitter().fit(
        lifelines_frame, duration_col="duration", event_col="event_indicator"
    )
    expected = proportional_hazard_test(
        fit, lifelines_frame, time_transform=["rank", "log"]
    ).summary
    transform_map = {"rank_lifelines_cumsum_event": "rank", "log1p_days": "log"}
    for row in actual.loc[actual["covariate"].ne("GLOBAL")].itertuples(index=False):
        expected_row = expected.loc[(row.covariate, transform_map[row.transform])]
        assert row.chi_square == pytest.approx(float(expected_row["test_statistic"]), rel=3e-5, abs=3e-6)
        assert row.p_value == pytest.approx(float(expected_row["p"]), rel=3e-5, abs=3e-6)

    one = pd.DataFrame(
        EXTENSIONS._fit_ph_rows(
            core=core,
            frame=frame,
            covariates=["x1"],
            event_column="event_indicator",
            common=_ph_common(),
        )
    )
    for transform in EXTENSIONS.PH_TRANSFORMS:
        covariate = one.loc[one["transform"].eq(transform) & one["covariate"].eq("x1")].iloc[0]
        global_row = one.loc[one["transform"].eq(transform) & one["covariate"].eq("GLOBAL")].iloc[0]
        assert global_row.chi_square == pytest.approx(covariate.chi_square, rel=0.0, abs=1e-12)
        assert global_row.p_value == pytest.approx(covariate.p_value, rel=0.0, abs=1e-12)


def test_ph_day0_zero_death_singular_and_strong_global_nonph_are_explicit() -> None:
    core = EXTENSIONS.load_survival_core(ROOT)
    frame = _synthetic_ph_frame(n=500)
    frame.loc[0, "observed_time_days"] = 0.0
    frame.loc[0, "event_indicator"] = 1
    day0 = pd.DataFrame(
        EXTENSIONS._fit_ph_rows(
            core=core,
            frame=frame,
            covariates=["x1", "x2"],
            event_column="event_indicator",
            common=_ph_common(),
        )
    )
    assert day0["estimability_status"].eq("estimable").all()
    assert set(day0["transform"]) == set(EXTENSIONS.PH_TRANSFORMS)

    zero_death = pd.DataFrame(
        EXTENSIONS._fit_ph_rows(
            core=core,
            frame=frame,
            covariates=["x1", "x2"],
            event_column="competing_death_indicator",
            common=_ph_common("death"),
        )
    )
    assert zero_death["estimability_status"].eq("zero_competing_deaths").all()
    assert zero_death["df"].isna().all()
    assert len(zero_death) == 6

    singular = frame.copy()
    singular["x2"] = 1.0
    failed = pd.DataFrame(
        EXTENSIONS._fit_ph_rows(
            core=core,
            frame=singular,
            covariates=["x1", "x2"],
            event_column="event_indicator",
            common=_ph_common(),
        )
    )
    assert len(failed) == 6
    assert failed["failure_code"].eq("nonfinite_or_zero_variance_covariate").all()
    assert failed["df"].isna().all()

    rng = np.random.default_rng(31)
    n = 1600
    x = rng.integers(0, 2, size=n).astype(float)
    uniform = rng.uniform(size=n)
    event_time = np.where(
        x == 0,
        (-np.log(uniform)) ** (1.0 / 0.4),
        (-np.log(uniform)) ** (1.0 / 3.0),
    )
    censor = np.full(n, 4.0)
    nonph = pd.DataFrame(
        {
            "observed_time_days": np.minimum(event_time, censor),
            "event_indicator": (event_time <= censor).astype(np.int8),
            "x": x,
        }
    )
    strong = pd.DataFrame(
        EXTENSIONS._fit_ph_rows(
            core=core,
            frame=nonph,
            covariates=["x"],
            event_column="event_indicator",
            common=_ph_common(),
        )
    )
    global_rows = strong.loc[strong["covariate"].eq("GLOBAL")]
    assert global_rows["estimability_status"].eq("estimable").all()
    assert global_rows["p_value"].max() < 1e-6


def test_complete_ph_exact_17136_key_universe_and_no_prediction_mutation() -> None:
    incident, ensemble, density = _v8_external_universe_fixture()
    before = ensemble.copy(deep=True)
    diagnostics = EXTENSIONS.build_complete_mimic_ph_diagnostics(
        root=ROOT, incident=incident, ensemble=ensemble, density=density
    )
    assert len(diagnostics) == 17136
    assert diagnostics["analysis_role"].eq("full_nonheldout_incident_eligible_association").sum() == 336
    assert diagnostics["analysis_role"].eq("heldout_training_model_diagnostic").sum() == 16800
    key = ["analysis_role", "input_variant", "organ_system", "horizon_months", "hash_fold", "model", "cause", "covariate", "transform"]
    assert not diagnostics.duplicated(key).any()
    assert diagnostics["diagnostic_beta_or_risk_reused_for_prediction"].eq(False).all()
    pd.testing.assert_frame_equal(ensemble, before, check_exact=True)


def test_nhanes_release_slot_qc_is_fixed_descriptive_and_fail_closed() -> None:
    mapping = pd.DataFrame(
        [
            {"release_id": release, "slot_index": slot, "feature_name": f"slot_{slot}", "target_unit": "ratio"}
            for release in EXTENSIONS.RELEASES
            for slot in range(78)
        ]
    )
    participants = pd.DataFrame(
        {"release_id": list(EXTENSIONS.RELEASES), "participant_id": ["P", "L"]}
    )
    raw = np.zeros((2, 78), dtype=np.float32)
    missing = np.ones((2, 78), dtype=bool)
    raw[:, 0] = [0.5, 0.7]
    missing[:, 0] = False
    qc = EXTENSIONS.build_nhanes_release_slot_qc(
        participants=participants, raw=raw, missing=missing, release_slot_map=mapping
    )
    assert len(qc) == 156
    assert qc["trimming_or_winsorization"].eq("prohibited").all()
    assert qc["quantiles_used_for_decision"].eq(False).all()
    EXTENSIONS.require_nhanes_qc_pass(qc)
    raw[0, 0] = 20.0
    failed = EXTENSIONS.build_nhanes_release_slot_qc(
        participants=participants, raw=raw, missing=missing, release_slot_map=mapping
    )
    with pytest.raises(AssertionError, match="fixed QC failed"):
        EXTENSIONS.require_nhanes_qc_pass(failed)
