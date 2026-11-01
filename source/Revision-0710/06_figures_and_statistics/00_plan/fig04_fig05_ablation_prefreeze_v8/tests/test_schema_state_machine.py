from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import v8_contract as contract
import v8_schema_validator as schemas
import v8_state_machine as state


ROOT = Path(__file__).resolve().parents[1]


def test_output_schema_is_executable_and_closes_required_tables() -> None:
    payload = schemas.load_schemas(ROOT / "OUTPUT_SCHEMAS_V8.json")
    assert set(payload["tables"]) == {
        "checkpoint_manifest",
        "prediction_rows",
        "age_correction",
        "cox_nuisance",
        "outer_read_ledger",
        "fold_metrics",
        "organ_metrics",
        "fold_receipt",
        "attempt_receipt",
        "global_aggregate",
    }
    prediction = payload["tables"]["prediction_rows"]
    assert prediction["cardinality"]["rows_per_participant"] == 7
    assert prediction["constraints"]["evaluation_epoch"]["const"] == 55
    fold = payload["tables"]["fold_metrics"]
    assert "strict_n_participants" in fold["columns"]
    assert "strict_n_participant_organ_rows" in fold["columns"]
    assert payload["tables"]["cox_nuisance"]["fields"]["penalizer"]["const"] == 1e-4
    assert {
        "opened_roles",
        "opened_paths",
        "outer_refit_calls",
    }.issubset(payload["tables"]["fold_receipt"]["fields"])


def test_execution_schema_only_relaxes_fixed_scaled_capture_constants() -> None:
    payload = schemas.load_schemas(ROOT / "OUTPUT_SCHEMAS_V8.json")
    formal = schemas.schema_for_execution(
        payload, "prediction_rows", execution_mode="formal", evaluation_epoch=55
    )
    assert formal["columns"]["evaluation_epoch"]["const"] == 55
    scaled = schemas.schema_for_execution(
        payload,
        "checkpoint_manifest",
        execution_mode="scaled_integration",
        evaluation_epoch=1,
    )
    assert scaled["fields"]["evaluation_epoch"]["const"] == 1
    assert scaled["fields"]["status"]["const"] == "SEALED_SCALED_CAPTURE_ONLY"
    with pytest.raises(schemas.SchemaError, match="E55"):
        schemas.schema_for_execution(
            payload, "prediction_rows", execution_mode="formal", evaluation_epoch=1
        )


def test_attempt_receipt_requires_exactly_one_status_hash() -> None:
    payload = schemas.load_schemas(ROOT / "OUTPUT_SCHEMAS_V8.json")
    schema = payload["tables"]["attempt_receipt"]
    receipt = {
        "schema_version": "revision0710.fig04_fig05.attempt_receipt.v8",
        "status": "COMPLETE",
        "run_id": contract.expected_run_ids()[0],
        "attempt_ordinal": 1,
        "seed": 7210,
        "restart_epoch": 1,
        "training_resume_used": False,
        "fold_receipt_sha256": "a" * 64,
        "failure_receipt_sha256": None,
        "selection_uses_metrics": False,
    }
    schemas.validate_json_payload(receipt, schema, table="attempt_receipt")
    broken = copy.deepcopy(receipt)
    broken["failure_receipt_sha256"] = "b" * 64
    with pytest.raises(schemas.SchemaError, match="failure"):
        schemas.validate_json_payload(broken, schema, table="attempt_receipt")


def test_closed_table_validator_checks_dtype_null_key_enum_formula_cardinality() -> None:
    schema = schemas.load_schemas(ROOT / "OUTPUT_SCHEMAS_V8.json")
    frame = schemas.synthetic_prediction_frame(run_id=contract.expected_run_ids()[0], participants=3)
    schemas.validate_table(frame, schema["tables"]["prediction_rows"], table="prediction_rows")
    mutations = []
    bad = frame.copy(); bad["outer_fold"] = bad["outer_fold"].astype(str); mutations.append(bad)
    bad = frame.copy(); bad.loc[0, "predicted_age"] = None; mutations.append(bad)
    bad = pd.concat([frame, frame.iloc[[0]]], ignore_index=True); mutations.append(bad)
    bad = frame.copy(); bad.loc[0, "split_role"] = "train"; mutations.append(bad)
    bad = frame.copy(); bad.loc[0, "raw_age_gap"] += 1.0; mutations.append(bad)
    bad = frame.iloc[:-1].copy(); mutations.append(bad)
    for bad in mutations:
        with pytest.raises(schemas.SchemaError):
            schemas.validate_table(bad, schema["tables"]["prediction_rows"], table="prediction_rows")
    with pytest.raises(schemas.SchemaError, match="minimum row"):
        schemas.validate_table(
            frame.iloc[0:0].copy(),
            schema["tables"]["prediction_rows"],
            table="prediction_rows",
        )


def test_bundle_cross_hash_validator_rejects_any_broken_link(tmp_path: Path) -> None:
    bundle = schemas.write_synthetic_run_bundle(tmp_path, contract.expected_run_ids()[0])
    schemas.validate_run_bundle(tmp_path, bundle)
    broken = copy.deepcopy(bundle)
    broken["fold_receipt"]["prediction_rows_sha256"] = "0" * 64
    with pytest.raises(schemas.SchemaError, match="prediction"):
        schemas.validate_run_bundle(tmp_path, broken)


def test_state_machine_is_create_only_and_selects_lowest_valid_attempt(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    first = state.claim_new_attempt(run_root, run_id="v__outer0", seed=7210)
    assert first.name == "attempt_0001"
    with pytest.raises(state.StateError):
        state.claim_specific_attempt(run_root, ordinal=1, run_id="v__outer0", seed=7210)
    state.write_failure_receipt(first, phase="train", reason="synthetic interruption")
    second = state.claim_new_attempt(run_root, run_id="v__outer0", seed=7210)
    assert second.name == "attempt_0002"
    claim = json.loads((second / "CLAIM.json").read_text(encoding="utf-8"))
    assert claim["restart_epoch"] == 1
    assert claim["seed"] == 7210
    second_fold = second / "FOLD_RECEIPT.json"
    second_fold.write_text("{}\n", encoding="utf-8")
    state.write_completion_receipt(
        second, run_id="v__outer0", seed=7210, fold_receipt_path=second_fold
    )
    third = state.claim_new_attempt(run_root, run_id="v__outer0", seed=7210)
    third_fold = third / "FOLD_RECEIPT.json"
    third_fold.write_text("{}\n", encoding="utf-8")
    state.write_completion_receipt(
        third, run_id="v__outer0", seed=7210, fold_receipt_path=third_fold
    )
    selected = state.select_completed_attempt(run_root)
    assert selected.name == "attempt_0002"


def test_resume_boundaries_and_complete_set_gate(tmp_path: Path) -> None:
    assert state.training_resume_allowed(completed_epochs=1, target_epochs=60) is False
    assert state.evaluation_resume_allowed(training_complete=True, e55_sealed=True) is True
    assert state.evaluation_resume_allowed(training_complete=False, e55_sealed=True) is False
    receipts = {run_id: "a" * 64 for run_id in contract.expected_run_ids()}
    state.validate_complete_set_for_aggregate(receipts)
    receipts.pop(next(iter(receipts)))
    with pytest.raises(state.StateError, match="60/60"):
        state.validate_complete_set_for_aggregate(receipts)
