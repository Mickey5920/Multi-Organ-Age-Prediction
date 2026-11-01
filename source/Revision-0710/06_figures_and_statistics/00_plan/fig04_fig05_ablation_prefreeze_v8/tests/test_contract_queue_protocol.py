from __future__ import annotations

import copy
from pathlib import Path

import pytest

import v8_contract as contract
import v8_inputs
import v8_loss
import v8_model


ROOT = Path(__file__).resolve().parents[1]


def test_v1_is_hash_bound_and_unchanged() -> None:
    payload = contract.load_contract(ROOT)
    observed = contract.verify_v1_immutable_bindings(payload, ROOT)
    expected_fixed = {
        "contract": "a1dbde3c0b684b45bdab75dc6ea7695a8bf854a9bc24e3ef3a2918e856bd602c",
        "registry": "1b4ec0e3f15bafdfcd103af68a2e54a3d2c1bdaff20ccc768a49b0831c18f574",
        "queue": "aa6ae492a56a0782b104591c6c145c017f2d5d5ad68edc515238f7050c734bc2",
        "receipt": "b288b164f2a94e81130db96041813ac016576b1444d7582cb293a60003994a74",
    }
    assert {key: observed[key] for key in expected_fixed} == expected_fixed
    runtime = contract.preflight_v1_runtime_bindings(ROOT)
    assert set(runtime) == {"ablation_model", "ablation_loss", "ablation_inputs"}
    assert all(observed[key] == value for key, value in runtime.items())


def test_contract_closes_every_production_protocol_field() -> None:
    result = contract.validate_prefreeze_bundle(ROOT, verify_hashes=True)
    assert result["status"] == "PASS"
    assert result["queue_rows"] == 60
    payload = contract.load_contract(ROOT)
    training = payload["training_protocol"]
    assert training == {
        "seed": 7210,
        "batch_size": 2048,
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": 3,
        "scheduler": "linear_warmup_then_cosine_decay",
        "scheduler_horizon_epochs": 60,
        "train_through_epoch": 60,
        "capture_epoch": 55,
        "evaluate_only_epoch": 55,
        "deterministic": True,
        "tf32": True,
        "amp": True,
        "restart_from_epoch_one_same_seed": True,
        "resume_training_state": False,
        "fresh_random_initialization_all_runs": True,
        "fold0_checkpoint_reuse": False,
        "warm_start": False,
        "fine_tune": False,
        "teacher": False,
        "distillation": False,
        "ensemble": False,
        "seed_retry": False,
        "performance_selection": False,
        "fold_drop": False,
    }
    roles = payload["role_protocol"]
    assert roles["training_worker_allowed_roles"] == ["train"]
    assert roles["training_worker_calibration_or_outer_paths_allowed"] is False
    assert roles["calibration_open_condition"] == "E60_COMPLETE_AND_E55_SEALED"
    assert roles["calibration_model_selection_allowed"] is False
    assert roles["outer_open_condition"] == "CALIBRATION_SEALED_AND_LEDGER_COMMITTED"
    assert roles["outer_refit_allowed"] is False
    evaluation = payload["evaluation_protocol"]
    assert evaluation["risk_metric"] == "age+sex Harrell delta C-index"
    assert evaluation["organs"] == list(contract.ORGAN_SYSTEMS)
    assert evaluation["organ_count"] == 7


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("training_protocol", "batch_size", 1024),
        ("training_protocol", "seed", 7211),
        ("training_protocol", "train_through_epoch", 55),
        ("training_protocol", "capture_epoch", 54),
        ("training_protocol", "learning_rate", 0.001),
        ("training_protocol", "weight_decay", 0.0),
        ("training_protocol", "gradient_clip_norm", 2.0),
        ("training_protocol", "warmup_epochs", 0),
        ("training_protocol", "scheduler_horizon_epochs", 55),
        ("training_protocol", "deterministic", False),
        ("training_protocol", "tf32", False),
        ("training_protocol", "amp", False),
        ("training_protocol", "resume_training_state", True),
        ("role_protocol", "training_worker_calibration_or_outer_paths_allowed", True),
        ("role_protocol", "calibration_model_selection_allowed", True),
        ("role_protocol", "outer_refit_allowed", True),
        ("evaluation_protocol", "risk_metric", "AUROC"),
        ("evaluation_protocol", "organ_count", 6),
    ],
)
def test_protocol_mutations_fail_closed(section: str, field: str, bad_value: object) -> None:
    payload = contract.load_contract(ROOT)
    broken = copy.deepcopy(payload)
    broken[section][field] = bad_value
    with pytest.raises(contract.ContractError, match=field):
        contract.validate_contract_payload(broken, ROOT, verify_hashes=False)


@pytest.mark.parametrize("section", ("role_protocol", "evaluation_protocol"))
def test_nested_protocol_extra_keys_fail_closed(section: str) -> None:
    payload = contract.load_contract(ROOT)
    broken = copy.deepcopy(payload)
    broken[section]["unreviewed_extra"] = True
    with pytest.raises(contract.ContractError, match="keys"):
        contract.validate_contract_payload(broken, ROOT, verify_hashes=False)


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("scope", "formal_training_executed", True),
        ("environment", "formal", {}),
        ("execution_roots", "formal_run_root", str(ROOT / "wrong")),
        ("variant_protocol", "unknown_variant_policy", "IGNORE"),
        ("state_machine", "training_resume_allowed", True),
        ("authorization_protocol", "partial_authorization_allowed", True),
        ("resource_protocol", "max_concurrent_training_runs", 2),
        ("publication_firewall", "formal_training_allowed", True),
        ("scaled_integration_protocol", "formal_gpu_allowed", True),
    ],
)
def test_all_remaining_contract_sections_fail_closed(
    section: str, field: str, bad_value: object
) -> None:
    payload = contract.load_contract(ROOT)
    broken = copy.deepcopy(payload)
    broken[section][field] = bad_value
    with pytest.raises(contract.ContractError):
        contract.validate_contract_payload(broken, ROOT, verify_hashes=False)


def test_binding_sets_reject_drop_duplicate_and_rename() -> None:
    payload = contract.load_contract(ROOT)
    for mutation in ("drop", "duplicate", "rename"):
        broken = copy.deepcopy(payload)
        records = broken["artifact_bindings"]["records"]
        if mutation == "drop":
            records.pop()
        elif mutation == "duplicate":
            records.append(copy.deepcopy(records[0]))
        else:
            records[0]["name"] = "renamed"
        with pytest.raises(contract.ContractError):
            contract.validate_contract_payload(broken, ROOT, verify_hashes=False)


def test_queue_is_exact_cartesian_product_with_formula_and_frozen_root() -> None:
    payload = contract.load_contract(ROOT)
    rows = contract.read_queue(ROOT / "ABLATION_QUEUE_V8.csv")
    contract.validate_queue(rows, payload)
    assert len(rows) == 60
    assert len({row["run_id"] for row in rows}) == 60
    root = Path(payload["execution_roots"]["formal_run_root"]).resolve()
    for row in rows:
        assert row["run_id"] == f"{row['variant_id']}__outer{row['outer_fold']}"
        intended = Path(row["intended_run_dir"]).resolve()
        intended.relative_to(root)
        assert intended == root / row["variant_id"] / f"outer_fold_{row['outer_fold']}"


@pytest.mark.parametrize(
    "mutation",
    ("run_id", "variant", "fold", "duplicate_path", "escape_root", "missing_row"),
)
def test_queue_mutations_fail_closed(mutation: str) -> None:
    payload = contract.load_contract(ROOT)
    rows = contract.read_queue(ROOT / "ABLATION_QUEUE_V8.csv")
    broken = copy.deepcopy(rows)
    if mutation == "run_id":
        broken[0]["run_id"] = "wrong"
    elif mutation == "variant":
        broken[0]["variant_id"] = broken[5]["variant_id"]
    elif mutation == "fold":
        broken[0]["outer_fold"] = 4
    elif mutation == "duplicate_path":
        broken[1]["intended_run_dir"] = broken[0]["intended_run_dir"]
    elif mutation == "escape_root":
        broken[0]["intended_run_dir"] = str(ROOT.parent / "escaped")
    else:
        broken.pop()
    with pytest.raises(contract.ContractError):
        contract.validate_queue(broken, payload)


@pytest.mark.parametrize("variant_id", ("", "unknown", "fig04_99_typo"))
def test_all_production_helpers_reject_unknown_variants(variant_id: str) -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        v8_inputs.variant_input_policy(variant_id)
    with pytest.raises(ValueError, match="unknown variant"):
        v8_loss.variant_loss_policy(variant_id)
    with pytest.raises(ValueError, match="unknown variant"):
        v8_model.variant_model_policy(variant_id)


def test_resource_budget_is_fixed_and_cannot_change_protocol() -> None:
    budget = contract.read_json_exact(ROOT / "RESOURCE_BUDGET_AND_PREFLIGHT_V8.json")
    assert budget["single_gpu_policy"]["max_concurrent_training_runs"] == 1
    assert budget["single_gpu_policy"]["gpu_model"] == "NVIDIA GeForce RTX 4090"
    assert budget["estimate"]["runs"] == 60
    assert budget["estimate"]["estimated_total_gpu_hours"] > 40
    assert budget["budget_may_change_epochs_batch_or_variants"] is False
    assert budget["micro_benchmark"]["authorized"] is False
    assert budget["micro_benchmark"]["heldout_roles_allowed"] == []
