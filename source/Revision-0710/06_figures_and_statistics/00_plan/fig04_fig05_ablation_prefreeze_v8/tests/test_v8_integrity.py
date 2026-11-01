from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import formal_executor
import v8_contract
import v8_data
import v8_schema_validator as schemas


ROOT = Path(__file__).resolve().parents[1]


def test_independent_replayers_do_not_import_production_deep_validator() -> None:
    for name in (
        "verify_prefreeze_v8_independent.py",
        "verify_formal_attempt_v8_independent.py",
        "v8_independent_replay.py",
    ):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "import v8_schema_validator" not in source
        assert "validate_scaled_attempt_bundle" not in source
        assert "validate_formal_attempt_bundle" not in source


def _sealed_nuisance() -> dict:
    models = []
    for organ in v8_contract.ORGAN_SYSTEMS:
        for model_name, covariates in (
            ("base", ["actual_age", "sex"]),
            ("full", ["actual_age", "sex", "corrected_age_gap"]),
        ):
            models.append(
                {
                    "organ_system": organ,
                    "model": model_name,
                    "covariates": covariates,
                    "means": {name: 0.0 for name in covariates},
                    "scales": {name: 1.0 for name in covariates},
                    "coefficients": {
                        name: {"actual_age": 0.03, "sex": 0.2, "corrected_age_gap": 0.7}[name]
                        for name in covariates
                    },
                    "fit_participants": 10,
                    "fit_events": 4,
                    "fit_participant_ids_sha256": "d" * 64,
                    "convergence_warning": False,
                }
            )
    return {"models": models}


def _reported_metric_frames(prediction: pd.DataFrame, nuisance: dict):
    fold, organs = schemas.independently_recompute_outer_metrics(prediction, nuisance)
    fold_frame = pd.DataFrame([fold])
    organ_frame = pd.DataFrame(organs)
    return fold_frame, organ_frame


def test_metrics_are_recomputed_from_prediction_and_sealed_nuisance() -> None:
    prediction = schemas.synthetic_prediction_frame(
        run_id=v8_contract.expected_run_ids()[0], participants=12
    )
    nuisance = _sealed_nuisance()
    fold, organs = _reported_metric_frames(prediction, nuisance)
    schemas.validate_independently_recomputed_metrics(
        prediction, nuisance, fold, organs
    )


def test_malicious_999_and_self_consistent_fake_c_are_rejected() -> None:
    prediction = schemas.synthetic_prediction_frame(
        run_id=v8_contract.expected_run_ids()[0], participants=12
    )
    nuisance = _sealed_nuisance()
    fold, organs = _reported_metric_frames(prediction, nuisance)
    malicious_fold = fold.copy()
    malicious_fold.loc[0, "raw_strict_mae"] = 999.0
    with pytest.raises(schemas.SchemaError, match="raw_strict_mae"):
        schemas.validate_independently_recomputed_metrics(
            prediction, nuisance, malicious_fold, organs
        )
    fake_c = organs.copy()
    fake_c["age_sex_harrell_base_c"] = 0.5
    fake_c["age_sex_gap_harrell_full_c"] = 0.5
    fake_c["age_sex_harrell_delta_c"] = 0.0
    fake_fold = fold.copy()
    fake_fold.loc[0, "age_sex_harrell_macro_delta_c"] = 0.0
    with pytest.raises(schemas.SchemaError, match="organ metric mismatch"):
        schemas.validate_independently_recomputed_metrics(
            prediction, nuisance, fake_fold, fake_c
        )


def test_numeric_participant_ids_roundtrip_as_exact_strings(tmp_path: Path) -> None:
    run_id = v8_contract.expected_run_ids()[0]
    frame = schemas.synthetic_prediction_frame(run_id=run_id, participants=3)
    identifiers = ("0000001", "1000002", "9007199254740993")
    mapping = {f"p{index:03d}": value for index, value in enumerate(identifiers)}
    frame["participant_id"] = frame["participant_id"].map(mapping)
    path = tmp_path / "numeric_participant_ids.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    schema = schemas.schema_for_execution(
        schemas.load_schemas(ROOT / "OUTPUT_SCHEMAS_V8.json"),
        "prediction_rows",
        execution_mode="formal",
        evaluation_epoch=55,
    )
    observed = schemas.read_csv_exact(path, schema, table="prediction_rows")
    assert tuple(observed["participant_id"].drop_duplicates().tolist()) == identifiers
    assert str(observed["participant_id"].dtype) == "string"


def test_cache_contract_binds_manifest_all_shards_and_all_transforms() -> None:
    contract = v8_contract.load_contract(ROOT)
    bindings = contract["cache_artifact_bindings"]
    v8_contract.validate_cache_artifact_bindings(bindings, ROOT, verify_hashes=True)
    assert len(bindings["shards"]) == 60
    assert len(bindings["fold_transforms"]) == 5
    assert [row["ordinal"] for row in bindings["shards"]] == list(range(60))
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in bindings["shards"])
    broken = copy.deepcopy(bindings)
    broken["shards"][0], broken["shards"][1] = broken["shards"][1], broken["shards"][0]
    with pytest.raises(v8_contract.ContractError, match="order"):
        v8_contract.validate_cache_artifact_bindings(broken, ROOT, verify_hashes=False)


def test_real_fold0_loader_rehashes_order_schema_content_and_numeric_ids() -> None:
    contract = v8_contract.load_contract(ROOT)
    cache_dir = (
        v8_contract.repo_root(ROOT) / v8_contract.CACHE_RELATIVE_ROOT
    )
    order = v8_data._validate_cache_files_for_load(
        cache_dir=cache_dir,
        bindings=contract["cache_artifact_bindings"],
        outer_fold=0,
        fold_ids=(0,),
    )
    expected_n = sum(
        row["n"]
        for row in contract["cache_artifact_bindings"]["shards"]
        if row["fold_id"] == 0
    )
    assert len(order) == expected_n == len(set(order))
    assert all(value.isdigit() and value == str(value) for value in order[:1000])
    broken = copy.deepcopy(contract["cache_artifact_bindings"])
    broken["shards"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content binding|bytes/SHA"):
        v8_data._validate_cache_files_for_load(
            cache_dir=cache_dir,
            bindings=broken,
            outer_fold=0,
            fold_ids=(0,),
        )


def test_formal_aggregate_executes_complete_schema_and_artifact_index_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    contract = copy.deepcopy(v8_contract.load_contract(ROOT))
    contract["execution_roots"]["formal_run_root"] = str(formal_root.resolve())
    rows = []
    for run_id in v8_contract.expected_run_ids():
        run_root = tmp_path / "runs" / run_id
        attempt = run_root / "attempt_0001"
        attempt.mkdir(parents=True)
        (attempt / "COMPLETE.json").write_text(
            json.dumps(
                {
                    "schema_version": "revision0710.fig04_fig05.attempt_complete.v8",
                    "status": "COMPLETE",
                    "run_id": run_id,
                    "seed": 7210,
                    "attempt_ordinal": 1,
                    "performance_metric_used_for_selection": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append({"run_id": run_id, "intended_run_dir": str(run_root.resolve())})

    monkeypatch.setattr(
        formal_executor,
        "authorize_formal_execution",
        lambda root, authorization_path: {
            "authorized_run_ids": list(v8_contract.expected_run_ids()),
            "formal_run_root": str(formal_root.resolve()),
        },
    )
    monkeypatch.setattr(formal_executor, "_activate_authorization", lambda *args: None)
    monkeypatch.setattr(formal_executor.v8_contract, "load_contract", lambda root: contract)
    monkeypatch.setattr(formal_executor.v8_contract, "read_queue", lambda path: rows)

    def fake_validate(attempt: Path, **kwargs):
        records = [
            {
                "name": f"artifact_{index:02d}",
                "path": f"artifact_{index:02d}",
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index in range(22)
        ]
        return {
            "fold_receipt_sha256": "f" * 64,
            "artifact_count": 22,
            "artifact_set_sha256": v8_contract.canonical_json_sha256(records),
            "artifacts": records,
        }

    monkeypatch.setattr(
        formal_executor.v8_schema_validator,
        "validate_formal_attempt_bundle",
        fake_validate,
    )
    monkeypatch.setattr(
        formal_executor,
        "_attempt_access_audit_records",
        lambda attempt: [
            {"phase": phase, "path": str(attempt / phase), "bytes": 1, "sha256": "a" * 64}
            for phase in ("train", "calibration", "outer_test")
        ],
    )
    monkeypatch.setattr(
        formal_executor,
        "_persistent_independent_replay",
        lambda **kwargs: {
            "schema_version": "revision0710.fig04_fig05.persistent_independent_replay.v8",
            "run_id": kwargs["run_id"],
            "path": str(formal_root / "00_independent_replays" / f"{kwargs['run_id']}.json"),
            "bytes": 1,
            "sha256": "b" * 64,
            "attempt_binding": {},
            "independent_replay_sha256": "c" * 64,
            "independent_replay": {
                "status": "PASS_SEED7210_FULL_TRAINING_AND_VALIDATION_OUTER_REPLAY",
                "run_id": kwargs["run_id"],
                "production_deep_validator_imported": False,
            },
        },
    )
    result = formal_executor._execute_formal_aggregate_under_lock(
        ROOT, authorization_path=tmp_path / "FAKE_NOT_FORMAL_AUTH.json"
    )
    aggregate_path = Path(result["global_aggregate_path"])
    index_path = Path(result["artifact_index_path"])
    assert aggregate_path.is_file() and index_path.is_file()
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["artifact_count"] == 1320
    assert aggregate["artifacts_per_run"] == 22
    assert aggregate["artifact_index_sha256"] == v8_contract.sha256_file(index_path)
    independent_path = Path(result["independent_replay_index_path"])
    assert aggregate["independent_replay_index_sha256"] == v8_contract.sha256_file(
        independent_path
    )
    assert aggregate["independent_training_replays_complete"] is True
