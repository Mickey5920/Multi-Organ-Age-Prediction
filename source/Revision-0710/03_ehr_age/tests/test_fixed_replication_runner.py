from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER_DIR = Path(__file__).resolve().parents[1] / "02_runner"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import fixed_replication_runner as runner  # noqa: E402
import fixed_replication_train_worker as train_worker  # noqa: E402
import run_fixed_replication as cli  # noqa: E402
from crossfit_metrics import (  # noqa: E402
    apply_linear_age_correction,
    apply_linear_age_correction_to_validation,
    evaluate_cox_cell,
    fit_linear_age_correction,
)


def _survival_frame(prefix: str, n: int, age_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for organ_index, organ in enumerate(runner.ORGAN_SYSTEMS):
        for index in range(n):
            age = 42.0 + (index % 29) + age_shift
            prediction = age + 0.08 * (age - 56.0) + 0.5 * np.sin(index / 3 + organ_index)
            rows.append(
                {
                    "participant_id": f"{prefix}{index:04d}",
                    "organ_system": organ,
                    "actual_age": age,
                    "predicted_age": prediction,
                    "strict_healthy_reference_60m": int(index % 3 != 0),
                    "sex": float(index % 2),
                    "duration": float(12 + (index * 7 + organ_index) % 48),
                    "event": int(index % 5 == organ_index % 5),
                    "at_risk": 1,
                    "outer_fold": 2,
                    "split_role": "validation" if prefix == "v" else "test",
                }
            )
    return pd.DataFrame(rows)


def test_cyclic_roles_are_exact_and_disjoint() -> None:
    expected = {
        0: (1, (2, 3, 4)),
        1: (2, (0, 3, 4)),
        2: (3, (0, 1, 4)),
        3: (4, (0, 1, 2)),
        4: (0, (1, 2, 3)),
    }
    for outer, (calibration, train) in expected.items():
        roles = runner.cyclic_fold_roles(outer)
        assert roles.calibration_fold == calibration
        assert roles.train_folds == train
        assert set(train).isdisjoint({outer, calibration})
        assert set(train) | {outer, calibration} == set(range(5))


def test_contract_role_parser_rejects_duplicate_outer() -> None:
    rows = [
        {
            "outer_fold": outer,
            "calibration_fold": (outer + 1) % 5,
            "train_folds": list(runner.cyclic_fold_roles(outer).train_folds),
            "mode": "reuse_d07_003_e55" if outer == 0 else "train_from_scratch",
            "seed": 7210,
        }
        for outer in range(5)
    ]
    rows[-1] = dict(rows[0])
    with pytest.raises(runner.FixedReplicationError, match="duplicate|exactly five"):
        runner._contract_roles({"roles": rows})


def test_train_worker_does_not_import_master_or_monolithic_runner() -> None:
    source = Path(train_worker.__file__).read_text(encoding="utf-8")
    assert "import fixed_replication_runner" not in source
    assert "load_replication_contract" not in source
    assert "DEFAULT_OUTCOMES" not in source
    assert "ROLE_ARTIFACT_REGISTRY" not in source


def test_cli_spawns_isolated_training_worker() -> None:
    source = inspect.getsource(cli._run_fold)
    assert "subprocess.run" in source
    assert "TRAIN_WORKER" in source
    assert "train_fixed_fold_worker" not in source


def test_cli_requires_expected_contract_sha_for_every_subcommand() -> None:
    parser = cli.build_parser()
    for command in (
        "build-role-artifacts",
        "validate-only",
        "run-fold",
        "run-sequence",
        "select-receipts",
        "aggregate-5fold",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([command])


def test_fixed_training_constants_are_not_metric_selected() -> None:
    source = Path(train_worker.__file__).read_text(encoding="utf-8")
    assert train_worker.FIXED_EPOCH == 55
    assert train_worker.MAX_EPOCHS == 60
    assert train_worker.SEED == 7210
    assert "early_stopping" not in source.lower()
    assert "best_epoch" not in source.lower()
    assert "selected_epoch" not in source.lower()


def test_stagewise_cox_matches_canonical_fit_score(tmp_path: Path) -> None:
    validation = _survival_frame("v", 90)
    test = _survival_frame("t", 75, age_shift=0.3)
    validation_path = tmp_path / "validation.parquet"
    validation.to_parquet(validation_path, index=False)
    correction_path, nuisance_path, validation_corrected = (
        runner.fit_and_seal_calibration_nuisance(
            validation,
            outer_fold=2,
            calibration_fold=3,
            validation_predictions_path=validation_path,
            output_dir=tmp_path / "nuisance",
            replication_contract_sha256="a" * 64,
            calibration_role_manifest_sha256="b" * 64,
            training_complete_sha256="c" * 64,
            fixed_e55_checkpoint_sha256="d" * 64,
        )
    )
    staged, test_corrected = runner.score_outer_with_sealed_nuisance(
        test,
        outer_fold=2,
        correction_path=correction_path,
        correction_sha256=runner.sha256_file(correction_path),
        nuisance_manifest_path=nuisance_path,
        nuisance_manifest_sha256=runner.sha256_file(nuisance_path),
    )
    for organ in runner.ORGAN_SYSTEMS:
        canonical = evaluate_cox_cell(
            validation_corrected.loc[validation_corrected["organ_system"].eq(organ)],
            test_corrected.loc[test_corrected["organ_system"].eq(organ)],
            fold=2,
            organ_system=organ,
        )
        observed = staged["organs"][organ]
        assert observed["c_base"] == pytest.approx(canonical.c_base, abs=1e-12)
        assert observed["c_plus"] == pytest.approx(canonical.c_plus, abs=1e-12)
        assert observed["delta_c"] == pytest.approx(canonical.delta_c, abs=1e-12)
    assert staged["outer_refit_calls"] == 0


def test_nuisance_fit_rejects_unbound_in_memory_frame(tmp_path: Path) -> None:
    persisted = _survival_frame("v", 30)
    path = tmp_path / "validation.parquet"
    persisted.to_parquet(path, index=False)
    changed = persisted.copy()
    changed.loc[0, "predicted_age"] += 1e-4
    with pytest.raises(runner.FixedReplicationError, match="hash-bound"):
        runner.fit_and_seal_calibration_nuisance(
            changed,
            outer_fold=2,
            calibration_fold=3,
            validation_predictions_path=path,
            output_dir=tmp_path / "bad",
            replication_contract_sha256="a" * 64,
            calibration_role_manifest_sha256="b" * 64,
            training_complete_sha256="c" * 64,
            fixed_e55_checkpoint_sha256="d" * 64,
        )


def test_score_only_source_contains_no_cox_fit_call() -> None:
    source = inspect.getsource(runner.score_outer_with_sealed_nuisance)
    assert ".fit(" not in source
    assert "CoxPHFitter" not in source
    assert "evaluate_cox_cell" not in source


def test_score_rejects_tampered_sealed_nuisance(tmp_path: Path) -> None:
    validation = _survival_frame("v", 45)
    test = _survival_frame("t", 40, age_shift=0.2)
    prediction_path = tmp_path / "validation.parquet"
    validation.to_parquet(prediction_path, index=False)
    correction, nuisance, _ = runner.fit_and_seal_calibration_nuisance(
        validation,
        outer_fold=2,
        calibration_fold=3,
        validation_predictions_path=prediction_path,
        output_dir=tmp_path / "sealed",
        replication_contract_sha256="a" * 64,
        calibration_role_manifest_sha256="b" * 64,
        training_complete_sha256="c" * 64,
        fixed_e55_checkpoint_sha256="d" * 64,
    )
    expected = runner.sha256_file(nuisance)
    payload = json.loads(nuisance.read_text(encoding="utf-8"))
    payload["model_count"] = 13
    nuisance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.FixedReplicationError, match="hash changed"):
        runner.score_outer_with_sealed_nuisance(
            test,
            outer_fold=2,
            correction_path=correction,
            correction_sha256=runner.sha256_file(correction),
            nuisance_manifest_path=nuisance,
            nuisance_manifest_sha256=expected,
        )


def test_builder_never_authorizes_fold0_training() -> None:
    source = inspect.getsource(runner._build_role_artifacts_impl)
    assert "if outer in FORMAL_FOLDS" in source
    assert '"mode": "reuse_d07_003_e55"' in source


def test_fold0_reuse_declares_no_retraining() -> None:
    source = inspect.getsource(runner.run_fold0_reuse)
    assert '"retrained": False' in source
    assert "historical Fold0" in source
    assert "restore_trainable_checkpoint" in source
    assert "_train_epoch" not in source


def test_authoritative_frame_uses_role_age_and_labels() -> None:
    source = inspect.getsource(runner._authoritative_prediction_frame)
    assert "age_targets" in source
    assert "joint_labels" in source
    assert "duration_months" in source


def test_aggregate_requires_all_five_receipts() -> None:
    with pytest.raises(runner.FixedReplicationError, match="folds 0..4"):
        runner.aggregate_fivefold(
            contract_path=Path("missing"),
            expected_contract_sha256="a" * 64,
            role_registry_path=Path("missing"),
            expected_role_registry_sha256="b" * 64,
            receipt_paths={},
        )
