from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REVISION_ROOT = WORKSPACE_ROOT / "Revision-0710"
EHR_ROOT = REVISION_ROOT / "03_ehr_age"
RUNNER_DIR = EHR_ROOT / "02_runner"
MODEL_DIR = EHR_ROOT / "01_model"
for path in (RUNNER_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from d07_open_fold0_development import _organ_age_metrics  # noqa: E402
from ehr_age_data import (  # noqa: E402
    Fold0DevelopmentData,
    JointLabels,
    JointVectorizedBatcher,
    canonical_hash,
    load_restricted_compact_cache,
    sha256_file,
)
from ehr_age_model import ORGAN_SYSTEMS  # noqa: E402
from ehr_age_training import (  # noqa: E402
    PredictionBundle,
    _evaluation_frame,
    _loss_config,
    _predict,
    _survival_multiplier,
    _train_epoch,
    pooled_age_metrics,
    trainable_parameter_hash,
)
from run_fold0_current import (  # noqa: E402
    DEFAULT_CACHE,
    SEED,
    SOURCE_CONFIG,
    VARIANTS,
    append_jsonl,
    atomic_json,
    make_model_and_optimizer,
    prediction_ids,
    save_checkpoint,
    utc_now,
)


HERE = Path(__file__).resolve().parent
FIVEFOLD_ROOT = HERE / "02_fivefold_frozen_huber1"
RUN_ROOT = FIVEFOLD_ROOT / "01_outer_folds"
SELECTION_PATH = FIVEFOLD_ROOT / "FINAL_SELECTION.json"
FOLD0_METRICS = (
    HERE
    / "01_runs"
    / "shared_private_independent_view_huber1_fold0"
    / "metrics.json"
)
FOLD0_CHECKPOINT = FOLD0_METRICS.parent / "checkpoint.pt"
BASELINE_TARGETS = (
    REVISION_ROOT / "05_downstream" / "11_codebook_v2" / "baseline_targets.json"
)
BASELINE_ENVELOPE = (
    REVISION_ROOT
    / "05_downstream"
    / "11_codebook_v2"
    / "02_summary"
    / "baseline_metric_envelope.json"
)
OUTCOMES_PATH = (
    REVISION_ROOT / "01_data_rebuild" / "09_v2_labels" / "organ_outcomes_60m.parquet"
)
ROLE_AGE_ROOT = (
    EHR_ROOT / "05_corrected_primary_v1" / "00_role_artifacts"
)

VARIANT = "shared_private_independent_view_huber1"
FROZEN_EPOCH = 47
TRAINING_HORIZON_EPOCHS = 60
MATCHED_VARIANT = "private_retain"
MATCHED_EPOCH = 60
MATCHED_ROOT = HERE / "04_matched_retain_joint"
MATCHED_RUN_ROOT = MATCHED_ROOT / "01_outer_folds"
MATCHED_SELECTION_PATH = MATCHED_ROOT / "FINAL_SELECTION.json"
MATCHED_FOLD0_METRICS = HERE / "01_runs" / "current_architecture_fold0" / "metrics.json"
MATCHED_FOLD0_CHECKPOINT = MATCHED_FOLD0_METRICS.parent / "checkpoint.pt"
ROLE_DIRECTORIES = {
    "train": "01_train",
    "calibration": "02_calibration",
    "outer_test": "03_outer_test",
}
OUTCOME_COLUMNS = (
    "participant_id",
    "organ_system",
    "duration_months",
    "event",
    "at_risk",
    "strict_healthy_reference_60m",
)


def role_fold_ids(outer_fold: int) -> dict[str, tuple[int, ...]]:
    calibration = (int(outer_fold) + 1) % 5
    train = tuple(
        fold for fold in range(5) if fold not in {int(outer_fold), calibration}
    )
    return {
        "train": train,
        "calibration": (calibration,),
        "outer_test": (int(outer_fold),),
    }


def _load_baseline_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    targets = json.loads(BASELINE_TARGETS.read_text(encoding="utf-8"))
    envelope = json.loads(BASELINE_ENVELOPE.read_text(encoding="utf-8"))
    if targets.get("status") != "PASS":
        raise RuntimeError("codebook-v2 baseline targets are not PASS")
    if len(envelope.get("organ_delta_c_envelope", [])) != len(ORGAN_SYSTEMS):
        raise RuntimeError("codebook-v2 organ baseline envelope is incomplete")
    return targets, envelope


def official_baseline_values(targets: Mapping[str, Any]) -> dict[str, float]:
    best = targets["best"]
    return {
        "strict_mae": float(best["strict_mae"]["value"]),
        "strict_pearson": float(best["strict_pearson"]["value"]),
        "strict_r2": float(best["strict_r2"]["value"]),
        "full_risk_macro_delta_c": float(best["full_risk_macro_delta_c"]["value"]),
    }


def organ_baseline_values(envelope: Mapping[str, Any]) -> dict[str, float]:
    values = {
        str(row["organ_system"]): float(row["baseline_mean_delta_c"])
        for row in envelope["organ_delta_c_envelope"]
    }
    if set(values) != set(ORGAN_SYSTEMS):
        raise RuntimeError("organ baseline envelope changed its seven-organ axis")
    return values


def _selection_payload() -> dict[str, Any]:
    metrics = json.loads(FOLD0_METRICS.read_text(encoding="utf-8"))
    targets, _ = _load_baseline_payloads()
    baseline = official_baseline_values(targets)
    observed = metrics.get("primary_same_epoch", {})
    if int(metrics.get("diagnostic_epoch", -1)) != FROZEN_EPOCH:
        raise RuntimeError("the selected Fold0 source is no longer epoch 47")
    comparisons = {
        "strict_mae": float(observed["strict_mae"]) < baseline["strict_mae"],
        "strict_pearson": float(observed["strict_pearson"])
        > baseline["strict_pearson"],
        "strict_r2": float(observed["strict_r2"]) > baseline["strict_r2"],
        "full_risk_macro_delta_c": float(observed["full_risk_macro_delta_c"])
        > baseline["full_risk_macro_delta_c"],
    }
    if not all(comparisons.values()):
        raise RuntimeError("Fold0 Huber1 does not beat all six official baselines")
    return {
        "schema_version": "revision0710.ehr_age.codebook_v2.final_selection.v1",
        "status": "FROZEN",
        "variant": VARIANT,
        "epoch": FROZEN_EPOCH,
        "training_scheduler_horizon_epochs": TRAINING_HORIZON_EPOCHS,
        "selection_basis": (
            "all four metrics strictly better than the metric-wise best of the six "
            "formal codebook-v2 baselines"
        ),
        "extra_three_model_equal_weight_mean": (
            "retained as a nonbinding design diagnostic, not promoted to a formal baseline"
        ),
        "from_scratch_for_every_outer_fold": True,
        "fine_tune": False,
        "distillation": False,
        "checkpoint_averaging": False,
        "fold0_source": {
            "metrics_path": str(FOLD0_METRICS.resolve()),
            "metrics_sha256": sha256_file(FOLD0_METRICS),
            "checkpoint_path": str(FOLD0_CHECKPOINT.resolve()),
            "checkpoint_sha256": sha256_file(FOLD0_CHECKPOINT),
        },
        "fold0_metrics": {
            key: float(observed[key])
            for key in (
                "strict_mae",
                "strict_pearson",
                "strict_r2",
                "full_risk_macro_delta_c",
            )
        },
        "formal_baseline_values": baseline,
        "fold0_strictly_better": comparisons,
        "inputs": {
            "baseline_targets": {
                "path": str(BASELINE_TARGETS.resolve()),
                "sha256": sha256_file(BASELINE_TARGETS),
            },
            "baseline_metric_envelope": {
                "path": str(BASELINE_ENVELOPE.resolve()),
                "sha256": sha256_file(BASELINE_ENVELOPE),
            },
            "codebook_v2_outcomes": {
                "path": str(OUTCOMES_PATH.resolve()),
                "sha256": sha256_file(OUTCOMES_PATH),
            },
        },
    }


def ensure_selection() -> dict[str, Any]:
    expected = _selection_payload()
    FIVEFOLD_ROOT.mkdir(parents=True, exist_ok=True)
    if SELECTION_PATH.exists():
        observed = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError("FINAL_SELECTION.json disagrees with the frozen inputs")
    else:
        atomic_json(SELECTION_PATH, expected)
    return expected


def ensure_matched_selection() -> dict[str, Any]:
    metrics = json.loads(MATCHED_FOLD0_METRICS.read_text(encoding="utf-8"))
    targets, _ = _load_baseline_payloads()
    baseline = official_baseline_values(targets)
    observed = metrics.get("primary_same_epoch", {})
    if int(metrics.get("diagnostic_epoch", -1)) != MATCHED_EPOCH:
        raise RuntimeError("the matched private-RETAIN Fold0 source is no longer epoch 60")
    comparisons = {
        "strict_mae": float(observed["strict_mae"]) < baseline["strict_mae"],
        "strict_pearson": float(observed["strict_pearson"])
        > baseline["strict_pearson"],
        "strict_r2": float(observed["strict_r2"]) > baseline["strict_r2"],
        "full_risk_macro_delta_c": float(observed["full_risk_macro_delta_c"])
        > baseline["full_risk_macro_delta_c"],
    }
    if not all(comparisons.values()):
        raise RuntimeError("matched private-RETAIN Fold0 does not beat all formal baselines")
    payload = {
        "schema_version": "revision0710.ehr_age.codebook_v2.matched_retain_selection.v1",
        "status": "FROZEN",
        "control_name": "RETAIN-organquery-joint",
        "variant": MATCHED_VARIANT,
        "epoch": MATCHED_EPOCH,
        "selection_basis": (
            "Fold0 same-epoch result strictly beats all six formal baseline metric values; "
            "highest evaluated full-risk delta-C retained"
        ),
        "from_scratch_for_every_outer_fold": True,
        "fine_tune": False,
        "distillation": False,
        "fold0_source": {
            "metrics_path": str(MATCHED_FOLD0_METRICS.resolve()),
            "metrics_sha256": sha256_file(MATCHED_FOLD0_METRICS),
            "checkpoint_path": str(MATCHED_FOLD0_CHECKPOINT.resolve()),
            "checkpoint_sha256": sha256_file(MATCHED_FOLD0_CHECKPOINT),
        },
        "fold0_metrics": {
            key: float(observed[key])
            for key in (
                "strict_mae",
                "strict_pearson",
                "strict_r2",
                "full_risk_macro_delta_c",
            )
        },
        "formal_baseline_values": baseline,
        "fold0_strictly_better": comparisons,
    }
    MATCHED_ROOT.mkdir(parents=True, exist_ok=True)
    if MATCHED_SELECTION_PATH.exists():
        stored = json.loads(MATCHED_SELECTION_PATH.read_text(encoding="utf-8"))
        if stored != payload:
            raise RuntimeError("matched RETAIN FINAL_SELECTION.json changed")
    else:
        atomic_json(MATCHED_SELECTION_PATH, payload)
    return payload


def _align_labels(
    frame: pd.DataFrame,
    participant_ids: Sequence[str],
    *,
    expected_role: str,
    expected_folds: Sequence[int],
) -> JointLabels:
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(frame["fold_id"].astype(int).unique()) != set(expected_folds):
        raise RuntimeError(f"{expected_role} labels contain unexpected folds")
    if set(frame["inner_role"].astype(str).unique()) != {expected_role}:
        raise RuntimeError(f"{expected_role} labels contain an unexpected role")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError(f"{expected_role} labels contain duplicate rows")
    expected_index = pd.MultiIndex.from_product(
        [pd.Index([str(value) for value in participant_ids]), list(ORGAN_SYSTEMS)],
        names=["participant_id", "organ_system"],
    )
    indexed = frame.set_index(["participant_id", "organ_system"]).reindex(
        expected_index
    )
    if indexed[["event", "at_risk"]].isna().any().any():
        raise RuntimeError(f"{expected_role} labels are incomplete after alignment")
    missing_duration = indexed["duration_months"].isna()
    if (missing_duration & indexed["at_risk"].ne(0)).any():
        raise RuntimeError(f"{expected_role} has missing at-risk duration")
    if (missing_duration & indexed["event"].ne(0)).any():
        raise RuntimeError(f"{expected_role} has an event with missing duration")
    indexed["duration_months"] = indexed["duration_months"].fillna(0.0)
    n = len(participant_ids)
    strict = indexed["strict_healthy_reference_60m"].to_numpy(
        dtype=np.int8
    ).reshape(n, -1)
    if not np.all(strict == strict[:, :1]):
        raise RuntimeError(f"{expected_role} strict Health differs across organs")
    labels = JointLabels(
        strict_healthy=torch.from_numpy(strict[:, 0].astype(bool)),
        duration_months=torch.from_numpy(
            indexed["duration_months"].to_numpy(dtype=np.float32).reshape(n, -1)
        ),
        event=torch.from_numpy(
            indexed["event"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)
        ),
        at_risk=torch.from_numpy(
            indexed["at_risk"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)
        ),
        inner_role=tuple(expected_role for _ in range(n)),
    )
    labels.validate(n)
    return labels


def load_role_data(
    outcomes: pd.DataFrame,
    *,
    outer_fold: int,
    role: str,
) -> Fold0DevelopmentData:
    expected_folds = role_fold_ids(outer_fold)[role]
    age_path = (
        ROLE_AGE_ROOT
        / f"outer_fold_{outer_fold}"
        / ROLE_DIRECTORIES[role]
        / "age_targets.parquet"
    )
    ages = pd.read_parquet(
        age_path, columns=["participant_id", "fold_id", "chronological_age"]
    )
    ages["participant_id"] = ages["participant_id"].astype(str)
    if ages["participant_id"].duplicated().any():
        raise RuntimeError(f"{role} age targets contain duplicate participants")
    if set(ages["fold_id"].astype(int).unique()) != set(expected_folds):
        raise RuntimeError(f"{role} age targets contain unexpected folds")
    cache = load_restricted_compact_cache(
        cache_dir=DEFAULT_CACHE,
        outer_fold=outer_fold,
        fold_ids=expected_folds,
        age_target_cache_path=age_path,
    )
    labels_frame = ages[["participant_id", "fold_id"]].merge(
        outcomes,
        on="participant_id",
        how="left",
        validate="one_to_many",
    )
    labels_frame["inner_role"] = role
    if len(labels_frame) != len(ages) * len(ORGAN_SYSTEMS):
        raise RuntimeError(f"{role} did not receive seven v2 organ labels per participant")
    labels = _align_labels(
        labels_frame,
        cache.dataset.participant_ids,
        expected_role=role,
        expected_folds=expected_folds,
    )
    indices = torch.arange(len(cache.dataset), dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(
        cache=cache,
        labels=labels,
        train_indices=indices if role == "train" else empty,
        calibration_indices=indices if role == "calibration" else empty,
        screen_indices=indices if role == "outer_test" else empty,
        label_manifest={
            "status": "PASS",
            "outer_fold": int(outer_fold),
            "role": role,
            "fold_ids": list(expected_folds),
            "age_targets_path": str(age_path.resolve()),
            "age_targets_sha256": sha256_file(age_path),
            "outcomes_path": str(OUTCOMES_PATH.resolve()),
            "outcomes_sha256": sha256_file(OUTCOMES_PATH),
        },
    )


def _serialize_outer_correction(correction: Any, outer_fold: int) -> dict[str, Any]:
    participant_ids = sorted(correction.validation_participant_ids)
    return {
        "fit_role": f"Fold{(outer_fold + 1) % 5} calibration nuisance only",
        "outer_fold": int(outer_fold),
        "calibration_fold": int((outer_fold + 1) % 5),
        "fit_participant_count": len(participant_ids),
        "fit_participant_ids_canonical_sha256": canonical_hash(
            {"participant_ids": participant_ids}
        ),
        "parameters": [asdict(parameter) for parameter in correction.parameters],
    }


def calibration_to_outer_cox(
    calibration: PredictionBundle,
    outer: PredictionBundle,
    *,
    calibration_participant_ids: list[str],
    outer_participant_ids: list[str],
    outer_fold: int,
    baseline_organ_delta_c: Mapping[str, float],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Any]:
    from crossfit_metrics import (
        apply_linear_age_correction,
        apply_linear_age_correction_to_validation,
        evaluate_cox_cell,
        fit_linear_age_correction,
    )

    calibration_frame = _evaluation_frame(
        calibration, calibration_participant_ids
    )
    outer_frame = _evaluation_frame(outer, outer_participant_ids)
    correction = fit_linear_age_correction(calibration_frame)
    calibration_corrected = apply_linear_age_correction_to_validation(
        calibration_frame, correction
    )
    outer_corrected = apply_linear_age_correction(outer_frame, correction)
    organs: dict[str, dict[str, Any]] = {}
    deltas: list[float] = []
    for organ in ORGAN_SYSTEMS:
        calibration_organ = calibration_corrected.loc[
            calibration_corrected["organ_system"].astype(str).eq(organ)
            & calibration_corrected["at_risk"].eq(1)
        ].copy()
        outer_organ = outer_corrected.loc[
            outer_corrected["organ_system"].astype(str).eq(organ)
            & outer_corrected["at_risk"].eq(1)
        ].copy()
        cell = evaluate_cox_cell(
            calibration_organ,
            outer_organ,
            fold=int(outer_fold),
            organ_system=organ,
        )
        organs[organ] = asdict(cell)
        deltas.append(float(cell.delta_c))
    return (
        {
            "definition": (
                f"Fold{(outer_fold + 1) % 5} calibration-fit strict-reference age "
                f"correction and Cox nuisance; Fold{outer_fold} Harrell C-index on "
                "each organ's complete at_risk=1 population"
            ),
            "fit_population": f"Fold{(outer_fold + 1) % 5} calibration",
            "score_population": f"Fold{outer_fold} organ-specific all at_risk=1",
            "strict_healthy_filter_applied_to_risk_score_population": False,
            "macro_delta_c": float(np.mean(deltas)),
            "positive_organs": int(sum(value > 0.0 for value in deltas)),
            "organs_beating_baseline": int(
                sum(
                    deltas[index] > float(baseline_organ_delta_c[organ])
                    for index, organ in enumerate(ORGAN_SYSTEMS)
                )
            ),
            "evaluable_organs": len(deltas),
            "organs": organs,
        },
        calibration_corrected,
        outer_corrected,
        correction,
    )


def _apply_variant(raw: dict[str, Any], variant: str = VARIANT) -> None:
    spec = VARIANTS[variant]
    raw["model"]["base_age_mode"] = spec["base_age_mode"]
    raw["loss"]["huber_beta_years"] = spec["huber_beta_years"]
    raw["loss"]["base_huber_weight"] = spec["base_huber_weight"]
    raw["loss"]["base_view_huber_weight"] = spec["base_view_huber_weight"]


def _comparison(
    age: Mapping[str, Any], risk: Mapping[str, Any], baseline: Mapping[str, float]
) -> dict[str, Any]:
    deltas = {
        "strict_mae_reduction_years": baseline["strict_mae"] - float(age["mae"]),
        "strict_pearson_gain": float(age["pearson"]) - baseline["strict_pearson"],
        "strict_r2_gain": float(age["r2"]) - baseline["strict_r2"],
        "full_risk_macro_delta_c_gain": float(risk["macro_delta_c"])
        - baseline["full_risk_macro_delta_c"],
    }
    passes = {
        "strict_mae": deltas["strict_mae_reduction_years"] > 0.0,
        "strict_pearson": deltas["strict_pearson_gain"] > 0.0,
        "strict_r2": deltas["strict_r2_gain"] > 0.0,
        "full_risk_macro_delta_c": deltas["full_risk_macro_delta_c_gain"] > 0.0,
    }
    return {
        "baseline_values": dict(baseline),
        "delta_vs_formal_baseline": deltas,
        "strictly_better": passes,
        "all_four_strictly_better": bool(all(passes.values())),
    }


def validate_outer_fold(
    outer_fold: int, *, variant: str = VARIANT, fixed_epoch: int = FROZEN_EPOCH
) -> dict[str, Any]:
    if variant == VARIANT:
        ensure_selection()
    elif variant == MATCHED_VARIANT:
        ensure_matched_selection()
    else:
        raise ValueError(f"unsupported frozen variant: {variant}")
    outcomes = pd.read_parquet(OUTCOMES_PATH, columns=list(OUTCOME_COLUMNS))
    outcomes["participant_id"] = outcomes["participant_id"].astype(str)
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("v2 outcomes contain duplicate participant-organ rows")
    data = {
        role: load_role_data(outcomes, outer_fold=outer_fold, role=role)
        for role in ROLE_DIRECTORIES
    }
    role_ids = {
        role: {str(value) for value in bundle.cache.dataset.participant_ids}
        for role, bundle in data.items()
    }
    if any(
        role_ids[left].intersection(role_ids[right])
        for left, right in (
            ("train", "calibration"),
            ("train", "outer_test"),
            ("calibration", "outer_test"),
        )
    ):
        raise RuntimeError("five-fold role participants overlap")
    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    _apply_variant(raw, variant)
    model, _, _, _, _, _, _ = make_model_and_optimizer(raw, data["train"])
    return {
        "status": "PASS",
        "outer_fold": int(outer_fold),
        "variant": variant,
        "fixed_epoch": int(fixed_epoch),
        "train_folds": list(role_fold_ids(outer_fold)["train"]),
        "calibration_fold": int((outer_fold + 1) % 5),
        "participant_counts": {
            role: int(len(bundle.cache.dataset)) for role, bundle in data.items()
        },
        "strict_health_counts": {
            role: int(bundle.labels.strict_healthy.sum())
            for role, bundle in data.items()
        },
        "initial_trainable_parameter_sha256": trainable_parameter_hash(model),
        "gpu": torch.cuda.get_device_name(0),
    }


def run_outer_fold(
    outer_fold: int,
    output_dir: Path,
    *,
    variant: str = VARIANT,
    fixed_epoch: int = FROZEN_EPOCH,
    selection_path: Path = SELECTION_PATH,
) -> Path:
    if outer_fold not in {1, 2, 3, 4}:
        raise ValueError("the frozen runner accepts outer folds 1-4 only")
    if variant == VARIANT:
        ensure_selection()
    elif variant == MATCHED_VARIANT:
        ensure_matched_selection()
    else:
        raise ValueError(f"unsupported frozen variant: {variant}")
    targets, envelope = _load_baseline_payloads()
    baseline = official_baseline_values(targets)
    organ_baseline = organ_baseline_values(envelope)
    outcomes = pd.read_parquet(OUTCOMES_PATH, columns=list(OUTCOME_COLUMNS))
    outcomes["participant_id"] = outcomes["participant_id"].astype(str)
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("v2 outcomes contain duplicate participant-organ rows")

    train_data = load_role_data(outcomes, outer_fold=outer_fold, role="train")
    calibration_data = load_role_data(
        outcomes, outer_fold=outer_fold, role="calibration"
    )
    outer_data = load_role_data(outcomes, outer_fold=outer_fold, role="outer_test")
    del outcomes
    role_ids = [
        {str(value) for value in bundle.cache.dataset.participant_ids}
        for bundle in (train_data, calibration_data, outer_data)
    ]
    if any(
        role_ids[left].intersection(role_ids[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("five-fold role participants overlap")

    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    _apply_variant(raw, variant)
    model, device, optimizer, scheduler, scaler, edges, age_anchor = (
        make_model_and_optimizer(raw, train_data)
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    training = raw["training"]
    run_config = {
        "schema_version": "revision0710.ehr_age.codebook_v2.frozen_outer_run.v1",
        "created_utc": utc_now(),
        "variant": f"{variant}_from_scratch",
        "outer_fold": int(outer_fold),
        "train_folds": list(role_fold_ids(outer_fold)["train"]),
        "calibration_fold": int((outer_fold + 1) % 5),
        "fixed_epoch": int(fixed_epoch),
        "scheduler_horizon_epochs": TRAINING_HORIZON_EPOCHS,
        "seed": SEED,
        "fine_tune": False,
        "distillation": False,
        "checkpoint_input": None,
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
        },
        "model": raw["model"],
        "training": raw["training"],
        "loss": raw["loss"],
        "role_inputs": {
            "train": train_data.label_manifest,
            "calibration": calibration_data.label_manifest,
            "outer_test": outer_data.label_manifest,
        },
        "initial_trainable_parameter_sha256": trainable_parameter_hash(model),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    atomic_json(output_dir / "run_config.json", run_config)

    train_batcher = JointVectorizedBatcher(
        train_data,
        train_data.train_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=SEED,
    )
    calibration_batcher = JointVectorizedBatcher(
        calibration_data,
        calibration_data.calibration_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=SEED,
    )
    outer_batcher = JointVectorizedBatcher(
        outer_data,
        outer_data.screen_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=SEED,
    )
    history_path = output_dir / "training_history.jsonl"
    for epoch in range(1, fixed_epoch + 1):
        epoch_started = time.perf_counter()
        losses = _train_epoch(
            model=model,
            batcher=train_batcher,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            interval_edges=edges,
            loss_config=_loss_config(raw["loss"]),
            survival_multiplier=_survival_multiplier(epoch, raw["loss"]),
            event_dropout_probability=float(training["event_dropout_probability"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            amp=bool(training["amp"]),
            max_batches=None,
        )
        scheduler.step()
        row = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - epoch_started,
            "losses": {key: float(value) for key, value in losses.items()},
        }
        append_jsonl(history_path, row)
        print(
            f"outer={outer_fold} epoch={epoch:02d}/{fixed_epoch} "
            f"loss={float(losses['total']):.6f} seconds={row['seconds']:.1f}",
            flush=True,
        )

    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, fixed_epoch, age_anchor)
    calibration = _predict(
        model=model,
        batcher=calibration_batcher,
        device=device,
        amp=bool(training["amp"]),
        max_batches=None,
    )
    outer = _predict(
        model=model,
        batcher=outer_batcher,
        device=device,
        amp=bool(training["amp"]),
        max_batches=None,
    )
    risk, calibration_frame, outer_frame, correction = calibration_to_outer_cox(
        calibration,
        outer,
        calibration_participant_ids=prediction_ids(calibration_data, calibration),
        outer_participant_ids=prediction_ids(outer_data, outer),
        outer_fold=outer_fold,
        baseline_organ_delta_c=organ_baseline,
    )
    calibration_frame["model_risk_score"] = calibration.risk_score.astype(
        np.float32
    ).reshape(-1)
    calibration_frame["outer_fold"] = int(outer_fold)
    calibration_frame.to_parquet(
        output_dir / "calibration_predictions.parquet", index=False
    )
    outer_frame["model_risk_score"] = outer.risk_score.astype(np.float32).reshape(-1)
    outer_frame["outer_fold"] = int(outer_fold)
    outer_frame.to_parquet(output_dir / "outer_predictions.parquet", index=False)
    atomic_json(
        output_dir / "age_correction.json",
        _serialize_outer_correction(correction, outer_fold),
    )

    raw_strict = pooled_age_metrics(
        outer.chronological_age,
        outer.predicted_age,
        participant_mask=outer.strict_healthy,
    )
    comparison = _comparison(raw_strict, risk, baseline)
    metrics = {
        "schema_version": "revision0710.ehr_age.codebook_v2.frozen_outer_metrics.v1",
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "outer_fold": int(outer_fold),
        "fixed_epoch": int(fixed_epoch),
        "strict_age_raw": {
            **raw_strict,
            "organs": _organ_age_metrics(
                outer.chronological_age,
                outer.predicted_age,
                outer.strict_healthy,
            ),
        },
        "full_risk": risk,
        "comparison_to_six_formal_baselines": comparison,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "selection_sha256": sha256_file(selection_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "outer_fold": outer_fold,
                "strict_age_raw": raw_strict,
                "full_risk_macro_delta_c": risk["macro_delta_c"],
                "all_four_strictly_better": comparison[
                    "all_four_strictly_better"
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one frozen Huber1 EHR-AGE outer fold from scratch on codebook-v2."
        )
    )
    parser.add_argument("--outer-fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument(
        "--model",
        choices=("ehr_age_huber1", "retain_organquery_joint"),
        default="ehr_age_huber1",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "ehr_age_huber1":
        variant = VARIANT
        fixed_epoch = FROZEN_EPOCH
        selection_path = SELECTION_PATH
        default_root = RUN_ROOT
    else:
        variant = MATCHED_VARIANT
        fixed_epoch = MATCHED_EPOCH
        selection_path = MATCHED_SELECTION_PATH
        default_root = MATCHED_RUN_ROOT
    if args.validate_only:
        print(
            json.dumps(
                validate_outer_fold(
                    args.outer_fold,
                    variant=variant,
                    fixed_epoch=fixed_epoch,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    output_dir = args.output_dir or default_root / f"outer_fold_{args.outer_fold}"
    print(
        str(
            run_outer_fold(
                args.outer_fold,
                output_dir,
                variant=variant,
                fixed_epoch=fixed_epoch,
                selection_path=selection_path,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
