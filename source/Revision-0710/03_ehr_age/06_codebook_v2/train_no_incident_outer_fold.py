from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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

from ehr_age_data import (  # noqa: E402
    Fold0DevelopmentData,
    JointLabels,
    JointVectorizedBatcher,
    load_restricted_compact_cache,
    sha256_file,
)
from ehr_age_model import ORGAN_SYSTEMS  # noqa: E402
from ehr_age_training import (  # noqa: E402
    PredictionBundle,
    _loss_config,
    _predict,
    _survival_multiplier,
    _train_epoch,
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
OUTPUT_ROOT = HERE / "03_no_incident_control" / "01_outer_folds"
ROLE_AGE_ROOT = EHR_ROOT / "05_corrected_primary_v1" / "00_role_artifacts"
FROZEN_EPOCH = 47
VARIANT = "shared_private_independent_view_huber1"
ROLE_DIRECTORIES = {
    "train": "01_train",
    "calibration": "02_calibration",
    "outer_test": "03_outer_test",
}


def role_fold_ids(outer_fold: int) -> dict[str, tuple[int, ...]]:
    calibration = (int(outer_fold) + 1) % 5
    return {
        "train": tuple(
            fold
            for fold in range(5)
            if fold not in {int(outer_fold), calibration}
        ),
        "calibration": (calibration,),
        "outer_test": (int(outer_fold),),
    }


def load_age_only_role(outer_fold: int, role: str) -> Fold0DevelopmentData:
    folds = role_fold_ids(outer_fold)[role]
    age_path = (
        ROLE_AGE_ROOT
        / f"outer_fold_{outer_fold}"
        / ROLE_DIRECTORIES[role]
        / "age_targets.parquet"
    )
    cache = load_restricted_compact_cache(
        cache_dir=DEFAULT_CACHE,
        outer_fold=outer_fold,
        fold_ids=folds,
        age_target_cache_path=age_path,
    )
    n = len(cache.dataset)
    labels = JointLabels(
        strict_healthy=torch.zeros(n, dtype=torch.bool),
        duration_months=torch.zeros((n, len(ORGAN_SYSTEMS)), dtype=torch.float32),
        event=torch.zeros((n, len(ORGAN_SYSTEMS)), dtype=torch.bool),
        at_risk=torch.zeros((n, len(ORGAN_SYSTEMS)), dtype=torch.bool),
        inner_role=tuple(role for _ in range(n)),
    )
    labels.validate(n)
    indices = torch.arange(n, dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(
        cache=cache,
        labels=labels,
        train_indices=indices if role == "train" else empty,
        calibration_indices=indices if role == "calibration" else empty,
        screen_indices=indices if role == "outer_test" else empty,
        label_manifest={
            "status": "PASS",
            "role": role,
            "fold_ids": list(folds),
            "age_targets_path": str(age_path.resolve()),
            "age_targets_sha256": sha256_file(age_path),
            "incident_outcome_file_read": False,
        },
    )


def no_incident_config() -> dict[str, Any]:
    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    spec = VARIANTS[VARIANT]
    raw["model"]["base_age_mode"] = spec["base_age_mode"]
    raw["loss"]["huber_beta_years"] = spec["huber_beta_years"]
    raw["loss"]["base_huber_weight"] = spec["base_huber_weight"]
    raw["loss"]["base_view_huber_weight"] = spec["base_view_huber_weight"]
    for key in (
        "strict_final_huber_weight",
        "strict_offset_weight",
        "strict_center_weight",
        "strict_slope_weight",
        "survival_weight",
        "evaluator_gap_cox_weight",
    ):
        raw["loss"][key] = 0.0
    return raw


def prediction_frame(
    data: Fold0DevelopmentData,
    prediction: PredictionBundle,
    *,
    outer_fold: int,
    role: str,
) -> pd.DataFrame:
    participant_ids = prediction_ids(data, prediction)
    n = len(participant_ids)
    return pd.DataFrame(
        {
            "participant_id": np.repeat(
                np.asarray(participant_ids, dtype=object), len(ORGAN_SYSTEMS)
            ),
            "outer_fold": int(outer_fold),
            "split_role": role,
            "organ_system": np.tile(np.asarray(ORGAN_SYSTEMS, dtype=object), n),
            "actual_age": np.repeat(
                prediction.chronological_age.astype(np.float64), len(ORGAN_SYSTEMS)
            ),
            "predicted_age": prediction.predicted_age.astype(np.float64).reshape(-1),
            "sex": np.repeat(prediction.sex.astype(np.float64), len(ORGAN_SYSTEMS)),
            "model_risk_score": prediction.risk_score.astype(np.float64).reshape(-1),
        }
    )


def validate_outer_fold(outer_fold: int) -> dict[str, Any]:
    bundles = {
        role: load_age_only_role(outer_fold, role) for role in ROLE_DIRECTORIES
    }
    id_sets = {
        role: {str(value) for value in bundle.cache.dataset.participant_ids}
        for role, bundle in bundles.items()
    }
    if any(
        id_sets[left].intersection(id_sets[right])
        for left, right in (
            ("train", "calibration"),
            ("train", "outer_test"),
            ("calibration", "outer_test"),
        )
    ):
        raise RuntimeError("age-only roles overlap")
    raw = no_incident_config()
    model, _, _, _, _, _, _ = make_model_and_optimizer(raw, bundles["train"])
    return {
        "status": "PASS",
        "outer_fold": int(outer_fold),
        "participant_counts": {
            role: int(len(bundle.cache.dataset))
            for role, bundle in bundles.items()
        },
        "incident_outcome_file_read": False,
        "outcome_derived_loss_weights": {
            key: raw["loss"][key]
            for key in (
                "strict_final_huber_weight",
                "strict_offset_weight",
                "strict_center_weight",
                "strict_slope_weight",
                "survival_weight",
                "evaluator_gap_cox_weight",
            )
        },
        "initial_trainable_parameter_sha256": trainable_parameter_hash(model),
    }


def run_outer_fold(outer_fold: int, output_dir: Path) -> Path:
    raw = no_incident_config()
    train_data = load_age_only_role(outer_fold, "train")
    calibration_data = load_age_only_role(outer_fold, "calibration")
    outer_data = load_age_only_role(outer_fold, "outer_test")
    id_sets = [
        {str(value) for value in bundle.cache.dataset.participant_ids}
        for bundle in (train_data, calibration_data, outer_data)
    ]
    if any(
        id_sets[left].intersection(id_sets[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("age-only roles overlap")
    model, device, optimizer, scheduler, scaler, edges, age_anchor = (
        make_model_and_optimizer(raw, train_data)
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    training = raw["training"]
    atomic_json(
        output_dir / "run_config.json",
        {
            "schema_version": "revision0710.ehr_age.no_incident.train.v1",
            "created_utc": utc_now(),
            "outer_fold": int(outer_fold),
            "train_folds": list(role_fold_ids(outer_fold)["train"]),
            "calibration_fold": int((outer_fold + 1) % 5),
            "fixed_epoch": FROZEN_EPOCH,
            "variant": VARIANT,
            "fine_tune": False,
            "distillation": False,
            "checkpoint_input": None,
            "incident_outcome_file_read": False,
            "strict_health_label_read": False,
            "model": raw["model"],
            "training": raw["training"],
            "loss": raw["loss"],
            "age_only_roles": {
                "train": train_data.label_manifest,
                "calibration": calibration_data.label_manifest,
                "outer_test": outer_data.label_manifest,
            },
            "initial_trainable_parameter_sha256": trainable_parameter_hash(model),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
    )
    train_batcher = JointVectorizedBatcher(
        train_data,
        train_data.train_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=SEED,
    )
    for epoch in range(1, FROZEN_EPOCH + 1):
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
        append_jsonl(output_dir / "training_history.jsonl", row)
        print(
            f"no_incident outer={outer_fold} epoch={epoch:02d}/{FROZEN_EPOCH} "
            f"loss={float(losses['total']):.6f} seconds={row['seconds']:.1f}",
            flush=True,
        )
    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, FROZEN_EPOCH, age_anchor)
    for role, data, indices in (
        ("calibration", calibration_data, calibration_data.calibration_indices),
        ("outer_test", outer_data, outer_data.screen_indices),
    ):
        batcher = JointVectorizedBatcher(
            data,
            indices,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            seed=SEED,
        )
        prediction = _predict(
            model=model,
            batcher=batcher,
            device=device,
            amp=bool(training["amp"]),
            max_batches=None,
        )
        prediction_frame(
            data, prediction, outer_fold=outer_fold, role=role
        ).to_parquet(output_dir / f"{role}_predictions.parquet", index=False)
    atomic_json(
        output_dir / "TRAINING_COMPLETE.json",
        {
            "schema_version": "revision0710.ehr_age.no_incident.complete.v1",
            "status": "COMPLETE",
            "completed_utc": utc_now(),
            "outer_fold": int(outer_fold),
            "fixed_epoch": FROZEN_EPOCH,
            "incident_outcome_file_read": False,
            "strict_health_label_read": False,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device)
            / 1024**2,
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one age-only EHR-AGE control without reading incident outcomes."
    )
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        print(json.dumps(validate_outer_fold(args.outer_fold), indent=2))
        return
    output_dir = args.output_dir or OUTPUT_ROOT / f"outer_fold_{args.outer_fold}"
    print(run_outer_fold(args.outer_fold, output_dir))


if __name__ == "__main__":
    main()
