from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
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

from d07_open_fold0_development import (  # noqa: E402
    _aligned_open_fold0_labels,
    _organ_age_metrics,
    _serialize_correction,
    calibration_to_open_fold0_cox,
)
from ehr_age_data import (  # noqa: E402
    Fold0DevelopmentData,
    JointVectorizedBatcher,
    load_fold0_development_data,
    load_restricted_compact_cache,
    sha256_file,
)
from ehr_age_model import EHRAGEModel, ORGAN_SYSTEMS  # noqa: E402
from ehr_age_training import (  # noqa: E402
    PredictionBundle,
    _loss_config,
    _model_config,
    _predict,
    _survival_multiplier,
    _train_epoch,
    configure_reproducibility,
    pooled_age_metrics,
    trainable_parameter_hash,
)


HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE / "00_roles"
ROLE_SUMMARY = ROLE_ROOT / "role_summary.json"
DEVELOPMENT_LABELS = ROLE_ROOT / "fold0_development_labels.parquet"
DEVELOPMENT_AGES = ROLE_ROOT / "fold0_development_ages.parquet"
OPEN_LABELS = ROLE_ROOT / "fold0_open_labels.parquet"
OPEN_AGES = ROLE_ROOT / "fold0_open_ages.parquet"
DEFAULT_OUTPUT = HERE / "01_runs" / "current_architecture_fold0"
DUAL_OUTPUT = HERE / "01_runs" / "shared_private_equal_mean_fold0"
DUAL_VIEW_LOSS_OUTPUT = (
    HERE / "01_runs" / "shared_private_equal_mean_view_loss_fold0"
)
INDEPENDENT_VIEW_LOSS_OUTPUT = (
    HERE / "01_runs" / "shared_private_independent_view_loss_fold0"
)
INDEPENDENT_VIEW_HUBER1_OUTPUT = (
    HERE / "01_runs" / "shared_private_independent_view_huber1_fold0"
)
DEFAULT_CACHE = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
SOURCE_CONFIG = EHR_ROOT / "00_contract" / "ehr_age_d07_003_private_retain_v1.json"

SEED = 7210
MAX_EPOCHS = 60
EXPECTED_INITIAL_TRAINABLE_SHA256 = {
    "private_retain": "cf54feb052f9b09556a1355e8e078c82dabbaa036c1e0387e05d85688373a8a8",
    "shared_private_equal_mean": "d14ab2dc3a080aa5803064d05fea7f6257d004cebff54796c3843e07945f8632",
}
VARIANTS = {
    "private_retain": {
        "base_age_mode": "private_retain",
        "huber_beta_years": 2.0,
        "base_huber_weight": 1.0,
        "base_view_huber_weight": 0.0,
        "output_dir": DEFAULT_OUTPUT,
    },
    "shared_private_equal_mean": {
        "base_age_mode": "shared_private_equal_mean",
        "huber_beta_years": 2.0,
        "base_huber_weight": 1.0,
        "base_view_huber_weight": 0.0,
        "output_dir": DUAL_OUTPUT,
    },
    "shared_private_equal_mean_view_loss": {
        "base_age_mode": "shared_private_equal_mean",
        "huber_beta_years": 2.0,
        "base_huber_weight": 1.0,
        "base_view_huber_weight": 1.0,
        "output_dir": DUAL_VIEW_LOSS_OUTPUT,
    },
    "shared_private_independent_view_loss": {
        "base_age_mode": "shared_private_equal_mean",
        "huber_beta_years": 2.0,
        "base_huber_weight": 0.0,
        "base_view_huber_weight": 2.0,
        "output_dir": INDEPENDENT_VIEW_LOSS_OUTPUT,
    },
    "shared_private_independent_view_huber1": {
        "base_age_mode": "shared_private_equal_mean",
        "huber_beta_years": 1.0,
        "base_huber_weight": 0.0,
        "base_view_huber_weight": 2.0,
        "output_dir": INDEPENDENT_VIEW_HUBER1_OUTPUT,
    },
}

# Fold0 development targets. These are provisional until the v2 baselines are rerun.
PROMOTION_TARGETS = {
    "strict_mae_max_exclusive": 3.681014,
    "strict_pearson_min_exclusive": 0.797574,
    "strict_r2_min_exclusive": 0.631047,
    "full_risk_macro_delta_c_min_exclusive": 0.022939,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def prediction_ids(data: Fold0DevelopmentData, prediction: PredictionBundle) -> list[str]:
    return [
        str(data.cache.dataset.participant_ids[int(index)])
        for index in prediction.participant_indices
    ]


def save_checkpoint(path: Path, model: EHRAGEModel, epoch: int, age_anchor: float) -> None:
    trainable = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    buffers = {
        name: value.detach().cpu().clone() for name, value in model.named_buffers()
    }
    torch.save(
        {
            "schema_version": "revision0710.ehr_age.codebook_v2.checkpoint.v1",
            "epoch": int(epoch),
            "age_anchor": float(age_anchor),
            "initialization": "from_scratch_except_frozen_cache_embeddings",
            "fine_tune": False,
            "distillation": False,
            "trainable_parameter_sha256": trainable_parameter_hash(model),
            "trainable_state_dict": trainable,
            "buffer_state_dict": buffers,
        },
        path,
    )


def restore_checkpoint(path: Path, model: EHRAGEModel) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    trainable = {
        name: value for name, value in model.named_parameters() if value.requires_grad
    }
    stored = payload.get("trainable_state_dict")
    buffers = {name: value for name, value in model.named_buffers()}
    stored_buffers = payload.get("buffer_state_dict")
    if not isinstance(stored, Mapping) or set(stored) != set(trainable):
        raise RuntimeError("checkpoint trainable state does not match the frozen architecture")
    if not isinstance(stored_buffers, Mapping) or set(stored_buffers) != set(buffers):
        raise RuntimeError("checkpoint buffer state does not match the frozen architecture")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(stored[name].to(parameter.device, dtype=parameter.dtype))
        for name, buffer in buffers.items():
            buffer.copy_(stored_buffers[name].to(buffer.device, dtype=buffer.dtype))
    if trainable_parameter_hash(model) != payload.get("trainable_parameter_sha256"):
        raise RuntimeError("checkpoint trainable hash changed after restore")
    return dict(payload)


def global_age_pareto_epochs(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    points = [
        (
            int(row["epoch"]),
            float(row["strict_mae"]),
            float(row["strict_pearson"]),
            float(row["strict_r2"]),
        )
        for row in rows
    ]
    frontier: list[int] = []
    for epoch, mae, pearson, r2 in points:
        dominated = any(
            other_epoch != epoch
            and other_mae <= mae
            and other_pearson >= pearson
            and other_r2 >= r2
            and (other_mae < mae or other_pearson > pearson or other_r2 > r2)
            for other_epoch, other_mae, other_pearson, other_r2 in points
        )
        if not dominated:
            frontier.append(epoch)
    return sorted(frontier)


def age_gate_count(row: Mapping[str, Any]) -> int:
    return int(float(row["strict_mae"]) < PROMOTION_TARGETS["strict_mae_max_exclusive"]) + int(
        float(row["strict_pearson"]) > PROMOTION_TARGETS["strict_pearson_min_exclusive"]
    ) + int(float(row["strict_r2"]) > PROMOTION_TARGETS["strict_r2_min_exclusive"])


def all_four_pass(row: Mapping[str, Any]) -> bool:
    return age_gate_count(row) == 3 and float(row["full_risk_macro_delta_c"]) > PROMOTION_TARGETS[
        "full_risk_macro_delta_c_min_exclusive"
    ]


def select_epoch(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], bool]:
    promotable = [row for row in rows if all_four_pass(row)]
    pool = promotable if promotable else list(rows)
    if not pool:
        raise RuntimeError("risk schedule produced no evaluable epoch")
    if promotable:
        selected = max(
            pool,
            key=lambda row: (
                float(row["full_risk_macro_delta_c"]),
                -float(row["strict_mae"]),
                float(row["strict_pearson"]),
                float(row["strict_r2"]),
                -int(row["epoch"]),
            ),
        )
        return selected, True
    diagnostic = max(
        pool,
        key=lambda row: (
            age_gate_count(row),
            float(row["full_risk_macro_delta_c"]),
            -float(row["strict_mae"]),
            float(row["strict_pearson"]),
            float(row["strict_r2"]),
            -int(row["epoch"]),
        ),
    )
    return diagnostic, False


def load_open_fold0(cache_dir: Path) -> Fold0DevelopmentData:
    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=0,
        fold_ids=(0,),
        age_target_cache_path=OPEN_AGES,
    )
    labels = _aligned_open_fold0_labels(
        pd.read_parquet(OPEN_LABELS), cache.dataset.participant_ids
    )
    indices = torch.arange(len(cache.dataset), dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(
        cache=cache,
        labels=labels,
        train_indices=empty,
        calibration_indices=empty,
        screen_indices=indices,
        label_manifest=json.loads(ROLE_SUMMARY.read_text(encoding="utf-8")),
    )


def make_model_and_optimizer(
    raw: Mapping[str, Any], data: Fold0DevelopmentData
) -> tuple[
    EHRAGEModel,
    torch.device,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR,
    torch.amp.GradScaler,
    torch.Tensor,
    float,
]:
    training = raw["training"]
    configure_reproducibility(SEED, deterministic=True, tf32=True)
    age_anchor = float(
        data.cache.dataset.chronological_age.index_select(0, data.train_indices).mean()
    )
    model = EHRAGEModel(
        data.cache.code_embedding,
        data.cache.text_embedding,
        age_anchor=age_anchor,
        config=_model_config(raw["model"]),
    )
    observed_initial = trainable_parameter_hash(model)
    expected_initial = EXPECTED_INITIAL_TRAINABLE_SHA256.get(
        str(raw["model"]["base_age_mode"])
    )
    if expected_initial is None or observed_initial != expected_initial:
        raise RuntimeError(
            "architecture initialization changed: "
            f"expected={expected_initial}, observed={observed_initial}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Fold0 formal training requires CUDA")
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup = int(training["warmup_epochs"])

    def lr_factor(index: int) -> float:
        if index < warmup:
            return float(index + 1) / max(warmup, 1)
        progress = (index - warmup) / max(MAX_EPOCHS - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    edges = torch.tensor(
        raw["model"]["survival_interval_edges_months"],
        dtype=torch.float32,
        device=device,
    )
    return model, device, optimizer, scheduler, scaler, edges, age_anchor


def baseline_organ_envelope(raw: Mapping[str, Any]) -> dict[str, float]:
    values = raw["baseline_envelope"]["gates"]["organ_delta_c_envelope"]
    return {organ: float(values[organ]) for organ in ORGAN_SYSTEMS}


def run(
    output_dir: Path, *, validate_only: bool, variant: str = "private_retain"
) -> Path | dict[str, Any]:
    role_summary = json.loads(ROLE_SUMMARY.read_text(encoding="utf-8"))
    if role_summary.get("status") != "PASS":
        raise RuntimeError("codebook-v2 role summary is not PASS")
    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if variant not in VARIANTS:
        raise ValueError(f"unknown Fold0 variant: {variant}")
    variant_spec = VARIANTS[variant]
    raw["model"]["base_age_mode"] = variant_spec["base_age_mode"]
    raw["loss"]["huber_beta_years"] = variant_spec["huber_beta_years"]
    raw["loss"]["base_huber_weight"] = variant_spec["base_huber_weight"]
    raw["loss"]["base_view_huber_weight"] = variant_spec[
        "base_view_huber_weight"
    ]
    data = load_fold0_development_data(
        cache_dir=DEFAULT_CACHE,
        label_cache_path=DEVELOPMENT_LABELS,
        age_target_cache_path=DEVELOPMENT_AGES,
        label_manifest_path=ROLE_SUMMARY,
    )
    open_data = load_open_fold0(DEFAULT_CACHE)
    if set(data.cache.dataset.participant_ids).intersection(open_data.cache.dataset.participant_ids):
        raise RuntimeError("training/calibration participants overlap open Fold0")
    model, device, optimizer, scheduler, scaler, edges, age_anchor = make_model_and_optimizer(
        raw, data
    )
    if validate_only:
        return {
            "status": "PASS",
            "train_participants": int(len(data.train_indices)),
            "calibration_participants": int(len(data.calibration_indices)),
            "fold1_screen_participants_not_used": int(len(data.screen_indices)),
            "open_fold0_participants": int(len(open_data.screen_indices)),
            "initial_trainable_parameter_sha256": trainable_parameter_hash(model),
            "gpu": torch.cuda.get_device_name(0),
        }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    epoch_cache = output_dir / "_epoch_cache"
    epoch_cache.mkdir()
    started = time.perf_counter()
    training = raw["training"]
    run_config = {
        "schema_version": "revision0710.ehr_age.codebook_v2.fold0_run.v1",
        "created_utc": utc_now(),
        "variant": f"{variant}_from_scratch",
        "outer_fold": 0,
        "train_folds": [2, 3, 4],
        "calibration_fold": 1,
        "open_development_fold": 0,
        "fold1_screen_used_for_training_or_selection": False,
        "seed": SEED,
        "epochs": MAX_EPOCHS,
        "fine_tune": False,
        "distillation": False,
        "checkpoint_input": None,
        "promotion_targets_provisional_until_v2_baselines": PROMOTION_TARGETS,
        "model": raw["model"],
        "training": raw["training"],
        "loss": raw["loss"],
        "inputs": {
            "role_summary": {"path": str(ROLE_SUMMARY), "sha256": sha256_file(ROLE_SUMMARY)},
            "source_config": {"path": str(SOURCE_CONFIG), "sha256": sha256_file(SOURCE_CONFIG)},
            "cache_batch_manifest": {
                "path": str(DEFAULT_CACHE / "batch_manifest.json"),
                "sha256": sha256_file(DEFAULT_CACHE / "batch_manifest.json"),
            },
        },
        "initial_trainable_parameter_sha256": trainable_parameter_hash(model),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    atomic_json(output_dir / "run_config.json", run_config)

    train_batcher = JointVectorizedBatcher(
        data,
        data.train_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=SEED,
    )
    calibration_batcher = JointVectorizedBatcher(
        data,
        data.calibration_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=SEED,
    )
    fold0_batcher = JointVectorizedBatcher(
        open_data,
        open_data.screen_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=SEED,
    )
    history_path = output_dir / "training_history.jsonl"
    age_rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
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
        checkpoint_path = epoch_cache / f"epoch_{epoch:02d}.pt"
        save_checkpoint(checkpoint_path, model, epoch, age_anchor)
        fold0_prediction = _predict(
            model=model,
            batcher=fold0_batcher,
            device=device,
            amp=bool(training["amp"]),
            max_batches=None,
        )
        strict = pooled_age_metrics(
            fold0_prediction.chronological_age,
            fold0_prediction.predicted_age,
            participant_mask=fold0_prediction.strict_healthy,
        )
        row = {
            "epoch": epoch,
            "strict_mae": float(strict["mae"]),
            "strict_pearson": float(strict["pearson"]),
            "strict_r2": float(strict["r2"]),
            "age_gate_count": 0,
            "seconds": time.perf_counter() - epoch_started,
        }
        row["age_gate_count"] = age_gate_count(row)
        age_rows.append(row)
        append_jsonl(
            history_path,
            {
                **row,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "losses": {key: float(value) for key, value in losses.items()},
            },
        )
        print(
            f"epoch={epoch:02d}/{MAX_EPOCHS} loss={float(losses['total']):.6f} "
            f"mae={row['strict_mae']:.6f} r={row['strict_pearson']:.6f} "
            f"r2={row['strict_r2']:.6f} age_gates={row['age_gate_count']}/3 "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )

    pareto_epochs = global_age_pareto_epochs(age_rows)
    risk_epochs = sorted(
        set(pareto_epochs)
        | {int(row["epoch"]) for row in age_rows if int(row["age_gate_count"]) == 3}
        | {MAX_EPOCHS}
    )
    age_by_epoch = {int(row["epoch"]): row for row in age_rows}
    risk_rows: list[dict[str, Any]] = []
    organ_envelope = baseline_organ_envelope(raw)
    for epoch in risk_epochs:
        restore_checkpoint(epoch_cache / f"epoch_{epoch:02d}.pt", model)
        calibration = _predict(
            model=model,
            batcher=calibration_batcher,
            device=device,
            amp=bool(training["amp"]),
            max_batches=None,
        )
        fold0_prediction = _predict(
            model=model,
            batcher=fold0_batcher,
            device=device,
            amp=bool(training["amp"]),
            max_batches=None,
        )
        risk, _, _, _ = calibration_to_open_fold0_cox(
            calibration,
            fold0_prediction,
            calibration_participant_ids=prediction_ids(data, calibration),
            fold0_participant_ids=prediction_ids(open_data, fold0_prediction),
            baseline_organ_delta_c=organ_envelope,
        )
        row = {
            **age_by_epoch[epoch],
            "full_risk_macro_delta_c": float(risk["macro_delta_c"]),
            "positive_organs": int(risk["positive_organs"]),
            "organs_beating_v1_baseline": int(risk["organs_beating_baseline"]),
            "all_four_pass": False,
            "organ_risk": risk["organs"],
        }
        row["all_four_pass"] = all_four_pass(row)
        risk_rows.append(row)
        print(
            f"risk_epoch={epoch:02d} delta_c={row['full_risk_macro_delta_c']:.6f} "
            f"positive={row['positive_organs']}/7 all_four={row['all_four_pass']}",
            flush=True,
        )

    selected, promotable = select_epoch(risk_rows)
    selected_epoch = int(selected["epoch"])
    selected_cache = epoch_cache / f"epoch_{selected_epoch:02d}.pt"
    final_checkpoint = output_dir / "checkpoint.pt"
    shutil.copy2(selected_cache, final_checkpoint)
    restore_checkpoint(final_checkpoint, model)
    calibration = _predict(
        model=model,
        batcher=calibration_batcher,
        device=device,
        amp=bool(training["amp"]),
        max_batches=None,
    )
    fold0_prediction = _predict(
        model=model,
        batcher=fold0_batcher,
        device=device,
        amp=bool(training["amp"]),
        max_batches=None,
    )
    final_risk, calibration_corrected, fold0_corrected, correction = (
        calibration_to_open_fold0_cox(
            calibration,
            fold0_prediction,
            calibration_participant_ids=prediction_ids(data, calibration),
            fold0_participant_ids=prediction_ids(open_data, fold0_prediction),
            baseline_organ_delta_c=organ_envelope,
        )
    )
    fold0_corrected["model_risk_score"] = fold0_prediction.risk_score.astype(
        np.float32
    ).reshape(-1)
    fold0_corrected.to_parquet(output_dir / "fold0_predictions.parquet", index=False)
    calibration_corrected["model_risk_score"] = calibration.risk_score.astype(
        np.float32
    ).reshape(-1)
    calibration_corrected.to_parquet(
        output_dir / "fold1_calibration_predictions.parquet", index=False
    )
    atomic_json(output_dir / "age_correction.json", _serialize_correction(correction))

    raw_strict = pooled_age_metrics(
        fold0_prediction.chronological_age,
        fold0_prediction.predicted_age,
        participant_mask=fold0_prediction.strict_healthy,
    )
    corrected_matrix = fold0_corrected["corrected_predicted_age"].to_numpy(
        dtype=np.float64
    ).reshape(len(fold0_prediction.chronological_age), len(ORGAN_SYSTEMS))
    corrected_strict = pooled_age_metrics(
        fold0_prediction.chronological_age,
        corrected_matrix,
        participant_mask=fold0_prediction.strict_healthy,
    )
    epoch_table = pd.DataFrame(age_rows)
    risk_by_epoch = {int(row["epoch"]): row for row in risk_rows}
    epoch_table["risk_evaluated"] = epoch_table["epoch"].isin(risk_by_epoch)
    epoch_table["full_risk_macro_delta_c"] = epoch_table["epoch"].map(
        {epoch: row["full_risk_macro_delta_c"] for epoch, row in risk_by_epoch.items()}
    )
    epoch_table["all_four_pass"] = epoch_table["epoch"].map(
        {epoch: row["all_four_pass"] for epoch, row in risk_by_epoch.items()}
    )
    epoch_table.to_csv(output_dir / "epoch_metrics.csv", index=False)

    metrics = {
        "schema_version": "revision0710.ehr_age.codebook_v2.fold0_metrics.v1",
        "status": "PROMOTABLE_ON_PROVISIONAL_TARGETS" if promotable else "NEEDS_OPTIMIZATION",
        "completed_utc": utc_now(),
        "fold_role": "open Fold0 development; repeated inspection allowed",
        "selected_epoch": selected_epoch if promotable else None,
        "diagnostic_epoch": None if promotable else selected_epoch,
        "promotion_targets_provisional_until_v2_baselines": PROMOTION_TARGETS,
        "primary_same_epoch": {
            "strict_mae": float(raw_strict["mae"]),
            "strict_pearson": float(raw_strict["pearson"]),
            "strict_r2": float(raw_strict["r2"]),
            "full_risk_macro_delta_c": float(final_risk["macro_delta_c"]),
            "positive_organs": int(final_risk["positive_organs"]),
            "all_four_pass": bool(promotable),
        },
        "strict_age_raw": {
            **raw_strict,
            "organs": _organ_age_metrics(
                fold0_prediction.chronological_age,
                fold0_prediction.predicted_age,
                fold0_prediction.strict_healthy,
            ),
        },
        "strict_age_fold1_corrected_diagnostic": {
            **corrected_strict,
            "organs": _organ_age_metrics(
                fold0_prediction.chronological_age,
                corrected_matrix,
                fold0_prediction.strict_healthy,
            ),
        },
        "full_risk": final_risk,
        "pareto_epochs": pareto_epochs,
        "risk_evaluated_epochs": risk_epochs,
        "risk_history": risk_rows,
        "checkpoint": {
            "path": str(final_checkpoint),
            "sha256": sha256_file(final_checkpoint),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    shutil.rmtree(epoch_cache)
    print(json.dumps(metrics["primary_same_epoch"], indent=2), flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current from-scratch EHR-AGE architecture on codebook-v2 Fold0."
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="private_retain",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = VARIANTS[args.variant]["output_dir"]
    result = run(
        output_dir,
        validate_only=args.validate_only,
        variant=args.variant,
    )
    if isinstance(result, Mapping):
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    else:
        print(str(result), flush=True)


if __name__ == "__main__":
    main()
