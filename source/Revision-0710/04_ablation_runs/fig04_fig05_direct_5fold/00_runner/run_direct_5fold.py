from __future__ import annotations

"""Run the fixed Fig4/Fig5 ablation queue without V8 authorization machinery.

The runner intentionally preserves the frozen data roles, model variants,
training budget, fixed E55 checkpoint, calibration-only correction, and outer
fold scoring. It writes a compact, restartable record for each queued task.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


RUNNER_ROOT = Path(__file__).resolve().parent
RUN_ROOT = RUNNER_ROOT.parent
REPO_ROOT = RUNNER_ROOT.parents[3]
V8_ROOT = (
    REPO_ROOT
    / "Revision-0710"
    / "06_figures_and_statistics"
    / "00_plan"
    / "fig04_fig05_ablation_prefreeze_v8"
)
MODEL_ROOT = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "01_model"
for location in (V8_ROOT, MODEL_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import v8_contract  # noqa: E402
import v8_data  # noqa: E402
import v8_model  # noqa: E402
import v8_training  # noqa: E402
from ehr_age_model import (  # noqa: E402
    BASE_AGE_HEAD_MODE_MLP,
    BASE_AGE_MODE_PRIVATE_RETAIN,
    EHRAGEConfig,
)


QUEUE_PATH = V8_ROOT / "ABLATION_QUEUE_V8.csv"
FROZEN_CONTRACT_PATH = V8_ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"
TASKS_ROOT = RUN_ROOT / "01_tasks"
SUMMARY_ROOT = RUN_ROOT / "02_summary"
LOG_ROOT = RUN_ROOT / "03_logs"
STATUS_PATH = RUN_ROOT / "RUN_STATUS.json"
STATUS_CSV_PATH = SUMMARY_ROOT / "task_status.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_queue() -> list[dict[str, str]]:
    with QUEUE_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 queue rows, found {len(rows)}")
    return rows


def load_contract() -> dict[str, Any]:
    return json.loads(FROZEN_CONTRACT_PATH.read_text(encoding="utf-8"))


def role_manifest(contract: dict[str, Any], outer_fold: int, role: str) -> tuple[Path, str]:
    name = f"outer_fold_{outer_fold}_{role}_manifest"
    matches = [item for item in contract["canonical_bindings"]["records"] if item["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"Missing canonical role manifest: {name}")
    record = matches[0]
    return REPO_ROOT / record["path"], str(record["sha256"])


def load_role(
    contract: dict[str, Any], *, outer_fold: int, role: str
) -> tuple[v8_data.RoleBundle, torch.Tensor, torch.Tensor]:
    manifest_path, manifest_sha256 = role_manifest(contract, outer_fold, role)
    cache_dir = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "04_cache_v1"
    bindings = contract["cache_artifact_bindings"]
    if sha256_file(manifest_path) != manifest_sha256:
        raise RuntimeError(f"Role manifest hash changed: {manifest_path}")
    v8_contract.validate_cache_artifact_bindings(bindings, V8_ROOT, verify_hashes=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("role") != role or int(manifest.get("outer_fold", -1)) != outer_fold:
        raise RuntimeError(f"Role manifest identity mismatch: {manifest_path}")
    artifact = manifest["artifact"]
    fold_ids = tuple(int(value) for value in artifact["fold_ids"])
    participant_order = v8_data._validate_cache_files_for_load(  # noqa: SLF001
        cache_dir=cache_dir,
        bindings=bindings,
        outer_fold=outer_fold,
        fold_ids=fold_ids,
    )
    from ehr_age_data import load_restricted_compact_cache

    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=outer_fold,
        fold_ids=fold_ids,
        age_target_cache_path=Path(artifact["age_targets"]["path"]),
    )
    participant_ids = tuple(cache.dataset.participant_ids)
    if participant_ids != participant_order or len(participant_ids) != int(artifact["participant_count"]):
        raise RuntimeError("Role cache participant coverage changed")
    labels = pd.read_parquet(Path(artifact["joint_labels"]["path"]))
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels["organ_system"] = labels["organ_system"].astype(str)
    index = pd.MultiIndex.from_product(
        [participant_ids, v8_contract.ORGAN_SYSTEMS], names=["participant_id", "organ_system"]
    )
    indexed = labels.set_index(["participant_id", "organ_system"]).reindex(index)
    required = ["duration_months", "event", "at_risk", "strict_healthy_reference_60m"]
    if indexed[required[1:]].isna().any().any():
        raise RuntimeError("Role labels are incomplete beyond duration placeholders")
    duration_missing = indexed["duration_months"].isna()
    non_risk_placeholder = duration_missing & indexed["at_risk"].eq(0) & indexed["event"].eq(0)
    if (duration_missing & ~non_risk_placeholder).any():
        raise RuntimeError("Risk-set duration labels are incomplete")
    # The canonical loss immediately maps all out-of-risk durations to zero.
    # This converts only its missing out-of-risk placeholders to that exact value.
    duration = indexed["duration_months"].fillna(0.0).to_numpy("float32").reshape(len(participant_ids), 7)
    event = indexed["event"].to_numpy("int64").reshape(len(participant_ids), 7)
    at_risk = indexed["at_risk"].to_numpy("int64").reshape(len(participant_ids), 7)
    strict_matrix = indexed["strict_healthy_reference_60m"].to_numpy("int64").reshape(len(participant_ids), 7)
    if not (strict_matrix == strict_matrix[:, :1]).all():
        raise RuntimeError("Strict-health flag differs across organ rows")
    return (
        v8_data.RoleBundle(
            role=role,
            outer_fold=outer_fold,
            participant_ids=participant_ids,
            inputs=cache.dataset.inputs,
            labels={
                "chronological_age": cache.dataset.chronological_age,
                "strict_healthy_reference": torch.from_numpy(strict_matrix[:, 0]),
                "duration_months": torch.from_numpy(duration),
                "event": torch.from_numpy(event.astype(bool)),
                "at_risk": torch.from_numpy(at_risk.astype(bool)),
            },
        ),
        cache.code_embedding,
        cache.text_embedding,
    )


def production_config() -> EHRAGEConfig:
    return EHRAGEConfig(
        hidden_dim=192,
        num_heads=6,
        num_transformer_layers=3,
        num_gru_layers=2,
        dropout=0.15,
        freeze_event_embeddings=True,
        risk_offset_cap_years=8.0,
        risk_temperature=2.0,
        survival_intervals=5,
        baseline_log_rate_init=-7.5,
        base_age_mode=BASE_AGE_MODE_PRIVATE_RETAIN,
        base_age_head_mode=BASE_AGE_HEAD_MODE_MLP,
    )


def restore_capture_model(
    checkpoint_path: Path, *, variant_id: str, code: torch.Tensor, text: torch.Tensor
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant_id") != variant_id or int(checkpoint.get("epoch", -1)) != 55:
        raise RuntimeError("The captured checkpoint does not match the frozen E55 protocol")
    model = v8_model.build_production_model(
        variant_id,
        code,
        text,
        age_anchor=float(checkpoint["age_anchor"]),
        config=EHRAGEConfig(**checkpoint["model_config"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def task_paths(row: dict[str, str]) -> dict[str, Path]:
    root = TASKS_ROOT / f"{int(row['ordinal']):02d}_{row['run_id']}"
    return {
        "root": root,
        "train": root / "01_train",
        "calibration": root / "02_calibration",
        "outer": root / "03_outer_test",
        "spec": root / "TASK_SPEC.json",
        "complete": root / "TASK_COMPLETE.json",
        "failure": root / "FAILED.json",
    }


def scan_status(rows: list[dict[str, str]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        paths = task_paths(row)
        state = "PENDING"
        detail = ""
        if paths["complete"].is_file():
            payload = json.loads(paths["complete"].read_text(encoding="utf-8"))
            state = "COMPLETE"
            detail = str(payload.get("completed_at", ""))
        elif paths["failure"].is_file():
            payload = json.loads(paths["failure"].read_text(encoding="utf-8"))
            state = "FAILED"
            detail = str(payload.get("error", ""))
        elif paths["root"].exists():
            state = "INCOMPLETE"
        records.append({**row, "run_state": state, "detail": detail})
    return pd.DataFrame(records)


def save_status(rows: list[dict[str, str]], *, active_run_id: str | None, message: str) -> None:
    table = scan_status(rows)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(STATUS_CSV_PATH, index=False, lineterminator="\n")
    counts = table["run_state"].value_counts().to_dict()
    write_json(
        STATUS_PATH,
        {
            "updated_at": utc_now(),
            "active_run_id": active_run_id,
            "message": message,
            "counts": {str(key): int(value) for key, value in counts.items()},
            "queue_total": int(len(table)),
            "queue_sha256": sha256_file(QUEUE_PATH),
            "frozen_contract_sha256": sha256_file(FROZEN_CONTRACT_PATH),
        },
    )


def run_one(row: dict[str, str], contract: dict[str, Any]) -> None:
    paths = task_paths(row)
    if paths["complete"].is_file():
        return
    if paths["root"].exists():
        raise RuntimeError(f"Incomplete task directory requires inspection: {paths['root']}")
    for key in ("train", "calibration", "outer"):
        paths[key].mkdir(parents=True, exist_ok=False)

    outer_fold = int(row["outer_fold"])
    variant_id = row["variant_id"]
    task_spec = {
        "runner": "fig04_fig05_direct_5fold",
        "started_at": utc_now(),
        "queue_row": row,
        "queue_sha256": sha256_file(QUEUE_PATH),
        "frozen_contract_sha256": sha256_file(FROZEN_CONTRACT_PATH),
        "protocol": {
            "seed": 7210,
            "batch_size": 2048,
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "warmup_epochs": 3,
            "scheduler_horizon_epochs": 60,
            "train_through_epoch": 60,
            "capture_epoch": 55,
            "evaluate_only_epoch": 55,
            "amp": True,
            "device": "cuda",
            "fresh_initialization": True,
            "resume": False,
        },
        "role_manifests": {
            role: str(role_manifest(contract, outer_fold, role)[0])
            for role in ("train", "calibration", "outer_test")
        },
    }
    write_json(paths["spec"], task_spec)

    try:
        # Training uses only the three canonical train folds for this outer fold.
        train_bundle, train_code, train_text = load_role(contract, outer_fold=outer_fold, role="train")
        v8_training.configure_reproducibility(7210, deterministic=True, tf32=True)
        model = v8_model.build_production_model(
            variant_id,
            train_code,
            train_text,
            age_anchor=float(train_bundle.labels["chronological_age"].float().mean()),
            config=production_config(),
        )
        checkpoint_path = paths["train"] / "fixed_epoch_0055.pt"
        train_result = v8_training.train_fresh(
            model,
            variant_id,
            v8_data.RoleBatcher(train_bundle, batch_size=2048, seed=7210, shuffle=True),
            device=torch.device("cuda"),
            seed=7210,
            epochs=60,
            capture_epoch=55,
            learning_rate=0.0002,
            weight_decay=0.0001,
            gradient_clip_norm=1.0,
            warmup_epochs=3,
            scheduler_horizon_epochs=60,
            amp=True,
            checkpoint_path=checkpoint_path,
        )
        write_json(paths["train"] / "TRAINING_COMPLETE.json", train_result)
        write_json(paths["train"] / "MODEL_CONFIG.json", asdict(production_config()))
        del train_bundle, train_code, train_text, model
        gc.collect()
        torch.cuda.empty_cache()

        # Calibration is isolated from both training and the outer fold.
        calibration_bundle, calibration_code, calibration_text = load_role(
            contract, outer_fold=outer_fold, role="calibration"
        )
        calibration_model = restore_capture_model(
            checkpoint_path,
            variant_id=variant_id,
            code=calibration_code,
            text=calibration_text,
        ).to(torch.device("cuda"))
        calibration_predictions = v8_data.prediction_frame(
            calibration_model,
            calibration_bundle,
            variant_id=variant_id,
            run_id=row["run_id"],
            evaluation_epoch=55,
            device=torch.device("cuda"),
            inference_batch_size=2048,
        )
        calibration_predictions["split_role"] = "calibration"
        correction = v8_data.fit_age_correction(calibration_predictions)
        corrected_calibration = v8_data.apply_age_correction(calibration_predictions, correction)
        nuisance = v8_data.fit_cox_nuisance(corrected_calibration, penalizer=1e-4)
        calibration_predictions.to_csv(paths["calibration"] / "calibration_predictions_raw.csv", index=False, lineterminator="\n")
        corrected_calibration.to_csv(paths["calibration"] / "calibration_predictions_corrected.csv", index=False, lineterminator="\n")
        write_json(paths["calibration"] / "age_correction.json", correction)
        write_json(paths["calibration"] / "cox_nuisance.json", nuisance)
        del calibration_bundle, calibration_code, calibration_text, calibration_model, calibration_predictions, corrected_calibration
        gc.collect()
        torch.cuda.empty_cache()

        # The outer fold is opened only for fixed-checkpoint prediction and scoring.
        outer_bundle, outer_code, outer_text = load_role(contract, outer_fold=outer_fold, role="outer_test")
        outer_model = restore_capture_model(
            checkpoint_path,
            variant_id=variant_id,
            code=outer_code,
            text=outer_text,
        ).to(torch.device("cuda"))
        outer_predictions = v8_data.prediction_frame(
            outer_model,
            outer_bundle,
            variant_id=variant_id,
            run_id=row["run_id"],
            evaluation_epoch=55,
            device=torch.device("cuda"),
            inference_batch_size=2048,
        )
        outer_predictions["split_role"] = "outer_test"
        corrected_outer = v8_data.apply_age_correction(outer_predictions, correction)
        fold_metrics, organ_metrics = v8_data.score_outer_metrics(corrected_outer, nuisance)
        fold_record = {
            "run_id": row["run_id"],
            "figure": row["figure"],
            "variant_id": variant_id,
            "outer_fold": outer_fold,
            "evaluation_epoch": 55,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            **fold_metrics,
        }
        organ_frame = pd.DataFrame(
            [
                {
                    "run_id": row["run_id"],
                    "figure": row["figure"],
                    "variant_id": variant_id,
                    "outer_fold": outer_fold,
                    "evaluation_epoch": 55,
                    **metric,
                }
                for metric in organ_metrics
            ]
        )
        corrected_outer.to_csv(paths["outer"] / "prediction_rows.csv", index=False, lineterminator="\n")
        pd.DataFrame([fold_record]).to_csv(paths["outer"] / "fold_metrics.csv", index=False, lineterminator="\n")
        organ_frame.to_csv(paths["outer"] / "organ_metrics.csv", index=False, lineterminator="\n")
        write_json(
            paths["complete"],
            {
                "status": "COMPLETE",
                "completed_at": utc_now(),
                "run_id": row["run_id"],
                "variant_id": variant_id,
                "outer_fold": outer_fold,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "fold_metrics": fold_record,
            },
        )
        del outer_bundle, outer_code, outer_text, outer_model, outer_predictions, corrected_outer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        write_json(
            paths["failure"],
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "run_id": row["run_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runs", type=int, default=None, help="Optional bounded launch for diagnostics.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Fig4/Fig5 ablation queue")
    rows = load_queue()
    contract = load_contract()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    save_status(rows, active_run_id=None, message="Direct formal queue initialized")
    completed = 0
    for row in rows:
        if task_paths(row)["complete"].is_file():
            continue
        if args.max_runs is not None and completed >= args.max_runs:
            break
        save_status(rows, active_run_id=row["run_id"], message="Training, calibration, and outer-fold scoring in progress")
        run_one(row, contract)
        completed += 1
        save_status(rows, active_run_id=None, message=f"Completed {row['run_id']}")
    save_status(rows, active_run_id=None, message="Direct queue process exited normally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
