from __future__ import annotations

"""Run the frozen corrected-primary Fig4/Fig5 12 x 5-fold queue."""

import argparse
import gc
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

from contract_runtime import (
    ContractError,
    canonical_json_sha256,
    get_role_record,
    read_json,
    read_queue,
    resolve_path,
    sha256_file,
    stop_requested,
    validate_frozen_contract,
    verify_file_record,
    write_json_atomic,
)


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


QUEUE_PATH = RUNNER_ROOT / "ABLATION_QUEUE_CORRECTED_PRIMARY_V1.csv"
CONTRACT_PATH = RUNNER_ROOT / "FROZEN_CONTRACT_CORRECTED_PRIMARY_V1.json"
TASKS_ROOT = RUN_ROOT / "01_tasks"
SUMMARY_ROOT = RUN_ROOT / "02_summary"
LOG_ROOT = RUN_ROOT / "03_logs"
STATUS_PATH = RUN_ROOT / "RUN_STATUS.json"
STATUS_CSV_PATH = SUMMARY_ROOT / "task_status.csv"
CACHE_DIR = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "04_cache_v1"
FOLD0_GATE_RUN_ID = "fig04_01_canonical_control__outer0"
FOLD0_GATE_JSON = SUMMARY_ROOT / "FOLD0_CANONICAL_GATE.json"
FOLD0_GATE_MARKDOWN = SUMMARY_ROOT / "FOLD0_CANONICAL_GATE.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _protocol_guard(contract: dict[str, Any]) -> None:
    expected = {
        "variant_count": 12,
        "outer_fold_count": 5,
        "run_count": 60,
        "seed": 7210,
        "initialization": "fresh_random",
        "batch_size": 2048,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": 3,
        "scheduler_horizon_epochs": 60,
        "train_through_epoch": 60,
        "capture_epoch": 55,
        "evaluate_only_epoch": 55,
        "age_correction_role": "calibration_only",
        "cox_nuisance_role": "calibration_only",
        "outer_test_use": "fixed_checkpoint_prediction_and_scoring_only",
        "performance_selection": False,
        "fold0_gate": {
            "run_id": FOLD0_GATE_RUN_ID,
            "automatic_pause_after_completion": True,
            "resume_requires_explicit_flag": "--continue-after-fold0-gate",
        },
    }
    if contract.get("protocol") != expected:
        raise ContractError("Corrected-primary training/evaluation protocol changed")
    if tuple(contract.get("variant_ids", [])) != tuple(v8_contract.VARIANT_IDS):
        raise ContractError("Corrected-primary variant IDs differ from V8 behavior modules")
    bindings = contract.get("cache_artifact_bindings")
    if not isinstance(bindings, dict):
        raise ContractError("Frozen cache artifact bindings missing")
    if canonical_json_sha256(bindings) != contract.get("cache_artifact_bindings_sha256"):
        raise ContractError("Frozen cache artifact binding digest changed")
    v8_contract.validate_cache_artifact_bindings(bindings, V8_ROOT, verify_hashes=False)


def launch_preflight() -> tuple[dict[str, Any], list[dict[str, str]]]:
    contract = validate_frozen_contract(
        contract_path=CONTRACT_PATH,
        queue_path=QUEUE_PATH,
        repo_root=REPO_ROOT,
        run_root=RUN_ROOT,
    )
    _protocol_guard(contract)
    rows = read_queue(QUEUE_PATH)
    return contract, rows


def load_role(
    contract: dict[str, Any], *, outer_fold: int, role: str
) -> tuple[v8_data.RoleBundle, torch.Tensor, torch.Tensor]:
    record = get_role_record(contract, outer_fold, role)
    manifest_path = verify_file_record(record["manifest"], REPO_ROOT)
    age_path = verify_file_record(record["age_targets"], REPO_ROOT)
    labels_path = verify_file_record(record["joint_labels"], REPO_ROOT)
    manifest = read_json(manifest_path)
    artifact = manifest["artifact"]
    if manifest.get("status") != "SEALED":
        raise ContractError(f"Role is not sealed: outer {outer_fold} {role}")
    if manifest.get("role") != role or int(manifest.get("outer_fold", -1)) != outer_fold:
        raise ContractError(f"Role manifest identity changed: outer {outer_fold} {role}")
    for name, path in (("age_targets", age_path), ("joint_labels", labels_path)):
        bound = artifact.get(name, {})
        if resolve_path(str(bound.get("path", "")), REPO_ROOT) != path:
            raise ContractError(f"Role {name} path changed: outer {outer_fold} {role}")
        if bound.get("sha256") != sha256_file(path):
            raise ContractError(f"Role {name} SHA changed: outer {outer_fold} {role}")
    fold_ids = tuple(int(value) for value in record["fold_ids"])
    if fold_ids != tuple(int(value) for value in artifact["fold_ids"]):
        raise ContractError(f"Role fold IDs changed: outer {outer_fold} {role}")

    bindings = contract["cache_artifact_bindings"]
    participant_order = v8_data._validate_cache_files_for_load(  # noqa: SLF001
        cache_dir=CACHE_DIR,
        bindings=bindings,
        outer_fold=outer_fold,
        fold_ids=fold_ids,
    )
    from ehr_age_data import load_restricted_compact_cache

    cache = load_restricted_compact_cache(
        cache_dir=CACHE_DIR,
        outer_fold=outer_fold,
        fold_ids=fold_ids,
        age_target_cache_path=age_path,
    )
    participant_ids = tuple(cache.dataset.participant_ids)
    if participant_ids != participant_order:
        raise ContractError(f"Role participant order changed: outer {outer_fold} {role}")
    if len(participant_ids) != int(record["participant_count"]):
        raise ContractError(f"Role participant count changed: outer {outer_fold} {role}")
    participant_hash = v8_contract.canonical_json_sha256(
        {"participant_ids": sorted(str(value) for value in participant_ids)}
    )
    if participant_hash != record["participant_ids_canonical_sha256"]:
        raise ContractError(f"Role participant coverage changed: outer {outer_fold} {role}")
    if int(record["participant_organ_rows"]) != len(participant_ids) * 7:
        raise ContractError(f"Role participant-organ cardinality changed: outer {outer_fold} {role}")

    labels = pd.read_parquet(labels_path)
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels["organ_system"] = labels["organ_system"].astype(str)
    index = pd.MultiIndex.from_product(
        [participant_ids, v8_contract.ORGAN_SYSTEMS],
        names=["participant_id", "organ_system"],
    )
    if labels.duplicated(["participant_id", "organ_system"]).any():
        raise ContractError(f"Role labels contain duplicate keys: outer {outer_fold} {role}")
    indexed = labels.set_index(["participant_id", "organ_system"]).reindex(index)
    required = ["duration_months", "event", "at_risk", "strict_healthy_reference_60m"]
    if indexed[required[1:]].isna().any().any():
        raise ContractError(f"Role labels are incomplete: outer {outer_fold} {role}")
    duration_missing = indexed["duration_months"].isna()
    nonrisk_placeholder = duration_missing & indexed["at_risk"].eq(0) & indexed["event"].eq(0)
    if (duration_missing & ~nonrisk_placeholder).any():
        raise ContractError(f"Risk-set duration labels are incomplete: outer {outer_fold} {role}")
    duration = (
        indexed["duration_months"]
        .fillna(0.0)
        .to_numpy("float32")
        .reshape(len(participant_ids), 7)
    )
    event = indexed["event"].to_numpy("int64").reshape(len(participant_ids), 7)
    at_risk = indexed["at_risk"].to_numpy("int64").reshape(len(participant_ids), 7)
    strict_matrix = (
        indexed["strict_healthy_reference_60m"]
        .to_numpy("int64")
        .reshape(len(participant_ids), 7)
    )
    if not (strict_matrix == strict_matrix[:, :1]).all():
        raise ContractError(f"Strict-health flag differs across organs: outer {outer_fold} {role}")
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
        raise RuntimeError("Captured checkpoint differs from the frozen E55 protocol")
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
    intended = Path(row["intended_run_dir"]).resolve()
    if intended != root.resolve():
        raise ContractError(f"Task path differs from frozen queue: {row['run_id']}")
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
            payload = read_json(paths["complete"])
            state = "COMPLETE"
            detail = str(payload.get("completed_at", ""))
        elif paths["failure"].is_file():
            payload = read_json(paths["failure"])
            state = "FAILED"
            detail = str(payload.get("error", ""))
        elif paths["root"].exists():
            state = "INCOMPLETE"
        records.append({**row, "run_state": state, "detail": detail})
    return pd.DataFrame(records)


def save_status(
    rows: list[dict[str, str]], *, active_run_id: str | None, message: str
) -> None:
    table = scan_status(rows)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(STATUS_CSV_PATH, index=False, lineterminator="\n")
    counts = table["run_state"].value_counts().to_dict()
    write_json_atomic(
        STATUS_PATH,
        {
            "schema_version": "revision0710.fig04_fig05.corrected_primary.run_status.v1",
            "updated_at": utc_now(),
            "active_run_id": active_run_id,
            "message": message,
            "stop_after_current_present": stop_requested(RUN_ROOT),
            "counts": {str(key): int(value) for key, value in counts.items()},
            "queue_total": int(len(table)),
            "queue_sha256": sha256_file(QUEUE_PATH),
            "frozen_contract_sha256": sha256_file(CONTRACT_PATH),
        },
    )


def write_fold0_gate(row: dict[str, str]) -> dict[str, Any]:
    if row["run_id"] != FOLD0_GATE_RUN_ID:
        raise ContractError("Fold0 gate requested for a non-canonical task")
    paths = task_paths(row)
    if not paths["complete"].is_file():
        raise ContractError("Canonical Fold0 task is not complete")
    complete = read_json(paths["complete"])
    metrics = complete.get("fold_metrics")
    if not isinstance(metrics, dict):
        raise ContractError("Canonical Fold0 completion lacks readable metrics")
    payload = {
        "schema_version": "revision0710.fig04_fig05.corrected_primary.fold0_gate.v1",
        "status": "AWAITING_REVIEW",
        "created_at": utc_now(),
        "run_id": row["run_id"],
        "variant_id": row["variant_id"],
        "outer_fold": int(row["outer_fold"]),
        "checkpoint_sha256": complete.get("checkpoint_sha256"),
        "metrics": metrics,
        "task_complete_path": str(paths["complete"].resolve()),
        "fold_metrics_csv": str((paths["outer"] / "fold_metrics.csv").resolve()),
        "frozen_contract_sha256": sha256_file(CONTRACT_PATH),
        "next_action": "Review metrics, then relaunch with --continue-after-fold0-gate.",
    }
    write_json_atomic(FOLD0_GATE_JSON, payload)
    lines = [
        "# Canonical control Fold0 gate",
        "",
        "Status: AWAITING_REVIEW",
        "",
        f"- Run: `{row['run_id']}`",
        f"- Checkpoint SHA256: `{complete.get('checkpoint_sha256')}`",
        f"- Contract SHA256: `{sha256_file(CONTRACT_PATH)}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "After review, relaunch with `--continue-after-fold0-gate`.",
            "",
        ]
    )
    FOLD0_GATE_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    temporary = FOLD0_GATE_MARKDOWN.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, FOLD0_GATE_MARKDOWN)
    return payload


def _binding_summary(contract: dict[str, Any]) -> dict[str, str]:
    names = {
        "corrected_mapping",
        "corrected_rebuild_manifest",
        "corrected_delta_and_leakage_audit",
        "corrected_organ_outcomes",
    }
    return {
        str(record["name"]): str(record["sha256"])
        for record in contract["corrected_primary_bindings"]
        if record["name"] in names
    }


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
    role_bindings = {
        role: {
            key: get_role_record(contract, outer_fold, role)[key]
            for key in ("manifest", "age_targets", "joint_labels")
        }
        for role in ("train", "calibration", "outer_test")
    }
    task_spec = {
        "schema_version": "revision0710.fig04_fig05.corrected_primary.task_spec.v1",
        "runner": "fig04_fig05_corrected_primary_v1",
        "started_at": utc_now(),
        "queue_row": row,
        "queue_sha256": sha256_file(QUEUE_PATH),
        "frozen_contract_sha256": sha256_file(CONTRACT_PATH),
        "corrected_primary_bindings": _binding_summary(contract),
        "protocol": contract["protocol"],
        "role_bindings": role_bindings,
    }
    write_json_atomic(paths["spec"], task_spec)

    try:
        train_bundle, train_code, train_text = load_role(
            contract, outer_fold=outer_fold, role="train"
        )
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
        write_json_atomic(paths["train"] / "TRAINING_COMPLETE.json", train_result)
        write_json_atomic(paths["train"] / "MODEL_CONFIG.json", asdict(production_config()))
        del train_bundle, train_code, train_text, model
        gc.collect()
        torch.cuda.empty_cache()

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
        calibration_predictions.to_csv(
            paths["calibration"] / "calibration_predictions_raw.csv",
            index=False,
            lineterminator="\n",
        )
        corrected_calibration.to_csv(
            paths["calibration"] / "calibration_predictions_corrected.csv",
            index=False,
            lineterminator="\n",
        )
        write_json_atomic(paths["calibration"] / "age_correction.json", correction)
        write_json_atomic(paths["calibration"] / "cox_nuisance.json", nuisance)
        del (
            calibration_bundle,
            calibration_code,
            calibration_text,
            calibration_model,
            calibration_predictions,
            corrected_calibration,
        )
        gc.collect()
        torch.cuda.empty_cache()

        outer_bundle, outer_code, outer_text = load_role(
            contract, outer_fold=outer_fold, role="outer_test"
        )
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
        corrected_outer.to_csv(
            paths["outer"] / "prediction_rows.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame([fold_record]).to_csv(
            paths["outer"] / "fold_metrics.csv", index=False, lineterminator="\n"
        )
        organ_frame.to_csv(
            paths["outer"] / "organ_metrics.csv", index=False, lineterminator="\n"
        )
        write_json_atomic(
            paths["complete"],
            {
                "schema_version": "revision0710.fig04_fig05.corrected_primary.task_complete.v1",
                "status": "COMPLETE",
                "completed_at": utc_now(),
                "run_id": row["run_id"],
                "variant_id": variant_id,
                "outer_fold": outer_fold,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "fold_metrics": fold_record,
                "frozen_contract_sha256": sha256_file(CONTRACT_PATH),
            },
        )
        del outer_bundle, outer_code, outer_text, outer_model, outer_predictions, corrected_outer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        write_json_atomic(
            paths["failure"],
            {
                "schema_version": "revision0710.fig04_fig05.corrected_primary.failure.v1",
                "status": "FAILED",
                "failed_at": utc_now(),
                "run_id": row["run_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "frozen_contract_sha256": sha256_file(CONTRACT_PATH),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all frozen hashes and exit without opening CUDA or creating task outputs.",
    )
    parser.add_argument(
        "--continue-after-fold0-gate",
        action="store_true",
        help="Continue only after the completed canonical Fold0 metrics have been reviewed.",
    )
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 0:
        raise ValueError("--max-runs must be non-negative")
    contract, rows = launch_preflight()
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "queue_rows": len(rows),
                    "contract_sha256": sha256_file(CONTRACT_PATH),
                    "queue_sha256": sha256_file(QUEUE_PATH),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal corrected-primary Fig4/Fig5 queue")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    save_status(rows, active_run_id=None, message="Corrected-primary formal queue initialized")
    fold0_row = next(row for row in rows if row["run_id"] == FOLD0_GATE_RUN_ID)
    if task_paths(fold0_row)["complete"].is_file() and not args.continue_after_fold0_gate:
        write_fold0_gate(fold0_row)
        save_status(
            rows,
            active_run_id=None,
            message="Canonical Fold0 complete; awaiting metric review before any later task",
        )
        return 0
    completed = 0
    stopped = False
    for row in rows:
        if task_paths(row)["complete"].is_file():
            continue
        if args.max_runs is not None and completed >= args.max_runs:
            break
        if stop_requested(RUN_ROOT):
            save_status(
                rows,
                active_run_id=None,
                message="STOP_AFTER_CURRENT observed between tasks; no new task started",
            )
            stopped = True
            break
        save_status(
            rows,
            active_run_id=row["run_id"],
            message="Training, calibration, and outer-fold scoring in progress",
        )
        run_one(row, contract)
        completed += 1
        save_status(rows, active_run_id=None, message=f"Completed {row['run_id']}")
        if row["run_id"] == FOLD0_GATE_RUN_ID:
            write_fold0_gate(row)
            save_status(
                rows,
                active_run_id=None,
                message="Canonical Fold0 complete; automatic pause for metric review",
            )
            stopped = True
            break
    if not stopped:
        save_status(rows, active_run_id=None, message="Corrected-primary queue process exited normally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
