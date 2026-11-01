from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import isolated_worker
import v8_contract
import v8_data
import v8_inputs
import v8_loss
import v8_model
import v8_training


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
CACHE_DIR = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "04_cache_v1"
BATCH_SIZE = 2048
SEQUENCE_LENGTH = 32
PHENOTYPE_FEATURES = 78


def _write_json_create_only(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _production_shape_synthetic_batch(seed: int = 7210) -> v8_training.TrainingBatch:
    generator = torch.Generator().manual_seed(seed)
    n = BATCH_SIZE
    seq = SEQUENCE_LENGTH
    event_type = torch.randint(1, 3, (n, seq), generator=generator)
    event_code = torch.randint(1, 9906, (n, seq), generator=generator)
    event_code[event_type.eq(2)] = 0
    inputs = {
        "phenotype": torch.randn((n, PHENOTYPE_FEATURES), generator=generator),
        "phenotype_missing_mask": torch.rand(
            (n, PHENOTYPE_FEATURES), generator=generator
        ) < 0.08,
        "sex": torch.randint(0, 2, (n,), generator=generator),
        "event_code_id": event_code,
        "event_text_id": torch.randint(1, 44565, (n, seq), generator=generator),
        "event_relative_days": torch.randint(-3650, 0, (n, seq), generator=generator),
        "event_type_id": event_type,
        "event_valid_mask": torch.ones((n, seq), dtype=torch.bool),
    }
    labels = {
        "chronological_age": 40.0 + 35.0 * torch.rand((n,), generator=generator),
        "strict_healthy_reference": torch.arange(n) % 3 == 0,
        "duration_months": 3.0 + 57.0 * torch.rand((n, 7), generator=generator),
        "event": torch.rand((n, 7), generator=generator) < 0.12,
        "at_risk": torch.ones((n, 7), dtype=torch.bool),
    }
    return v8_training.TrainingBatch(
        participant_ids=tuple(f"shape_only_{index:04d}" for index in range(n)),
        inputs=inputs,
        labels=labels,
    )


def _load_bound_embeddings() -> tuple[torch.Tensor, torch.Tensor, dict]:
    path = CACHE_DIR / "embedding_tables.npz"
    contract = v8_contract.load_contract(ROOT)
    records = {
        row["name"]: row for row in contract["canonical_bindings"]["records"]
    }
    manifest = json.loads((CACHE_DIR / "embedding_manifest.json").read_text(encoding="utf-8"))
    table_record = records.get("embedding_tables")
    if table_record is None:
        raise RuntimeError("contract does not bind the production embedding table")
    table_path = REPO_ROOT / table_record["path"]
    if (
        v8_contract.sha256_file(path) != manifest["embedding_numpy_sha256"]
        or v8_contract.sha256_file(table_path) != table_record["sha256"]
        or table_record["sha256"] != manifest["embedding_table_sha256"]
    ):
        raise RuntimeError("production embedding table or numpy mirror changed")
    with np.load(path, allow_pickle=False) as payload:
        code = torch.from_numpy(np.asarray(payload["code_embedding"]).copy())
        text = torch.from_numpy(np.asarray(payload["text_embedding"]).copy())
    if tuple(code.shape) != (9906, 512) or tuple(text.shape) != (44565, 512):
        raise RuntimeError("production embedding table shapes changed")
    if code.dtype != torch.float16 or text.dtype != torch.float16:
        raise RuntimeError("production embedding table dtype changed")
    return code, text, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": v8_contract.sha256_file(path),
        "manifest_path": str((CACHE_DIR / "embedding_manifest.json").resolve()),
        "manifest_sha256": v8_contract.sha256_file(CACHE_DIR / "embedding_manifest.json"),
        "manifest": manifest,
        "contract_canonical_record_count": len(records),
        "contract_embedding_table": table_record,
    }


def run_preflight(output_dir: Path) -> dict:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be set before torch/CUDA startup")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA production-shape preflight requires an available GPU")
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = v8_contract.load_contract(ROOT)
    contract_sha256 = v8_contract.sha256_file(ROOT / "FROZEN_ABLATION_CONTRACT_V8.json")
    formal_root = Path(contract["execution_roots"]["formal_run_root"]).resolve()
    try:
        output_dir.resolve().relative_to(formal_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("CUDA preflight output must not be inside the formal run root")

    code, text, embedding_record = _load_bound_embeddings()
    config = isolated_worker._model_config(tiny=False)
    batch = _production_shape_synthetic_batch()
    device = torch.device("cuda")
    variant_receipts = []
    peak_bytes = 0
    for ordinal, variant_id in enumerate(v8_contract.VARIANT_IDS, start=1):
        started = time.perf_counter()
        v8_training.configure_reproducibility(7210, deterministic=True, tf32=True)
        model = v8_model.build_production_model(
            variant_id, code, text, age_anchor=58.0, config=config
        ).to(device)
        manifest = v8_training.build_optimizer_manifest(model, variant_id)
        optimizer = v8_training.build_adamw(
            model, manifest, lr=0.0002, weight_decay=0.0001
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        initial_state_sha = v8_training.canonical_state_sha256(model.state_dict())
        initial_optimizer_sha = v8_training.canonical_state_sha256(optimizer.state_dict())
        inputs = {
            key: value.to(device)
            for key, value in v8_inputs.apply_variant_inputs(batch.inputs, variant_id).items()
        }
        labels = {key: value.to(device) for key, value in batch.labels.items()}
        policy = v8_loss.variant_loss_policy(variant_id)
        dropout_generator = torch.Generator(device="cuda").manual_seed(8219)
        thinned = v8_inputs.compact_event_dropout(
            inputs,
            probability=float(policy["event_dropout_probability"]),
            generator=dropout_generator,
        )
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(**inputs)
                augmented = (
                    model(**thinned)
                    if float(policy["event_dropout_probability"]) > 0.0
                    else None
                )
                losses, _ = v8_loss.compute_variant_loss(
                    variant_id,
                    output,
                    labels=labels,
                    interval_edges_months=v8_training.INTERVAL_EDGES.to(device),
                    survival_multiplier=0.1,
                    augmented_output=augmented,
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"non-finite CUDA AMP loss: {variant_id}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss_value = float(losses["total"].detach())
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"production-shape CUDA OOM at batch {BATCH_SIZE}: {variant_id}"
            ) from exc
        post_state_sha = v8_training.canonical_state_sha256(model.state_dict())
        if post_state_sha == initial_state_sha:
            raise RuntimeError(f"CUDA backward/step did not change parameters: {variant_id}")
        checkpoint_path = output_dir / f"v{ordinal:02d}_shape_checkpoint.pt"
        torch.save(
            {
                "schema_version": "revision0710.fig04_fig05.ablation_checkpoint.v8",
                "status": "SYNTHETIC_TARGET_PRODUCTION_SHAPE_CUDA_PREFLIGHT_ONLY",
                "variant_id": variant_id,
                "epoch": 55,
                "seed": 7210,
                "model_config": asdict(config),
                "age_anchor": 58.0,
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_parameter_names": manifest["optimizer_parameter_names"],
                "initial_trainable_sha256": manifest["initial_trainable_sha256"],
                "initial_full_model_sha256": initial_state_sha,
                "initial_optimizer_sha256": initial_optimizer_sha,
                "training_trajectory_sha256_at_capture": "0" * 64,
            },
            checkpoint_path,
        )
        variant_peak = int(torch.cuda.max_memory_allocated())
        peak_bytes = max(peak_bytes, variant_peak)
        elapsed = time.perf_counter() - started
        variant_receipts.append(
            {
                "ordinal": ordinal,
                "variant_id": variant_id,
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "phenotype_features": PHENOTYPE_FEATURES,
                "amp_autocast": True,
                "backward": True,
                "optimizer_step": True,
                "gradient_clip": True,
                "loss": loss_value,
                "preclip_gradient_norm": float(gradient_norm),
                "initial_model_sha256": initial_state_sha,
                "post_step_model_sha256": post_state_sha,
                "checkpoint_path": checkpoint_path.name,
                "checkpoint_sha256": v8_contract.sha256_file(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "peak_cuda_memory_bytes": variant_peak,
                "elapsed_seconds_one_step_and_checkpoint": elapsed,
            }
        )
        del model, optimizer, output, augmented, losses, inputs, labels, thinned
        torch.cuda.empty_cache()

    receipt = {
        "schema_version": "revision0710.fig04_fig05.cuda_production_shape_preflight.v8",
        "status": "PASS_12_OF_12_BATCH2048_PRODUCTION_SHAPE_AMP_STEPS",
        "contract_sha256": contract_sha256,
        "preflight_script_sha256": v8_contract.sha256_file(Path(__file__).resolve()),
        "variant_count": 12,
        "variant_ids": list(v8_contract.VARIANT_IDS),
        "variants": variant_receipts,
        "shape_contract": {
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "phenotype_features": PHENOTYPE_FEATURES,
            "code_embedding_shape": [9906, 512],
            "text_embedding_shape": [44565, 512],
            "model_config": asdict(config),
        },
        "input_content": "deterministic_synthetic_inputs_and_targets_no_participant_rows",
        "participant_labels_opened": False,
        "calibration_role_opened": False,
        "outer_role_opened": False,
        "embedding_record": embedding_record,
        "all_amp_autocast": True,
        "all_backward": True,
        "all_optimizer_step": True,
        "all_gradient_clip": True,
        "all_parameter_state_changed": True,
        "oom_policy": "HARD_FAIL_NO_BATCH_OR_SEQUENCE_DOWNGRADE",
        "oom_observed": False,
        "peak_cuda_memory_bytes": peak_bytes,
        "capacity_scope": "one_full_batch_optimizer_step_per_variant_not_e60_runtime_benchmark",
        "formal_protocol_covers_same_shape": True,
        "formal_authorization_used": False,
        "formal_output_used": False,
    }
    receipt_path = output_dir / "CUDA_PRODUCTION_SHAPE_PREFLIGHT_RECEIPT.json"
    _write_json_create_only(receipt_path, receipt)
    return {
        **receipt,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": v8_contract.sha256_file(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
