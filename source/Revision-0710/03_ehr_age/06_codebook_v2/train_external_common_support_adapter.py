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
EHR_ROOT = WORKSPACE_ROOT / "Revision-0710/03_ehr_age"
for module_path in (EHR_ROOT / "02_runner", EHR_ROOT / "01_model"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from ehr_age_data import JointVectorizedBatcher, sha256_file
from ehr_age_model import ORGAN_SYSTEMS
from ehr_age_training import (
    _loss_config,
    _predict,
    _survival_multiplier,
    _train_epoch,
    pooled_age_metrics,
)
from run_fold0_current import (
    SEED,
    append_jsonl,
    atomic_json,
    make_model_and_optimizer,
    prediction_ids,
    save_checkpoint,
    utc_now,
)
from run_frozen_outer_fold import (
    BASELINE_ENVELOPE,
    BASELINE_TARGETS,
    OUTCOME_COLUMNS,
    OUTCOMES_PATH,
    ROLE_DIRECTORIES,
    _comparison,
    _organ_age_metrics,
    _serialize_outer_correction,
    calibration_to_outer_cox,
    load_role_data,
    official_baseline_values,
    organ_baseline_values,
    role_fold_ids,
)


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
MAIN_ROOT = HERE / "02_fivefold_frozen_huber1"
OUTPUT_ROOT = HERE / "05_external_common_support_adapters"
MIMIC_COVERAGE = (
    REVISION_ROOT / "Final_Fig_0716/Fig13/Source_Data/MIMIC_feature_coverage.csv"
)
NHANES_COVERAGE = (
    REVISION_ROOT / "Final_Fig_0716/Fig14/Source_Data/NHANES_feature_coverage.csv"
)
ADAPTER_EPOCHS = 8
BASE_EPOCH = 47
ADAPTER_LR = 2e-5
VARIANTS = ("mimic_common53", "nhanes_common_eventless")


def main_directory(fold: int) -> Path:
    if fold == 0:
        return HERE / "01_runs/shared_private_independent_view_huber1_fold0"
    return MAIN_ROOT / f"01_outer_folds/outer_fold_{fold}"


def load_support(variant: str) -> pd.DataFrame:
    if variant == "mimic_common53":
        coverage = pd.read_csv(MIMIC_COVERAGE)
        primary = coverage.loc[
            coverage["input_variant"].astype(str).eq(
                "mimic_primary_hospital_icd_only_explicit_unit_phenotype"
            )
        ].copy()
        if len(primary) != 78 or int(primary["slot_index"].nunique()) != 78:
            raise RuntimeError("MIMIC primary coverage must adjudicate 78 slots")
        support = primary[["slot_index", "feature_name", "observed_fraction"]].copy()
        support["available"] = primary["observed_n"].gt(0).to_numpy()
        support["event_mode"] = "hospital_icd_only"
        expected = 53
    elif variant == "nhanes_common_eventless":
        coverage = pd.read_csv(NHANES_COVERAGE)
        if set(coverage["release_id"].astype(str)) != {
            "P_2017_MARCH2020_PREPANDEMIC",
            "L_AUG2021_AUG2023",
        }:
            raise RuntimeError("NHANES P/L release coverage changed")
        rows = []
        for slot, group in coverage.groupby("slot_index", observed=True):
            if len(group) != 2:
                raise RuntimeError(f"NHANES slot {slot} is not present in both releases")
            available = bool(group["observed_n"].gt(0).all())
            rows.append(
                {
                    "slot_index": int(slot),
                    "feature_name": str(group["feature_name"].iloc[0]),
                    "observed_fraction": (
                        float(group["observed_fraction"].min()) if available else 0.0
                    ),
                    "available": available,
                    "event_mode": "none",
                }
            )
        support = pd.DataFrame(rows)
        expected = int(
            coverage.pivot_table(
                index="slot_index", columns="release_id", values="observed_n"
            ).gt(0).all(axis=1).sum()
        )
    else:
        raise ValueError(variant)
    support = support.sort_values("slot_index").reset_index(drop=True)
    if support["slot_index"].tolist() != list(range(78)):
        raise RuntimeError("common-support table changed the 78-slot axis")
    if int(support["available"].sum()) != expected:
        raise RuntimeError(
            f"{variant} support count changed: {int(support['available'].sum())} != {expected}"
        )
    return support


def apply_support(
    data: Any,
    support: pd.DataFrame,
    *,
    outer_fold: int,
    role: str,
    variant: str,
) -> dict[str, Any]:
    inputs = data.cache.dataset.inputs
    phenotype = inputs["phenotype"]
    missing = inputs["phenotype_missing_mask"]
    if phenotype.shape[1] != 78 or missing.shape != phenotype.shape:
        raise RuntimeError("adapter expected the frozen 78-slot phenotype axis")
    transform = data.cache.phenotype_transform
    median_z = torch.from_numpy(
        ((transform.median - transform.mean) / transform.std).astype(np.float32)
    )
    role_index = {"train": 1, "calibration": 2, "outer_test": 3}[role]
    variant_index = VARIANTS.index(variant) + 1
    generator = torch.Generator().manual_seed(
        SEED + outer_fold * 1_000 + role_index * 100 + variant_index * 10
    )
    original_observed = int((~missing).sum())
    for row in support.itertuples(index=False):
        slot = int(row.slot_index)
        target_fraction = (
            float(row.observed_fraction) if bool(row.available) else 0.0
        )
        source_fraction = float((~missing[:, slot]).float().mean())
        if target_fraction <= 0.0 or source_fraction <= 0.0:
            missing[:, slot] = True
        elif target_fraction < source_fraction:
            retention_probability = min(1.0, target_fraction / source_fraction)
            retained = torch.rand(len(missing), generator=generator).lt(
                retention_probability
            )
            missing[:, slot] |= ~retained
        phenotype[:, slot] = torch.where(
            missing[:, slot], median_z[slot], phenotype[:, slot]
        )
    event_mode = str(support["event_mode"].iloc[0])
    valid_before = inputs["event_valid_mask"].bool().clone()
    type_before = inputs["event_type_id"].clone()
    icd_before = int((valid_before & type_before.eq(1)).sum())
    gp_before = int((valid_before & type_before.eq(2)).sum())
    if event_mode == "hospital_icd_only":
        keep = valid_before & type_before.eq(1)
        destination = keep.long().cumsum(dim=1) - 1
        rows, columns = torch.nonzero(keep, as_tuple=True)
        target_columns = destination[rows, columns]
        for key in (
            "event_code_id",
            "event_text_id",
            "event_relative_days",
            "event_type_id",
            "event_valid_mask",
        ):
            source = inputs[key]
            compacted = torch.zeros_like(source)
            compacted[rows, target_columns] = source[rows, columns]
            inputs[key] = compacted
    elif event_mode == "none":
        for key in (
            "event_code_id",
            "event_text_id",
            "event_relative_days",
            "event_type_id",
        ):
            inputs[key].zero_()
        inputs["event_valid_mask"].zero_()
    else:
        raise RuntimeError(f"unsupported external support event mode: {event_mode}")
    valid_after = inputs["event_valid_mask"].bool()
    type_after = inputs["event_type_id"]
    observed_after = int((~missing).sum())
    phenotype_cell_n = int(missing.numel())
    return {
        "participants": len(data.cache.dataset),
        "phenotype_cells_observed_before": original_observed,
        "phenotype_cells_observed_after": observed_after,
        "phenotype_observed_fraction_before": original_observed / phenotype_cell_n,
        "phenotype_observed_fraction_after": observed_after / phenotype_cell_n,
        "external_target_mean_observed_fraction": float(
            support["observed_fraction"].mean()
        ),
        "available_slots": int(support["available"].sum()),
        "event_mode": event_mode,
        "active_events_before": int(valid_before.sum()),
        "hospital_icd_events_before": icd_before,
        "gp_prescription_events_before": gp_before,
        "active_events_after": int(valid_after.sum()),
        "hospital_icd_events_after": int((valid_after & type_after.eq(1)).sum()),
        "gp_prescription_events_after": int((valid_after & type_after.eq(2)).sum()),
    }


def restore_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> float:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != set(checkpoint["trainable_state_dict"]):
        raise RuntimeError("adapter source checkpoint trainable state is incompatible")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(
                checkpoint["trainable_state_dict"][name].to(dtype=parameter.dtype)
            )
        buffers = dict(model.named_buffers())
        if set(buffers) != set(checkpoint["buffer_state_dict"]):
            raise RuntimeError("adapter source checkpoint buffers are incompatible")
        for name, buffer in buffers.items():
            buffer.copy_(checkpoint["buffer_state_dict"][name].to(dtype=buffer.dtype))
    keys = [key for key in checkpoint if "age_anchor" in key]
    if len(keys) != 1:
        raise RuntimeError("adapter source checkpoint must contain one age anchor")
    return float(checkpoint[keys[0]])


def run_adapter(variant: str, outer_fold: int, output_dir: Path) -> Path:
    source_dir = main_directory(outer_fold)
    source_checkpoint = source_dir / "checkpoint.pt"
    source_config = source_dir / "run_config.json"
    source_metrics = source_dir / "metrics.json"
    for path in (source_checkpoint, source_config, source_metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    support = load_support(variant)
    outcomes = pd.read_parquet(OUTCOMES_PATH, columns=list(OUTCOME_COLUMNS))
    outcomes["participant_id"] = outcomes["participant_id"].astype(str)
    bundles = {
        role: load_role_data(outcomes, outer_fold=outer_fold, role=role)
        for role in ROLE_DIRECTORIES
    }
    del outcomes
    adaptations = {
        role: apply_support(
            bundle,
            support,
            outer_fold=outer_fold,
            role=role,
            variant=variant,
        )
        for role, bundle in bundles.items()
    }
    raw = json.loads(source_config.read_text(encoding="utf-8"))
    raw["training"]["learning_rate"] = ADAPTER_LR
    raw["training"]["warmup_epochs"] = 1
    model, device, optimizer, scheduler, scaler, edges, age_anchor = (
        make_model_and_optimizer(raw, bundles["train"])
    )
    restored_anchor = restore_checkpoint(model, source_checkpoint)
    if not np.isclose(age_anchor, restored_anchor, atol=1e-5, rtol=0.0):
        raise RuntimeError("adapter source and current train-fold age anchors differ")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    support.to_csv(output_dir / "common_support.csv", index=False)
    atomic_json(
        output_dir / "run_config.json",
        {
            "schema_version": "revision0710.external_common_support_adapter.v1",
            "created_utc": utc_now(),
            "variant": variant,
            "outer_fold": outer_fold,
            "train_folds": list(role_fold_ids(outer_fold)["train"]),
            "calibration_fold": (outer_fold + 1) % 5,
            "adapter_epochs": ADAPTER_EPOCHS,
            "adapter_learning_rate": ADAPTER_LR,
            "fine_tune": True,
            "distillation": False,
            "external_participant_level_data_read": False,
            "external_aggregate_feature_coverage_read": True,
            "external_age_labels_read": False,
            "external_outcome_labels_read": False,
            "selection_on_external_results": False,
            "source_checkpoint": {
                "path": str(source_checkpoint.resolve()),
                "sha256": sha256_file(source_checkpoint),
                "epoch": BASE_EPOCH,
            },
            "support_sources": {
                "mimic": {
                    "path": str(MIMIC_COVERAGE.resolve()),
                    "sha256": sha256_file(MIMIC_COVERAGE),
                },
                "nhanes": {
                    "path": str(NHANES_COVERAGE.resolve()),
                    "sha256": sha256_file(NHANES_COVERAGE),
                },
            },
            "adapted_role_counts": adaptations,
            "model": raw["model"],
            "loss": raw["loss"],
            "training": raw["training"],
        },
    )
    started = time.perf_counter()
    train_data = bundles["train"]
    batcher = JointVectorizedBatcher(
        train_data,
        train_data.train_indices,
        batch_size=int(raw["training"]["batch_size"]),
        shuffle=True,
        seed=SEED,
    )
    for local_epoch in range(1, ADAPTER_EPOCHS + 1):
        epoch_started = time.perf_counter()
        continued_epoch = BASE_EPOCH + local_epoch
        losses = _train_epoch(
            model=model,
            batcher=batcher,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            interval_edges=edges,
            loss_config=_loss_config(raw["loss"]),
            survival_multiplier=_survival_multiplier(continued_epoch, raw["loss"]),
            event_dropout_probability=float(
                raw["training"]["event_dropout_probability"]
            ),
            gradient_clip_norm=float(raw["training"]["gradient_clip_norm"]),
            amp=bool(raw["training"]["amp"]),
            max_batches=None,
        )
        scheduler.step()
        row = {
            "adapter_epoch": local_epoch,
            "continued_epoch": continued_epoch,
            "seconds": time.perf_counter() - epoch_started,
            "losses": {key: float(value) for key, value in losses.items()},
        }
        append_jsonl(output_dir / "training_history.jsonl", row)
        print(
            f"adapter={variant} outer={outer_fold} epoch={local_epoch:02d}/"
            f"{ADAPTER_EPOCHS} loss={float(losses['total']):.6f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, BASE_EPOCH + ADAPTER_EPOCHS, age_anchor)
    predictions = {}
    for role, index_name in (
        ("calibration", "calibration_indices"),
        ("outer_test", "screen_indices"),
    ):
        data = bundles[role]
        prediction_batcher = JointVectorizedBatcher(
            data,
            getattr(data, index_name),
            batch_size=int(raw["training"]["batch_size"]),
            shuffle=False,
            seed=SEED,
        )
        predictions[role] = _predict(
            model=model,
            batcher=prediction_batcher,
            device=device,
            amp=bool(raw["training"]["amp"]),
            max_batches=None,
        )
    targets = json.loads(BASELINE_TARGETS.read_text(encoding="utf-8"))
    envelope = json.loads(BASELINE_ENVELOPE.read_text(encoding="utf-8"))
    baseline = official_baseline_values(targets)
    organ_baseline = organ_baseline_values(envelope)
    risk, calibration_frame, outer_frame, correction = calibration_to_outer_cox(
        predictions["calibration"],
        predictions["outer_test"],
        calibration_participant_ids=prediction_ids(
            bundles["calibration"], predictions["calibration"]
        ),
        outer_participant_ids=prediction_ids(
            bundles["outer_test"], predictions["outer_test"]
        ),
        outer_fold=outer_fold,
        baseline_organ_delta_c=organ_baseline,
    )
    calibration_frame["model_risk_score"] = predictions[
        "calibration"
    ].risk_score.astype(np.float32).reshape(-1)
    calibration_frame["outer_fold"] = outer_fold
    outer_frame["model_risk_score"] = predictions["outer_test"].risk_score.astype(
        np.float32
    ).reshape(-1)
    outer_frame["outer_fold"] = outer_fold
    calibration_frame.to_parquet(
        output_dir / "calibration_predictions.parquet", index=False
    )
    outer_frame.to_parquet(output_dir / "outer_predictions.parquet", index=False)
    atomic_json(
        output_dir / "age_correction.json",
        _serialize_outer_correction(correction, outer_fold),
    )
    outer = predictions["outer_test"]
    strict = pooled_age_metrics(
        outer.chronological_age,
        outer.predicted_age,
        participant_mask=outer.strict_healthy,
    )
    metrics = {
        "schema_version": "revision0710.external_common_support_adapter.metrics.v1",
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "variant": variant,
        "outer_fold": outer_fold,
        "strict_age_raw": {
            **strict,
            "organs": _organ_age_metrics(
                outer.chronological_age, outer.predicted_age, outer.strict_healthy
            ),
        },
        "full_risk": risk,
        "comparison_to_six_formal_baselines": _comparison(strict, risk, baseline),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
    }
    atomic_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics["strict_age_raw"], indent=2), flush=True)
    print(json.dumps({"full_risk_macro_delta_c": risk["macro_delta_c"]}), flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune frozen UKB folds to external common input support."
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir or (
        OUTPUT_ROOT / args.variant / "01_outer_folds" / f"outer_fold_{args.outer_fold}"
    )
    print(run_adapter(args.variant, args.outer_fold, output))


if __name__ == "__main__":
    main()
