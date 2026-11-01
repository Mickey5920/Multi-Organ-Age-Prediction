from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

import compute_external_statistics_0716 as statistics
import render_fig13_fig14_0716 as renderer
import run_external_core_0716 as core
import validate_external_artifacts_0716 as validator


ROOT = Path(__file__).resolve().parents[4]
MAIN_MODEL_ROOT = (
    ROOT / "Revision-0710/03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1"
)
ADAPTER_ROOT = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/"
    "05_external_common_support_adapters"
)
MAIN_SUMMARY = MAIN_MODEL_ROOT / "02_summary/fivefold_summary.json"
CHECKPOINT_SET = "main"
WORK = Path()
FINAL = Path()
INPUTS = Path()
MAIN_EXTERNAL_WORK = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/04_external_validation/"
    "core_0717_codebook_v2_huber1"
)

_ORIGINAL_REGISTRY_LOADER = core.load_registry
_ORIGINAL_STAGE_CACHE = core.stage_cache


def checkpoint_directory(fold: int) -> Path:
    if CHECKPOINT_SET == "main":
        if fold == 0:
            return (
                ROOT
                / "Revision-0710/03_ehr_age/06_codebook_v2/01_runs/"
                "shared_private_independent_view_huber1_fold0"
            )
        return MAIN_MODEL_ROOT / f"01_outer_folds/outer_fold_{fold}"
    return ADAPTER_ROOT / CHECKPOINT_SET / f"01_outer_folds/outer_fold_{fold}"


def require_final_model() -> None:
    if not MAIN_SUMMARY.is_file():
        raise FileNotFoundError(
            "Run summarize_frozen_fivefold.py before external validation"
        )
    summary = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    passed = summary["comparison_to_six_formal_baselines"][
        "all_four_strictly_better"
    ]
    if not passed:
        raise RuntimeError("The frozen five-fold EHR-AGE model did not pass all four gates")
    for fold in range(5):
        directory = checkpoint_directory(fold)
        for name in ("checkpoint.pt", "run_config.json", "age_correction.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def compatibility_correction(fold: int) -> Path:
    source = checkpoint_directory(fold) / "age_correction.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    parameters = []
    for row in payload["parameters"]:
        slope = row.get("slope", row.get("age_slope"))
        if slope is None:
            raise RuntimeError(f"age correction lacks slope for fold {fold}")
        parameters.append(
            {
                "organ_system": row["organ_system"],
                "intercept": float(row["intercept"]),
                "slope": float(slope),
                "fit_population_count": int(
                    row.get("fit_population_count", row.get("n_reference_rows", 0))
                ),
            }
        )
    output = INPUTS / f"age_correction_fold_{fold}.json"
    core.write_json(
        output,
        {
            "schema_version": "revision0710.external.codebook_v2.correction.v1",
            "outer_fold": fold,
            "source_path": relative(source),
            "source_sha256": core.sha256_file(source),
            "parameters": parameters,
        },
    )
    return output


def effective_registry() -> dict[str, Any]:
    registry = copy.deepcopy(_ORIGINAL_REGISTRY_LOADER())
    replacements: dict[str, dict[str, Any]] = {}
    for fold in range(5):
        directory = checkpoint_directory(fold)
        checkpoint = directory / "checkpoint.pt"
        correction = compatibility_correction(fold)
        replacements[f"checkpoint_fold_{fold}"] = {
            "artifact_id": f"checkpoint_fold_{fold}",
            "role": f"frozen codebook-v2 checkpoint set {CHECKPOINT_SET}",
            "path_kind": "repo_relative",
            "path": relative(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": core.sha256_file(checkpoint),
            "outer_fold": fold,
            "run_config_path": relative(directory / "run_config.json"),
            "run_config_sha256": core.sha256_file(directory / "run_config.json"),
        }
        replacements[f"correction_fold_{fold}"] = {
            "artifact_id": f"correction_fold_{fold}",
            "role": "codebook-v2 validation-fold age-gap correction",
            "path_kind": "repo_relative",
            "path": relative(correction),
            "bytes": correction.stat().st_size,
            "sha256": core.sha256_file(correction),
            "outer_fold": fold,
        }
    registry["locked_artifacts"] = [
        replacements.get(row["artifact_id"], row)
        for row in registry["locked_artifacts"]
    ]
    return registry


def checkpoint_anchors() -> dict[int, tuple[str, float]]:
    import torch

    result: dict[int, tuple[str, float]] = {}
    for fold in range(5):
        checkpoint = torch.load(
            checkpoint_directory(fold) / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        keys = [key for key in checkpoint if "age_anchor" in key]
        if len(keys) != 1:
            raise RuntimeError(f"fold {fold} has {len(keys)} age-anchor keys")
        result[fold] = (keys[0], float(checkpoint[keys[0]]))
    return result


def shared_adapter_cache(registry: dict[str, Any]) -> dict[str, Path]:
    directory = MAIN_EXTERNAL_WORK / "shared/02_label_blind_cache"
    manifest = directory / "MANIFEST.json"
    source_registry = (
        directory
        / "inference_worker_artifacts/INFERENCE_LOCKED_ARTIFACT_REGISTRY_V8.json"
    )
    if not manifest.is_file() or not source_registry.is_file():
        raise FileNotFoundError(
            "Run the main codebook-v2 external cache before an adapter checkpoint set"
        )
    cached = json.loads(source_registry.read_text(encoding="utf-8"))
    replacements = {row["artifact_id"]: row for row in registry["locked_artifacts"]}
    cached["locked_artifacts"] = [
        replacements.get(row["artifact_id"], row)
        for row in cached["locked_artifacts"]
    ]
    cached["external_sources"] = []
    cached["schema_version"] = "revision0710.external.adapter_inference_registry.v1"
    cached["status"] = "LABEL_BLIND_CACHE_REUSED_WITH_CHECKPOINT_SET_REGISTRY"
    adapter_registry = INPUTS / "INFERENCE_LOCKED_ARTIFACT_REGISTRY_V8.json"
    core.write_json(adapter_registry, cached)
    return {
        "directory": directory,
        "manifest": manifest,
        "inference_artifact_dir": directory / "inference_worker_artifacts",
        "inference_registry": adapter_registry,
        "mimic_density": directory / "mimic_participant_density_label_blind.parquet",
        "nhanes_release_slot_qc": directory / "nhanes_release_slot_qc.csv",
    }


def restore_v2_model(
    root: Path,
    checkpoint_path: Path,
    code_embedding: Any,
    text_embedding: Any,
    *,
    outer_fold: int,
) -> Any:
    import torch

    model_dir = root / "Revision-0710/03_ehr_age/01_model"
    runner_dir = root / "Revision-0710/03_ehr_age/02_runner"
    for path in (model_dir, runner_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from ehr_age_model import EHRAGEModel
    from ehr_age_training import _model_config

    run_config_path = checkpoint_path.parent / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if int(run_config["outer_fold"]) != int(outer_fold):
        raise RuntimeError("checkpoint and requested outer fold disagree")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    keys = [key for key in checkpoint if "age_anchor" in key]
    if len(keys) != 1:
        raise RuntimeError("checkpoint must contain exactly one age anchor")
    model = EHRAGEModel(
        code_embedding,
        text_embedding,
        age_anchor=float(checkpoint[keys[0]]),
        config=_model_config(run_config["model"]),
    )
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = checkpoint["trainable_state_dict"]
    if set(trainable) != set(state):
        raise RuntimeError("codebook-v2 checkpoint trainable tensor names changed")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state[name].to(dtype=parameter.dtype))
        buffers = dict(model.named_buffers())
        stored = checkpoint["buffer_state_dict"]
        if set(buffers) != set(stored):
            raise RuntimeError("codebook-v2 checkpoint buffer names changed")
        for name, buffer in buffers.items():
            buffer.copy_(stored[name].to(dtype=buffer.dtype))
    return model.cpu().eval()


def configure_modules(checkpoint_set: str) -> None:
    global CHECKPOINT_SET, WORK, FINAL, INPUTS
    CHECKPOINT_SET = checkpoint_set
    suffix = (
        "core_0717_codebook_v2_huber1"
        if checkpoint_set == "main"
        else f"core_0717_adapter_{checkpoint_set}"
    )
    WORK = (
        ROOT
        / "Revision-0710/06_figures_and_statistics/04_external_validation"
        / suffix
    )
    FINAL = ROOT / "Revision-0710/Final_Fig_0716/Codebook_v2_Huber1"
    if checkpoint_set != "main":
        FINAL = FINAL / "External_Adaptation" / checkpoint_set
    INPUTS = WORK / "00_model_inputs"
    require_final_model()
    INPUTS.mkdir(parents=True, exist_ok=True)
    registry = effective_registry()
    anchors = checkpoint_anchors()
    cache = None
    core.stage_cache = _ORIGINAL_STAGE_CACHE
    if checkpoint_set != "main":
        cache = shared_adapter_cache(registry)
        core.stage_cache = lambda: dict(cache)

    core.WORK_DIR = WORK
    core.FINAL_DIR = FINAL
    core.CONTRACT_PATH = WORK / "CORE_CONTRACT_0717_V2.json"
    core.STATUS_PATH = WORK / "RUN_STATUS.json"
    core.load_registry = lambda: copy.deepcopy(registry)
    core.v8.CHECKPOINT_AGE_ANCHORS = anchors
    core.v8._restore_fixed_model = restore_v2_model

    statistics.WORK = WORK
    statistics.CACHE = (
        cache["directory"] if cache is not None else WORK / "shared/02_label_blind_cache"
    )
    statistics.INFERENCE = WORK / "shared/04_inference_work"
    statistics.OUTCOMES = WORK / "shared/05_outcomes_after_prediction_seal"
    statistics.STATS = WORK / "shared/06_statistics"
    statistics.FINAL = FINAL
    statistics.FIG13 = FINAL / "Fig13/Source_Data"
    statistics.FIG14 = FINAL / "Fig14/Source_Data"

    renderer.FINAL = FINAL
    renderer.FIG13 = FINAL / "Fig13"
    renderer.FIG14 = FINAL / "Fig14"
    renderer.WORK = WORK

    validator.FINAL = FINAL
    validator.FIG13 = FINAL / "Fig13"
    validator.FIG14 = FINAL / "Fig14"
    validator.WORK = WORK


def configure_analysis_modules(checkpoint_set: str) -> None:
    global CHECKPOINT_SET, WORK, FINAL, INPUTS
    CHECKPOINT_SET = checkpoint_set
    suffix = (
        "core_0717_codebook_v2_huber1"
        if checkpoint_set == "main"
        else f"core_0717_adapter_{checkpoint_set}"
    )
    WORK = (
        ROOT
        / "Revision-0710/06_figures_and_statistics/04_external_validation"
        / suffix
    )
    FINAL = ROOT / "Revision-0710/Final_Fig_0716/Codebook_v2_Huber1"
    if checkpoint_set != "main":
        FINAL = FINAL / "External_Adaptation" / checkpoint_set
    INPUTS = WORK / "00_model_inputs"
    cache = (
        MAIN_EXTERNAL_WORK / "shared/02_label_blind_cache"
        if checkpoint_set != "main"
        else WORK / "shared/02_label_blind_cache"
    )
    statistics.WORK = WORK
    statistics.CACHE = cache
    statistics.INFERENCE = WORK / "shared/04_inference_work"
    statistics.OUTCOMES = WORK / "shared/05_outcomes_after_prediction_seal"
    statistics.STATS = WORK / "shared/06_statistics"
    statistics.FINAL = FINAL
    statistics.FIG13 = FINAL / "Fig13/Source_Data"
    statistics.FIG14 = FINAL / "Fig14/Source_Data"
    renderer.FINAL = FINAL
    renderer.FIG13 = FINAL / "Fig13"
    renderer.FIG14 = FINAL / "Fig14"
    renderer.WORK = WORK
    validator.FINAL = FINAL
    validator.FIG13 = FINAL / "Fig13"
    validator.FIG14 = FINAL / "Fig14"
    validator.WORK = WORK


def freeze() -> None:
    core.freeze_contract()
    contract = json.loads(core.CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["schema_version"] = "revision0710.external.codebook_v2_huber1.v1"
    contract["model_policy"].update(
        {
            "checkpoint_selection": (
                f"fixed UKB codebook-v2 five-fold checkpoint set: {CHECKPOINT_SET}"
            ),
            "internal_model_fine_tuning": CHECKPOINT_SET != "main",
            "ukb_common_support_adaptation": CHECKPOINT_SET != "main",
            "external_aggregate_feature_coverage_used": CHECKPOINT_SET != "main",
            "external_participant_level_training": False,
            "external_label_supervision": False,
            "external_fine_tuning": False,
            "external_distillation": False,
            "external_adapter_training": False,
            "label_blind_external_cache": (
                "checkpoint-set local"
                if CHECKPOINT_SET == "main"
                else "reused from main codebook-v2 run with a checkpoint-set-specific inference registry"
            ),
            "ensemble": "unweighted mean of five frozen codebook-v2 checkpoints",
        }
    )
    core.write_json(core.CONTRACT_PATH, contract)
    core.update_status(
        "00_freeze",
        "COMPLETE",
        contract=str(core.CONTRACT_PATH),
        contract_sha256=core.sha256_file(core.CONTRACT_PATH),
    )


def reuse_adapter_outcomes() -> None:
    output_dir = WORK / "shared/05_outcomes_after_prediction_seal"
    receipt_path = output_dir / "OUTCOME_READ_RECEIPT.json"
    if receipt_path.is_file():
        core.update_status(
            "04_outcomes",
            "COMPLETE_REUSED",
            receipt=str(receipt_path),
            receipt_sha256=core.sha256_file(receipt_path),
        )
        print(json.dumps({"receipt": str(receipt_path)}))
        return
    adapter_seal = core.stage_sealed()["receipt"]
    main_outcomes = MAIN_EXTERNAL_WORK / "shared/05_outcomes_after_prediction_seal"
    main_receipt_path = main_outcomes / "OUTCOME_READ_RECEIPT.json"
    if not adapter_seal.is_file() or not main_receipt_path.is_file():
        raise FileNotFoundError(
            "Adapter predictions must be sealed and main outcomes completed before reuse"
        )
    main_receipt = json.loads(main_receipt_path.read_text(encoding="utf-8"))
    files = {
        "incident_sha256": "incident_endpoints.parquet",
        "recurrent_sha256": "recurrent_endpoints.parquet",
        "questionnaire_sha256": "nhanes_questionnaire_labels.parquet",
        "eligibility_sha256": "mimic_endpoint_subject_eligibility.parquet",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for hash_key, name in files.items():
        source = main_outcomes / name
        if core.sha256_file(source) != main_receipt[hash_key]:
            raise RuntimeError(f"Main external outcome hash changed: {name}")
        shutil.copy2(source, output_dir / name)
    receipt = {
        **main_receipt,
        "schema_version": "revision0710.external_validation.outcome_reuse.v1",
        "status": "MODEL_INDEPENDENT_OUTCOMES_REUSED_AFTER_ADAPTER_PREDICTION_SEAL",
        "checkpoint_set": CHECKPOINT_SET,
        "prediction_seal_sha256": core.sha256_file(adapter_seal),
        "main_outcome_receipt_path": str(main_receipt_path.resolve()),
        "main_outcome_receipt_sha256": core.sha256_file(main_receipt_path),
        "raw_external_outcomes_reread": False,
    }
    core.write_json(receipt_path, receipt)
    core.update_status(
        "04_outcomes",
        "COMPLETE_REUSED_MODEL_INDEPENDENT_TABLES",
        receipt=str(receipt_path),
        receipt_sha256=core.sha256_file(receipt_path),
    )
    print(json.dumps({"receipt": str(receipt_path)}))


def run_stage(stage: str, batch_size: int, bootstrap_replicates: int) -> None:
    if stage == "freeze":
        freeze()
    elif stage == "cache":
        core.build_cache()
    elif stage == "infer":
        core.gpu_inference(batch_size)
    elif stage == "seal":
        core.seal_predictions()
    elif stage == "outcomes":
        if CHECKPOINT_SET == "main":
            core.build_outcomes()
        else:
            reuse_adapter_outcomes()
    elif stage == "basic":
        statistics.compute_basic()
    elif stage == "risk":
        statistics.compute_risk(bootstrap_replicates)
    elif stage == "render":
        renderer.main()
    elif stage == "validate":
        validator.main()
    else:
        raise ValueError(stage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external validation with the final codebook-v2 Huber1 model."
    )
    parser.add_argument(
        "stage",
        choices=(
            "freeze",
            "cache",
            "infer",
            "seal",
            "outcomes",
            "basic",
            "risk",
            "render",
            "validate",
            "all",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument(
        "--checkpoint-set",
        choices=("main", "mimic_common53", "nhanes_common_eventless"),
        default="main",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"basic", "risk", "render", "validate"}:
        configure_analysis_modules(args.checkpoint_set)
    else:
        configure_modules(args.checkpoint_set)
    stages = (
        "freeze",
        "cache",
        "infer",
        "seal",
        "outcomes",
        "basic",
        "risk",
        "render",
        "validate",
    )
    if args.stage == "all":
        for stage in stages:
            run_stage(stage, args.batch_size, args.bootstrap_replicates)
    else:
        run_stage(args.stage, args.batch_size, args.bootstrap_replicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
