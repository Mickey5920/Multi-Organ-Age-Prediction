from __future__ import annotations

import json
import hashlib
import math
import re
import shlex
import time
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from d07_open_fold0_development import (
    CANDIDATE_SCHEMA,
    DEFAULT_BASELINE_ARTIFACT_MANIFEST,
    DEFAULT_BASELINE_ENVELOPE,
    DEFAULT_BASELINE_VALIDATOR,
    DEFAULT_CACHE_DIR,
    DEFAULT_DEVELOPMENT_AGES,
    DEFAULT_DEVELOPMENT_LABELS,
    DEFAULT_DEVELOPMENT_MANIFEST,
    DEFAULT_FOLD0_AGES,
    DEFAULT_FOLD0_LABELS,
    DEFAULT_FOLD0_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    D07ContractError,
    D07_PROTOCOL_PATH,
    D07_PROTOCOL_SHA256,
    D04_INITIAL_TRAINABLE_SHA256,
    _organ_age_metrics,
    _serialize_correction,
    _source_hashes,
    _utc_now,
    _verify_sealed_manifest_metadata,
    calibration_to_open_fold0_cox,
    commit_open_development_read,
    load_authorized_open_fold0_data,
    validate_d07_fold0_baseline_envelope,
    validate_d07_protocol,
    write_artifact_hash_registry,
    write_fold0_read_completion_receipt,
    write_fold0_read_failure_receipt,
)
from ehr_age_data import (
    MODEL_DIR,
    REVISION_ROOT,
    Fold0DevelopmentData,
    JointVectorizedBatcher,
    canonical_hash,
    load_fold0_development_data,
    sha256_file,
)
from ehr_age_model import (
    BASE_AGE_MODE_PRIVATE_RETAIN,
    BASE_AGE_MODE_V1,
    BASE_AGE_POOLING_MODE_GLOBAL_FUSED,
    EHRAGEModel,
    ORGAN_SYSTEMS,
)
from private_retain_age import (
    FORMAL_COMMON_RISK_HIDDEN_DIM,
    PRIVATE_RETAIN_HIDDEN_DIM,
    PrivateRETAINAgeTower,
    PrivateRETAINBackbone,
    PrivateRETAINEventTokenEncoder,
    PrivateRETAINStaticEncoder,
)
from ehr_age_training import (
    PredictionBundle,
    SelectionGates,
    _evaluation_frame,
    _loss_config,
    _model_config,
    _predict,
    _survival_multiplier,
    _train_epoch,
    configure_reproducibility,
    pooled_age_metrics,
    trainable_parameter_hash,
)


RUNNER_DIR = Path(__file__).resolve().parent
EHR_AGE_ROOT = RUNNER_DIR.parent
WORKSPACE_ROOT = REVISION_ROOT.parent
DEFAULT_D07_CANDIDATE_CONFIG = (
    EHR_AGE_ROOT
    / "00_contract"
    / "ehr_age_d07_002_baseline_native_objective_v1.json"
)
DEFAULT_D07_002_CONFIG_SHA256 = (
    "bb52ebb59cdb6facc928f87a7bb7e5bd1712e8dba61bbf8de11745423d640e9e"
)

GENERIC_RUN_SCHEMA = "revision0710.d07.generic_candidate_run.v1"
GENERIC_CHECKPOINT_SCHEMA = "revision0710.d07.trainable_checkpoint.v1"
GENERIC_CHECKPOINT_SET_SCHEMA = "revision0710.d07.checkpoint_set.v1"
GENERIC_EPOCH_HISTORY_SCHEMA = "revision0710.d07.epoch_history.v1"
GENERIC_METRICS_SCHEMA = "revision0710.d07.open_fold0_metrics.v2"
GENERIC_FAILURE_SCHEMA = "revision0710.d07.candidate_failure.v1"
EVALUATION_MANIFEST_SCHEMA = "revision0710.d07.fold0_evaluation_manifest.v1"

_MODEL_EFFECTIVE_KEYS = {
    "hidden_dim",
    "num_heads",
    "num_transformer_layers",
    "num_gru_layers",
    "dropout",
    "freeze_event_embeddings",
    "risk_offset_cap_years",
    "risk_temperature",
    "baseline_log_rate_init",
    "base_age_mode",
    "base_age_head_mode",
    "base_age_pooling_mode",
    "organ_axis_order",
    "survival_interval_edges_months",
    "architecture_change_from_d04",
}
_PRIVATE_MODEL_METADATA = {
    "private_age_module_class": "PrivateRETAINAgeTower",
    "private_age_hidden_dim": PRIVATE_RETAIN_HIDDEN_DIM,
    "private_age_event_encoder_class": "PrivateRETAINEventTokenEncoder",
    "private_age_backbone_class": "PrivateRETAINBackbone",
    "private_age_static_encoder_class": "PrivateRETAINStaticEncoder",
    "private_age_type_vocab_size": 3,
    "private_age_event_time_features": [
        "log1p_days_before_index_div_log_36526",
        "years_before_index_div_100",
        "exp_negative_years_before_index_div_10",
    ],
    "private_age_pooling": "reverse_time_retain_alpha_beta_attention",
    "private_age_alpha_gru": {"layers": 1, "bidirectional": False},
    "private_age_beta_gru": {"layers": 1, "bidirectional": False},
    "private_age_static_input_dim": 157,
    "private_age_fusion": "linear_256_128_layernorm_gelu_dropout",
    "private_age_head": "linear_128_128_gelu_dropout_linear_128_1_shared_scalar",
    "private_age_terminal_zero_init": True,
    "private_age_organ_conditioned": False,
    "private_age_embedding_storage": "independent_clones_equal_values",
    "private_age_embeddings_trainable": False,
    "private_age_rng_initialization": (
        "namespaced_sha256_entry_cpu_state_seeded_cpu_default_generator_substream"
    ),
    "private_age_rng_policy": {
        "version": "revision0710.d07.private_retain.initialization.v1",
        "global_training_seed": 7210,
        "entry_state": "cpu_rng_after_common_event_temporal_static_modules",
        "d07_002_shadow": (
            "shared_fusion_h192_plus_mlp_base_age_head_in_cpu_fork_keep_post_init_state_only"
        ),
        "private_seed_derivation": {
            "algorithm": "sha256",
            "input": "namespace_bytes_concat_entry_cpu_rng_state_bytes",
            "digest_slice": "first_8_bytes",
            "integer_encoding": "unsigned_little_endian",
        },
        "private_construction": {
            "fork_rng_devices": [],
            "seed_api": "torch.random.default_generator.manual_seed",
            "device": "cpu",
            "cuda_rng_mutated": False,
        },
        "post_private_global_cpu_state": "d07_002_shared_fusion_mlp_post_init_state",
        "common_risk_initialization_matches_d07_002": True,
        "shadow_modules_registered": False,
        "state_dict_copy": False,
        "learned_state_loaded": False,
    },
    "private_age_expected_parameter_counts": {
        "full_model": {"total": 59091046, "trainable": 3312742},
        "private_age_tower": {"total": 28356226, "trainable": 467074},
        "private_age_backbone": {"total": 28269185, "trainable": 380033},
        "private_age_static_encoder": {"total": 37248, "trainable": 37248},
        "private_age_fusion": {"total": 33152, "trainable": 33152},
        "private_age_head": {"total": 16641, "trainable": 16641},
    },
}
_PRIVATE_MODEL_METADATA_KEYS = frozenset(_PRIVATE_MODEL_METADATA)
_MODEL_EFFECTIVE_KEYS.update(_PRIVATE_MODEL_METADATA_KEYS)
_PRIVATE_COMMON_MODEL_CONTRACT = {
    "hidden_dim": 192,
    "num_heads": 6,
    "num_transformer_layers": 3,
    "num_gru_layers": 2,
    "dropout": 0.15,
    "freeze_event_embeddings": True,
    "risk_offset_cap_years": 8.0,
    "risk_temperature": 2.0,
    "baseline_log_rate_init": -7.5,
    "base_age_head_mode": "mlp",
    "base_age_pooling_mode": "global_fused",
    "survival_interval_edges_months": [0.0, 12.0, 24.0, 36.0, 48.0, 60.0],
}
_LOSS_EFFECTIVE_KEYS = {
    "objective_name",
    "age_estimator",
    "huber_beta_years",
    "age_scale_years",
    "base_huber_weight",
    "base_mse_weight",
    "base_correlation_weight",
    "final_all_huber_weight",
    "strict_final_huber_weight",
    "strict_offset_weight",
    "strict_center_weight",
    "strict_slope_weight",
    "survival_weight",
    "evaluator_gap_cox_weight",
    "evaluator_gap_cox_scale_years",
    "consistency_weight",
    "survival_ramp_start_fraction",
    "survival_ramp_epochs",
    "strict_offset_beta_years",
    "risk_terms_changed_from_d04",
    "augmentation_terms_changed_from_d04",
}
_FORBIDDEN_LEARNED_INPUT_KEYS = {
    "resume_from",
    "resume_checkpoint",
    "initial_checkpoint",
    "pretrained_checkpoint",
    "teacher_checkpoint",
    "baseline_checkpoint_path",
    "baseline_prediction_path",
    "baseline_predictions_path",
    "d07_checkpoint_path",
    "d07_002_checkpoint_path",
}
_D07_002_EVIDENCE_ARTIFACT_KEYS = frozenset(
    {
        "contract",
        "run_config",
        "training_history",
        "checkpoint_set",
        "fold0_metrics",
        "candidate_summary",
        "artifact_hash_registry",
        "fold0_read_ledger",
        "completion_receipt",
    }
)
_D07_002_RESULT = {
    "status": "NON_PROMOTABLE",
    "diagnostic_epoch": 35,
    "strict_mae": 3.7142194679437215,
    "strict_pearson": 0.7968304178625415,
    "strict_r2": 0.6307445954098906,
    "age_sex_harrell_macro_delta_c": 0.047816447106148185,
    "positive_organs": 7,
    "organs_beating_baseline": 7,
    "age_gates_passed": False,
    "risk_gates_passed": True,
    "promotable": False,
    "strict_age_evaluator_calls": 60,
    "harrell_evaluator_calls": 5,
}


@dataclass(frozen=True)
class EvaluationSchedule:
    strict_age_epochs: tuple[int, ...]
    fixed_risk_epochs: tuple[int, ...]
    add_age_gate_eligible_epochs: bool
    add_new_age_pareto_epochs: bool
    include_final_epoch: bool


@dataclass(frozen=True)
class CandidateContract:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    candidate_id: str
    candidate_number: int
    max_epochs: int
    schedule: EvaluationSchedule


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()


def _resolve_contract_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0].lower() == REVISION_ROOT.name.lower():
        return (WORKSPACE_ROOT / path).resolve()
    return (REVISION_ROOT / path).resolve()


def _require_hash(path: Path, expected: object, *, field: str) -> str:
    if not path.is_file():
        raise D07ContractError(f"D07 candidate {field} is missing: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise D07ContractError(
            f"D07 candidate {field} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise D07ContractError(f"D07 candidate contract lacks mapping {key}")
    return value


def _require_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise D07ContractError(f"D07 candidate protocol requires {key}=false")


def _walk_forbidden_learned_inputs(value: Any, *, path: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in _FORBIDDEN_LEARNED_INPUT_KEYS and child not in (None, False, ""):
                raise D07ContractError(
                    f"D07 candidate declares forbidden learned input at {child_path}"
                )
            _walk_forbidden_learned_inputs(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_learned_inputs(child, path=f"{path}[{index}]")


def validate_private_model_metadata(model: Mapping[str, Any]) -> None:
    private_keys = {key for key in model if key.startswith("private_age_")}
    mode = str(model.get("base_age_mode", BASE_AGE_MODE_V1))
    if mode != BASE_AGE_MODE_PRIVATE_RETAIN:
        if private_keys:
            raise D07ContractError(
                f"non-private model declares private metadata: {sorted(private_keys)}"
            )
        return
    common_mismatches = {
        key: {"expected": expected, "observed": model.get(key)}
        for key, expected in _PRIVATE_COMMON_MODEL_CONTRACT.items()
        if model.get(key) != expected
    }
    if common_mismatches:
        raise D07ContractError(
            f"formal private_retain common D07-002 model contract changed: {common_mismatches}"
        )
    missing = _PRIVATE_MODEL_METADATA_KEYS.difference(private_keys)
    if missing:
        raise D07ContractError(
            f"private_retain model lacks implemented metadata: {sorted(missing)}"
        )
    mismatches = {
        key: {"expected": expected, "observed": model.get(key)}
        for key, expected in _PRIVATE_MODEL_METADATA.items()
        if model.get(key) != expected
    }
    if mismatches:
        raise D07ContractError(f"private_retain metadata mismatch: {mismatches}")


def private_parameter_counts(model: EHRAGEModel) -> dict[str, dict[str, int]]:
    if not hasattr(model, "private_retain_age_tower"):
        raise D07ContractError("private parameter counts require a private_retain model")
    tower = model.private_retain_age_tower

    def count(module: torch.nn.Module) -> dict[str, int]:
        return {
            "total": int(sum(parameter.numel() for parameter in module.parameters())),
            "trainable": int(
                sum(
                    parameter.numel()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
            ),
        }

    return {
        "full_model": count(model),
        "private_age_tower": count(tower),
        "private_age_backbone": count(tower.backbone),
        "private_age_static_encoder": count(tower.static_encoder),
        "private_age_fusion": count(tower.fusion),
        "private_age_head": count(tower.age_head),
    }


def validate_private_model_instance(
    model: EHRAGEModel,
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    validate_private_model_metadata(declaration)
    if declaration.get("base_age_mode") != BASE_AGE_MODE_PRIVATE_RETAIN:
        return {"status": "NOT_APPLICABLE_SHARED_MODEL"}
    tower = getattr(model, "private_retain_age_tower", None)
    if not isinstance(tower, PrivateRETAINAgeTower):
        raise D07ContractError("private model did not instantiate PrivateRETAINAgeTower")
    if (
        not isinstance(tower.backbone, PrivateRETAINBackbone)
        or not isinstance(tower.backbone.token_encoder, PrivateRETAINEventTokenEncoder)
        or not isinstance(tower.static_encoder, PrivateRETAINStaticEncoder)
    ):
        raise D07ContractError("private model instantiated the wrong RETAIN module classes")
    token = tower.backbone.token_encoder
    if (
        tower.backbone.output_dim != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.backbone.alpha_gru.input_size != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.backbone.alpha_gru.hidden_size != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.backbone.alpha_gru.num_layers != 1
        or tower.backbone.alpha_gru.bidirectional
        or tower.backbone.beta_gru.input_size != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.backbone.beta_gru.hidden_size != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.backbone.beta_gru.num_layers != 1
        or tower.backbone.beta_gru.bidirectional
        or token.type_embedding.num_embeddings != 3
        or token.time_projection.in_features != 3
        or tower.static_encoder.network[0].in_features != 157
        or tower.static_encoder.network[0].out_features != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.fusion[0].in_features != 2 * PRIVATE_RETAIN_HIDDEN_DIM
        or tower.fusion[0].out_features != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.age_head[0].in_features != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.age_head[0].out_features != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.age_head[-1].in_features != PRIVATE_RETAIN_HIDDEN_DIM
        or tower.age_head[-1].out_features != 1
    ):
        raise D07ContractError("private RETAIN instance dimensions disagree with metadata")
    if torch.count_nonzero(tower.age_head[-1].weight).item() != 0 or torch.count_nonzero(
        tower.age_head[-1].bias
    ).item() != 0:
        raise D07ContractError("private RETAIN terminal age head is not zero initialized")
    private_code = token.code_embedding.weight
    private_text = token.text_embedding.weight
    common_code = model.event_encoder.code_embedding.weight
    common_text = model.event_encoder.text_embedding.weight
    if (
        private_code.requires_grad
        or private_text.requires_grad
        or not torch.equal(private_code, common_code)
        or not torch.equal(private_text, common_text)
        or private_code.untyped_storage().data_ptr()
        == common_code.untyped_storage().data_ptr()
        or private_text.untyped_storage().data_ptr()
        == common_text.untyped_storage().data_ptr()
        or hasattr(tower, "organ_queries")
        or hasattr(tower, "organ_embeddings")
    ):
        raise D07ContractError("private embedding storage or organ-independence contract failed")
    counts = private_parameter_counts(model)
    declared_counts = declaration.get("private_age_expected_parameter_counts")
    if declared_counts != counts:
        raise D07ContractError(
            f"private parameter counts changed: expected {declared_counts}, observed {counts}"
        )
    return {
        "status": "PASS",
        "module_classes": {
            "tower": type(tower).__name__,
            "event_encoder": type(token).__name__,
            "backbone": type(tower.backbone).__name__,
            "static_encoder": type(tower.static_encoder).__name__,
        },
        "parameter_counts": counts,
        "terminal_zero_init": True,
        "organ_conditioned": False,
        "embedding_storage": "independent_clones_equal_values",
        "embeddings_trainable": False,
    }


def audit_private_common_initialization(
    *,
    model: EHRAGEModel,
    code_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    age_anchor: float,
    seed: int,
) -> dict[str, Any]:
    if model.config.base_age_mode != BASE_AGE_MODE_PRIVATE_RETAIN:
        return {"status": "NOT_APPLICABLE_SHARED_MODEL"}
    rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    shared_config = replace(
        model.config,
        base_age_mode=BASE_AGE_MODE_V1,
        base_age_head_mode="mlp",
        base_age_pooling_mode=BASE_AGE_POOLING_MODE_GLOBAL_FUSED,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        reference = EHRAGEModel(
            code_embedding,
            text_embedding,
            age_anchor=age_anchor,
            config=shared_config,
        )
    reference_trainable_hash = trainable_parameter_hash(reference)
    if reference_trainable_hash != D04_INITIAL_TRAINABLE_SHA256:
        raise D07ContractError(
            "ephemeral same-seed D07-002 reference initial trainable hash changed"
        )
    rng_after = torch.random.get_rng_state().clone()
    cuda_rng_after = [state.clone() for state in torch.cuda.get_rng_state_all()]
    if not torch.equal(rng_before, rng_after):
        raise D07ContractError("private common-init audit mutated the training CPU RNG state")
    if len(cuda_rng_before) != len(cuda_rng_after) or any(
        not torch.equal(before, after)
        for before, after in zip(cuda_rng_before, cuda_rng_after)
    ):
        raise D07ContractError("private common-init audit mutated a CUDA RNG state")
    shared_common = {
        name: tensor.detach().cpu()
        for name, tensor in reference.state_dict().items()
        if not name.startswith(("base_fusion.", "base_age_head."))
    }
    private_common = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("private_retain_age_tower.")
    }
    if set(shared_common) != set(private_common):
        raise D07ContractError("shared/private common initialization tensor names differ")
    mismatches = [
        name for name in shared_common if not torch.equal(shared_common[name], private_common[name])
    ]
    if mismatches:
        raise D07ContractError(
            f"private common risk initialization differs from D07-002: {mismatches[:8]}"
        )
    module_selectors: dict[str, Callable[[str], bool]] = {
        "event_encoder": lambda name: name.startswith("event_encoder."),
        "temporal_encoder": lambda name: name.startswith("temporal_encoder."),
        "static_encoder": lambda name: name.startswith("static_encoder."),
        "organ_queries": lambda name: name == "organ_queries",
        "organ_cross_attention": lambda name: name.startswith("organ_cross_attention."),
        "organ_fusion": lambda name: name.startswith("organ_fusion."),
        "organ_relation": lambda name: name.startswith("organ_relation."),
        "risk_head": lambda name: name.startswith("risk_head."),
        "baseline_log_rate": lambda name: name == "baseline_log_rate",
        "age_anchor": lambda name: name == "age_anchor",
    }
    module_hashes: dict[str, dict[str, Any]] = {}
    for module_name, selector in module_selectors.items():
        shared_values = {
            name: value for name, value in shared_common.items() if selector(name)
        }
        private_values = {
            name: value for name, value in private_common.items() if selector(name)
        }
        if not shared_values or set(shared_values) != set(private_values):
            raise D07ContractError(
                f"private common-init audit lacks module tensors for {module_name}"
            )
        shared_hash = _tensor_mapping_hash(shared_values)
        private_hash = _tensor_mapping_hash(private_values)
        if shared_hash != private_hash:
            raise D07ContractError(
                f"private common-init module hash differs for {module_name}"
            )
        module_hashes[module_name] = {
            "tensor_count": len(shared_values),
            "d07_002_reference_sha256": shared_hash,
            "private_candidate_sha256": private_hash,
            "exact_match": True,
        }
    audit = {
        "status": "PASS",
        "reference": "same-seed random D07-002 shared_fusion+MLP construction",
        "learned_state_loaded": False,
        "d07_002_reference_trainable_parameter_sha256": reference_trainable_hash,
        "d07_002_expected_trainable_parameter_sha256": D04_INITIAL_TRAINABLE_SHA256,
        "d07_002_reference_trainable_hash_exact_match": True,
        "common_tensor_count": len(shared_common),
        "common_state_sha256": _tensor_mapping_hash(shared_common),
        "per_module_hashes": module_hashes,
        "reference_audit_cpu_rng_state_preserved": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_rng_state_count": len(cuda_rng_before),
        "reference_audit_cuda_rng_states_preserved": True,
    }
    del reference
    return audit


def audit_optimizer_parameter_binding(
    model: EHRAGEModel, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable_ids = [id(parameter) for parameter in trainable]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if (
        len(optimizer_ids) != len(set(optimizer_ids))
        or set(optimizer_ids) != set(trainable_ids)
    ):
        raise D07ContractError(
            "optimizer parameters do not equal the model's trainable parameter set"
        )
    shadow_names = [
        name
        for name, _ in list(model.named_modules()) + list(model.named_parameters())
        if "shadow" in name.lower()
    ]
    if shadow_names:
        raise D07ContractError(f"discarded RNG shadow modules were registered: {shadow_names}")
    trainable_numel = int(sum(parameter.numel() for parameter in trainable))
    optimizer_numel = int(sum(parameter.numel() for parameter in optimizer_parameters))
    if trainable_numel != optimizer_numel:
        raise D07ContractError("optimizer trainable parameter count changed")
    return {
        "status": "PASS",
        "model_trainable_tensor_count": len(trainable),
        "optimizer_parameter_tensor_count": len(optimizer_parameters),
        "model_trainable_parameter_count": trainable_numel,
        "optimizer_parameter_count": optimizer_numel,
        "exact_parameter_identity_set_match": True,
        "duplicate_optimizer_parameters": 0,
        "registered_shadow_modules_or_parameters": [],
    }


def normalize_evaluation_schedule(
    payload: Mapping[str, Any], *, max_epochs: int
) -> EvaluationSchedule:
    evaluation = _require_mapping(payload, "evaluation")
    strict = _require_mapping(evaluation, "strict_age")
    epoch_schedule = strict.get("epoch_schedule")
    if epoch_schedule == "every_epoch":
        strict_epochs = tuple(range(1, max_epochs + 1))
        if int(strict.get("start_epoch", -1)) != 1 or int(
            strict.get("end_epoch", -1)
        ) != max_epochs:
            raise D07ContractError("strict-age every-epoch bounds disagree with training budget")
    elif isinstance(epoch_schedule, list):
        strict_epochs = tuple(int(epoch) for epoch in epoch_schedule)
    else:
        raise D07ContractError("strict_age.epoch_schedule must be every_epoch or an epoch list")
    if (
        not strict_epochs
        or len(strict_epochs) != len(set(strict_epochs))
        or tuple(sorted(strict_epochs)) != strict_epochs
        or strict_epochs[0] < 1
        or strict_epochs[-1] > max_epochs
    ):
        raise D07ContractError("strict-age epoch schedule is invalid")
    if strict.get("uses_unrounded_values") is not True:
        raise D07ContractError("strict-age gate decisions must use unrounded values")

    risk = _require_mapping(evaluation, "harrell_risk")
    fixed = tuple(int(epoch) for epoch in risk.get("fixed_epochs", []))
    if (
        len(fixed) != len(set(fixed))
        or tuple(sorted(fixed)) != fixed
        or any(epoch not in strict_epochs for epoch in fixed)
    ):
        raise D07ContractError("Harrell fixed epochs must be unique strict-age epochs")
    schedule = EvaluationSchedule(
        strict_age_epochs=strict_epochs,
        fixed_risk_epochs=fixed,
        add_age_gate_eligible_epochs=risk.get("add_age_gate_eligible_epochs") is True,
        add_new_age_pareto_epochs=risk.get("add_new_age_pareto_epochs") is True,
        include_final_epoch=risk.get("include_final_epoch") is True,
    )
    if not (
        schedule.fixed_risk_epochs
        or schedule.add_age_gate_eligible_epochs
        or schedule.add_new_age_pareto_epochs
        or schedule.include_final_epoch
    ):
        raise D07ContractError("Harrell schedule would evaluate no epoch")
    if schedule.include_final_epoch and max_epochs not in strict_epochs:
        raise D07ContractError("final Harrell epoch lacks a strict-age evaluation")
    if risk.get("deduplicate_epoch_union") is not True:
        raise D07ContractError("Harrell schedule must deduplicate the epoch union")
    if risk.get("evaluation_order") != "ascending_epoch":
        raise D07ContractError("Harrell epochs must be evaluated in ascending order")
    if risk.get("retain_every_evaluator_call") is not True:
        raise D07ContractError("every Harrell evaluator call must be retained")
    if schedule.add_new_age_pareto_epochs:
        definition = _require_mapping(risk, "new_age_pareto_definition")
        expected_definition = {
            "scope": (
                "global_nondominated_frontier_after_all_"
                f"{max_epochs}_strict_age_rows_are_available"
            ),
            "objectives": {
                "strict_mae": "minimize",
                "strict_pearson": "maximize",
                "strict_r2": "maximize",
            },
            "values": "unrounded",
            "dominance": (
                "weakly_no_worse_in_all_three_and_strictly_better_in_at_least_one"
            ),
        }
        if dict(definition) != expected_definition:
            raise D07ContractError("global age-Pareto definition changed")
    return schedule


def _validate_declared_baseline(
    raw: Mapping[str, Any], gates: SelectionGates, baseline_payload: Mapping[str, Any]
) -> None:
    declared = _require_mapping(raw, "baseline_envelope")
    declared_path = _resolve_contract_path(declared.get("path"))
    declared_manifest = _resolve_contract_path(declared.get("artifact_manifest_path"))
    if declared_path != DEFAULT_BASELINE_ENVELOPE.resolve():
        raise D07ContractError("candidate does not declare the canonical true-Fold-0 envelope")
    if declared_manifest != DEFAULT_BASELINE_ARTIFACT_MANIFEST.resolve():
        raise D07ContractError("candidate does not declare the canonical baseline manifest")
    _require_hash(declared_path, declared.get("sha256"), field="baseline_envelope")
    _require_hash(
        declared_manifest,
        declared.get("artifact_manifest_sha256"),
        field="baseline_artifact_manifest",
    )
    if declared.get("schema_version") != baseline_payload.get("schema_version"):
        raise D07ContractError("candidate baseline-envelope schema changed")

    declared_gates = _require_mapping(declared, "gates")
    expected = {
        "strict_mae": ("<", Decimal(str(gates.strict_mae_max))),
        "strict_pearson": (">", Decimal(str(gates.strict_pearson_min))),
        "strict_r2": (">", Decimal(str(gates.strict_r2_min))),
        "age_sex_harrell_macro_delta_c": (">", Decimal(str(gates.delta_c_min))),
    }
    for name, (operator, threshold) in expected.items():
        row = declared_gates.get(name)
        if (
            not isinstance(row, Mapping)
            or row.get("operator") != operator
            or Decimal(str(row.get("threshold"))) != threshold
        ):
            raise D07ContractError(f"candidate baseline gate changed: {name}")
    if int(declared_gates.get("positive_organs_required", -1)) != gates.positive_organs_min:
        raise D07ContractError("candidate positive-organ gate changed")
    if (
        int(declared_gates.get("organs_beating_baseline_required", -1))
        != gates.organs_beating_baseline_min
    ):
        raise D07ContractError("candidate organ-superiority gate changed")
    organ_values = declared_gates.get("organ_delta_c_envelope")
    if not isinstance(organ_values, Mapping) or gates.baseline_organ_delta_c is None:
        raise D07ContractError("candidate lacks the seven-organ baseline envelope")
    for organ in ORGAN_SYSTEMS:
        if Decimal(str(organ_values.get(organ))) != Decimal(
            str(gates.baseline_organ_delta_c[organ])
        ):
            raise D07ContractError(f"candidate organ baseline changed: {organ}")


def _validate_contract_lineage(raw: Mapping[str, Any]) -> None:
    for field in ("plan", "d07_protocol"):
        declaration = _require_mapping(raw, field)
        _require_hash(
            _resolve_contract_path(declaration.get("path")),
            declaration.get("sha256"),
            field=field,
        )
    d07_protocol = _require_mapping(raw, "d07_protocol")
    if (
        _resolve_contract_path(d07_protocol.get("path")) != D07_PROTOCOL_PATH.resolve()
        or str(d07_protocol.get("sha256")) != D07_PROTOCOL_SHA256
    ):
        raise D07ContractError("candidate declares a noncanonical D07 protocol")

    parent = _require_mapping(raw, "parent_d04")
    _require_hash(
        _resolve_contract_path(parent.get("config_path")),
        parent.get("config_sha256"),
        field="parent_d04.config",
    )
    if parent.get("architecture_loaded_from_checkpoint") is not False:
        raise D07ContractError("D04 may provide architecture evidence, not a checkpoint")

    preimplementation = _require_mapping(raw, "preimplementation_code_lineage")
    current = {
        "ehr_age_model": MODEL_DIR / "ehr_age_model.py",
        "ehr_age_losses": MODEL_DIR / "ehr_age_losses.py",
        "ehr_age_data": RUNNER_DIR / "ehr_age_data.py",
        "ehr_age_training": RUNNER_DIR / "ehr_age_training.py",
    }
    if str(raw.get("model", {}).get("base_age_mode")) == BASE_AGE_MODE_PRIVATE_RETAIN:
        current.update(
            {
                "private_retain_age": MODEL_DIR / "private_retain_age.py",
                "d07_candidate_runner": Path(__file__).resolve(),
                "compact_cache_dataset": (
                    REVISION_ROOT
                    / "02_baselines_5fold"
                    / "common"
                    / "compact_cache_dataset.py"
                ),
                "crossfit_metrics": (
                    REVISION_ROOT / "05_downstream" / "common" / "crossfit_metrics.py"
                ),
            }
        )
    for name, path in current.items():
        _require_hash(path, preimplementation.get(name), field=f"code.{name}")
    baseline_reference = _require_mapping(
        preimplementation, "baseline_training_loss_reference"
    )
    _require_hash(
        _resolve_contract_path(baseline_reference.get("path")),
        baseline_reference.get("sha256"),
        field="baseline_training_loss_reference",
    )
    evidence = _require_mapping(_require_mapping(raw, "d07_001_evidence"), "artifacts")
    for name, declaration in evidence.items():
        if not isinstance(declaration, Mapping):
            raise D07ContractError(f"D07-001 evidence declaration is malformed: {name}")
        _require_hash(
            _resolve_contract_path(declaration.get("path")),
            declaration.get("sha256"),
            field=f"d07_001_evidence.{name}",
        )
    supporting = _require_mapping(raw, "supporting_estimator_evidence")
    for name in ("d06_0_outcome_plan", "d06_1_stop_decision"):
        declaration = _require_mapping(supporting, name)
        _require_hash(
            _resolve_contract_path(declaration.get("path")),
            declaration.get("sha256"),
            field=f"supporting_estimator_evidence.{name}",
        )


def _validate_d07_002_parent_evidence(raw: Mapping[str, Any]) -> None:
    number = int(raw.get("candidate_number", -1))
    if number < 3:
        return
    evidence = _require_mapping(raw, "d07_002_evidence")
    result = _require_mapping(evidence, "result")
    if dict(result) != _D07_002_RESULT:
        raise D07ContractError("D07-002 result evidence changed")
    declarations = _require_mapping(evidence, "artifacts")
    if set(declarations) != set(_D07_002_EVIDENCE_ARTIFACT_KEYS):
        raise D07ContractError(
            "D07-002 evidence must contain exactly the nine frozen artifacts"
        )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in sorted(_D07_002_EVIDENCE_ARTIFACT_KEYS):
        declaration = _require_mapping(declarations, name)
        path = _resolve_contract_path(declaration.get("path"))
        observed = _require_hash(
            path, declaration.get("sha256"), field=f"d07_002_evidence.{name}"
        )
        paths[name] = path
        hashes[name] = observed

    payloads = {
        name: json.loads(paths[name].read_text(encoding="utf-8"))
        for name in _D07_002_EVIDENCE_ARTIFACT_KEYS
        if name != "artifact_hash_registry"
    }
    contract = payloads["contract"]
    run_config = payloads["run_config"]
    training = payloads["training_history"]
    checkpoint_set = payloads["checkpoint_set"]
    metrics = payloads["fold0_metrics"]
    summary = payloads["candidate_summary"]
    ledger = payloads["fold0_read_ledger"]
    receipt = payloads["completion_receipt"]
    candidate_id = "002_baseline_native_objective"
    if (
        contract.get("candidate_id") != candidate_id
        or sha256_file(paths["contract"]) != DEFAULT_D07_002_CONFIG_SHA256
        or run_config.get("candidate_id") != candidate_id
        or run_config.get("candidate_contract_sha256") != DEFAULT_D07_002_CONFIG_SHA256
        or run_config.get("learned_checkpoint_inputs") != []
        or run_config.get("baseline_prediction_inputs") != []
        or training.get("candidate_id") != candidate_id
        or training.get("full_budget_completed") is not True
        or training.get("fold0_read_count_at_training_completion") != 0
        or len(training.get("epochs", [])) != 60
        or checkpoint_set.get("candidate_id") != candidate_id
        or checkpoint_set.get("status") != "SEALED_BEFORE_FOLD0_READ"
        or len(checkpoint_set.get("checkpoints", [])) != 60
    ):
        raise D07ContractError("D07-002 contract, run, training, or checkpoint evidence is invalid")

    expected_row = {
        key: _D07_002_RESULT[key]
        for key in (
            "strict_mae",
            "strict_pearson",
            "strict_r2",
            "age_sex_harrell_macro_delta_c",
            "positive_organs",
            "organs_beating_baseline",
        )
    }
    expected_row = {"epoch": 35, **expected_row}
    for name, payload in (("metrics", metrics), ("summary", summary)):
        if (
            payload.get("candidate_id") != candidate_id
            or payload.get("status") != "NON_PROMOTABLE"
            or payload.get("selected_epoch") is not None
            or payload.get("diagnostic_epoch") != 35
            or payload.get("primary_same_epoch_row") != expected_row
            or payload.get("d07_fold0_gate_eligible") is not False
        ):
            raise D07ContractError(f"D07-002 {name} semantic evidence changed")
    if (
        summary.get("strict_age_evaluator_calls") != 60
        or summary.get("harrell_evaluator_calls") != 5
        or summary.get("baseline_predictions_or_checkpoints_loaded") is not False
        or summary.get("from_random_initialization") is not True
    ):
        raise D07ContractError("D07-002 summary lineage or evaluator counts changed")

    if (
        ledger.get("schema_version") != "revision0710.d07.fold0_read_ledger_entry.v1"
        or ledger.get("status") != "READ_COMMITTED_BEFORE_FOLD0_ACCESS"
        or ledger.get("sequence") != 2
        or ledger.get("candidate_id") != candidate_id
        or receipt.get("schema_version") != "revision0710.d07.fold0_read_receipt.v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("sequence") != 2
        or receipt.get("candidate_id") != candidate_id
        or receipt.get("stage") != "COMPLETE"
    ):
        raise D07ContractError("D07-002 ledger or completion receipt identity changed")
    precommit = receipt.get("precommit_read_ledger", {})
    if (
        precommit.get("path") != str(paths["fold0_read_ledger"].resolve())
        or precommit.get("sha256") != hashes["fold0_read_ledger"]
        or precommit.get("checkpoint_sha256") != ledger.get("checkpoint_sha256")
        or precommit.get("evaluation_manifest_sha256")
        != ledger.get("evaluation_manifest_sha256")
    ):
        raise D07ContractError("D07-002 ledger and completion receipt disagree")

    registry = json.loads(paths["artifact_hash_registry"].read_text(encoding="utf-8"))
    registry_names = {
        "contract": "frozen_candidate_contract",
        "run_config": "run_config",
        "training_history": "training_history",
        "checkpoint_set": "checkpoint_set_before_fold0",
        "fold0_metrics": "fold0_open_development_metrics",
        "candidate_summary": "candidate_summary",
        "fold0_read_ledger": "fold0_read_ledger_entry",
    }
    for evidence_name, registry_name in registry_names.items():
        binding = registry.get(registry_name)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str(paths[evidence_name].resolve())
            or binding.get("sha256") != hashes[evidence_name]
        ):
            raise D07ContractError(
                f"D07-002 artifact registry disagrees for {evidence_name}"
            )
    receipt_bindings = receipt.get("artifact_bindings", {})
    if (
        receipt_bindings.get("artifact_hashes", {}).get("sha256")
        != hashes["artifact_hash_registry"]
        or receipt_bindings.get("candidate_summary", {}).get("sha256")
        != hashes["candidate_summary"]
        or receipt_bindings.get("fold0_open_development_metrics", {}).get("sha256")
        != hashes["fold0_metrics"]
    ):
        raise D07ContractError("D07-002 completion receipt artifact bindings changed")


def load_frozen_candidate_contract(
    path: Path = DEFAULT_D07_CANDIDATE_CONFIG,
    *,
    expected_sha256: str | None = None,
    gates: SelectionGates | None = None,
    baseline_payload: Mapping[str, Any] | None = None,
) -> CandidateContract:
    path = path.resolve()
    observed_hash = sha256_file(path)
    if expected_sha256 is not None and observed_hash != expected_sha256:
        raise D07ContractError(
            f"candidate contract hash mismatch: expected {expected_sha256}, observed {observed_hash}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not str(raw.get("schema_version", "")).startswith(
        "revision0710_ehr_age_d07_"
    ):
        raise D07ContractError("unsupported D07 candidate schema")
    if raw.get("status") != "FROZEN_BEFORE_EXECUTION" or raw.get("executed") is not False:
        raise D07ContractError("D07 candidate contract is not frozen and unexecuted")
    candidate_id = str(raw.get("candidate_id", ""))
    if re.fullmatch(r"\d{3}_[a-z0-9][a-z0-9_]*", candidate_id) is None:
        raise D07ContractError("D07 candidate_id is invalid")
    number = int(raw.get("candidate_number", -1))
    if number < 2 or candidate_id[:3] != f"{number:03d}":
        raise D07ContractError("D07 candidate number and ID disagree")
    if raw.get("candidate_role") != "OPEN_FOLD0_DEVELOPMENT":
        raise D07ContractError("generic D07 candidates must use open Fold0 development")

    model = _require_mapping(raw, "model")
    unknown_model = set(model).difference(_MODEL_EFFECTIVE_KEYS)
    if unknown_model:
        raise D07ContractError(
            f"candidate model fields are not implemented by this runner: {sorted(unknown_model)}"
        )
    validate_private_model_metadata(model)
    if tuple(model.get("organ_axis_order", ())) != tuple(ORGAN_SYSTEMS):
        raise D07ContractError("candidate changed the organ axis")
    _model_config(model).validate()

    loss = _require_mapping(raw, "loss")
    unknown_loss = set(loss).difference(_LOSS_EFFECTIVE_KEYS)
    if unknown_loss:
        raise D07ContractError(
            f"candidate loss fields are not implemented by this runner: {sorted(unknown_loss)}"
        )
    _loss_config(loss).validate()
    if float(loss.get("strict_offset_beta_years", math.nan)) != 1.0:
        raise D07ContractError("candidate strict-offset beta is not implemented")

    training = _require_mapping(raw, "training")
    max_epochs = int(training.get("max_epochs", -1))
    if (
        int(training.get("outer_fold", -1)) != 0
        or list(training.get("train_folds", [])) != [2, 3, 4]
        or training.get("fold1_role")
        != "CALIBRATION_NUISANCE_ONLY_NO_CANDIDATE_METRICS"
        or training.get("fold0_role") != "OPEN_DEVELOPMENT"
        or max_epochs < 1
        or int(training.get("scheduler_horizon_epochs", -1)) != max_epochs
        or training.get("full_budget_required") is not True
        or training.get("performance_early_stopping_enabled") is not False
        or training.get("second_seed_allowed") is not False
        or training.get("deterministic") is not True
        or training.get("optimizer") != "AdamW"
        or training.get("scheduler") != "cosine"
    ):
        raise D07ContractError("candidate training/fold/full-budget contract is invalid")
    if (
        model.get("base_age_mode") == BASE_AGE_MODE_PRIVATE_RETAIN
        and int(training.get("seed", -1))
        != int(model["private_age_rng_policy"]["global_training_seed"])
    ):
        raise D07ContractError("private RNG policy and training seed disagree")

    selection = _require_mapping(raw, "selection")
    if list(selection.get("allowed_checkpoint_epochs", [])) != list(
        range(1, max_epochs + 1)
    ):
        raise D07ContractError("allowed checkpoint epochs must cover the full budget")
    if (
        selection.get("selection_pool") != "risk_evaluated_epochs_only"
        or selection.get("all_same_epoch_gates_required") is not True
        or selection.get("comparisons_use_unrounded_values") is not True
        or int(selection.get("candidate_count_to_freeze", -1)) != 1
        or selection.get("fallback_checkpoint_allowed") is not False
        or selection.get("checkpoint_averaging_allowed") is not False
    ):
        raise D07ContractError("candidate checkpoint-selection contract is invalid")
    expected_ranking = [
        "age_sex_harrell_macro_delta_c_descending",
        "strict_mae_ascending",
        "strict_pearson_descending",
        "strict_r2_descending",
        "epoch_ascending",
    ]
    if list(selection.get("ranking_if_multiple_promotable", [])) != expected_ranking:
        raise D07ContractError("candidate checkpoint ranking changed")

    evaluation = _require_mapping(raw, "evaluation")
    nuisance = _require_mapping(evaluation, "nuisance_fit")
    if (
        int(nuisance.get("fold", -1)) != 1
        or nuisance.get("inner_role") != "calibration"
        or nuisance.get("fold0_fitted_parameters") is not False
        or evaluation.get("fold1_candidate_metrics_computed") is not False
        or evaluation.get("fold2_to_4_heldout_ehr_metrics_computed") is not False
    ):
        raise D07ContractError("candidate nuisance or hidden-fold metric contract changed")

    protocol = _require_mapping(raw, "protocol")
    if protocol.get("initialization") != "random_except_frozen_cache_embedding_tables":
        raise D07ContractError("candidate initialization is not from random trainable weights")
    for key in (
        "warm_start",
        "fine_tune",
        "teacher",
        "distillation",
        "baseline_predictions_as_targets",
        "baseline_checkpoint_input_allowed",
        "d07_001_checkpoint_input_allowed",
        "any_learned_checkpoint_input_allowed",
        "ensemble",
        "checkpoint_averaging",
        "formal_run_started",
    ):
        _require_false(protocol, key)
    if number >= 3:
        _require_false(protocol, "d07_002_checkpoint_input_allowed")
    _walk_forbidden_learned_inputs(raw)

    schedule = normalize_evaluation_schedule(raw, max_epochs=max_epochs)
    _validate_contract_lineage(raw)
    _validate_d07_002_parent_evidence(raw)
    if gates is not None and baseline_payload is not None:
        _validate_declared_baseline(raw, gates, baseline_payload)
    return CandidateContract(
        path=path,
        sha256=observed_hash,
        payload=raw,
        candidate_id=candidate_id,
        candidate_number=number,
        max_epochs=max_epochs,
        schedule=schedule,
    )


def _decimal_above(value: Any, threshold: float, decimal: str | None) -> bool:
    if value is None or not math.isfinite(float(value)):
        return False
    return Decimal(str(value)) > Decimal(decimal if decimal is not None else str(threshold))


def _decimal_below(value: Any, threshold: float, decimal: str | None) -> bool:
    if value is None or not math.isfinite(float(value)):
        return False
    return Decimal(str(value)) < Decimal(decimal if decimal is not None else str(threshold))


def evaluate_raw_age_gates(
    metrics: Mapping[str, Any], gates: SelectionGates
) -> tuple[int, list[str]]:
    decimals = gates.threshold_decimals or {}
    passed = {
        "strict_mae": _decimal_below(
            metrics.get("mae"), gates.strict_mae_max, decimals.get("strict_mae")
        ),
        "strict_pearson": _decimal_above(
            metrics.get("pearson"),
            gates.strict_pearson_min,
            decimals.get("strict_pearson"),
        ),
        "strict_r2": _decimal_above(
            metrics.get("r2"), gates.strict_r2_min, decimals.get("strict_r2")
        ),
    }
    failures = [name for name, success in passed.items() if not success]
    return sum(passed.values()), failures


def global_age_pareto_epochs(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    points: list[tuple[int, float, float, float]] = []
    for row in rows:
        metrics = row.get("raw_strict_age", {})
        try:
            point = (
                int(row["epoch"]),
                float(metrics["mae"]),
                float(metrics["pearson"]),
                float(metrics["r2"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in point[1:]):
            points.append(point)
    frontier: list[int] = []
    for epoch, mae, pearson, r2 in points:
        dominated = False
        for other_epoch, other_mae, other_pearson, other_r2 in points:
            if other_epoch == epoch:
                continue
            weakly_better = (
                other_mae <= mae and other_pearson >= pearson and other_r2 >= r2
            )
            strictly_better = (
                other_mae < mae or other_pearson > pearson or other_r2 > r2
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(epoch)
    return tuple(sorted(frontier))


def build_risk_epoch_schedule(
    age_rows: Sequence[Mapping[str, Any]],
    schedule: EvaluationSchedule,
    *,
    max_epochs: int,
) -> tuple[tuple[int, ...], dict[int, list[str]]]:
    epochs: dict[int, set[str]] = {
        epoch: {"fixed"} for epoch in schedule.fixed_risk_epochs
    }
    if schedule.add_age_gate_eligible_epochs:
        for row in age_rows:
            if row.get("all_age_gates_passed") is True:
                epochs.setdefault(int(row["epoch"]), set()).add("all_age_gates_passed")
    if schedule.add_new_age_pareto_epochs:
        for epoch in global_age_pareto_epochs(age_rows):
            epochs.setdefault(epoch, set()).add("global_age_pareto")
    if schedule.include_final_epoch:
        epochs.setdefault(max_epochs, set()).add("final_epoch")
    allowed = set(schedule.strict_age_epochs)
    if not set(epochs).issubset(allowed):
        raise D07ContractError("risk schedule contains an epoch without raw strict-age metrics")
    ordered = tuple(sorted(epochs))
    return ordered, {epoch: sorted(epochs[epoch]) for epoch in ordered}


def choose_candidate_epoch(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    evaluated = [row for row in rows if row.get("risk_evaluated") is True]
    promotable = [row for row in evaluated if row.get("all_same_epoch_gates_passed") is True]

    def promotion_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
        age = row["raw_strict_age"]
        risk = row["harrell_risk"]
        return (
            float(risk["macro_delta_c"]),
            -float(age["mae"]),
            float(age["pearson"]),
            float(age["r2"]),
            -int(row["epoch"]),
        )

    selected = max(promotable, key=promotion_key) if promotable else None
    diagnostic = None
    if evaluated:
        diagnostic = max(
            evaluated,
            key=lambda row: (
                int(row.get("age_gate_pass_count", 0)),
                float(row["harrell_risk"]["macro_delta_c"]),
                -float(row["raw_strict_age"]["mae"]),
                float(row["raw_strict_age"]["pearson"]),
                float(row["raw_strict_age"]["r2"]),
                -int(row["epoch"]),
            ),
        )
    return selected, diagnostic


def _trainable_state(model: EHRAGEModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _buffer_state(model: EHRAGEModel) -> dict[str, torch.Tensor]:
    return {
        name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()
    }


def _tensor_mapping_hash(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_trainable_checkpoint(
    *,
    path: Path,
    model: EHRAGEModel,
    epoch: int,
    candidate_id: str,
    contract_sha256: str,
    materialized_config_hash: str,
    age_anchor: float,
    initial_trainable_sha256: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"candidate checkpoint already exists: {path}")
    state = _trainable_state(model)
    buffers = _buffer_state(model)
    torch.save(
        {
            "schema_version": GENERIC_CHECKPOINT_SCHEMA,
            "candidate_id": candidate_id,
            "epoch": int(epoch),
            "candidate_contract_sha256": contract_sha256,
            "materialized_config_hash": materialized_config_hash,
            "age_anchor_train_folds_2_to_4_only": float(age_anchor),
            "initial_trainable_parameter_sha256": initial_trainable_sha256,
            "trainable_parameter_sha256": trainable_parameter_hash(model),
            "trainable_state_dict": state,
            "buffer_state_sha256": _tensor_mapping_hash(buffers),
            "buffer_state_dict": buffers,
            "frozen_cache_embeddings_omitted_and_hash_locked": True,
            "optimizer_state_omitted_not_resumable": True,
            "from_random_initialization": True,
            "warm_start": False,
            "fine_tune": False,
            "teacher": False,
            "distillation": False,
            "baseline_predictions_or_checkpoints_loaded": False,
        },
        path,
    )
    return {
        "epoch": int(epoch),
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "trainable_parameter_sha256": trainable_parameter_hash(model),
    }


def restore_trainable_checkpoint(
    *,
    path: Path,
    model: EHRAGEModel,
    candidate_id: str,
    contract_sha256: str,
    materialized_config_hash: str,
) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != GENERIC_CHECKPOINT_SCHEMA
        or checkpoint.get("candidate_id") != candidate_id
        or checkpoint.get("candidate_contract_sha256") != contract_sha256
        or checkpoint.get("materialized_config_hash") != materialized_config_hash
        or checkpoint.get("from_random_initialization") is not True
        or checkpoint.get("baseline_predictions_or_checkpoints_loaded") is not False
    ):
        raise D07ContractError("candidate checkpoint provenance mismatch")
    state = checkpoint.get("trainable_state_dict")
    if not isinstance(state, Mapping):
        raise D07ContractError("candidate checkpoint lacks trainable state")
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(parameters):
        raise D07ContractError("candidate checkpoint trainable parameter names changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise D07ContractError(f"candidate checkpoint tensor changed: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        buffers = {name: buffer for name, buffer in model.named_buffers()}
        stored_buffers = checkpoint.get("buffer_state_dict")
        if not isinstance(stored_buffers, Mapping) or set(stored_buffers) != set(buffers):
            raise D07ContractError("candidate checkpoint buffer names changed")
        if _tensor_mapping_hash(stored_buffers) != checkpoint.get("buffer_state_sha256"):
            raise D07ContractError("candidate checkpoint buffer hash mismatch")
        for name, buffer in buffers.items():
            value = stored_buffers[name]
            if not isinstance(value, torch.Tensor) or value.shape != buffer.shape:
                raise D07ContractError(f"candidate checkpoint buffer changed: {name}")
            buffer.copy_(value.to(device=buffer.device, dtype=buffer.dtype))
    if trainable_parameter_hash(model) != checkpoint.get("trainable_parameter_sha256"):
        raise D07ContractError("restored trainable parameter hash mismatch")
    return checkpoint


def _checkpoint_set_manifest(
    *,
    path: Path,
    contract: CandidateContract,
    checkpoints: Sequence[Mapping[str, Any]],
    materialized_config_hash: str,
    initial_trainable_sha256: str,
    final_trainable_sha256: str,
) -> None:
    expected_epochs = list(range(1, contract.max_epochs + 1))
    if [int(row["epoch"]) for row in checkpoints] != expected_epochs:
        raise D07ContractError("checkpoint set does not cover every allowed epoch")
    _write_json_exclusive(
        path,
        {
            "schema_version": GENERIC_CHECKPOINT_SET_SCHEMA,
            "status": "SEALED_BEFORE_FOLD0_READ",
            "created_utc": _utc_now(),
            "candidate_id": contract.candidate_id,
            "candidate_contract_sha256": contract.sha256,
            "materialized_config_hash": materialized_config_hash,
            "checkpoint_representation": (
                "all trainable parameters; frozen cache embeddings omitted and reconstructed "
                "only from hash-locked Revision-0710 cache"
            ),
            "full_epoch_coverage": True,
            "optimizer_resume_supported": False,
            "initial_trainable_parameter_sha256": initial_trainable_sha256,
            "final_trainable_parameter_sha256": final_trainable_sha256,
            "checkpoints": list(checkpoints),
        },
    )


def verify_checkpoint_set(path: Path, contract: CandidateContract) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints")
    if (
        payload.get("schema_version") != GENERIC_CHECKPOINT_SET_SCHEMA
        or payload.get("status") != "SEALED_BEFORE_FOLD0_READ"
        or payload.get("candidate_id") != contract.candidate_id
        or payload.get("candidate_contract_sha256") != contract.sha256
        or not isinstance(rows, list)
        or [int(row.get("epoch", -1)) for row in rows]
        != list(range(1, contract.max_epochs + 1))
    ):
        raise D07ContractError("invalid candidate checkpoint-set manifest")
    for row in rows:
        checkpoint = Path(str(row.get("path", "")))
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != row.get("sha256")
            or checkpoint.stat().st_size != int(row.get("bytes", -1))
        ):
            raise D07ContractError(
                f"candidate checkpoint changed before Fold0 access: epoch {row.get('epoch')}"
            )
    return payload


def _training_code_hashes() -> dict[str, str]:
    paths = {
        "ehr_age_model": MODEL_DIR / "ehr_age_model.py",
        "private_retain_age": MODEL_DIR / "private_retain_age.py",
        "ehr_age_losses": MODEL_DIR / "ehr_age_losses.py",
        "ehr_age_data": RUNNER_DIR / "ehr_age_data.py",
        "ehr_age_training": RUNNER_DIR / "ehr_age_training.py",
        "compact_cache_dataset": (
            REVISION_ROOT
            / "02_baselines_5fold"
            / "common"
            / "compact_cache_dataset.py"
        ),
        "d07_open_fold0_development": RUNNER_DIR / "d07_open_fold0_development.py",
        "d07_candidate_runner": Path(__file__).resolve(),
        "crossfit_metrics": REVISION_ROOT / "05_downstream" / "common" / "crossfit_metrics.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _verify_hashes_unchanged(expected: Mapping[str, str]) -> None:
    observed = _training_code_hashes()
    if observed != dict(expected):
        raise D07ContractError("candidate runner/model/evaluator code changed during execution")


def _materialized_source_hashes(
    *,
    cache_dir: Path,
    development_labels_path: Path,
    development_age_path: Path,
    development_manifest_path: Path,
    fold0_manifest_path: Path,
) -> dict[str, str]:
    """Bind Fold0 only through its historical seal before the read ledger exists."""

    development = _source_hashes(
        cache_dir=cache_dir,
        development_labels_path=development_labels_path,
        development_age_path=development_age_path,
        development_manifest_path=development_manifest_path,
    )
    sealed = _verify_sealed_manifest_metadata(fold0_manifest_path)
    return {
        **development,
        "fold0_age_targets": str(sealed["artifacts"]["age_targets"]["sha256"]),
        "fold0_joint_labels": str(sealed["artifacts"]["joint_labels"]["sha256"]),
    }


def _validate_declared_data_lineage(
    raw: Mapping[str, Any], observed: Mapping[str, str]
) -> None:
    declared = _require_mapping(_require_mapping(raw, "data_lineage"), "source_hashes")
    key_map = {
        "cache_hard_gate": "cache_hard_gate",
        "leakage_gate": "leakage_gate",
        "batch_manifest": "batch_manifest",
        "embedding_tables": "embedding_tables",
        "development_joint_labels": "development_labels",
        "development_age_targets": "development_age_targets",
        "development_label_manifest": "development_label_manifest",
        "fold0_age_targets": "fold0_age_targets",
        "fold0_joint_labels": "fold0_joint_labels",
    }
    mismatches = {
        declared_key: {
            "declared": declared.get(declared_key),
            "observed": observed.get(observed_key),
        }
        for declared_key, observed_key in key_map.items()
        if declared.get(declared_key) != observed.get(observed_key)
    }
    if mismatches:
        raise D07ContractError(f"candidate data lineage changed: {mismatches}")


def _make_model_and_training(
    contract: CandidateContract,
    data: Fold0DevelopmentData,
) -> tuple[
    EHRAGEModel,
    torch.device,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR,
    torch.amp.GradScaler,
    torch.Tensor,
    JointVectorizedBatcher,
    JointVectorizedBatcher,
    float,
    dict[str, Any],
]:
    raw = contract.payload
    training = raw["training"]
    seed = int(training["seed"])
    configure_reproducibility(
        seed,
        deterministic=bool(training["deterministic"]),
        tf32=bool(training["tf32"]),
    )
    age_anchor = float(
        data.cache.dataset.chronological_age.index_select(0, data.train_indices).mean()
    )
    cuda_rng_before_model = [state.clone() for state in torch.cuda.get_rng_state_all()]
    model = EHRAGEModel(
        data.cache.code_embedding,
        data.cache.text_embedding,
        age_anchor=age_anchor,
        config=_model_config(raw["model"]),
    )
    cuda_rng_after_model = [state.clone() for state in torch.cuda.get_rng_state_all()]
    cuda_preserved = len(cuda_rng_before_model) == len(cuda_rng_after_model) and all(
        torch.equal(before, after)
        for before, after in zip(cuda_rng_before_model, cuda_rng_after_model)
    )
    if model.config.base_age_mode == BASE_AGE_MODE_PRIVATE_RETAIN and not cuda_preserved:
        raise D07ContractError("private model construction mutated a CUDA RNG generator")
    construction_rng_audit = {
        "status": (
            "PASS"
            if model.config.base_age_mode == BASE_AGE_MODE_PRIVATE_RETAIN
            else "NOT_APPLICABLE_SHARED_MODEL"
        ),
        "model_constructed_on_cpu": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_rng_state_count": len(cuda_rng_before_model),
        "private_constructor_cuda_rng_states_preserved": cuda_preserved,
        "private_constructor_cuda_next_random_preserved_by_state_identity": cuda_preserved,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_epochs = int(training["warmup_epochs"])
    horizon = int(training["scheduler_horizon_epochs"])

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / max(warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs) / max(horizon - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(training["amp"]) and device.type == "cuda"
    )
    interval_edges = torch.tensor(
        raw["model"]["survival_interval_edges_months"],
        dtype=torch.float32,
        device=device,
    )
    train_batcher = JointVectorizedBatcher(
        data,
        data.train_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    calibration_batcher = JointVectorizedBatcher(
        data,
        data.calibration_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=seed,
    )
    return (
        model,
        device,
        optimizer,
        scheduler,
        scaler,
        interval_edges,
        train_batcher,
        calibration_batcher,
        age_anchor,
        construction_rng_audit,
    )


def _create_evaluation_manifest(
    *,
    path: Path,
    contract: CandidateContract,
    checkpoint_set_path: Path,
    checkpoint_set: Mapping[str, Any],
    anchor_checkpoint: Path,
    fold0_manifest_path: Path,
    materialized_config_hash: str,
) -> None:
    sealed = _verify_sealed_manifest_metadata(fold0_manifest_path)
    _write_json_exclusive(
        path,
        {
            "schema_version": EVALUATION_MANIFEST_SCHEMA,
            "status": "SEALED_BEFORE_FOLD0_READ",
            "created_utc": _utc_now(),
            "candidate_id": contract.candidate_id,
            "d07_protocol_sha256": D07_PROTOCOL_SHA256,
            "candidate_contract_path": str(contract.path),
            "candidate_contract_sha256": contract.sha256,
            "materialized_config_hash": materialized_config_hash,
            "checkpoint_path": str(anchor_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(anchor_checkpoint),
            "checkpoint_role": "epoch60_anchor_for_hash_locked_all_epoch_checkpoint_set",
            "candidate_checkpoint_set_manifest_path": str(checkpoint_set_path.resolve()),
            "candidate_checkpoint_set_manifest_sha256": sha256_file(checkpoint_set_path),
            "candidate_checkpoint_count": len(checkpoint_set["checkpoints"]),
            "strict_age_epoch_schedule": list(contract.schedule.strict_age_epochs),
            "harrell_schedule_is_derived_after_all_raw_strict_rows": True,
            "harrell_schedule_contract": contract.payload["evaluation"]["harrell_risk"],
            "selection_contract": contract.payload["selection"],
            "baseline_envelope_path": str(DEFAULT_BASELINE_ENVELOPE.resolve()),
            "baseline_envelope_sha256": sha256_file(DEFAULT_BASELINE_ENVELOPE),
            "baseline_artifact_manifest_sha256": sha256_file(
                DEFAULT_BASELINE_ARTIFACT_MANIFEST
            ),
            "declared_fold0_artifact_hashes": {
                name: sealed["artifacts"][name]["sha256"]
                for name in ("age_targets", "joint_labels")
            },
            "fold0_fitted_parameters": False,
            "fold1_screen_metrics_computed": False,
            "candidate_fallback_allowed": False,
            "checkpoint_averaging_allowed": False,
        },
    )


def _verify_generic_evaluation_manifest(
    path: Path,
    *,
    contract: CandidateContract,
    checkpoint_set_path: Path,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != EVALUATION_MANIFEST_SCHEMA
        or payload.get("status") != "SEALED_BEFORE_FOLD0_READ"
        or payload.get("candidate_id") != contract.candidate_id
        or payload.get("candidate_contract_sha256") != contract.sha256
        or payload.get("candidate_checkpoint_set_manifest_sha256")
        != sha256_file(checkpoint_set_path)
        or payload.get("candidate_checkpoint_count") != contract.max_epochs
        or payload.get("strict_age_epoch_schedule")
        != list(contract.schedule.strict_age_epochs)
        or payload.get("fold0_fitted_parameters") is not False
        or payload.get("fold1_screen_metrics_computed") is not False
    ):
        raise D07ContractError("generic candidate evaluation manifest is invalid")


def _prediction_ids(data: Fold0DevelopmentData, prediction: PredictionBundle) -> list[str]:
    return [
        data.cache.dataset.participant_ids[int(index)]
        for index in prediction.participant_indices.tolist()
    ]


def _assert_recomputed_metrics(
    expected: Mapping[str, Any], observed_age: Mapping[str, Any], observed_risk: Mapping[str, Any]
) -> None:
    comparisons = {
        "mae": (expected["raw_strict_age"]["mae"], observed_age["mae"]),
        "pearson": (expected["raw_strict_age"]["pearson"], observed_age["pearson"]),
        "r2": (expected["raw_strict_age"]["r2"], observed_age["r2"]),
        "macro_delta_c": (
            expected["harrell_risk"]["macro_delta_c"],
            observed_risk["macro_delta_c"],
        ),
    }
    for name, (left, right) in comparisons.items():
        if left is None or right is None or not math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-12
        ):
            raise D07ContractError(f"final same-epoch metric recomputation changed: {name}")


def _record_failure_before_fold0(
    *, run_dir: Path, stage: str, exception: BaseException, log_path: Path | None
) -> None:
    failure_path = run_dir / "candidate_failure_before_fold0.json"
    if failure_path.exists():
        return
    payload = {
        "schema_version": GENERIC_FAILURE_SCHEMA,
        "status": "FAILED_BEFORE_FOLD0_READ",
        "created_utc": _utc_now(),
        "stage": stage,
        "fold0_read_count": 0,
        "exception": {"type": type(exception).__name__, "message": str(exception)},
    }
    if log_path is not None and log_path.is_file():
        payload["run_log_sha256_at_failure"] = sha256_file(log_path)
    _write_json_exclusive(failure_path, payload)


def validate_d07_candidate_preflight(
    *,
    config_path: Path = DEFAULT_D07_CANDIDATE_CONFIG,
    expected_config_sha256: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    development_labels_path: Path = DEFAULT_DEVELOPMENT_LABELS,
    development_age_path: Path = DEFAULT_DEVELOPMENT_AGES,
    development_manifest_path: Path = DEFAULT_DEVELOPMENT_MANIFEST,
    fold0_manifest_path: Path = DEFAULT_FOLD0_MANIFEST,
    baseline_envelope_path: Path = DEFAULT_BASELINE_ENVELOPE,
) -> dict[str, Any]:
    """Validate a candidate without creating a run, loading data, or opening Fold0 artifacts."""

    validate_d07_protocol()
    gates, baseline_payload, baseline_validation = validate_d07_fold0_baseline_envelope(
        baseline_envelope_path.resolve()
    )
    contract = load_frozen_candidate_contract(
        config_path.resolve(),
        expected_sha256=expected_config_sha256,
        gates=gates,
        baseline_payload=baseline_payload,
    )
    source_hashes = _materialized_source_hashes(
        cache_dir=cache_dir.resolve(),
        development_labels_path=development_labels_path.resolve(),
        development_age_path=development_age_path.resolve(),
        development_manifest_path=development_manifest_path.resolve(),
        fold0_manifest_path=fold0_manifest_path.resolve(),
    )
    _validate_declared_data_lineage(contract.payload, source_hashes)
    return {
        "status": "VALID_NO_RUN_CREATED_NO_FOLD0_ARTIFACT_READ",
        "candidate_id": contract.candidate_id,
        "candidate_contract_sha256": contract.sha256,
        "plan_sha256": contract.payload["plan"]["sha256"],
        "d07_protocol_sha256": D07_PROTOCOL_SHA256,
        "baseline_validation": baseline_validation,
        "baseline_envelope_sha256": sha256_file(DEFAULT_BASELINE_ENVELOPE),
        "source_hashes": source_hashes,
        "fold0_hash_source": "historical sealed-manifest declarations only",
        "code_hashes": _training_code_hashes(),
        "max_epochs": contract.max_epochs,
        "strict_age_epoch_count": len(contract.schedule.strict_age_epochs),
        "fixed_risk_epochs": list(contract.schedule.fixed_risk_epochs),
        "add_age_gate_eligible_epochs": contract.schedule.add_age_gate_eligible_epochs,
        "add_global_age_pareto_epochs": contract.schedule.add_new_age_pareto_epochs,
        "include_final_epoch": contract.schedule.include_final_epoch,
    }


def run_d07_generic_candidate(
    *,
    config_path: Path = DEFAULT_D07_CANDIDATE_CONFIG,
    expected_config_sha256: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    development_labels_path: Path = DEFAULT_DEVELOPMENT_LABELS,
    development_age_path: Path = DEFAULT_DEVELOPMENT_AGES,
    development_manifest_path: Path = DEFAULT_DEVELOPMENT_MANIFEST,
    fold0_age_path: Path = DEFAULT_FOLD0_AGES,
    fold0_labels_path: Path = DEFAULT_FOLD0_LABELS,
    fold0_manifest_path: Path = DEFAULT_FOLD0_MANIFEST,
    baseline_envelope_path: Path = DEFAULT_BASELINE_ENVELOPE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    command_line: Sequence[str] = (),
) -> Path:
    started = time.perf_counter()
    stage = "VALIDATE_FROZEN_CONTRACT"
    run_dir: Path | None = None
    log_path: Path | None = None
    ledger_path: Path | None = None
    artifacts: dict[str, Path] = {}

    try:
        validate_d07_protocol()
        gates, baseline_payload, baseline_validation = (
            validate_d07_fold0_baseline_envelope(baseline_envelope_path.resolve())
        )
        contract = load_frozen_candidate_contract(
            config_path,
            expected_sha256=expected_config_sha256,
            gates=gates,
            baseline_payload=baseline_payload,
        )
        run_dir = output_root.resolve() / "candidates" / contract.candidate_id
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "run.log"
        log_path.write_text("", encoding="utf-8")

        def emit(message: str) -> None:
            print(message, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
                handle.flush()

        stage = "VALIDATE_DATA_AND_CODE_LINEAGE"
        cache_dir = cache_dir.resolve()
        development_labels_path = development_labels_path.resolve()
        development_age_path = development_age_path.resolve()
        development_manifest_path = development_manifest_path.resolve()
        fold0_age_path = fold0_age_path.resolve()
        fold0_labels_path = fold0_labels_path.resolve()
        fold0_manifest_path = fold0_manifest_path.resolve()
        source_hashes = _materialized_source_hashes(
            cache_dir=cache_dir,
            development_labels_path=development_labels_path,
            development_age_path=development_age_path,
            development_manifest_path=development_manifest_path,
            fold0_manifest_path=fold0_manifest_path,
        )
        _validate_declared_data_lineage(contract.payload, source_hashes)
        code_hashes = _training_code_hashes()
        materialized = {
            "schema_version": GENERIC_RUN_SCHEMA,
            "status": "MATERIALIZED_BEFORE_FIRST_OPTIMIZER_STEP",
            "created_utc": _utc_now(),
            "candidate_id": contract.candidate_id,
            "candidate_number": contract.candidate_number,
            "candidate_contract_path": str(contract.path),
            "candidate_contract_sha256": contract.sha256,
            "d07_protocol_path": str(D07_PROTOCOL_PATH.resolve()),
            "d07_protocol_sha256": D07_PROTOCOL_SHA256,
            "model": contract.payload["model"],
            "loss": contract.payload["loss"],
            "training": contract.payload["training"],
            "evaluation": contract.payload["evaluation"],
            "selection": contract.payload["selection"],
            "fold_roles": {
                "fold0": "OPEN_DEVELOPMENT",
                "fold1": "CALIBRATION_NUISANCE_ONLY_NO_CANDIDATE_METRICS",
                "fold2_to_4": "TRAINING_ONLY_NO_HELDOUT_EHR_METRICS",
            },
            "source_hashes": source_hashes,
            "fold0_hash_binding_before_read": (
                "historical sealed-manifest declarations only; Fold0 age/label artifact "
                "bytes are not opened until the candidate read ledger is committed"
            ),
            "code_hashes": code_hashes,
            "baseline_envelope_path": str(DEFAULT_BASELINE_ENVELOPE.resolve()),
            "baseline_envelope_sha256": sha256_file(DEFAULT_BASELINE_ENVELOPE),
            "baseline_artifact_manifest_path": str(
                DEFAULT_BASELINE_ARTIFACT_MANIFEST.resolve()
            ),
            "baseline_artifact_manifest_sha256": sha256_file(
                DEFAULT_BASELINE_ARTIFACT_MANIFEST
            ),
            "baseline_validator_path": str(DEFAULT_BASELINE_VALIDATOR.resolve()),
            "baseline_validator_sha256": sha256_file(DEFAULT_BASELINE_VALIDATOR),
            "baseline_validation": baseline_validation,
            "evaluation_command": shlex.join(list(command_line)) if command_line else None,
            "from_random_initialization": True,
            "learned_checkpoint_inputs": [],
            "baseline_prediction_inputs": [],
            "fold1_screen_metrics_computed": False,
            "fold2_to_4_heldout_ehr_metrics_computed": False,
        }
        materialized_config_hash = canonical_hash(materialized)
        materialized["materialized_config_hash"] = materialized_config_hash
        run_config_path = run_dir / "run_config.json"
        _write_json_exclusive(run_config_path, materialized)
        artifacts.update(
            {
                "frozen_candidate_contract": contract.path,
                "run_config": run_config_path,
                "run_log": log_path,
            }
        )
        emit(
            f"START candidate={contract.candidate_id} full_budget={contract.max_epochs} "
            "train_folds=2,3,4 fold1=calibration_nuisance_only fold0=open_development"
        )

        stage = "LOAD_TRAINING_AND_CALIBRATION_DATA"
        data = load_fold0_development_data(
            cache_dir=cache_dir,
            label_cache_path=development_labels_path,
            age_target_cache_path=development_age_path,
            label_manifest_path=development_manifest_path,
        )
        (
            model,
            device,
            optimizer,
            scheduler,
            scaler,
            interval_edges,
            train_batcher,
            calibration_batcher,
            age_anchor,
            model_construction_rng_audit,
        ) = _make_model_and_training(contract, data)
        private_instance_audit = validate_private_model_instance(
            model, contract.payload["model"]
        )
        private_common_init_audit = audit_private_common_initialization(
            model=model,
            code_embedding=data.cache.code_embedding,
            text_embedding=data.cache.text_embedding,
            age_anchor=age_anchor,
            seed=int(contract.payload["training"]["seed"]),
        )
        optimizer_parameter_audit = audit_optimizer_parameter_binding(model, optimizer)
        model_preflight_path = run_dir / "model_preflight_before_optimizer.json"
        _write_json_exclusive(
            model_preflight_path,
            {
                "schema_version": "revision0710.d07.model_preflight.v1",
                "status": "PASS",
                "created_utc": _utc_now(),
                "candidate_id": contract.candidate_id,
                "optimizer_steps_at_preflight": 0,
                "fold0_read_count_at_preflight": 0,
                "private_instance_audit": private_instance_audit,
                "model_construction_rng_audit": model_construction_rng_audit,
                "private_common_initialization_audit": private_common_init_audit,
                "optimizer_parameter_audit": optimizer_parameter_audit,
                "learned_checkpoint_or_prediction_loaded": False,
            },
        )
        artifacts["model_preflight_before_optimizer"] = model_preflight_path
        emit(
            f"MODEL_PREFLIGHT base_age_mode={model.config.base_age_mode} "
            f"private_instance={private_instance_audit['status']} "
            f"common_init={private_common_init_audit['status']}"
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        initial_trainable_hash = trainable_parameter_hash(model)
        checkpoints_dir = run_dir / "epoch_checkpoints"
        checkpoints_dir.mkdir(exist_ok=False)
        training_jsonl = run_dir / "training_history.jsonl"
        training_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []

        stage = "TRAIN_FULL_BUDGET_AND_SNAPSHOT_EVERY_EPOCH"
        training = contract.payload["training"]
        for epoch in range(1, contract.max_epochs + 1):
            epoch_started = time.perf_counter()
            survival_multiplier = _survival_multiplier(epoch, contract.payload["loss"])
            losses = _train_epoch(
                model=model,
                batcher=train_batcher,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                interval_edges=interval_edges,
                loss_config=_loss_config(contract.payload["loss"]),
                survival_multiplier=survival_multiplier,
                event_dropout_probability=float(training["event_dropout_probability"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                amp=bool(training["amp"]),
                max_batches=None,
            )
            if losses["optimizer_steps"] < 1:
                raise RuntimeError(f"epoch {epoch} completed without an optimizer update")
            scheduler.step()
            training_row = {
                "epoch": epoch,
                "survival_multiplier": survival_multiplier,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "losses": losses,
                "seconds": time.perf_counter() - epoch_started,
            }
            checkpoint_path = checkpoints_dir / f"trainable_checkpoint_epoch_{epoch:04d}.pt"
            checkpoint_row = save_trainable_checkpoint(
                path=checkpoint_path,
                model=model,
                epoch=epoch,
                candidate_id=contract.candidate_id,
                contract_sha256=contract.sha256,
                materialized_config_hash=materialized_config_hash,
                age_anchor=age_anchor,
                initial_trainable_sha256=initial_trainable_hash,
            )
            training_rows.append(training_row)
            checkpoint_rows.append(checkpoint_row)
            _append_jsonl(training_jsonl, training_row)
            emit(
                f"TRAIN epoch={epoch} loss={losses['total']:.9f} "
                f"lr={training_row['learning_rate']:.12g} "
                f"optimizer_steps={int(losses['optimizer_steps'])} "
                f"amp_skipped={int(losses['amp_skipped_steps'])} "
                f"snapshot_sha256={checkpoint_row['sha256']} "
                f"seconds={training_row['seconds']:.1f}"
            )

        final_trainable_hash = trainable_parameter_hash(model)
        if final_trainable_hash == initial_trainable_hash:
            raise RuntimeError("full-budget training did not change trainable parameters")
        training_history_path = run_dir / "training_history.json"
        _write_json_exclusive(
            training_history_path,
            {
                "schema_version": GENERIC_EPOCH_HISTORY_SCHEMA,
                "candidate_id": contract.candidate_id,
                "full_budget_completed": True,
                "fold0_read_count_at_training_completion": 0,
                "epochs": training_rows,
            },
        )
        checkpoint_set_path = run_dir / "checkpoint_set_before_fold0.json"
        _checkpoint_set_manifest(
            path=checkpoint_set_path,
            contract=contract,
            checkpoints=checkpoint_rows,
            materialized_config_hash=materialized_config_hash,
            initial_trainable_sha256=initial_trainable_hash,
            final_trainable_sha256=final_trainable_hash,
        )
        checkpoint_set = verify_checkpoint_set(checkpoint_set_path, contract)
        artifacts.update(
            {
                "training_history_jsonl": training_jsonl,
                "training_history": training_history_path,
                "checkpoint_set_before_fold0": checkpoint_set_path,
            }
        )

        stage = "SEAL_EVALUATION_AND_COMMIT_OPEN_FOLD0_READ"
        _verify_hashes_unchanged(code_hashes)
        if sha256_file(contract.path) != contract.sha256:
            raise D07ContractError("candidate contract changed during training")
        anchor_checkpoint = Path(checkpoint_rows[-1]["path"])
        evaluation_manifest_path = run_dir / "EVALUATION_MANIFEST_BEFORE_FOLD0.json"
        _create_evaluation_manifest(
            path=evaluation_manifest_path,
            contract=contract,
            checkpoint_set_path=checkpoint_set_path,
            checkpoint_set=checkpoint_set,
            anchor_checkpoint=anchor_checkpoint,
            fold0_manifest_path=fold0_manifest_path,
            materialized_config_hash=materialized_config_hash,
        )
        _verify_generic_evaluation_manifest(
            evaluation_manifest_path,
            contract=contract,
            checkpoint_set_path=checkpoint_set_path,
        )
        role_path, ledger_path = commit_open_development_read(
            output_root=output_root.resolve(),
            candidate_id=contract.candidate_id,
            sealed_manifest_path=fold0_manifest_path,
            checkpoint_path=anchor_checkpoint,
            evaluation_manifest_path=evaluation_manifest_path,
        )
        _verify_generic_evaluation_manifest(
            evaluation_manifest_path,
            contract=contract,
            checkpoint_set_path=checkpoint_set_path,
        )
        verify_checkpoint_set(checkpoint_set_path, contract)
        artifacts.update(
            {
                "evaluation_manifest_before_fold0": evaluation_manifest_path,
                "fold0_role_manifest": role_path,
                "fold0_read_ledger_entry": ledger_path,
            }
        )
        emit(
            f"FOLD0_READ_COMMITTED candidate={contract.candidate_id} "
            f"ledger={ledger_path.name} checkpoints={contract.max_epochs}"
        )

        stage = "LOAD_AUTHORIZED_OPEN_FOLD0_DATA"
        fold0_data = load_authorized_open_fold0_data(
            cache_dir=cache_dir,
            fold0_age_path=fold0_age_path,
            fold0_labels_path=fold0_labels_path,
            sealed_manifest_path=fold0_manifest_path,
            read_ledger_entry_path=ledger_path,
            candidate_id=contract.candidate_id,
        )
        actual_fold0_hashes = {
            "age_targets": sha256_file(fold0_age_path),
            "joint_labels": sha256_file(fold0_labels_path),
        }
        declared_fold0_hashes = {
            "age_targets": source_hashes["fold0_age_targets"],
            "joint_labels": source_hashes["fold0_joint_labels"],
        }
        if actual_fold0_hashes != declared_fold0_hashes:
            raise D07ContractError(
                "authorized Fold0 artifacts disagree with their historical sealed manifest"
            )
        fold0_actual_verification_path = run_dir / "fold0_actual_hash_verification_after_ledger.json"
        _write_json_exclusive(
            fold0_actual_verification_path,
            {
                "schema_version": "revision0710.d07.fold0_actual_hash_verification.v1",
                "status": "PASS_AFTER_COMMITTED_READ",
                "created_utc": _utc_now(),
                "candidate_id": contract.candidate_id,
                "fold0_read_ledger_entry": str(ledger_path.resolve()),
                "fold0_read_ledger_sha256": sha256_file(ledger_path),
                "actual_sha256": actual_fold0_hashes,
                "historical_sealed_manifest_declared_sha256": declared_fold0_hashes,
            },
        )
        artifacts["fold0_actual_hash_verification_after_ledger"] = (
            fold0_actual_verification_path
        )
        fold0_batcher = JointVectorizedBatcher(
            fold0_data,
            fold0_data.screen_indices,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            seed=int(training["seed"]),
        )
        checkpoint_by_epoch = {
            int(row["epoch"]): Path(str(row["path"])) for row in checkpoint_rows
        }
        checkpoint_hash_by_epoch = {
            int(row["epoch"]): str(row["sha256"]) for row in checkpoint_rows
        }

        def restore_epoch(epoch: int) -> None:
            checkpoint_path = checkpoint_by_epoch[epoch]
            if sha256_file(checkpoint_path) != checkpoint_hash_by_epoch[epoch]:
                raise D07ContractError(
                    f"candidate checkpoint changed after the pre-read seal: epoch {epoch}"
                )
            restore_trainable_checkpoint(
                path=checkpoint_path,
                model=model,
                candidate_id=contract.candidate_id,
                contract_sha256=contract.sha256,
                materialized_config_hash=materialized_config_hash,
            )

        stage = "EVALUATE_RAW_STRICT_AGE_SCHEDULE"
        strict_jsonl = run_dir / "fold0_raw_strict_age_history.jsonl"
        age_rows: list[dict[str, Any]] = []
        fold0_prediction_calls = 0
        for epoch in contract.schedule.strict_age_epochs:
            restore_epoch(epoch)
            fold0_prediction = _predict(
                model=model,
                batcher=fold0_batcher,
                device=device,
                amp=bool(training["amp"]),
                max_batches=None,
            )
            fold0_prediction_calls += 1
            raw_strict = pooled_age_metrics(
                fold0_prediction.chronological_age,
                fold0_prediction.predicted_age,
                participant_mask=fold0_prediction.strict_healthy,
            )
            pass_count, failures = evaluate_raw_age_gates(raw_strict, gates)
            row = {
                "epoch": epoch,
                "checkpoint_sha256": sha256_file(checkpoint_by_epoch[epoch]),
                "raw_strict_age": raw_strict,
                "age_gate_pass_count": pass_count,
                "age_gate_failures": failures,
                "all_age_gates_passed": not failures,
            }
            age_rows.append(row)
            _append_jsonl(strict_jsonl, row)
            emit(
                f"FOLD0_AGE epoch={epoch} mae={raw_strict['mae']} "
                f"pearson={raw_strict['pearson']} r2={raw_strict['r2']} "
                f"age_gates={pass_count}/3"
            )

        pareto_epochs = global_age_pareto_epochs(age_rows)
        risk_epochs, risk_reasons = build_risk_epoch_schedule(
            age_rows, contract.schedule, max_epochs=contract.max_epochs
        )
        risk_schedule_path = run_dir / "harrell_risk_schedule_after_strict_age.json"
        _write_json_exclusive(
            risk_schedule_path,
            {
                "schema_version": "revision0710.d07.harrell_schedule.v1",
                "candidate_id": contract.candidate_id,
                "derived_only_after_all_strict_age_rows": True,
                "strict_age_row_count": len(age_rows),
                "global_age_pareto_epochs": list(pareto_epochs),
                "risk_epochs": list(risk_epochs),
                "risk_reasons": {str(epoch): risk_reasons[epoch] for epoch in risk_epochs},
                "fixed_epochs": list(contract.schedule.fixed_risk_epochs),
                "age_gate_eligible_epochs": [
                    int(row["epoch"])
                    for row in age_rows
                    if row["all_age_gates_passed"]
                ],
                "final_epoch_included": contract.schedule.include_final_epoch,
                "deduplicated": True,
                "ascending": True,
            },
        )
        artifacts.update(
            {
                "fold0_raw_strict_age_history_jsonl": strict_jsonl,
                "harrell_risk_schedule_after_strict_age": risk_schedule_path,
            }
        )

        stage = "EVALUATE_DECLARED_HARRELL_EPOCHS"
        if gates.baseline_organ_delta_c is None:
            raise D07ContractError("canonical D07 envelope lacks organ delta-C values")
        calibration_ids_reference: list[str] | None = None
        risk_jsonl = run_dir / "fold0_harrell_risk_history.jsonl"
        age_by_epoch = {int(row["epoch"]): row for row in age_rows}
        risk_rows: list[dict[str, Any]] = []
        for epoch in risk_epochs:
            restore_epoch(epoch)
            calibration = _predict(
                model=model,
                batcher=calibration_batcher,
                device=device,
                amp=bool(training["amp"]),
                max_batches=None,
            )
            calibration_ids = _prediction_ids(data, calibration)
            if calibration_ids_reference is None:
                calibration_ids_reference = calibration_ids
            elif calibration_ids != calibration_ids_reference:
                raise D07ContractError("Fold1 calibration participant order changed")
            fold0_prediction = _predict(
                model=model,
                batcher=fold0_batcher,
                device=device,
                amp=bool(training["amp"]),
                max_batches=None,
            )
            fold0_prediction_calls += 1
            fold0_ids = _prediction_ids(fold0_data, fold0_prediction)
            risk_metrics, _, _, _ = calibration_to_open_fold0_cox(
                calibration,
                fold0_prediction,
                calibration_participant_ids=calibration_ids,
                fold0_participant_ids=fold0_ids,
                baseline_organ_delta_c=gates.baseline_organ_delta_c,
            )
            raw_strict = age_by_epoch[epoch]["raw_strict_age"]
            eligible, failures = gates.evaluate(raw_strict, risk_metrics)
            row = {
                **age_by_epoch[epoch],
                "global_age_pareto": epoch in pareto_epochs,
                "risk_evaluated": True,
                "risk_schedule_reasons": risk_reasons[epoch],
                "harrell_risk": risk_metrics,
                "all_same_epoch_gates_passed": eligible,
                "all_same_epoch_gate_failures": failures,
            }
            risk_rows.append(row)
            _append_jsonl(risk_jsonl, row)
            emit(
                f"FOLD0_RISK epoch={epoch} delta_c={risk_metrics['macro_delta_c']} "
                f"positive={risk_metrics['positive_organs']}/7 "
                f"beat={risk_metrics['organs_beating_baseline']}/7 "
                f"same_epoch_eligible={eligible}"
            )

        selected, diagnostic = choose_candidate_epoch(risk_rows)
        final_row = selected if selected is not None else diagnostic
        if final_row is None:
            raise D07ContractError("declared Harrell schedule produced no evaluable epoch")
        final_epoch = int(final_row["epoch"])

        stage = "RECOMPUTE_AND_EXPORT_SELECTED_OR_DIAGNOSTIC_EPOCH"
        restore_epoch(final_epoch)
        calibration = _predict(
            model=model,
            batcher=calibration_batcher,
            device=device,
            amp=bool(training["amp"]),
            max_batches=None,
        )
        calibration_ids = _prediction_ids(data, calibration)
        fold0_prediction = _predict(
            model=model,
            batcher=fold0_batcher,
            device=device,
            amp=bool(training["amp"]),
            max_batches=None,
        )
        fold0_prediction_calls += 1
        fold0_ids = _prediction_ids(fold0_data, fold0_prediction)
        final_risk, calibration_corrected, fold0_corrected, correction = (
            calibration_to_open_fold0_cox(
                calibration,
                fold0_prediction,
                calibration_participant_ids=calibration_ids,
                fold0_participant_ids=fold0_ids,
                baseline_organ_delta_c=gates.baseline_organ_delta_c,
            )
        )
        final_raw_strict = pooled_age_metrics(
            fold0_prediction.chronological_age,
            fold0_prediction.predicted_age,
            participant_mask=fold0_prediction.strict_healthy,
        )
        _assert_recomputed_metrics(final_row, final_raw_strict, final_risk)

        risk_by_epoch = {int(row["epoch"]): row for row in risk_rows}
        combined_rows = []
        for age_row in age_rows:
            epoch = int(age_row["epoch"])
            combined = {
                **age_row,
                "global_age_pareto": epoch in pareto_epochs,
                "risk_evaluated": epoch in risk_by_epoch,
            }
            if epoch in risk_by_epoch:
                combined.update(
                    {
                        "risk_schedule_reasons": risk_by_epoch[epoch][
                            "risk_schedule_reasons"
                        ],
                        "harrell_risk": risk_by_epoch[epoch]["harrell_risk"],
                        "all_same_epoch_gates_passed": risk_by_epoch[epoch][
                            "all_same_epoch_gates_passed"
                        ],
                        "all_same_epoch_gate_failures": risk_by_epoch[epoch][
                            "all_same_epoch_gate_failures"
                        ],
                    }
                )
            combined_rows.append(combined)
        epoch_history_path = run_dir / "fold0_epoch_history.json"
        _write_json_exclusive(
            epoch_history_path,
            {
                "schema_version": GENERIC_EPOCH_HISTORY_SCHEMA,
                "candidate_id": contract.candidate_id,
                "all_strict_age_rows_retained": True,
                "all_computed_harrell_rows_retained": True,
                "epochs": combined_rows,
            },
        )

        corrected_matrix = fold0_corrected["corrected_predicted_age"].to_numpy(
            dtype=np.float64
        ).reshape(len(fold0_ids), len(ORGAN_SYSTEMS))
        corrected_strict = pooled_age_metrics(
            fold0_prediction.chronological_age,
            corrected_matrix,
            participant_mask=fold0_prediction.strict_healthy,
        )
        final_status = "PROMOTABLE" if selected is not None else "NON_PROMOTABLE"
        primary_row = {
            "epoch": final_epoch,
            "strict_mae": final_raw_strict["mae"],
            "strict_pearson": final_raw_strict["pearson"],
            "strict_r2": final_raw_strict["r2"],
            "age_sex_harrell_macro_delta_c": final_risk["macro_delta_c"],
            "positive_organs": final_risk["positive_organs"],
            "organs_beating_baseline": final_risk["organs_beating_baseline"],
        }
        metrics_path = run_dir / "fold0_open_development_metrics.json"
        _write_json_exclusive(
            metrics_path,
            {
                "schema_version": GENERIC_METRICS_SCHEMA,
                "candidate_id": contract.candidate_id,
                "status": final_status,
                "fold": 0,
                "fold_role": "OPEN_DEVELOPMENT",
                "confirmatory_interpretation_allowed": False,
                "selected_epoch": None if selected is None else final_epoch,
                "diagnostic_epoch": final_epoch if selected is None else None,
                "primary_same_epoch_row": primary_row,
                "strict_age_raw_primary": {
                    "population": "strict_healthy_reference_60m=1 only",
                    **final_raw_strict,
                    "organs": _organ_age_metrics(
                        fold0_prediction.chronological_age,
                        fold0_prediction.predicted_age,
                        fold0_prediction.strict_healthy,
                    ),
                },
                "strict_age_fold1_corrected_diagnostic_only": {
                    "population": "strict_healthy_reference_60m=1 only",
                    **corrected_strict,
                    "organs": _organ_age_metrics(
                        fold0_prediction.chronological_age,
                        corrected_matrix,
                        fold0_prediction.strict_healthy,
                    ),
                },
                "age_sex_harrell_primary_development": final_risk,
                "d07_fold0_gate_eligible": selected is not None,
                "d07_fold0_gate_failures": final_row["all_same_epoch_gate_failures"],
                "d07_fold0_thresholds": asdict(gates),
                "global_age_pareto_epochs": list(pareto_epochs),
                "risk_evaluated_epochs": list(risk_epochs),
                "all_epoch_history_path": str(epoch_history_path.resolve()),
            },
        )

        correction_path = run_dir / "fold1_calibration_age_correction.json"
        _write_json_exclusive(correction_path, _serialize_correction(correction))
        fold0_corrected["model_risk_score"] = fold0_prediction.risk_score.astype(
            np.float32
        ).reshape(-1)
        fold0_corrected["fold_role"] = "OPEN_DEVELOPMENT"
        fold0_prediction_path = run_dir / "fold0_open_development_predictions_long.parquet"
        fold0_corrected.to_parquet(fold0_prediction_path, index=False)
        calibration_corrected["model_risk_score"] = calibration.risk_score.astype(
            np.float32
        ).reshape(-1)
        calibration_corrected["fold_role"] = "FOLD1_CALIBRATION_NUISANCE_ONLY"
        calibration_path = run_dir / "fold1_calibration_nuisance_corrected_long.parquet"
        calibration_corrected.to_parquet(calibration_path, index=False)

        checkpoint_decision_path = run_dir / "checkpoint_decision.json"
        _write_json_exclusive(
            checkpoint_decision_path,
            {
                "schema_version": "revision0710.d07.checkpoint_decision.v1",
                "status": final_status,
                "candidate_id": contract.candidate_id,
                "selection_pool": "risk_evaluated_epochs_only",
                "same_epoch_four_gates_required": True,
                "selected_epoch": None if selected is None else final_epoch,
                "diagnostic_epoch": final_epoch if selected is None else None,
                "checkpoint_path": str(checkpoint_by_epoch[final_epoch].resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_by_epoch[final_epoch]),
                "primary_same_epoch_row": primary_row,
                "fallback_used": False,
                "checkpoint_averaging_used": False,
                "second_seed_used": False,
            },
        )

        summary_path = run_dir / "candidate_summary.json"
        _write_json_exclusive(
            summary_path,
            {
                "schema_version": CANDIDATE_SCHEMA,
                "status": final_status,
                "created_utc": _utc_now(),
                "candidate_id": contract.candidate_id,
                "candidate_contract_sha256": contract.sha256,
                "materialized_config_hash": materialized_config_hash,
                "full_training_budget_completed": True,
                "training_epochs": contract.max_epochs,
                "selected_epoch": None if selected is None else final_epoch,
                "diagnostic_epoch": final_epoch if selected is None else None,
                "primary_same_epoch_row": primary_row,
                "d07_fold0_gate_eligible": selected is not None,
                "d07_fold0_gate_failures": final_row["all_same_epoch_gate_failures"],
                "strict_age_evaluator_calls": len(age_rows),
                "harrell_evaluator_calls": len(risk_rows) + 1,
                "fold0_prediction_calls": fold0_prediction_calls,
                "fold0_read_ledger_entry": str(ledger_path.resolve()),
                "fold0_read_count_this_candidate": 1,
                "fold1_screen_metrics_computed": False,
                "fold2_to_4_heldout_ehr_metrics_computed": False,
                "from_random_initialization": True,
                "baseline_predictions_or_checkpoints_loaded": False,
                "fine_tune": False,
                "distillation": False,
                "initial_trainable_parameter_sha256": initial_trainable_hash,
                "final_trainable_parameter_sha256": final_trainable_hash,
                "peak_cuda_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        artifacts.update(
            {
                "fold0_harrell_risk_history_jsonl": risk_jsonl,
                "fold0_epoch_history": epoch_history_path,
                "fold0_open_development_metrics": metrics_path,
                "fold1_calibration_age_correction": correction_path,
                "fold0_open_development_predictions_long": fold0_prediction_path,
                "fold1_calibration_nuisance_corrected_long": calibration_path,
                "checkpoint_decision": checkpoint_decision_path,
                "selected_or_diagnostic_checkpoint": checkpoint_by_epoch[final_epoch],
                "candidate_summary": summary_path,
            }
        )

        stage = "FINALIZE_RUN_LOG_AND_HASHES"
        _verify_hashes_unchanged(code_hashes)
        verify_checkpoint_set(checkpoint_set_path, contract)
        if sha256_file(contract.path) != contract.sha256:
            raise D07ContractError("candidate contract changed before completion")
        emit(
            f"COMPLETE candidate={contract.candidate_id} status={final_status} "
            f"epoch={final_epoch} strict_mae={final_raw_strict['mae']} "
            f"strict_r={final_raw_strict['pearson']} strict_r2={final_raw_strict['r2']} "
            f"harrell_delta_c={final_risk['macro_delta_c']}"
        )
        artifact_hashes_path = run_dir / "artifact_hashes.json"
        write_artifact_hash_registry(artifact_hashes_path, artifacts)
        artifacts["artifact_hashes"] = artifact_hashes_path
        stage = "WRITE_COMPLETION_RECEIPT"
        write_fold0_read_completion_receipt(
            output_root=output_root.resolve(),
            read_ledger_entry_path=ledger_path,
            artifacts=artifacts,
        )
        return run_dir
    except BaseException as exc:
        if run_dir is not None:
            if ledger_path is None:
                if log_path is not None and log_path.is_file():
                    try:
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(
                                f"FAILED_BEFORE_FOLD0_READ stage={stage} "
                                f"error={type(exc).__name__}: {exc}\n"
                            )
                    except Exception:
                        pass
                try:
                    _record_failure_before_fold0(
                        run_dir=run_dir,
                        stage=stage,
                        exception=exc,
                        log_path=log_path,
                    )
                except Exception:
                    pass
            else:
                try:
                    completed_registry = run_dir / "artifact_hashes.json"
                    if (
                        not completed_registry.exists()
                        and log_path is not None
                        and log_path.is_file()
                    ):
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(
                                f"FAILED_AFTER_COMMITTED_READ stage={stage} "
                                f"error={type(exc).__name__}: {exc}\n"
                            )
                    failure_path = run_dir / "candidate_failure_after_fold0.json"
                    if not completed_registry.exists() and not failure_path.exists():
                        _write_json_exclusive(
                            failure_path,
                            {
                                "schema_version": GENERIC_FAILURE_SCHEMA,
                                "status": "FAILED_AFTER_COMMITTED_READ",
                                "created_utc": _utc_now(),
                                "stage": stage,
                                "exception": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                },
                            },
                        )
                    if failure_path.exists():
                        artifacts["candidate_failure_after_fold0"] = failure_path
                    if completed_registry.exists():
                        artifacts["artifact_hashes"] = completed_registry
                    write_fold0_read_failure_receipt(
                        output_root=output_root.resolve(),
                        read_ledger_entry_path=ledger_path,
                        stage=stage,
                        exception=exc,
                        artifacts=artifacts,
                    )
                except BaseException as receipt_exc:
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"failure receipt could not be written: {receipt_exc}")
        raise
