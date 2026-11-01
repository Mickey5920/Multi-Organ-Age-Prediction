from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

import v8_contract
import v8_inputs


REPO_ROOT = Path(__file__).resolve().parents[4]
V1_ROOT = (
    REPO_ROOT
    / "Revision-0710"
    / "06_figures_and_statistics"
    / "00_plan"
    / "fig04_fig05_ablation_prefreeze_v1"
)
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

# This must run before importlib executes any immutable V1 runtime module.
V1_RUNTIME_PREFLIGHT = v8_contract.preflight_v1_runtime_bindings(Path(__file__).resolve().parent)


def _load_v1_model_module():
    name = "revision0710_fig0405_v1_model_for_v8"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V1_ROOT / "ablation_model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable V1 model wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V1 = _load_v1_model_module()
IndependentOrganAgeOutput = _V1.IndependentOrganAgeOutput


def _known(variant_id: str) -> str:
    value = str(variant_id)
    if value not in v8_inputs.KNOWN_VARIANTS:
        raise ValueError(f"unknown variant: {value!r}")
    return value


def variant_model_policy(variant_id: str) -> dict[str, object]:
    value = _known(variant_id)
    policies: dict[str, dict[str, object]] = {
        "fig04_01_canonical_control": {
            "canonical_ehr_age_full_method": True,
            "base_age_mode": "private_retain",
            "same_executor_seed_fold_protocol": True,
        },
        "fig04_02_no_organ_queries": {
            "organ_queries": "seven_identical_exact_zero_frozen",
        },
        "fig04_03_no_common_bidirectional_gru": {
            "shared_bidir_gru": False,
            "shared_gru_output": "exact_zero_tensor",
            "private_retain_grus": True,
        },
        "fig04_07_seven_independent_organ_age_heads": {
            "independent_age_heads": 7,
            "shared_scalar_base_age": False,
            "survival_risk_head": True,
        },
    }
    return dict(policies.get(value, {"canonical_e55_architecture": True}))


def _v1_model_variant(variant_id: str) -> str:
    if variant_id in {
        "fig04_01_canonical_control",
        "fig04_04_no_survival_supervision",
        "fig04_05_no_strict_offset_shrinkage",
        "fig04_06_no_consistency_objective",
    }:
        # The immutable V1 builder uses this recognized ID only to instantiate
        # the canonical architecture. Its input wrapper is deliberately discarded.
        return "fig05_01_no_phenotypes_or_missingness"
    return variant_id


class ProductionAblationModel(nn.Module):
    def __init__(self, variant_id: str, model: nn.Module) -> None:
        super().__init__()
        self.variant_id = _known(variant_id)
        self.model = model

    def forward(self, **inputs: torch.Tensor):
        transformed = v8_inputs.apply_variant_inputs(inputs, self.variant_id)
        return self.model(**transformed)


def build_production_model(
    variant_id: str,
    code_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    *,
    age_anchor: float,
    config,
) -> ProductionAblationModel:
    value = _known(variant_id)
    v1_wrapper = _V1.build_variant_model(
        _v1_model_variant(value),
        code_embedding,
        text_embedding,
        age_anchor=age_anchor,
        config=config,
    )
    return ProductionAblationModel(value, v1_wrapper.model)


def build_tiny_production_model(variant_id: str) -> ProductionAblationModel:
    value = _known(variant_id)
    v1_wrapper = _V1.build_tiny_variant_model(_v1_model_variant(value))
    return ProductionAblationModel(value, v1_wrapper.model)


def model_wiring_receipt(model: ProductionAblationModel) -> dict[str, object]:
    variant = model.variant_id
    underlying = model.model
    receipt: dict[str, object] = {
        "variant_id": variant,
        "policy": variant_model_policy(variant),
    }
    if variant == "fig04_02_no_organ_queries":
        query = underlying.organ_queries.detach()
        receipt.update(
            {
                "query_exact_zero": bool(torch.count_nonzero(query).item() == 0),
                "query_identical": bool(torch.equal(query, query[0].expand_as(query))),
                "query_requires_grad": bool(underlying.organ_queries.requires_grad),
            }
        )
    if variant == "fig04_03_no_common_bidirectional_gru":
        private = underlying.private_retain_age_tower.backbone
        receipt.update(
            {
                "shared_gru_module_present": hasattr(underlying.temporal_encoder, "gru"),
                "shared_gru_output": getattr(underlying.temporal_encoder, "shared_gru_output", None),
                "private_alpha_gru_present": isinstance(private.alpha_gru, nn.GRU),
                "private_beta_gru_present": isinstance(private.beta_gru, nn.GRU),
            }
        )
    if variant == "fig04_07_seven_independent_organ_age_heads":
        receipt.update(
            {
                "independent_age_heads": len(underlying.independent_age_heads),
                "base_age_output": None,
                "survival_risk_head_present": hasattr(underlying, "risk_head"),
            }
        )
    return receipt
