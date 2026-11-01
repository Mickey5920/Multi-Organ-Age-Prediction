from __future__ import annotations

from collections.abc import Mapping

import torch


FIG05_VARIANTS = (
    "fig05_01_no_phenotypes_or_missingness",
    "fig05_02_no_hospital_icd",
    "fig05_03_no_gp_prescriptions",
    "fig05_04_no_event_text",
    "fig05_05_no_longitudinal_events",
)
FIG04_VARIANTS = (
    "fig04_01_canonical_control",
    "fig04_02_no_organ_queries",
    "fig04_03_no_common_bidirectional_gru",
    "fig04_04_no_survival_supervision",
    "fig04_05_no_strict_offset_shrinkage",
    "fig04_06_no_consistency_objective",
    "fig04_07_seven_independent_organ_age_heads",
)
KNOWN_VARIANTS = (*FIG04_VARIANTS, *FIG05_VARIANTS)
EVENT_KEYS = (
    "event_code_id",
    "event_text_id",
    "event_relative_days",
    "event_type_id",
    "event_valid_mask",
)
MODEL_INPUT_KEYS = (
    "phenotype",
    "phenotype_missing_mask",
    "sex",
    *EVENT_KEYS,
)


def _known(variant_id: str) -> str:
    value = str(variant_id)
    if value not in KNOWN_VARIANTS:
        raise ValueError(f"unknown variant: {value!r}")
    return value


def variant_input_policy(variant_id: str) -> dict[str, object]:
    value = _known(variant_id)
    policies: dict[str, dict[str, object]] = {
        "fig05_01_no_phenotypes_or_missingness": {
            "phenotype": "zero",
            "phenotype_missing_mask": "zero",
            "sex": "retain",
        },
        "fig05_02_no_hospital_icd": {
            "remove_event_type": 1,
            "stable_left_compaction": True,
        },
        "fig05_03_no_gp_prescriptions": {
            "remove_event_type": 2,
            "stable_left_compaction": True,
        },
        "fig05_04_no_event_text": {
            "event_text_id": "padding_zero",
            "mask_unchanged": True,
        },
        "fig05_05_no_longitudinal_events": {
            "all_event_arrays": "padding_zero",
            "static_and_sex": "retain",
        },
    }
    return dict(policies.get(value, {"identity": True}))


def assert_left_aligned(mask: torch.Tensor) -> None:
    if mask.ndim != 2:
        raise ValueError("event_valid_mask must have shape [batch, sequence]")
    valid = mask.bool()
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("valid events must form a left-aligned prefix")


def validate_inputs(inputs: Mapping[str, torch.Tensor]) -> None:
    if set(inputs) != set(MODEL_INPUT_KEYS):
        raise ValueError("model input keys do not match the closed schema")
    batch = len(inputs["sex"])
    if inputs["phenotype"].shape != (batch, 78):
        raise ValueError("phenotype must have shape [batch,78]")
    if inputs["phenotype_missing_mask"].shape != (batch, 78):
        raise ValueError("phenotype missingness must have shape [batch,78]")
    shape = inputs["event_valid_mask"].shape
    if len(shape) != 2 or shape[0] != batch:
        raise ValueError("event arrays must have shape [batch,sequence]")
    if any(inputs[key].shape != shape for key in EVENT_KEYS):
        raise ValueError("event tuple arrays are not aligned")
    assert_left_aligned(inputs["event_valid_mask"])
    invalid = ~inputs["event_valid_mask"].bool()
    for key in EVENT_KEYS[:-1]:
        if torch.count_nonzero(inputs[key][invalid]).item():
            raise ValueError(f"{key} has a non-padding value outside the valid mask")
    valid = inputs["event_valid_mask"].bool()
    if (valid & inputs["event_relative_days"].ge(0)).any():
        raise ValueError("valid events must be strictly pre-index")
    if (valid & inputs["event_type_id"].eq(2) & inputs["event_code_id"].ne(0)).any():
        raise ValueError("GP prescription events must use code_id=0")


def _clone(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def stable_left_compact(
    inputs: Mapping[str, torch.Tensor], keep: torch.Tensor
) -> dict[str, torch.Tensor]:
    validate_inputs(inputs)
    valid = inputs["event_valid_mask"].bool()
    keep = keep.bool()
    if keep.shape != valid.shape or (keep & ~valid).any():
        raise ValueError("invalid event keep mask")
    result = _clone(inputs)
    destination = keep.long().cumsum(dim=1) - 1
    rows, source_columns = torch.nonzero(keep, as_tuple=True)
    target_columns = destination[rows, source_columns]
    for key in EVENT_KEYS:
        compacted = torch.zeros_like(inputs[key])
        compacted[rows, target_columns] = inputs[key][rows, source_columns]
        result[key] = compacted
    validate_inputs(result)
    return result


def apply_variant_inputs(
    inputs: Mapping[str, torch.Tensor], variant_id: str
) -> dict[str, torch.Tensor]:
    value = _known(variant_id)
    validate_inputs(inputs)
    result = _clone(inputs)
    if value == "fig05_01_no_phenotypes_or_missingness":
        result["phenotype"].zero_()
        result["phenotype_missing_mask"].zero_()
    elif value == "fig05_02_no_hospital_icd":
        valid = inputs["event_valid_mask"].bool()
        result = stable_left_compact(inputs, valid & inputs["event_type_id"].ne(1))
    elif value == "fig05_03_no_gp_prescriptions":
        valid = inputs["event_valid_mask"].bool()
        result = stable_left_compact(inputs, valid & inputs["event_type_id"].ne(2))
    elif value == "fig05_04_no_event_text":
        result["event_text_id"].zero_()
    elif value == "fig05_05_no_longitudinal_events":
        for key in EVENT_KEYS:
            result[key].zero_()
    validate_inputs(result)
    return result


def compact_event_dropout(
    inputs: Mapping[str, torch.Tensor],
    *,
    probability: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= probability < 1.0:
        raise ValueError("event dropout probability must lie in [0,1)")
    validate_inputs(inputs)
    if probability == 0.0:
        return _clone(inputs)
    valid = inputs["event_valid_mask"].bool()
    random = torch.rand(valid.shape, device=valid.device, generator=generator)
    return stable_left_compact(inputs, valid & random.ge(probability))

