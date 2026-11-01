from __future__ import annotations

from collections.abc import Mapping

import torch


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
INPUT_ABLATION_IDS = (
    "fig05_01_no_phenotypes_or_missingness",
    "fig05_02_no_hospital_icd",
    "fig05_03_no_gp_prescriptions",
    "fig05_04_no_event_text",
    "fig05_05_no_longitudinal_events",
)


def assert_left_aligned(mask: torch.Tensor) -> None:
    if mask.ndim != 2:
        raise ValueError("event_valid_mask must have shape [batch, sequence]")
    valid = mask.bool()
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("valid events must form a left-aligned prefix")


def _validate_inputs(inputs: Mapping[str, torch.Tensor]) -> None:
    if set(inputs) != set(MODEL_INPUT_KEYS):
        raise ValueError("model inputs do not match the closed cache-input schema")
    batch = len(inputs["sex"])
    if inputs["phenotype"].shape != (batch, 78):
        raise ValueError("phenotype must have shape [batch, 78]")
    if inputs["phenotype_missing_mask"].shape != (batch, 78):
        raise ValueError("phenotype_missing_mask must have shape [batch, 78]")
    event_shape = inputs["event_valid_mask"].shape
    if len(event_shape) != 2 or event_shape[0] != batch:
        raise ValueError("event arrays must have shape [batch, sequence]")
    if any(inputs[key].shape != event_shape for key in EVENT_KEYS):
        raise ValueError("all event arrays must share one [batch, sequence] shape")
    assert_left_aligned(inputs["event_valid_mask"])
    invalid = ~inputs["event_valid_mask"].bool()
    for key in EVENT_KEYS[:-1]:
        if torch.count_nonzero(inputs[key][invalid]).item():
            raise ValueError(f"{key} contains non-padding values outside event_valid_mask")


def _clone_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def stable_left_compact_events(
    inputs: Mapping[str, torch.Tensor], keep_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Keep an ordered event subsequence and compact all tuple fields identically."""

    _validate_inputs(inputs)
    valid = inputs["event_valid_mask"].bool()
    keep = keep_mask.bool()
    if keep.shape != valid.shape:
        raise ValueError("event keep mask shape changed")
    if (keep & ~valid).any():
        raise ValueError("cannot retain a padded event")
    result = _clone_inputs(inputs)
    destination = keep.long().cumsum(dim=1) - 1
    rows, source_columns = torch.nonzero(keep, as_tuple=True)
    destination_columns = destination[rows, source_columns]
    for key in EVENT_KEYS:
        source = inputs[key]
        compacted = torch.zeros_like(source)
        compacted[rows, destination_columns] = source[rows, source_columns]
        result[key] = compacted
    assert_left_aligned(result["event_valid_mask"])
    return result


def apply_input_ablation(
    inputs: Mapping[str, torch.Tensor], variant_id: str
) -> dict[str, torch.Tensor]:
    _validate_inputs(inputs)
    result = _clone_inputs(inputs)
    if variant_id == "fig05_01_no_phenotypes_or_missingness":
        result["phenotype"] = torch.zeros_like(inputs["phenotype"])
        result["phenotype_missing_mask"] = torch.zeros_like(
            inputs["phenotype_missing_mask"]
        )
    elif variant_id == "fig05_02_no_hospital_icd":
        valid = inputs["event_valid_mask"].bool()
        result = stable_left_compact_events(
            inputs, valid & inputs["event_type_id"].ne(1)
        )
    elif variant_id == "fig05_03_no_gp_prescriptions":
        valid = inputs["event_valid_mask"].bool()
        result = stable_left_compact_events(
            inputs, valid & inputs["event_type_id"].ne(2)
        )
    elif variant_id == "fig05_04_no_event_text":
        result["event_text_id"] = torch.zeros_like(inputs["event_text_id"])
    elif variant_id == "fig05_05_no_longitudinal_events":
        for key in EVENT_KEYS:
            result[key] = torch.zeros_like(inputs[key])
    elif variant_id.startswith("fig05_"):
        raise ValueError(f"unknown Fig05 input ablation: {variant_id}")
    _validate_inputs(result)
    return result

