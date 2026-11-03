from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import torch

SEED = 7210

VARIANTS = ("mimic_common53", "nhanes_common_eventless")

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
