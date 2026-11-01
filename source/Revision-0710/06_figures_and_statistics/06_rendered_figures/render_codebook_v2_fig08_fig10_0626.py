from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "Revision-0710").is_dir())
REVISION = ROOT / "Revision-0710"
SOURCE_SCRIPT = HERE / "render_corrected_canonical_fig08_fig12_0626_exact_v2.py"
MAIN_OOF = (
    REVISION
    / "03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1/"
    "02_summary/oof_predictions.parquet"
)
V2_OUTCOMES = REVISION / "01_data_rebuild/09_v2_labels/organ_outcomes_60m.parquet"
V2_LABELS = REVISION / "01_data_rebuild/09_v2_labels/icd_participant_labels_v2.parquet"
FOLLOWUP = (
    REVISION
    / "01_data_rebuild/07_corrected_primary_v1/02_outcomes_60m/"
    "participant_followup_and_evaluation_masks.parquet"
)
DEFAULT_OUTPUT = (
    REVISION / "Final_Fig_0716/Codebook_v2_Huber1/Legacy_0626_Package"
)


def load_source_module() -> Any:
    spec = importlib.util.spec_from_file_location("legacy_exact_v2_runtime", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_compatibility_oof(path: Path) -> None:
    if path.is_file():
        return
    frame = pd.read_parquet(MAIN_OOF)
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("final codebook-v2 OOF contains duplicate rows")
    frame["variant_id"] = "codebook_v2_huber1_epoch47"
    frame["split_role"] = "outer_test"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def v2_outcomes(module: Any) -> pd.DataFrame:
    columns = [
        "participant_id",
        "organ_system",
        "index_date",
        "has_prevalent_target",
        "cohort_state_60m",
        "event",
        "duration_months",
        "subsequent_event",
        "subsequent_duration_months",
        "at_risk",
        "subsequent_at_risk",
    ]
    outcomes = pd.read_parquet(V2_OUTCOMES, columns=columns)
    outcomes["participant_id"] = module.normalize_id(outcomes["participant_id"])
    outcomes["organ_system"] = outcomes["organ_system"].astype(str)
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("v2 outcomes contain duplicate participant-organ rows")
    prevalent = outcomes["has_prevalent_target"].astype(bool)
    selected_event = np.where(
        prevalent, outcomes["subsequent_event"], outcomes["event"]
    ).astype(np.float64)
    selected_duration = np.where(
        prevalent,
        outcomes["subsequent_duration_months"],
        outcomes["duration_months"],
    )
    risk = np.where(
        prevalent, outcomes["subsequent_at_risk"], outcomes["at_risk"]
    ).astype(bool)
    outcomes["event_0626"] = np.where(risk, selected_event, np.nan)
    outcomes["duration_0626"] = np.where(risk, selected_duration, np.nan)
    evaluable_duration = pd.to_numeric(
        outcomes.loc[risk, "duration_0626"], errors="coerce"
    )
    if evaluable_duration.isna().any():
        raise RuntimeError("v2 evaluable mixed outcome has missing duration")
    outcomes.loc[risk, "duration_0626"] = np.clip(
        evaluable_duration, 0.001, 60.0
    )
    outcomes["incident_event_0626"] = (
        (~prevalent) & outcomes["at_risk"].eq(1) & outcomes["event"].eq(1)
    ).astype(np.int8)
    outcomes["recurrent_event_0626"] = (
        prevalent
        & outcomes["subsequent_at_risk"].eq(1)
        & outcomes["subsequent_event"].eq(1)
    ).astype(np.int8)
    return outcomes


def v2_groups(
    module: Any, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, set[str]], pd.DataFrame]:
    followup = pd.read_parquet(
        FOLLOWUP,
        columns=[
            "participant_id",
            "has_prevalent_any_icd",
            "strict_healthy_reference_60m",
            "followup_ascertained_60m",
        ],
    )
    labels = pd.read_parquet(
        V2_LABELS,
        columns=[
            "participant_id",
            *[f"has_prevalent_{organ}" for organ in module.SYSTEMS],
        ],
    )
    for frame in (followup, labels):
        frame["participant_id"] = module.normalize_id(frame["participant_id"])
        if frame["participant_id"].duplicated().any():
            raise RuntimeError("v2 group input contains duplicate participant IDs")
    event_any = (
        outcomes.groupby("participant_id", observed=True)["incident_event_0626"]
        .max()
        .rename("any_future_event_60m")
    )
    participant = (
        followup.set_index("participant_id")
        .join(labels.set_index("participant_id"), how="inner")
        .join(event_any, how="left")
        .reset_index()
    )
    participant["any_future_event_60m"] = participant[
        "any_future_event_60m"
    ].fillna(0).astype(np.int8)
    prevalent_columns = [f"has_prevalent_{organ}" for organ in module.SYSTEMS]
    participant["any_prevalent_organ"] = participant[prevalent_columns].any(axis=1)
    group_rows: list[pd.DataFrame] = []
    group_map: dict[str, set[str]] = {}

    def add_group(name: str, kind: str, organ: str, ids: pd.Series) -> None:
        unique = pd.Series(
            module.normalize_id(ids).dropna().unique(),
            dtype="string",
            name="participant_id",
        )
        group_map[name] = set(unique.astype(str))
        group_rows.append(
            pd.DataFrame(
                {
                    "analysis_group": name,
                    "group_type": kind,
                    "group_organ": organ,
                    "participant_id": unique,
                }
            )
        )

    baseline_icd_free = ~participant["has_prevalent_any_icd"].astype(bool)
    health_display = participant["strict_healthy_reference_60m"].eq(1) | participant[
        "any_future_event_60m"
    ].eq(1)
    add_group(
        "Health",
        "baseline_any_icd_free_strict_or_future_organ_event",
        "",
        participant.loc[baseline_icd_free & health_display, "participant_id"],
    )
    add_group(
        "Other_Disease",
        "baseline_other_icd_without_seven_organ_disease",
        "",
        participant.loc[
            participant["has_prevalent_any_icd"].astype(bool)
            & ~participant["any_prevalent_organ"],
            "participant_id",
        ],
    )
    for organ in module.SYSTEMS:
        target = participant[f"has_prevalent_{organ}"].astype(bool)
        other_columns = [
            f"has_prevalent_{item}" for item in module.SYSTEMS if item != organ
        ]
        other = participant[other_columns].any(axis=1)
        add_group(
            organ,
            "prevalent_target_organ_disease",
            organ,
            participant.loc[target, "participant_id"],
        )
        add_group(
            f"{organ}_Health",
            "target_organ_disease_free_with_other_organ_disease",
            organ,
            participant.loc[~target & other, "participant_id"],
        )
    membership = pd.concat(group_rows, ignore_index=True)
    membership["analysis_group"] = pd.Categorical(
        membership["analysis_group"], categories=module.GROUP_ORDER, ordered=True
    )
    membership = membership.sort_values(
        ["analysis_group", "participant_id"]
    ).reset_index(drop=True)
    if set(membership["analysis_group"].astype(str)) != set(module.GROUP_ORDER):
        raise RuntimeError("v2 membership lost a 0626 display group")
    return membership, group_map, participant


def configure(module: Any, output_root: Path) -> None:
    compatibility_oof = output_root / "00_inputs/oof_predictions_compat.parquet"
    build_compatibility_oof(compatibility_oof)
    module.OOF = compatibility_oof
    module.OUTCOMES = V2_OUTCOMES
    module.FOLLOWUP = FOLLOWUP
    module.ICD_SUMMARY = V2_LABELS
    module.OUTPUT_ROOT = output_root
    module.CONTRACT_DIR = output_root / "00_group_contract"
    module.LEGACY_TABLE = module.CONTRACT_DIR / "legacy_input_table_0626_exact.csv.gz"
    module.NETWORK_WIDE = module.CONTRACT_DIR / "canonical_age_gap_wide_0626_exact.csv.gz"
    module.MEMBERSHIP = (
        module.CONTRACT_DIR / "revised_incident_recurrence_group_membership_long.csv.gz"
    )
    module.load_outcomes = lambda: v2_outcomes(module)
    module.build_groups = lambda outcomes: v2_groups(module, outcomes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--steps",
        default="prepare,fig8,fig8_target_only,fig8_single_gap_full,fig9,fig10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_source_module()
    configure(module, args.output_root.resolve())
    sys.argv = [str(SOURCE_SCRIPT), "--steps", args.steps]
    module.main()
    counts = pd.read_csv(module.CONTRACT_DIR / "group_membership_counts.csv")
    expected = {
        "Health": 134078,
        "Other_Disease": 162573,
        "cardiovascular": 55846,
        "pulmonary": 18193,
        "musculoskeletal": 60135,
        "immune": 5787,
        "renal": 5360,
        "hepatic": 2272,
        "metabolic": 32950,
        "cardiovascular_Health": 62722,
        "pulmonary_Health": 100375,
        "musculoskeletal_Health": 58433,
        "immune_Health": 112781,
        "renal_Health": 113208,
        "hepatic_Health": 116296,
        "metabolic_Health": 85618,
    }
    observed = dict(zip(counts["analysis_group"], counts["n"]))
    if any(int(observed.get(key, -1)) != value for key, value in expected.items()):
        raise RuntimeError("rendered v2 group counts differ from the rebuilt codebook audit")
    (module.CONTRACT_DIR / "GROUP_DEFINITION_CONTRACT.md").write_text(
        """# Codebook-v2 Fig8-Fig10 grouping

- `Health`: baseline any-ICD-free participants who are either in strict Health or have a future included-organ event within 60 months.
- `{organ}`: participants with prevalent disease in that organ.
- `{organ}_Health`: participants free of prevalent disease in the target organ and with prevalent disease in at least one of the other six organs.
- `Other_Disease`: baseline ICD present but no prevalent disease in any of the seven organs.
- For each target-organ panel, disease-free participants use the incident endpoint and participants with prevalent target-organ disease use the subsequent-event endpoint.
- Groups may overlap, matching the 0626 display structure. The full target-organ disease-free incident risk set is reserved for full-risk Delta C and is not relabeled as `{organ}_Health`.
""",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "group_counts": observed}, indent=2))


if __name__ == "__main__":
    main()
