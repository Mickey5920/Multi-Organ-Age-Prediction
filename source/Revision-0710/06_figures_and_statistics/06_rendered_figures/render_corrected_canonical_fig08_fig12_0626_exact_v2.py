#!/usr/bin/env python
"""Rebuild Revision-0710 Fig8-Fig12 with the exact 0626 statistical contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / "Revision-0710").exists())
REVISION = REPO_ROOT / "Revision-0710"

OOF = (
    REVISION
    / "03_ehr_age/05_corrected_primary_v1/01_canonical_5fold/03_predictions/oof_predictions.parquet"
)
OUTCOMES = REVISION / "01_data_rebuild/07_corrected_primary_v1/02_outcomes_60m/organ_outcomes_60m.parquet"
FOLLOWUP = (
    REVISION
    / "01_data_rebuild/07_corrected_primary_v1/02_outcomes_60m/participant_followup_and_evaluation_masks.parquet"
)
ICD_SUMMARY = (
    REVISION
    / "01_data_rebuild/07_corrected_primary_v1/01_index_and_events/icd_participant_summary.parquet"
)
CONTEXT = REVISION / "05_downstream/06_metric_v2/01_contextual_covariates/participant_context_covariates.parquet"
TEMPLATE_SCRIPT = (
    REPO_ROOT
    / "Revision-0626/18_final_ehr_age_downstream_gp_medication_20260707/"
    / "source_scripts_snapshot/render_legacy_style_figures.py"
)
TARGET_GAP_TEMPLATE_SCRIPT = (
    REPO_ROOT / "Revision-0626/Final_Figure_Data_0708/scripts/render_fig8_target_gap_only.py"
)

OUTPUT_ROOT = (
    REVISION
    / "06_figures_and_statistics/06_rendered_figures/corrected_canonical_0626_exact_v2"
)
CONTRACT_DIR = OUTPUT_ROOT / "00_group_contract"
LEGACY_TABLE = CONTRACT_DIR / "legacy_input_table_0626_exact.csv.gz"
NETWORK_WIDE = CONTRACT_DIR / "canonical_age_gap_wide_0626_exact.csv.gz"
MEMBERSHIP = CONTRACT_DIR / "revised_incident_recurrence_group_membership_long.csv.gz"

SYSTEMS = [
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
]
GROUP_ORDER = [
    "Health",
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
    "cardiovascular_Health",
    "pulmonary_Health",
    "musculoskeletal_Health",
    "immune_Health",
    "renal_Health",
    "hepatic_Health",
    "metabolic_Health",
    "Other_Disease",
]
INCIDENT_RISK_STATES = {
    "baseline_free_future_target_event",
    "strict_disease_free_through_followup",
    "target_disease_free_no_included_disease",
    "target_disease_free_other_system_disease",
}
CONTEXT_COLUMNS = [
    "participant_id",
    "sex_male",
    "smoking_current",
    "education_higher",
    "townsend",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "sleep_duration",
    "sensitivity_pm25_2010",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def load_template() -> Any:
    spec = importlib.util.spec_from_file_location("legacy_0626_fig08_fig12_exact", TEMPLATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 0626 template: {TEMPLATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LEGACY_OUT = OUTPUT_ROOT
    module.LEGACY_INPUT = CONTRACT_DIR
    module.CURRENT_WIDE = NETWORK_WIDE

    # The 0626 renderer builds node order from a set, so Python hash randomization
    # can move the longest label to the canvas edge. Freeze the reference 0626
    # circular order while leaving edge selection and all coefficients unchanged.
    original_circular_layout = module.nx.circular_layout
    reference_node_order = ["Pulm.", "Immune", "Musculo.", "Hepatic", "Cardio.", "Metabolic", "Renal"]

    def reference_circular_layout(graph: Any, *args: Any, **kwargs: Any) -> dict[Any, Any]:
        present = list(graph.nodes)
        ordered = [node for node in reference_node_order if node in present]
        ordered.extend(node for node in present if node not in ordered)
        layout_graph = module.nx.Graph()
        layout_graph.add_nodes_from(ordered)
        return original_circular_layout(layout_graph, *args, **kwargs)

    module.nx.circular_layout = reference_circular_layout
    return module


def load_target_gap_template() -> Any:
    spec = importlib.util.spec_from_file_location("legacy_0626_fig08_target_gap_only", TARGET_GAP_TEMPLATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 0626 target-gap-only template: {TARGET_GAP_TEMPLATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig8 = OUTPUT_ROOT / "Fig-8-Health-Group-HR-Forest-Bootstrap"
    module.OUT_CSV_DIR = fig8 / "Cox_results_target_gap_only"
    module.OUT_PLOT_DIR = fig8 / "Forest_plots_target_gap_only"
    return module


def load_single_gap_full_template() -> Any:
    spec = importlib.util.spec_from_file_location("legacy_0626_fig08_single_gap_full", TARGET_GAP_TEMPLATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 0626 single-gap-full template: {TARGET_GAP_TEMPLATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig8 = OUTPUT_ROOT / "Fig-8-Health-Group-HR-Forest-Bootstrap"
    module.OUT_CSV_DIR = fig8 / "Cox_results_single_gap_full"
    module.OUT_PLOT_DIR = fig8 / "Forest_plots_single_gap_full"
    return module


def require_inputs() -> None:
    missing = [
        path
        for path in (OOF, OUTCOMES, FOLLOWUP, ICD_SUMMARY, CONTEXT, TEMPLATE_SCRIPT, TARGET_GAP_TEMPLATE_SCRIPT)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


def load_oof_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "participant_id",
        "variant_id",
        "outer_fold",
        "split_role",
        "organ_system",
        "actual_age",
        "raw_age_gap",
        "corrected_age_gap",
        "sex",
    ]
    oof = pd.read_parquet(OOF, columns=columns)
    oof["participant_id"] = normalize_id(oof["participant_id"])
    oof["organ_system"] = oof["organ_system"].astype(str)
    if oof.duplicated(["participant_id", "organ_system"]).any():
        raise AssertionError("Canonical OOF has duplicate participant-organ rows")
    if set(oof["organ_system"].unique()) != set(SYSTEMS):
        raise AssertionError("Canonical OOF organ set differs from the 0626 seven-organ contract")
    if not oof.groupby("participant_id", observed=True).size().eq(7).all():
        raise AssertionError("Canonical OOF is not complete seven-organ OOF")
    if not oof["split_role"].eq("outer_test").all():
        raise AssertionError("Canonical OOF contains rows outside outer_test")

    participant = (
        oof.sort_values(["participant_id", "organ_system"])
        .drop_duplicates("participant_id")[["participant_id", "actual_age", "sex", "outer_fold"]]
        .rename(columns={"actual_age": "Actual_Age", "sex": "Sex"})
    )
    raw = oof.pivot(index="participant_id", columns="organ_system", values="raw_age_gap")
    raw = raw.rename(columns={organ: f"{organ}_Age_Gap" for organ in SYSTEMS}).reset_index()
    corrected = oof.pivot(index="participant_id", columns="organ_system", values="corrected_age_gap")
    corrected = corrected.rename(columns={organ: f"Corrected_{organ}_Age_Gap" for organ in SYSTEMS}).reset_index()
    analysis = participant.merge(raw, on="participant_id", validate="one_to_one").merge(
        corrected, on="participant_id", validate="one_to_one"
    )

    network = corrected.rename(
        columns={f"Corrected_{organ}_Age_Gap": f"{organ}_linear_corrected_age_gap" for organ in SYSTEMS}
    )
    return analysis, network


def load_context() -> pd.DataFrame:
    context = pd.read_parquet(CONTEXT, columns=CONTEXT_COLUMNS)
    context["participant_id"] = normalize_id(context["participant_id"])
    if context["participant_id"].duplicated().any():
        raise AssertionError("Context table has duplicate participant IDs")
    out = context.rename(
        columns={
            "smoking_current": "Current tobacco smoking_Representation",
            "education_higher": "Qualifications_Representation",
            "townsend": "Townsend deprivation index at recruitment",
            "sleep_duration": "Sleep Duration",
            "sensitivity_pm25_2010": "Particulate matter air pollution (pm2.5); 2010",
        }
    ).copy()
    out["Current_employment_status"] = np.select(
        [context["employment_unemployed"].eq(1), context["employment_employed"].eq(1)],
        [1.0, 2.0],
        default=0.0,
    )
    keep = [
        "participant_id",
        "Current tobacco smoking_Representation",
        "Current_employment_status",
        "Particulate matter air pollution (pm2.5); 2010",
        "Qualifications_Representation",
        "Sleep Duration",
        "Townsend deprivation index at recruitment",
    ]
    return out[keep]


def load_outcomes() -> pd.DataFrame:
    columns = [
        "participant_id",
        "organ_system",
        "index_date",
        "first_incident_target_date",
        "has_prevalent_target",
        "cohort_state_60m",
        "event",
    ]
    outcomes = pd.read_parquet(OUTCOMES, columns=columns)
    outcomes["participant_id"] = normalize_id(outcomes["participant_id"])
    outcomes["organ_system"] = outcomes["organ_system"].astype(str)
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise AssertionError("Corrected-primary outcomes have duplicate participant-organ rows")
    if not outcomes.groupby("participant_id", observed=True).size().eq(7).all():
        raise AssertionError("Corrected-primary outcomes are not complete seven-organ rows")

    months = (outcomes["first_incident_target_date"] - outcomes["index_date"]).dt.days / (365.25 / 12.0)
    within_60 = months.notna() & months.ge(0.0) & months.le(60.0)
    prevalent = pd.to_numeric(outcomes["has_prevalent_target"], errors="coerce").fillna(0).eq(1)
    incident_event = (~prevalent) & within_60
    recurrent_event = outcomes["cohort_state_60m"].eq("current_target_disease") & within_60
    outcomes["event_0626"] = (incident_event | recurrent_event).astype(np.int8)
    outcomes["duration_0626"] = np.where(outcomes["event_0626"].eq(1), months, 60.0)
    outcomes["duration_0626"] = np.clip(
        pd.to_numeric(outcomes["duration_0626"], errors="coerce").fillna(60.0), 0.001, 60.0
    )
    outcomes["incident_event_0626"] = incident_event.astype(np.int8)
    outcomes["recurrent_event_0626"] = recurrent_event.astype(np.int8)
    return outcomes


def build_groups(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]], pd.DataFrame]:
    followup = pd.read_parquet(
        FOLLOWUP,
        columns=[
            "participant_id",
            "has_prevalent_any_icd",
            "strict_healthy_reference_60m",
            "followup_ascertained_60m",
        ],
    )
    icd = pd.read_parquet(
        ICD_SUMMARY,
        columns=["participant_id", "has_prevalent_any_icd", *[f"has_prevalent_{organ}" for organ in SYSTEMS]],
    )
    for frame in (followup, icd):
        frame["participant_id"] = normalize_id(frame["participant_id"])
    if followup["participant_id"].duplicated().any() or icd["participant_id"].duplicated().any():
        raise AssertionError("Participant contract source contains duplicate IDs")

    event_any = outcomes.groupby("participant_id", observed=True)["incident_event_0626"].max().rename("any_future_event_60m")
    prevalent_any_organ = (
        outcomes.groupby("participant_id", observed=True)["has_prevalent_target"].max().rename("any_prevalent_organ")
    )
    participant = followup.set_index("participant_id").join(event_any).join(prevalent_any_organ).reset_index()
    icd_check = icd.set_index("participant_id")["has_prevalent_any_icd"].reindex(participant["participant_id"])
    if not participant["has_prevalent_any_icd"].astype(bool).to_numpy().tolist() == icd_check.astype(bool).to_numpy().tolist():
        raise AssertionError("Follow-up and ICD summaries disagree on baseline ICD presence")

    group_rows: list[pd.DataFrame] = []
    group_map: dict[str, set[str]] = {}

    def add_group(name: str, group_type: str, group_organ: str, ids: pd.Series) -> None:
        unique = pd.Series(normalize_id(ids).dropna().unique(), dtype="string", name="participant_id")
        group_map[name] = set(unique.astype(str))
        group_rows.append(
            pd.DataFrame(
                {
                    "analysis_group": name,
                    "group_type": group_type,
                    "group_organ": group_organ,
                    "participant_id": unique,
                }
            )
        )

    no_prebaseline_icd = ~participant["has_prevalent_any_icd"].astype(bool)
    health_state = participant["strict_healthy_reference_60m"].eq(1) | participant["any_future_event_60m"].eq(1)
    add_group(
        "Health",
        "baseline_no_prebaseline_icd_included_risk",
        "",
        participant.loc[no_prebaseline_icd & health_state, "participant_id"],
    )
    add_group(
        "Other_Disease",
        "baseline_nonincluded_icd_inclusive_future_event",
        "",
        participant.loc[
            participant["has_prevalent_any_icd"].astype(bool) & participant["any_prevalent_organ"].eq(0),
            "participant_id",
        ],
    )
    for organ in SYSTEMS:
        organ_frame = outcomes.loc[outcomes["organ_system"].eq(organ)]
        add_group(
            organ,
            "prevalent_current_target_disease_recurrence_risk",
            organ,
            organ_frame.loc[organ_frame["cohort_state_60m"].eq("current_target_disease"), "participant_id"],
        )
        add_group(
            f"{organ}_Health",
            "baseline_target_disease_free_incident_risk_set",
            organ,
            organ_frame.loc[organ_frame["cohort_state_60m"].isin(INCIDENT_RISK_STATES), "participant_id"],
        )

    membership = pd.concat(group_rows, ignore_index=True)
    membership["analysis_group"] = pd.Categorical(
        membership["analysis_group"], categories=GROUP_ORDER, ordered=True
    )
    membership = membership.sort_values(["analysis_group", "participant_id"]).reset_index(drop=True)
    if set(membership["analysis_group"].astype(str).unique()) != set(GROUP_ORDER):
        raise AssertionError("Exact 0626 membership does not contain all 16 groups")
    if membership.duplicated(["analysis_group", "participant_id"]).any():
        raise AssertionError("Exact 0626 membership contains duplicate group-participant rows")
    return membership, group_map, participant


def build_analysis_table(
    age: pd.DataFrame,
    context: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    time_wide = outcomes.pivot(index="participant_id", columns="organ_system", values="duration_0626")
    time_wide = time_wide.rename(columns={organ: f"{organ}_Time_to_First_Event" for organ in SYSTEMS}).reset_index()
    event_wide = outcomes.pivot(index="participant_id", columns="organ_system", values="event_0626")
    event_wide = event_wide.rename(columns={organ: f"{organ}_Event_Indicator" for organ in SYSTEMS}).reset_index()
    frame = age.merge(context, on="participant_id", how="inner", validate="one_to_one")
    frame = frame.merge(time_wide, on="participant_id", how="inner", validate="one_to_one")
    frame = frame.merge(event_wide, on="participant_id", how="inner", validate="one_to_one")
    frame["Participant_ID"] = frame["participant_id"]

    ordered = ["Participant_ID", "Sex", "Actual_Age"]
    for organ in ["cardiovascular", "pulmonary", "musculoskeletal", "immune", "renal", "hepatic", "metabolic"]:
        ordered.extend(
            [
                f"{organ}_Time_to_First_Event",
                f"{organ}_Event_Indicator",
                f"{organ}_Age_Gap",
                f"Corrected_{organ}_Age_Gap",
            ]
        )
    ordered.extend(
        [
            "Current tobacco smoking_Representation",
            "Current_employment_status",
            "Particulate matter air pollution (pm2.5); 2010",
            "Qualifications_Representation",
            "Sleep Duration",
            "Townsend deprivation index at recruitment",
        ]
    )
    missing = [column for column in ordered if column not in frame.columns]
    if missing:
        raise AssertionError(f"Legacy analysis table is missing columns: {missing}")
    return frame[ordered]


def build_group_event_audit(
    analysis: pd.DataFrame,
    group_map: dict[str, set[str]],
) -> pd.DataFrame:
    indexed = analysis.set_index("Participant_ID", drop=False)
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        ids = indexed.index.intersection(pd.Index(list(group_map[group])))
        sub = indexed.loc[ids]
        for target in SYSTEMS:
            events = pd.to_numeric(sub[f"{target}_Event_Indicator"], errors="coerce").fillna(0)
            rows.append(
                {
                    "analysis_group": group,
                    "target_organ": target,
                    "n_group": int(len(sub)),
                    "n_events": int(events.sum()),
                    "event_rate": float(events.mean()) if len(events) else np.nan,
                    "same_target_prevalent_cell": bool(group == target),
                }
            )
    return pd.DataFrame(rows)


def run_fig8_target_gap_only(analysis: pd.DataFrame, group_map: dict[str, set[str]]) -> None:
    """Run the 0626 primary Fig8 contract: clinical covariates plus target gap."""
    target_template = load_target_gap_template()
    target_template.OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    target_template.OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    frame = analysis.copy()
    frame = frame[pd.to_numeric(frame["Current_employment_status"], errors="coerce").ne(1)].copy()
    frame["Smoked"] = (
        pd.to_numeric(frame["Current tobacco smoking_Representation"], errors="coerce").eq(1).astype(int)
    )
    frame["Education_Higher"] = (
        pd.to_numeric(frame["Qualifications_Representation"], errors="coerce").eq(1).astype(int)
    )
    frame["Employed"] = pd.to_numeric(frame["Current_employment_status"], errors="coerce").eq(2).astype(int)
    frame["Sex"] = pd.to_numeric(frame["Sex"], errors="coerce").eq(1).astype(int)

    group_labels = {
        "Health": "Health group",
        "Other_Disease": "Other disease group",
        **{organ: f"{target_template.ORGAN_LONG[organ]} disease group" for organ in SYSTEMS},
        **{f"{organ}_Health": f"{target_template.ORGAN_LONG[organ]} health group" for organ in SYSTEMS},
    }
    exact_columns = [
        "model_source",
        "model_spec",
        "System_Group",
        "Group_Label",
        "Organ",
        "Organ_Label",
        "Target_Gap",
        "N_Raw_After_Employment_Filter",
        "N_Events_Raw",
        "N_Samples",
        "N_Events",
        "status",
        "covariate",
        "variable_name",
        "variable_type",
        "coef",
        "HR",
        "HR_ci95_low",
        "HR_ci95_high",
        "se",
        "z",
        "p",
        "minus_log2_p",
        "mean_before_scaling",
        "sd_before_scaling",
    ]

    all_results: list[pd.DataFrame] = []
    for organ in SYSTEMS:
        target_gap = f"Corrected_{organ}_Age_Gap"
        for group in ("Health", organ, f"{organ}_Health", "Other_Disease"):
            sub = frame.loc[frame["Participant_ID"].isin(group_map[group])].copy()
            result = target_template.fit_single_gap_model(
                sub,
                organ,
                group,
                group_labels[group],
                target_gap,
            )
            result["model_source"] = "Revision-0710 canonical 5-fold OOF corrected EHR-AGE"
            result["model_spec"] = "clinical_covariates_plus_target_gap_only"
            result = result[exact_columns]
            result.to_csv(
                target_template.OUT_CSV_DIR / f"{group}_{organ}_cox_results.csv",
                index=False,
            )
            target_template.plot_forest(result, group, organ)
            all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)
    fig8 = OUTPUT_ROOT / "Fig-8-Health-Group-HR-Forest-Bootstrap"
    combined.to_csv(fig8 / "Combined_Target_Gap_Only_Cox_Results.csv", index=False)
    target_rows = combined.loc[combined["covariate"].eq(combined["Target_Gap"])].copy()
    target_rows = target_rows.rename(
        columns={
            "HR": "HR_per_SD",
            "mean_before_scaling": "gap_mean_before_scaling",
            "sd_before_scaling": "gap_sd_before_scaling",
        }
    )
    target_rows.to_csv(fig8 / "target_gap_only_group_cox_latest_ehr_age.csv", index=False)
    target_rows.to_csv(fig8 / "target_gap_only_group_cox_best_model.csv", index=False)

    role = {
        "schema_version": "revision0710.fig08.analysis_role.v1",
        "created_utc": utc_now(),
        "primary": {
            "model_spec": "clinical_covariates_plus_target_gap_only",
            "groups_per_organ": ["Health", "target-organ disease", "target-organ Health", "Other_Disease"],
            "panel_count": 28,
            "combined_csv": "Combined_Target_Gap_Only_Cox_Results.csv",
            "cox_results": "Cox_results_target_gap_only",
            "forest_plots": "Forest_plots_target_gap_only",
        },
        "sensitivity": {
            "model_spec": "clinical_covariates_plus_all_seven_age_gaps",
            "panel_count": 112,
            "combined_csv": "Combined_Cox_Bootstrap_Results.csv",
            "cox_results": "Cox_results",
            "forest_plots": "Forest_plots",
        },
        "reason": (
            "Seven corrected organ age gaps are severely collinear; primary HRs therefore use "
            "the existing 0626 target-gap-only contract."
        ),
    }
    (fig8 / "FIG8_ANALYSIS_ROLE.json").write_text(json.dumps(role, indent=2), encoding="utf-8")


def run_fig8_single_gap_full(analysis: pd.DataFrame, group_map: dict[str, set[str]]) -> None:
    """Render 14-row panels from seven separately fitted clinical-plus-gap models."""
    single_template = load_single_gap_full_template()
    single_template.OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    single_template.OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    frame = analysis.copy()
    frame = frame[pd.to_numeric(frame["Current_employment_status"], errors="coerce").ne(1)].copy()
    frame["Smoked"] = (
        pd.to_numeric(frame["Current tobacco smoking_Representation"], errors="coerce").eq(1).astype(int)
    )
    frame["Education_Higher"] = (
        pd.to_numeric(frame["Qualifications_Representation"], errors="coerce").eq(1).astype(int)
    )
    frame["Employed"] = pd.to_numeric(frame["Current_employment_status"], errors="coerce").eq(2).astype(int)
    frame["Sex"] = pd.to_numeric(frame["Sex"], errors="coerce").eq(1).astype(int)

    group_labels = {
        "Health": "Health group",
        "Other_Disease": "Other disease group",
        **{organ: f"{single_template.ORGAN_LONG[organ]} disease group" for organ in SYSTEMS},
        **{f"{organ}_Health": f"{single_template.ORGAN_LONG[organ]} health group" for organ in SYSTEMS},
    }
    exact_columns = [
        "model_source",
        "model_spec",
        "System_Group",
        "Group_Label",
        "Organ",
        "Organ_Label",
        "Target_Gap",
        "Fitted_Gap",
        "N_Raw_After_Employment_Filter",
        "N_Events_Raw",
        "N_Samples",
        "N_Events",
        "status",
        "covariate",
        "variable_name",
        "variable_type",
        "coef",
        "HR",
        "HR_ci95_low",
        "HR_ci95_high",
        "se",
        "z",
        "p",
        "minus_log2_p",
        "mean_before_scaling",
        "sd_before_scaling",
        "display_order",
        "is_target_gap_row",
        "Panel_N_Samples",
        "Panel_N_Events",
    ]

    all_results: list[pd.DataFrame] = []
    for target_organ in SYSTEMS:
        target_gap = f"Corrected_{target_organ}_Age_Gap"
        other_gaps = [gap for gap in single_template.AGE_GAP_COLUMNS if gap != target_gap]
        for group in GROUP_ORDER:
            sub = frame.loc[frame["Participant_ID"].isin(group_map[group])].copy()
            target_model = single_template.fit_single_gap_model(
                sub,
                target_organ,
                group,
                group_labels[group],
                target_gap,
            )
            target_model["model_source"] = "Revision-0710 canonical 5-fold OOF corrected EHR-AGE"
            target_model["model_spec"] = "clinical_covariates_plus_one_age_gap_per_age_gap_row"
            target_ok = target_model.loc[target_model["status"].eq("ok")].copy()
            if target_ok.empty:
                raise RuntimeError(f"Target-gap model failed for {group}/{target_organ}")

            target_clinical = target_ok.loc[
                target_ok["covariate"].isin(single_template.CLINICAL_COVARIATES)
            ].copy()
            target_gap_row = target_ok.loc[target_ok["covariate"].eq(target_gap)].copy()
            panel_parts = [target_clinical, target_gap_row]
            for fitted_gap in other_gaps:
                gap_model = single_template.fit_single_gap_model(
                    sub,
                    target_organ,
                    group,
                    group_labels[group],
                    fitted_gap,
                )
                gap_model["model_source"] = "Revision-0710 canonical 5-fold OOF corrected EHR-AGE"
                gap_model["model_spec"] = "clinical_covariates_plus_one_age_gap_per_age_gap_row"
                gap_row = gap_model.loc[
                    gap_model["status"].eq("ok") & gap_model["covariate"].eq(fitted_gap)
                ].copy()
                if gap_row.empty:
                    raise RuntimeError(f"Single-gap model failed for {group}/{target_organ}/{fitted_gap}")
                panel_parts.append(gap_row)

            panel = pd.concat(panel_parts, ignore_index=True)
            display_covariates = [*single_template.CLINICAL_COVARIATES, target_gap, *other_gaps]
            display_order = {covariate: index for index, covariate in enumerate(display_covariates)}
            panel["display_order"] = panel["covariate"].map(display_order)
            panel["is_target_gap_row"] = panel["covariate"].eq(target_gap)
            panel["Panel_N_Samples"] = int(target_gap_row["N_Samples"].iloc[0])
            panel["Panel_N_Events"] = float(target_gap_row["N_Events"].iloc[0])
            panel = panel.sort_values("display_order", kind="stable").reset_index(drop=True)
            panel = panel[exact_columns]
            panel.to_csv(
                single_template.OUT_CSV_DIR / f"{group}_{target_organ}_cox_results.csv",
                index=False,
            )
            single_template.plot_forest(panel, group, target_organ)
            all_results.append(panel)

    combined = pd.concat(all_results, ignore_index=True)
    fig8 = OUTPUT_ROOT / "Fig-8-Health-Group-HR-Forest-Bootstrap"
    combined.to_csv(fig8 / "Combined_Single_Gap_Full_Cox_Results.csv", index=False)
    age_gap_rows = combined.loc[
        combined["covariate"].str.startswith("Corrected_", na=False) & combined["status"].eq("ok")
    ].copy()
    age_gap_rows = age_gap_rows.rename(
        columns={
            "HR": "HR_per_SD",
            "mean_before_scaling": "gap_mean_before_scaling",
            "sd_before_scaling": "gap_sd_before_scaling",
        }
    )
    age_gap_rows.to_csv(fig8 / "single_gap_full_age_gap_rows_latest_ehr_age.csv", index=False)
    age_gap_rows.to_csv(fig8 / "single_gap_full_age_gap_rows_best_model.csv", index=False)

    role = {
        "schema_version": "revision0710.fig08.analysis_role.v2",
        "created_utc": utc_now(),
        "primary": {
            "model_spec": "clinical_covariates_plus_one_age_gap_per_age_gap_row",
            "panel_count": 112,
            "rows_per_panel": 14,
            "fit_contract": (
                "The seven clinical rows come from the clinical-plus-target-gap Cox model. "
                "Each of the seven age-gap rows comes from a separate clinical-plus-one-gap Cox model "
                "for the panel's target-organ event."
            ),
            "combined_csv": "Combined_Single_Gap_Full_Cox_Results.csv",
            "cox_results": "Cox_results_single_gap_full",
            "forest_plots": "Forest_plots_single_gap_full",
        },
        "compact_secondary": {
            "model_spec": "clinical_covariates_plus_target_gap_only",
            "panel_count": 28,
            "combined_csv": "Combined_Target_Gap_Only_Cox_Results.csv",
            "cox_results": "Cox_results_target_gap_only",
            "forest_plots": "Forest_plots_target_gap_only",
        },
        "collinearity_sensitivity": {
            "model_spec": "clinical_covariates_plus_all_seven_age_gaps_simultaneously",
            "panel_count": 112,
            "combined_csv": "Combined_Cox_Bootstrap_Results.csv",
            "cox_results": "Cox_results",
            "forest_plots": "Forest_plots",
        },
    }
    (fig8 / "FIG8_ANALYSIS_ROLE.json").write_text(json.dumps(role, indent=2), encoding="utf-8")


def prepare() -> tuple[pd.DataFrame, dict[str, set[str]], pd.DataFrame]:
    require_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    age, network = load_oof_tables()
    context = load_context()
    outcomes = load_outcomes()
    membership, group_map, participant = build_groups(outcomes)
    analysis = build_analysis_table(age, context, outcomes)
    audit = build_group_event_audit(analysis, group_map)

    health = audit.loc[audit["analysis_group"].eq("Health")]
    diagonal = audit.loc[audit["same_target_prevalent_cell"]]
    if len(audit) != 112 or len(health) != 7 or len(diagonal) != 7:
        raise AssertionError("Group-target audit is not the expected 16 x 7 contract")
    if health["n_events"].le(0).any() or diagonal["n_events"].le(0).any():
        raise AssertionError("Health or prevalent same-target cells have zero events under the 0626 contract")

    membership.to_csv(MEMBERSHIP, index=False, compression="gzip")
    analysis.to_csv(LEGACY_TABLE, index=False, compression="gzip")
    network.to_csv(NETWORK_WIDE, index=False, compression="gzip")
    audit.to_csv(CONTRACT_DIR / "group_target_event_counts.csv", index=False)
    membership.groupby(["analysis_group", "group_type", "group_organ"], observed=True).size().rename("n").reset_index().to_csv(
        CONTRACT_DIR / "group_membership_counts.csv", index=False
    )

    contract = """# Revision-0710 Fig8-Fig12 exact 0626 contract

- Health: no baseline ICD and either strict disease-free reference or a future included-organ event within 60 months.
- Other_Disease: baseline ICD present, but no baseline disease in any of the seven included organs.
- {organ}: baseline current target-organ disease; same-target outcomes are post-baseline/recurrent ICD events.
- {organ}_Health: the full 0626 target-organ incident risk states.
- Event: first post-baseline same-organ ICD within 60 months; non-events are assigned 60 months.
- Groups are not forced to be mutually exclusive.
"""
    (CONTRACT_DIR / "GROUP_DEFINITION_CONTRACT.md").write_text(contract, encoding="utf-8")
    manifest = {
        "schema_version": "revision0710.fig08_fig12.corrected_canonical_0626_exact_v2",
        "created_utc": utc_now(),
        "status": "INPUTS_READY",
        "policy": "Revision-0626 statistical and visual code; Revision-0710 corrected-primary outcomes and canonical OOF values",
        "inputs": {
            str(path.relative_to(REPO_ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (OOF, OUTCOMES, FOLLOWUP, ICD_SUMMARY, CONTEXT, TEMPLATE_SCRIPT, TARGET_GAP_TEMPLATE_SCRIPT)
        },
        "rows": {
            "participants": int(analysis["Participant_ID"].nunique()),
            "membership_rows": int(len(membership)),
            "group_target_cells": int(len(audit)),
            "health_n": int(len(group_map["Health"])),
            "other_disease_n": int(len(group_map["Other_Disease"])),
        },
        "oof_variant_ids": sorted(pd.read_parquet(OOF, columns=["variant_id"])["variant_id"].astype(str).unique()),
        "group_order": GROUP_ORDER,
        "organ_order": SYSTEMS,
    }
    (CONTRACT_DIR / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return analysis, group_map, audit


def load_prepared() -> tuple[pd.DataFrame, dict[str, set[str]], pd.DataFrame]:
    required = [LEGACY_TABLE, NETWORK_WIDE, MEMBERSHIP, CONTRACT_DIR / "group_target_event_counts.csv"]
    if not all(path.exists() for path in required):
        return prepare()
    analysis = pd.read_csv(LEGACY_TABLE, dtype={"Participant_ID": str}, low_memory=False)
    analysis["Participant_ID"] = normalize_id(analysis["Participant_ID"])
    membership = pd.read_csv(MEMBERSHIP, dtype={"participant_id": str}, low_memory=False)
    membership["participant_id"] = normalize_id(membership["participant_id"])
    group_map = {
        str(group): set(sub["participant_id"].astype(str))
        for group, sub in membership.groupby("analysis_group", observed=True, sort=False)
    }
    audit = pd.read_csv(CONTRACT_DIR / "group_target_event_counts.csv")
    return analysis, group_map, audit


def image_qa(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        return {
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "width": int(rgb.width),
            "height": int(rgb.height),
            "channel_std_mean": float(np.mean(stat.stddev)),
            "nonblank": bool(np.mean(stat.stddev) > 1.0 and rgb.width > 100 and rgb.height > 100),
        }


def validate() -> dict[str, Any]:
    fig8 = OUTPUT_ROOT / "Fig-8-Health-Group-HR-Forest-Bootstrap"
    fig9 = OUTPUT_ROOT / "Fig-9-KM-Curve-Bootstrap"
    fig10 = OUTPUT_ROOT / "Fig-10-CumRisk_Heatmap"
    fig12 = OUTPUT_ROOT / "Fig-12-Path_Mean_Heatmap"

    cox_files = sorted((fig8 / "Cox_results").glob("*_cox_results.csv"))
    forest_png = sorted((fig8 / "Forest_plots").glob("*_forest_plot.png"))
    forest_svg = sorted((fig8 / "Forest_plots").glob("*_forest_plot.svg"))
    target_cox_files = sorted((fig8 / "Cox_results_target_gap_only").glob("*_cox_results.csv"))
    target_forest_png = sorted((fig8 / "Forest_plots_target_gap_only").glob("*_forest_plot.png"))
    target_forest_svg = sorted((fig8 / "Forest_plots_target_gap_only").glob("*_forest_plot.svg"))
    single_cox_files = sorted((fig8 / "Cox_results_single_gap_full").glob("*_cox_results.csv"))
    single_forest_png = sorted((fig8 / "Forest_plots_single_gap_full").glob("*_forest_plot.png"))
    single_forest_svg = sorted((fig8 / "Forest_plots_single_gap_full").glob("*_forest_plot.svg"))
    km_files = sorted((fig9 / "KM-Curve-Bootstrap-Data").glob("*_Group/*/Summary_*.csv"))
    km_png = sorted((fig9 / "Panel_Plots").glob("*_Group/*_panel_2x7_from_csv.png"))
    km_svg = sorted((fig9 / "Panel_Plots").glob("*_Group/*_panel_2x7_from_csv.svg"))

    combined_cox = pd.read_csv(fig8 / "Combined_Cox_Bootstrap_Results.csv")
    target_combined = pd.read_csv(fig8 / "Combined_Target_Gap_Only_Cox_Results.csv")
    single_combined = pd.read_csv(fig8 / "Combined_Single_Gap_Full_Cox_Results.csv")
    target_gap_rows = target_combined.loc[
        target_combined["covariate"].str.startswith("Corrected_", na=False)
    ]
    single_gap_rows = single_combined.loc[
        single_combined["covariate"].str.startswith("Corrected_", na=False)
    ]
    combined_km = pd.read_csv(fig9 / "Combined_All_KM_Curve_Bootstrap_Data.csv")
    curves = pd.read_csv(fig10 / "CumRisk_curves/All_Organ_CumRisk_Data.csv")
    mwu = pd.read_csv(fig10 / "MWU_FDR_results/All_Pairwise_MWU_FDR_results.csv")
    path_source = pd.read_csv(fig12 / "revised_network_sources/revised_network_path_coefficients.csv")
    boot_source = pd.read_csv(fig12 / "revised_network_sources/revised_network_path_coefficients_bootstrap_long.csv")
    comparisons = pd.read_csv(fig12 / "AllPaths_Bootstrap_Compare_AllPairs.csv")
    matrix = pd.read_csv(fig12 / "AllGroups_Path_Mean_Matrix.csv", index_col=0)

    checks = {
        "fig8_cox_csv_112": len(cox_files) == 112,
        "fig8_png_112": len(forest_png) == 112,
        "fig8_svg_112": len(forest_svg) == 112,
        "fig8_cells_112": combined_cox[["System_Group", "Organ"]].drop_duplicates().shape[0] == 112,
        "fig8_primary_target_cox_csv_28": len(target_cox_files) == 28,
        "fig8_primary_target_png_28": len(target_forest_png) == 28,
        "fig8_primary_target_svg_28": len(target_forest_svg) == 28,
        "fig8_primary_target_rows_224": len(target_combined) == 224,
        "fig8_primary_target_cells_28": target_combined[["System_Group", "Organ"]].drop_duplicates().shape[0] == 28,
        "fig8_primary_target_eight_rows_per_cell": target_combined.groupby(
            ["System_Group", "Organ"], observed=True
        ).size().eq(8).all().item(),
        "fig8_primary_only_target_gap": bool(
            len(target_gap_rows) == 28
            and target_gap_rows["covariate"].eq(target_gap_rows["Target_Gap"]).all()
        ),
        "fig8_primary_single_gap_cox_csv_112": len(single_cox_files) == 112,
        "fig8_primary_single_gap_png_112": len(single_forest_png) == 112,
        "fig8_primary_single_gap_svg_112": len(single_forest_svg) == 112,
        "fig8_primary_single_gap_rows_1568": len(single_combined) == 1568,
        "fig8_primary_single_gap_cells_112": single_combined[
            ["System_Group", "Organ"]
        ].drop_duplicates().shape[0] == 112,
        "fig8_primary_single_gap_fourteen_rows_per_cell": bool(
            single_combined.groupby(["System_Group", "Organ"], observed=True).size().eq(14).all()
        ),
        "fig8_primary_single_gap_seven_gap_rows_per_cell": bool(
            single_gap_rows.groupby(["System_Group", "Organ"], observed=True).size().eq(7).all()
        ),
        "fig8_primary_single_gap_rows_from_matching_models": bool(
            len(single_gap_rows) == 784
            and single_gap_rows["covariate"].eq(single_gap_rows["Fitted_Gap"]).all()
        ),
        "fig8_primary_single_gap_one_target_row_per_cell": bool(
            single_combined.groupby(["System_Group", "Organ"], observed=True)[
                "is_target_gap_row"
            ].sum().eq(1).all()
        ),
        "fig9_summary_csv_112": len(km_files) == 112,
        "fig9_png_112": len(km_png) == 112,
        "fig9_svg_112": len(km_svg) == 112,
        "fig9_cells_112": combined_km[["Group", "Target_System"]].drop_duplicates().shape[0] == 112,
        "fig10_curve_rows_6832": len(curves) == 6832,
        "fig10_mwu_rows_840": len(mwu) == 840,
        "fig11_path_rows_672": len(path_source) == 672,
        "fig11_bootstrap_rows_67200": len(boot_source) == 67200,
        "fig11_matrix_42_by_16": matrix.shape == (42, 16),
        "fig11_pair_comparisons_5040": len(comparisons) == 5040,
        "fig12_top17_csv_4": len(list(fig12.glob("Top17Paths_health_vs_*.csv"))) == 4,
        "fig12_edge_csv_4": len(list(fig12.glob("*_network_edges.csv"))) == 4,
        "fig12_formats_3": all((fig12 / f"Fig-12-SEM-Network.{ext}").exists() for ext in ("png", "pdf", "svg")),
    }
    key_png = [
        fig8 / "Forest_plots_single_gap_full/Health_cardiovascular_forest_plot.png",
        fig8 / "Forest_plots_single_gap_full/cardiovascular_cardiovascular_forest_plot.png",
        fig8 / "Forest_plots_target_gap_only/Health_cardiovascular_forest_plot.png",
        fig8 / "Forest_plots_target_gap_only/cardiovascular_cardiovascular_forest_plot.png",
        fig8 / "Forest_plots/Health_cardiovascular_forest_plot.png",
        fig8 / "Forest_plots/cardiovascular_cardiovascular_forest_plot.png",
        fig9 / "Panel_Plots/Health_Group/cardiovascular_panel_2x7_from_csv.png",
        fig10 / "Fig-10-CumRisk_Heatmap.png",
        fig12 / "Fig-11-Path_Mean_Heatmap.png",
        fig12 / "Fig-12-SEM-Network.png",
    ]
    images = [image_qa(path) for path in key_png if path.exists()]
    checks["key_png_nonblank"] = len(images) == len(key_png) and all(item["nonblank"] for item in images)
    payload = {
        "schema_version": "revision0710.fig08_fig12.corrected_canonical_0626_exact_v2.validation",
        "created_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "fig8_combined_rows": int(len(combined_cox)),
            "fig8_primary_single_gap_full_rows": int(len(single_combined)),
            "fig8_primary_single_gap_full_panels": int(
                single_combined[["System_Group", "Organ"]].drop_duplicates().shape[0]
            ),
            "fig8_primary_single_gap_full_age_gap_rows": int(len(single_gap_rows)),
            "fig8_primary_target_gap_rows": int(len(target_combined)),
            "fig8_primary_target_gap_panels": int(
                target_combined[["System_Group", "Organ"]].drop_duplicates().shape[0]
            ),
            "fig9_combined_rows": int(len(combined_km)),
            "fig10_curve_rows": int(len(curves)),
            "fig10_mwu_rows": int(len(mwu)),
            "fig11_path_rows": int(len(path_source)),
            "fig11_bootstrap_rows": int(len(boot_source)),
        },
        "image_qa": images,
    }
    (OUTPUT_ROOT / "FINAL_VALIDATION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Final validation failed: {failed}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        default="prepare,fig8,fig8_target_only,fig8_single_gap_full,fig9,fig10,fig11_12,validate",
        help=(
            "Comma-separated: prepare,fig8,fig8_target_only,fig8_single_gap_full,"
            "fig9,fig10,fig11_12,validate"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [item.strip() for item in args.steps.split(",") if item.strip()]
    valid = {
        "prepare",
        "fig8",
        "fig8_target_only",
        "fig8_single_gap_full",
        "fig9",
        "fig10",
        "fig11_12",
        "validate",
    }
    unknown = set(steps) - valid
    if unknown:
        raise ValueError(f"Unknown steps: {sorted(unknown)}")

    if "prepare" in steps:
        analysis, group_map, audit = prepare()
    else:
        analysis, group_map, audit = load_prepared()
    print(
        f"Prepared {analysis['Participant_ID'].nunique():,} participants; "
        f"Health={len(group_map['Health']):,}; Other_Disease={len(group_map['Other_Disease']):,}; "
        f"cells={len(audit)}"
    )

    template = load_template()
    template.setup_legacy_style()
    if "fig8" in steps:
        template.run_fig8_cox_forest(analysis, group_map, force=True)
    if "fig8_target_only" in steps:
        run_fig8_target_gap_only(analysis, group_map)
    if "fig8_single_gap_full" in steps:
        run_fig8_single_gap_full(analysis, group_map)
    if "fig9" in steps:
        template.run_fig9_km_panels(analysis, group_map, force=True)
    if "fig10" in steps:
        template.run_fig10_cumrisk(analysis, group_map, force=True)
    if "fig11_12" in steps:
        template.run_fig12_network(force=True)
    if "validate" in steps:
        payload = validate()
        print(json.dumps(payload["counts"], indent=2))
        print(f"Validation: {payload['status']}")


if __name__ == "__main__":
    main()
