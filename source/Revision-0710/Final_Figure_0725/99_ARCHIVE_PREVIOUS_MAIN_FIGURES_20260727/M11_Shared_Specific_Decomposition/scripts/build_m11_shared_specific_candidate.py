#!/usr/bin/env python3
"""Build a candidate main figure for shared and organ-specific EHR-AGE signals.

No neural-network model is trained or refit.  All components are derived from
the frozen participant-isolated OOF predictions.  The four panels show:
  A. the model-derived base-gap plus organ-offset decomposition;
  B. empirical shared versus within-participant organ-specific variance;
  C. covariate-adjusted full-gap and relative-offset correlation matrices; and
  D. full-risk-adjusted target-organ Cox associations for relative offsets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
ORGAN_LABELS = {
    "cardiovascular": "Cardio.",
    "hepatic": "Hepatic",
    "immune": "Immune",
    "metabolic": "Metabolic",
    "musculoskeletal": "Musculo.",
    "pulmonary": "Pulm.",
    "renal": "Renal",
}
ORGAN_COLORS = {
    "cardiovascular": "#F06455",
    "hepatic": "#2CB7A5",
    "immune": "#F4A589",
    "metabolic": "#9BAACB",
    "musculoskeletal": "#A08A78",
    "pulmonary": "#4DBBD5",
    "renal": "#4C6496",
}
GROUP_ORDER = (
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
)
GROUP_LABELS = {
    "Health": "Baseline ICD-free",
    "cardiovascular": "Cardio. disease",
    "pulmonary": "Pulm. disease",
    "musculoskeletal": "Musculo. disease",
    "immune": "Immune disease",
    "renal": "Renal disease",
    "hepatic": "Hepatic disease",
    "metabolic": "Metabolic disease",
    "cardiovascular_Health": "Cardio.-free / other organ",
    "pulmonary_Health": "Pulm.-free / other organ",
    "musculoskeletal_Health": "Musculo.-free / other organ",
    "immune_Health": "Immune-free / other organ",
    "renal_Health": "Renal-free / other organ",
    "hepatic_Health": "Hepatic-free / other organ",
    "metabolic_Health": "Metabolic-free / other organ",
    "Other_Disease": "Other recorded disease",
}

CORRELATION_CONTROLS = (
    "chronological_age",
    "sex_male",
    "employment_employed",
    "sleep_duration",
    "smoking_current",
    "sensitivity_pm25_2010",
    "education_higher",
    "townsend",
)
FULL_RISK_COVARIATES = (
    "chronological_age",
    "sex_male",
    "smoking_current",
    "smoking_missing",
    "education_higher",
    "education_missing",
    "townsend",
    "townsend_missing",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "employment_missing",
    "sleep_duration",
    "sleep_missing",
    "phenotype_missing_fraction",
    "log1p_preindex_icd_count",
    "log1p_preindex_gp_count",
    "has_preindex_icd",
    "has_preindex_gp",
    "prevalent_organ_count",
    "sensitivity_pm25_2010",
    "sensitivity_pm25_2010_missing",
)
SHARED_COLOR = "#4C78A8"
SPECIFIC_COLOR = "#E76F51"
COX_PENALIZER = 0.1


def find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "Revision-0710").is_dir():
            return candidate
    raise RuntimeError("Workspace root not found")


ROOT = find_root()
SCRIPT = Path(__file__).resolve()
FIGURE_ROOT = SCRIPT.parent.parent
OUTPUT_ROOT = FIGURE_ROOT / "CANDIDATE_v1"
SOURCE_DIR = OUTPUT_ROOT / "source_data"
OUTPUT_DIR = OUTPUT_ROOT / "outputs"
QA_DIR = OUTPUT_ROOT / "qa"
OOF = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1/"
    "02_summary/oof_predictions.parquet"
)
RUN_CONFIG = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/01_runs/"
    "shared_private_independent_view_huber1_fold0/run_config.json"
)
FINAL_SELECTION = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/"
    "02_fivefold_frozen_huber1/FINAL_SELECTION.json"
)
CANONICAL_WIDE = (
    ROOT
    / "Revision-0710/Final_Fig_0716/Codebook_v2_Huber1/00_inputs/fig11_fig12/"
    "canonical_age_gap_wide.parquet"
)
MEMBERSHIP = (
    ROOT
    / "Revision-0710/Final_Fig_0716/Codebook_v2_Huber1/00_inputs/fig11_fig12/"
    "display_group_membership.parquet"
)
CONTEXT = (
    ROOT
    / "Revision-0710/05_downstream/06_metric_v2/01_contextual_covariates/"
    "participant_context_covariates.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--reuse-statistics",
        action="store_true",
        help="Reuse existing candidate CSVs for layout-only redraws.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def setup_style() -> None:
    sns.set_theme(style="white", font="Times New Roman", font_scale=1.0)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_components() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object]]:
    config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    cap = float(config["model"]["risk_offset_cap_years"])
    temperature = float(config["model"]["risk_temperature"])
    columns = [
        "participant_id",
        "organ_system",
        "actual_age",
        "raw_age_gap",
        "corrected_age_gap",
        "model_risk_score",
        "duration",
        "event",
        "at_risk",
        "outer_fold",
        "strict_healthy_reference_60m",
    ]
    long = pd.read_parquet(OOF, columns=columns)
    long["participant_id"] = long["participant_id"].astype(str)
    if len(long) != 456_773 * 7:
        raise RuntimeError(f"Unexpected OOF row count: {len(long)}")
    if long.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("OOF participant-organ rows are not unique")
    if set(long["organ_system"].astype(str)) != set(ORGANS):
        raise RuntimeError("OOF organ contract changed")
    long["organ_offset"] = cap * np.tanh(
        pd.to_numeric(long["model_risk_score"], errors="raise").to_numpy(float)
        / temperature
    )
    long["base_gap_reconstructed"] = (
        pd.to_numeric(long["raw_age_gap"], errors="raise").to_numpy(float)
        - long["organ_offset"].to_numpy(float)
    )

    values: dict[str, pd.DataFrame] = {}
    for column in (
        "raw_age_gap",
        "corrected_age_gap",
        "organ_offset",
        "base_gap_reconstructed",
        "duration",
        "event",
        "at_risk",
    ):
        values[column] = long.pivot(
            index="participant_id", columns="organ_system", values=column
        ).loc[:, list(ORGANS)]
    participant = (
        long.groupby("participant_id", sort=False)
        .agg(
            outer_fold=("outer_fold", "first"),
            actual_age=("actual_age", "first"),
            strict_healthy_reference_60m=("strict_healthy_reference_60m", "first"),
        )
        .reindex(values["corrected_age_gap"].index)
    )
    base_matrix = values["base_gap_reconstructed"].to_numpy(float)
    participant["base_gap"] = base_matrix.mean(axis=1)
    offset_matrix = values["organ_offset"].to_numpy(float)
    participant["mean_offset"] = offset_matrix.mean(axis=1)
    for index, organ in enumerate(ORGANS):
        other_mean = (offset_matrix.sum(axis=1) - offset_matrix[:, index]) / 6.0
        participant[f"relative_offset_{organ}"] = offset_matrix[:, index] - other_mean

    base_span = np.ptp(base_matrix, axis=1)
    audit = {
        "participants": int(len(participant)),
        "oof_rows": int(len(long)),
        "risk_offset_cap_years": cap,
        "risk_temperature": temperature,
        "base_gap_reconstruction_span_p99_years": float(np.quantile(base_span, 0.99)),
        "base_gap_reconstruction_span_max_years": float(np.max(base_span)),
        "strict_healthy_n": int(participant["strict_healthy_reference_60m"].sum()),
    }
    return participant, values, audit


def decomposition_contributions(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    shared = centered.mean(axis=1)
    specific = centered - shared[:, None]
    shared_contribution = np.square(shared)
    specific_contribution = np.mean(np.square(specific), axis=1)
    total_contribution = np.mean(np.square(centered), axis=1)
    if not np.allclose(
        shared_contribution + specific_contribution,
        total_contribution,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise RuntimeError("Variance decomposition identity failed")
    return shared_contribution, specific_contribution, total_contribution


def decomposition_row(group: str, ids: pd.Index, corrected: pd.DataFrame) -> dict[str, object]:
    selected = corrected.loc[corrected.index.intersection(ids)].to_numpy(float)
    shared, specific, total = decomposition_contributions(selected)
    denominator = float(total.sum())
    return {
        "analysis_group": group,
        "analysis_group_label": "Full cohort" if group == "full_cohort" else GROUP_LABELS[group],
        "n": int(len(selected)),
        "shared_variance_fraction": float(shared.sum() / denominator),
        "specific_variance_fraction": float(specific.sum() / denominator),
    }


def bootstrap_full_decomposition(
    participant: pd.DataFrame,
    corrected: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    shared, specific, total = decomposition_contributions(corrected.to_numpy(float))
    folds = participant["outer_fold"].to_numpy(int)
    rng = np.random.default_rng(seed)
    by_fold = [np.flatnonzero(folds == fold) for fold in range(5)]
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        shared_sum = 0.0
        specific_sum = 0.0
        total_sum = 0.0
        for indices in by_fold:
            sampled = indices[rng.integers(0, len(indices), size=len(indices))]
            shared_sum += float(shared[sampled].sum())
            specific_sum += float(specific[sampled].sum())
            total_sum += float(total[sampled].sum())
        rows.append(
            {
                "bootstrap_replicate": replicate + 1,
                "shared_variance_fraction": shared_sum / total_sum,
                "specific_variance_fraction": specific_sum / total_sum,
                "seed": seed,
            }
        )
        if (replicate + 1) % 100 == 0:
            print(f"Variance bootstrap {replicate + 1}/{replicates}", flush=True)
    return pd.DataFrame(rows)


def build_decomposition(
    participant: pd.DataFrame,
    corrected: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = pd.read_parquet(MEMBERSHIP)
    membership["participant_id"] = membership["participant_id"].astype(str)
    rows = [decomposition_row("full_cohort", corrected.index, corrected)]
    for group in GROUP_ORDER:
        ids = pd.Index(
            membership.loc[membership["analysis_group"].eq(group), "participant_id"]
        )
        rows.append(decomposition_row(group, ids, corrected))
    summary = pd.DataFrame(rows)
    bootstrap = bootstrap_full_decomposition(
        participant,
        corrected,
        replicates=replicates,
        seed=seed,
    )
    full = summary.loc[summary["analysis_group"].eq("full_cohort")].index[0]
    summary.loc[full, "shared_ci95_low"] = bootstrap["shared_variance_fraction"].quantile(0.025)
    summary.loc[full, "shared_ci95_high"] = bootstrap["shared_variance_fraction"].quantile(0.975)
    summary.loc[full, "specific_ci95_low"] = bootstrap["specific_variance_fraction"].quantile(0.025)
    summary.loc[full, "specific_ci95_high"] = bootstrap["specific_variance_fraction"].quantile(0.975)
    return summary, bootstrap


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = values.std(axis=0, ddof=0)
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise RuntimeError("Cannot z-score constant/non-finite values")
    return (values - values.mean(axis=0)) / scale


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(controls)), controls])
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != design.shape[1]:
        raise RuntimeError("Correlation control design is rank deficient")
    return values - design @ coefficients


def build_correlations(
    corrected: pd.DataFrame,
    offsets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    context = pd.read_parquet(CONTEXT, columns=["participant_id", *CORRELATION_CONTROLS])
    context["participant_id"] = context["participant_id"].astype(str)
    context = context.set_index("participant_id").reindex(corrected.index)
    complete = context[list(CORRELATION_CONTROLS)].notna().all(axis=1)
    controls = context.loc[complete, list(CORRELATION_CONTROLS)].to_numpy(float)
    controls = zscore(controls)
    matrices: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    n = int(complete.sum())
    control_rank = 1 + len(CORRELATION_CONTROLS)
    df = n - control_rank - 2
    relative_offsets = offsets.sub(offsets.mean(axis=1), axis=0)
    for component, frame in (("full_gap", corrected), ("relative_organ_offset", relative_offsets)):
        values = zscore(frame.loc[complete].to_numpy(float))
        residual = residualize(values, controls)
        correlation = np.corrcoef(residual, rowvar=False)
        matrices[component] = correlation
        component_rows: list[dict[str, object]] = []
        for left in range(len(ORGANS)):
            for right in range(left + 1, len(ORGANS)):
                r = float(correlation[left, right])
                statistic = r * np.sqrt(df / max(1e-15, 1.0 - r * r))
                p = float(2.0 * stats.t.sf(abs(statistic), df=df))
                component_rows.append(
                    {
                        "component": component,
                        "organ_1": ORGANS[left],
                        "organ_2": ORGANS[right],
                        "n_complete": n,
                        "partial_pearson_r": r,
                        "p_value": p,
                    }
                )
        q = bh_adjust(np.asarray([row["p_value"] for row in component_rows]))
        for row, q_value in zip(component_rows, q):
            row["q_value_bh_21"] = float(q_value)
        rows.extend(component_rows)
    return pd.DataFrame(rows), matrices, n


def matrices_from_correlation_table(correlations: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for component in ("full_gap", "relative_organ_offset"):
        matrix = np.eye(len(ORGANS), dtype=float)
        subset = correlations.loc[correlations["component"].eq(component)]
        for row in subset.itertuples(index=False):
            left = ORGANS.index(str(row.organ_1))
            right = ORGANS.index(str(row.organ_2))
            matrix[left, right] = float(row.partial_pearson_r)
            matrix[right, left] = float(row.partial_pearson_r)
        matrices[component] = matrix
    return matrices


def relative_offset_vif(frame: pd.DataFrame, target: str, predictors: list[str]) -> float:
    target_values = frame[target].to_numpy(float)
    other = [column for column in predictors if column != target]
    design = np.column_stack([np.ones(len(frame)), frame[other].to_numpy(float)])
    coefficients, _, _, _ = np.linalg.lstsq(design, target_values, rcond=None)
    residual = target_values - design @ coefficients
    total = target_values - target_values.mean()
    r_squared = 1.0 - float(np.dot(residual, residual) / np.dot(total, total))
    return float(1.0 / max(1e-12, 1.0 - r_squared))


def fit_target_cox(
    organ: str,
    participant: pd.DataFrame,
    values: dict[str, pd.DataFrame],
    context: pd.DataFrame,
) -> dict[str, object]:
    from lifelines import CoxPHFitter

    at_risk = values["at_risk"][organ].to_numpy(int).astype(bool)
    outcome = pd.DataFrame(
        {
            "duration": values["duration"].loc[participant.index, organ].to_numpy(float),
            "event": values["event"].loc[participant.index, organ].to_numpy(int),
        },
        index=participant.index,
    )
    relative = f"relative_offset_{organ}"
    features = [*FULL_RISK_COVARIATES, "base_gap", "mean_offset", relative]
    analysis = context.loc[participant.index, list(FULL_RISK_COVARIATES)].copy()
    analysis["base_gap"] = participant["base_gap"]
    analysis["mean_offset"] = participant["mean_offset"]
    analysis[relative] = participant[relative]
    analysis = analysis.loc[at_risk].copy()
    outcome = outcome.loc[at_risk]
    for column in features:
        values_column = pd.to_numeric(analysis[column], errors="coerce")
        median = float(values_column.median())
        analysis[column] = values_column.fillna(median)
    retained = [column for column in features if analysis[column].std(ddof=0) > 0.0]
    means = analysis[retained].mean(axis=0)
    scales = analysis[retained].std(axis=0, ddof=0)
    standardized = (analysis[retained] - means) / scales
    vif = relative_offset_vif(standardized, relative, retained)
    model_frame = standardized.copy()
    model_frame["duration"] = outcome["duration"].to_numpy(float)
    model_frame["event"] = outcome["event"].to_numpy(int)
    model = CoxPHFitter(penalizer=COX_PENALIZER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            model_frame,
            duration_col="duration",
            event_col="event",
            show_progress=False,
            batch_mode=True,
            fit_options={"max_steps": 100},
        )
    item = model.summary.loc[relative]
    return {
        "target_organ": organ,
        "target_organ_label": ORGAN_LABELS[organ],
        "n_at_risk": int(len(model_frame)),
        "events": int(model_frame["event"].sum()),
        "model": "cause-specific Cox: full-risk baseline + base gap + mean offset + target-relative offset",
        "relative_offset_definition": "target offset minus mean of the other six offsets",
        "relative_offset_sd_years": float(scales[relative]),
        "coef_per_sd": float(item["coef"]),
        "hr_per_sd": float(item["exp(coef)"]),
        "hr_ci95_low": float(item["exp(coef) lower 95%"]),
        "hr_ci95_high": float(item["exp(coef) upper 95%"]),
        "se": float(item["se(coef)"]),
        "z": float(item["z"]),
        "p_value": float(item["p"]),
        "relative_offset_vif": vif,
        "penalizer": COX_PENALIZER,
    }


def build_cox(
    participant: pd.DataFrame,
    values: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    context = pd.read_parquet(CONTEXT, columns=["participant_id", *FULL_RISK_COVARIATES])
    context["participant_id"] = context["participant_id"].astype(str)
    context = context.set_index("participant_id")
    rows: list[dict[str, object]] = []
    for organ in ORGANS:
        print(f"Fitting target-relative offset Cox model: {organ}", flush=True)
        rows.append(fit_target_cox(organ, participant, values, context))
    result = pd.DataFrame(rows)
    result["q_value_bh_7"] = bh_adjust(result["p_value"].to_numpy(float))
    return result


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_panel_a(axis: plt.Axes) -> None:
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)
    axis.axis("off")
    panel_label(axis, "a")
    axis.set_title("Hierarchical decomposition of organ age gaps", fontsize=14, fontweight="bold", pad=8)
    base_box = FancyBboxPatch(
        (0.3, 6.1), 3.0, 2.1, boxstyle="round,pad=0.18", facecolor="#DCE8F3", edgecolor=SHARED_COLOR, linewidth=1.6
    )
    offset_box = FancyBboxPatch(
        (0.3, 2.0), 3.0, 2.8, boxstyle="round,pad=0.18", facecolor="#FBE4DA", edgecolor=SPECIFIC_COLOR, linewidth=1.6
    )
    output_box = FancyBboxPatch(
        (6.5, 3.25), 3.0, 3.2, boxstyle="round,pad=0.18", facecolor="#F7F7F7", edgecolor="#555555", linewidth=1.5
    )
    for patch in (base_box, offset_box, output_box):
        axis.add_patch(patch)
    axis.text(1.8, 7.45, "Consensus base-age gap", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#2F5F8F")
    axis.text(1.8, 6.7, r"$\widehat{Age}_{base}-Age$", ha="center", va="center", fontsize=14)
    axis.text(1.8, 4.15, "Seven prognostic offsets", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#A7432D")
    axis.text(1.8, 3.3, r"$8\tanh(RiskScore_k/2)$", ha="center", va="center", fontsize=13)
    for index, organ in enumerate(ORGANS):
        x = 0.68 + (index % 4) * 0.74
        y = 2.45 if index < 4 else 2.08
        axis.scatter(x, y, s=42, color=ORGAN_COLORS[organ], edgecolor="white", linewidth=0.4)
    axis.text(8.0, 5.65, "Seven organ gaps", ha="center", va="center", fontsize=13, fontweight="bold")
    axis.text(8.0, 4.65, r"$Gap_k=BaseGap+Offset_k$", ha="center", va="center", fontsize=13)
    axis.text(8.0, 3.8, "shared systemic signal\n+ smaller organ deviations", ha="center", va="center", fontsize=11, color="#444444")
    for start in ((3.35, 7.1), (3.35, 3.4)):
        axis.add_patch(
            FancyArrowPatch(start, (6.45, 5.0), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#666666")
        )
    axis.text(4.9, 5.35, "+", fontsize=20, fontweight="bold", ha="center")


def draw_panel_b(
    bar_axis: plt.Axes,
    dot_axis: plt.Axes,
    decomposition: pd.DataFrame,
) -> None:
    panel_label(bar_axis, "b")
    bar_axis.set_title("Shared and organ-specific variance", fontsize=14, fontweight="bold", pad=16)
    full = decomposition.loc[decomposition["analysis_group"].eq("full_cohort")].iloc[0]
    shared = 100.0 * float(full["shared_variance_fraction"])
    specific = 100.0 * float(full["specific_variance_fraction"])
    bar_axis.barh([0], [shared], color=SHARED_COLOR, height=0.55)
    bar_axis.barh([0], [specific], left=[shared], color=SPECIFIC_COLOR, height=0.55)
    bar_axis.text(shared / 2, 0, f"Shared participant component  {shared:.1f}%", color="white", ha="center", va="center", fontsize=11.5, fontweight="bold")
    bar_axis.text(shared + specific / 2, 0, f"{specific:.1f}%", color="white", ha="center", va="center", fontsize=9, fontweight="bold", rotation=90)
    bar_axis.set_xlim(0, 100)
    bar_axis.set_yticks([])
    bar_axis.set_xlabel("Variance fraction (%)", fontsize=10)
    bar_axis.spines[["top", "right", "left"]].set_visible(False)

    groups = decomposition.loc[decomposition["analysis_group"].ne("full_cohort")].copy()
    groups["order"] = groups["analysis_group"].map({group: index for index, group in enumerate(GROUP_ORDER)})
    groups = groups.sort_values("order", ascending=False)
    y = np.arange(len(groups))
    values = 100.0 * groups["specific_variance_fraction"].to_numpy(float)
    dot_axis.hlines(y, 0, values, color="#D6D6D6", linewidth=1.2)
    dot_axis.scatter(values, y, color=SPECIFIC_COLOR, s=34, edgecolor="white", linewidth=0.6, zorder=3)
    dot_axis.axvline(specific, color=SHARED_COLOR, linestyle="--", linewidth=1.2, alpha=0.9)
    dot_axis.text(
        0.985,
        0.985,
        f"Dashed line: full cohort ({specific:.2f}%)",
        transform=dot_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=SHARED_COLOR,
    )
    dot_axis.set_yticks(y, groups["analysis_group_label"], fontsize=8.2)
    dot_axis.set_xlabel("Organ-specific variance (%)", fontsize=10)
    dot_axis.set_xlim(0, max(5.2, values.max() * 1.12))
    dot_axis.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    dot_axis.spines[["top", "right", "left"]].set_visible(False)
    dot_axis.tick_params(axis="y", length=0)


def draw_panel_c(
    axes: list[plt.Axes],
    color_axis: plt.Axes,
    matrices: dict[str, np.ndarray],
) -> None:
    panel_label(axes[0], "c")
    labels = [ORGAN_LABELS[organ] for organ in ORGANS]
    titles = ("Full organ gaps", "Relative organ offsets")
    for axis, component, title in zip(axes, ("full_gap", "relative_organ_offset"), titles):
        matrix = matrices[component]
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="coolwarm",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 7.2},
            linewidths=0.7,
            linecolor="white",
            cbar=False,
        )
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticklabels(labels, rotation=0, fontsize=8)
        axis.set_title(title, fontsize=12, fontweight="bold", pad=8)
        axis.set_xlabel("")
        axis.set_ylabel("")
    # A discrete vector key keeps the SVG fully editable. Matplotlib otherwise
    # embeds a continuous colorbar as a small PNG inside the SVG.
    norm = Normalize(-1.0, 1.0)
    cmap = matplotlib.colormaps["coolwarm"]
    color_axis.set_xlim(0.0, 1.0)
    color_axis.set_ylim(-1.0, 1.0)
    boundaries = np.linspace(-1.0, 1.0, 21)
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        color_axis.add_patch(
            Rectangle(
                (0.0, low),
                1.0,
                high - low,
                facecolor=cmap(norm((low + high) / 2.0)),
                edgecolor="none",
            )
        )
    color_axis.set_xticks([])
    color_axis.yaxis.tick_right()
    color_axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    color_axis.tick_params(axis="y", labelsize=8)
    color_axis.set_ylabel("Adjusted partial correlation", fontsize=9, rotation=90, labelpad=10)
    color_axis.yaxis.set_label_position("right")
    axes[0].text(
        0.0,
        1.19,
        "Correlation structure before and after shared-component removal",
        transform=axes[0].transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
    )


def draw_panel_d(axis: plt.Axes, cox: pd.DataFrame) -> None:
    panel_label(axis, "d")
    axis.set_title("Prospective associations of relative organ offsets", fontsize=14, fontweight="bold", pad=10)
    display = cox.set_index("target_organ").loc[list(reversed(ORGANS))].reset_index()
    y = np.arange(len(display))
    for index, row in display.iterrows():
        organ = str(row["target_organ"])
        axis.errorbar(
            float(row["hr_per_sd"]),
            index,
            xerr=np.asarray(
                [[float(row["hr_per_sd"] - row["hr_ci95_low"])], [float(row["hr_ci95_high"] - row["hr_per_sd"])]]
            ),
            fmt="o",
            color=ORGAN_COLORS[organ],
            markersize=7,
            capsize=3,
            linewidth=1.6,
            zorder=3,
        )
        axis.text(
            float(row["hr_ci95_high"]) + 0.01,
            index,
            f"{row['hr_per_sd']:.2f} ({row['hr_ci95_low']:.2f}–{row['hr_ci95_high']:.2f})",
            va="center",
            fontsize=8.5,
        )
    axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1.1)
    axis.set_yticks(y, display["target_organ_label"], fontsize=10)
    axis.set_xlabel("Adjusted hazard ratio per 1-SD relative offset", fontsize=10)
    lower = min(0.95, float(display["hr_ci95_low"].min()) - 0.03)
    upper = float(display["hr_ci95_high"].max()) + 0.09
    axis.set_xlim(lower, upper)
    axis.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.text(
        0.0,
        -0.17,
        "Cause-specific Cox models adjust for the full clinical and record-density baseline,\nshared base gap, and mean organ offset; Wald 95% CIs.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )


def render_figure(
    decomposition: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    cox: pd.DataFrame,
) -> list[Path]:
    figure = plt.figure(figsize=(15.4, 10.6))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(0.94, 1.06),
        height_ratios=(0.94, 1.06),
        left=0.055,
        right=0.98,
        top=0.96,
        bottom=0.08,
        wspace=0.28,
        hspace=0.29,
    )
    axis_a = figure.add_subplot(outer[0, 0])
    b_grid = outer[0, 1].subgridspec(2, 1, height_ratios=(1.0, 3.25), hspace=0.34)
    axis_b_bar = figure.add_subplot(b_grid[0, 0])
    axis_b_dot = figure.add_subplot(b_grid[1, 0])
    c_grid = outer[1, 0].subgridspec(1, 3, width_ratios=(1.0, 1.0, 0.055), wspace=0.25)
    axes_c = [figure.add_subplot(c_grid[0, 0]), figure.add_subplot(c_grid[0, 1])]
    cbar_c = figure.add_subplot(c_grid[0, 2])
    axis_d = figure.add_subplot(outer[1, 1])
    draw_panel_a(axis_a)
    draw_panel_b(axis_b_bar, axis_b_dot, decomposition)
    draw_panel_c(axes_c, cbar_c, matrices)
    draw_panel_d(axis_d, cox)
    outputs: list[Path] = []
    for extension in ("svg", "pdf", "png"):
        path = OUTPUT_DIR / f"M11_candidate_shared_and_organ_specific_signals.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def write_readme() -> None:
    text = """# M11共享与器官特异性候选正文图

本目录是待作者审核的候选图，不替换当前Fig.11指针。所有分析均来自冻结的参与者隔离OOF预测，没有重新训练神经网络。

- A：展示`器官gap=共享基础gap+器官offset`的模型结构；
- B：以正交的参与者内分解量化共享成分和器官特异性变异，全队列95%区间使用1,000次按outer fold分层的参与者bootstrap；
- C：在年龄、性别及七项预设临床协变量调整后，对比完整gap与去除个体内平均offset后的相对器官offset偏相关；
- D：使用完整临床、疾病负担和医疗记录密度基线，并进一步调整共享基础gap与平均offset，估计靶器官相对offset的cause-specific Cox HR。

当前D面板用于先判断器官相对offset是否保留独立前瞻性关联。增量Harrell C-index及7×7靶器官/非靶器官敏感性将在确认该候选布局后补充。

注意：相对器官offset在每位参与者内进行了中心化，因此C面板中的负相关表示器官之间的相对对比结构，不表示器官间抑制、传播方向或因果关系。
"""
    (OUTPUT_ROOT / "README_CN.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    for path in (OOF, RUN_CONFIG, FINAL_SELECTION, CANONICAL_WIDE, MEMBERSHIP, CONTEXT):
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in (OUTPUT_ROOT, SOURCE_DIR, OUTPUT_DIR, QA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    setup_style()
    participant, values, reconstruction_audit = load_components()

    canonical = pd.read_parquet(CANONICAL_WIDE)
    canonical["participant_id"] = canonical["participant_id"].astype(str)
    canonical = canonical.set_index("participant_id").loc[participant.index]
    canonical_columns = [f"{organ}_linear_corrected_age_gap" for organ in ORGANS]
    canonical_gap = canonical[canonical_columns].copy()
    canonical_gap.columns = list(ORGANS)
    oof_gap = values["corrected_age_gap"].loc[participant.index, list(ORGANS)]
    parity = float(np.max(np.abs(canonical_gap.to_numpy(float) - oof_gap.to_numpy(float))))
    if parity > 1e-8:
        raise RuntimeError(f"Canonical/OOF corrected-gap mismatch: {parity}")

    decomposition_path = SOURCE_DIR / "shared_specific_variance_full_and_16groups.csv"
    bootstrap_path = SOURCE_DIR / "shared_specific_variance_bootstrap_1000.csv"
    correlations_path = SOURCE_DIR / "full_gap_vs_relative_offset_partial_correlations.csv"
    cox_path = SOURCE_DIR / "target_relative_offset_fullrisk_cox.csv"
    reusable = all(
        path.exists()
        for path in (decomposition_path, bootstrap_path, correlations_path, cox_path)
    )
    if args.reuse_statistics and reusable:
        decomposition = pd.read_csv(decomposition_path)
        bootstrap = pd.read_csv(bootstrap_path)
        correlations = pd.read_csv(correlations_path)
        matrices = matrices_from_correlation_table(correlations)
        correlation_n = int(correlations["n_complete"].iloc[0])
        cox = pd.read_csv(cox_path)
    else:
        decomposition, bootstrap = build_decomposition(
            participant,
            canonical_gap,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        correlations, matrices, correlation_n = build_correlations(
            canonical_gap,
            values["organ_offset"].loc[participant.index, list(ORGANS)],
        )
        cox = build_cox(participant, values)
        decomposition.to_csv(decomposition_path, index=False)
        bootstrap.to_csv(bootstrap_path, index=False)
        correlations.to_csv(correlations_path, index=False)
        cox.to_csv(cox_path, index=False)
    reconstruction_audit.update(
        {
            "canonical_oof_corrected_gap_max_abs_difference": parity,
            "correlation_complete_cases": correlation_n,
        }
    )
    (SOURCE_DIR / "component_reconstruction_audit.json").write_text(
        json.dumps(reconstruction_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs = render_figure(decomposition, matrices, cox)
    write_readme()

    svg = next(path for path in outputs if path.suffix == ".svg")
    svg_text = svg.read_text(encoding="utf-8")
    full = decomposition.loc[decomposition["analysis_group"].eq("full_cohort")].iloc[0]
    qa = {
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "candidate_only_pointer_unchanged": True,
        "participants": int(len(participant)),
        "organs": len(ORGANS),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "shared_variance_fraction": float(full["shared_variance_fraction"]),
        "specific_variance_fraction": float(full["specific_variance_fraction"]),
        "variance_identity_sum": float(full["shared_variance_fraction"] + full["specific_variance_fraction"]),
        "full_gap_partial_r_range": [
            float(correlations.loc[correlations["component"].eq("full_gap"), "partial_pearson_r"].min()),
            float(correlations.loc[correlations["component"].eq("full_gap"), "partial_pearson_r"].max()),
        ],
        "relative_offset_partial_r_range": [
            float(correlations.loc[correlations["component"].eq("relative_organ_offset"), "partial_pearson_r"].min()),
            float(correlations.loc[correlations["component"].eq("relative_organ_offset"), "partial_pearson_r"].max()),
        ],
        "cox_hr_range": [float(cox["hr_per_sd"].min()), float(cox["hr_per_sd"].max())],
        "cox_max_relative_offset_vif": float(cox["relative_offset_vif"].max()),
        "svg_editable_text": "<text" in svg_text,
        "svg_embedded_raster": "<image" in svg_text or "data:image" in svg_text,
        "canonical_oof_corrected_gap_max_abs_difference": parity,
    }
    if not qa["svg_editable_text"] or qa["svg_embedded_raster"]:
        raise RuntimeError("SVG editability QA failed")
    if abs(qa["variance_identity_sum"] - 1.0) > 1e-10:
        raise RuntimeError("Variance fraction QA failed")
    (QA_DIR / "QA_REPORT.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    inputs = [OOF, RUN_CONFIG, FINAL_SELECTION, CANONICAL_WIDE, MEMBERSHIP, CONTEXT]
    artifacts = sorted(
        path for path in OUTPUT_ROOT.rglob("*") if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "schema_version": "final_figure_0725.m11.shared_specific.candidate.v1",
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "candidate_only": True,
        "pointer_updated": False,
        "random_seed": int(args.seed),
        "inputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ],
        "code": {
            "path": SCRIPT.relative_to(ROOT).as_posix(),
            "bytes": SCRIPT.stat().st_size,
            "sha256": sha256(SCRIPT),
        },
        "artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }
    (OUTPUT_ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
