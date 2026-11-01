#!/usr/bin/env python3
"""Build the Figure 6 shared-component residual covariance atlas preview.

The left heatmap shows all 21 unordered organ pairs in the 16 prespecified
clinical groups. The right side shows the Full cohort reference plus four
representative clinical-state networks. Edges are undirected covariance
residuals from the current one-factor DWLS SEM, not causal paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd
import polars as pl
import semopy


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

NETWORK_LABELS = {
    **ORGAN_LABELS,
    "metabolic": "Metab.",
}

# The 16 clinical groups retain the order used in the supplied reference.
CLINICAL_GROUPS = (
    ("health", "Baseline-free Health", "Health", "Baseline-free\nHealth"),
    ("cardiovascular_disease", "Cardiovascular disease", "cardiovascular", "Cardio.\ndisease"),
    ("pulmonary_disease", "Pulmonary disease", "pulmonary", "Pulm.\ndisease"),
    ("musculoskeletal_disease", "Musculoskeletal disease", "musculoskeletal", "Musculo.\ndisease"),
    ("immune_disease", "Immune disease", "immune", "Immune\ndisease"),
    ("renal_disease", "Renal disease", "renal", "Renal\ndisease"),
    ("hepatic_disease", "Hepatic disease", "hepatic", "Hepatic\ndisease"),
    ("metabolic_disease", "Metabolic disease", "metabolic", "Metabolic\ndisease"),
    ("cardiovascular_health", "Cardiovascular Health", "cardiovascular_Health", "Cardio.\nHealth"),
    ("pulmonary_health", "Pulmonary Health", "pulmonary_Health", "Pulm.\nHealth"),
    ("musculoskeletal_health", "Musculoskeletal Health", "musculoskeletal_Health", "Musculo.\nHealth"),
    ("immune_health", "Immune Health", "immune_Health", "Immune\nHealth"),
    ("renal_health", "Renal Health", "renal_Health", "Renal\nHealth"),
    ("hepatic_health", "Hepatic Health", "hepatic_Health", "Hepatic\nHealth"),
    ("metabolic_health", "Metabolic Health", "metabolic_Health", "Metabolic\nHealth"),
    ("other_disease", "Other recorded disease", "Other_Disease", "Other recorded\ndisease"),
)

NETWORK_GROUPS = (
    ("full_cohort", "Full cohort", None),
    ("health", "Baseline-free\nHealth", "Health"),
    ("cardiovascular_health", "Cardiovascular\nHealth", "cardiovascular_Health"),
    ("cardiovascular_disease", "Cardiovascular\ndisease", "cardiovascular"),
)

HEATMAP_GROUPS = (
    ("full_cohort", "Full cohort", None, "Full\ncohort"),
    *CLINICAL_GROUPS,
)

MODEL = "Shared =~ " + " + ".join(ORGANS)

POSITIONS = {
    "cardiovascular": (-1.02, 0.23),
    "pulmonary": (-1.02, -0.58),
    "hepatic": (-0.30, 0.83),
    "renal": (-0.30, -0.97),
    "metabolic": (0.58, 0.62),
    "immune": (1.02, -0.05),
    "musculoskeletal": (0.58, -0.73),
}

RED = "#FF1717"
BLUE = "#4F79C7"
NEUTRAL = "#F2F2F2"
DARK = "#252525"
MID = "#666666"
WHITE = "#FFFFFF"

BASE = Path(__file__).resolve().parent.parent
ROOT = next(
    candidate
    for candidate in (BASE, *BASE.parents)
    if (candidate / "Revision-0730").is_dir()
)
SOURCE = BASE / "source_data"
OUTPUT = BASE / "outputs" / "preview_figure6_residual_atlas"
WIDE = ROOT / (
    "Revision-0710/06_figures_and_statistics/02_canonical_analysis_data/"
    "02_oof_wide/canonical_age_gap_wide.parquet"
)
MEMBERSHIP = ROOT / (
    "Revision-0710/Final_Fig_0716/Codebook_v2_Huber1/00_inputs/"
    "fig11_fig12/display_group_membership.parquet"
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = (
        pl.read_parquet(WIDE)
        .select(
            [
                "participant_id",
                "outer_fold",
                *[f"{organ}__corrected_age_gap" for organ in ORGANS],
            ]
        )
        .rename({f"{organ}__corrected_age_gap": organ for organ in ORGANS})
    )
    membership = pl.read_parquet(MEMBERSHIP)
    expected = {group for _, _, group, _ in CLINICAL_GROUPS}
    observed = set(membership["analysis_group"].unique().to_list())
    if observed != expected:
        raise RuntimeError(
            "clinical group contract mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if membership.select(
        pl.struct(["participant_id", "analysis_group"]).n_unique()
    ).item() != membership.height:
        raise RuntimeError("duplicate participant-group membership rows")
    data = pd.DataFrame(wide.to_dict(as_series=False))
    groups = pd.DataFrame(membership.to_dict(as_series=False))
    data["outer_fold"] = data["outer_fold"].astype(int)
    if data["participant_id"].duplicated().any():
        raise RuntimeError("wide OSAG table contains duplicate participant IDs")
    if not np.isfinite(data[list(ORGANS)].to_numpy(float)).all():
        raise RuntimeError("corrected OSAG matrix contains missing or non-finite values")
    return data, groups


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[list(ORGANS)].astype(float)
    scale = values.std(axis=0, ddof=0)
    if (scale <= 0).any():
        raise RuntimeError("non-positive within-group OSAG scale")
    return (values - values.mean(axis=0)) / scale


def fit_one_factor(frame: pd.DataFrame) -> dict[str, object]:
    standardized = standardize(frame)
    model = semopy.Model(MODEL)
    optimizer = model.fit(data=standardized, obj="DWLS", solver="SLSQP")
    if not optimizer.success:
        raise RuntimeError("one-factor DWLS SEM did not converge")
    # Standard errors are not required for this descriptive residual atlas.
    # Skipping them also avoids an information-matrix inversion when a tightly
    # aligned subgroup lies on a residual-variance boundary.
    estimates = model.inspect(std_est=True, information=None)
    loadings = np.array(
        [
            float(
                estimates.loc[
                    (estimates["op"] == "~")
                    & (estimates["rval"] == "Shared")
                    & (estimates["lval"] == organ),
                    "Est. Std",
                ].iloc[0]
            )
            for organ in ORGANS
        ]
    )
    residual_variances = np.array(
        [
            float(
                estimates.loc[
                    (estimates["op"] == "~~")
                    & (estimates["lval"] == organ)
                    & (estimates["rval"] == organ),
                    "Est. Std",
                ].iloc[0]
            )
            for organ in ORGANS
        ]
    )
    observed = standardized.corr().to_numpy(float)
    residual = observed - np.outer(loadings, loadings)
    np.fill_diagonal(residual, 0.0)
    fit = semopy.calc_stats(model).iloc[0]
    off_diagonal = ~np.eye(len(ORGANS), dtype=bool)
    srmr = float(np.sqrt(np.mean(np.square(residual[off_diagonal]))))
    return {
        "loadings": loadings,
        "residual_variances": residual_variances,
        "residual": residual,
        "optimizer_success": bool(optimizer.success),
        "df": int(fit["DoF"]),
        "chi_square": float(fit["chi2"]),
        "cfi": float(fit["CFI"]),
        "tli": float(fit["TLI"]),
        "rmsea": float(fit["RMSEA"]),
        "srmr": srmr,
    }


def sample_frame(
    data: pd.DataFrame, membership: pd.DataFrame, membership_group: str | None
) -> pd.DataFrame:
    if membership_group is None:
        return data.copy()
    ids = membership.loc[
        membership["analysis_group"].eq(membership_group), "participant_id"
    ]
    frame = data.loc[data["participant_id"].isin(ids)].copy()
    if len(frame) != len(ids):
        raise RuntimeError(f"membership join lost rows for {membership_group}")
    return frame


def analyse_groups(
    data: pd.DataFrame, membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_specs = [(key, label, raw) for key, label, raw, _ in HEATMAP_GROUPS]
    residual_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    for sample, label, membership_group in group_specs:
        frame = sample_frame(data, membership, membership_group)
        print(f"Fitting residual atlas sample: {sample} (n={len(frame):,})", flush=True)
        pooled = fit_one_factor(frame)
        fold_results = []
        for fold in range(5):
            fold_frame = frame.loc[frame["outer_fold"].eq(fold)]
            if len(fold_frame) < 100:
                raise RuntimeError(f"insufficient fold sample for {sample}, fold {fold}")
            fold_results.append(fit_one_factor(fold_frame))
        fold_matrices = np.stack([result["residual"] for result in fold_results])

        fit_rows.append(
            {
                "sample": sample,
                "group_label": label,
                "membership_group": "Full cohort" if membership_group is None else membership_group,
                "n": int(len(frame)),
                "estimator": "DWLS",
                "optimizer_success": pooled["optimizer_success"],
                "df": pooled["df"],
                "chi_square": pooled["chi_square"],
                "cfi": pooled["cfi"],
                "tli": pooled["tli"],
                "rmsea": pooled["rmsea"],
                "srmr": pooled["srmr"],
                "loading_min": float(np.min(pooled["loadings"])),
                "loading_max": float(np.max(pooled["loadings"])),
                "residual_variance_min": float(np.min(pooled["residual_variances"])),
                "residual_variance_max": float(np.max(pooled["residual_variances"])),
            }
        )
        for index, organ in enumerate(ORGANS):
            parameter_rows.append(
                {
                    "sample": sample,
                    "group_label": label,
                    "n": int(len(frame)),
                    "organ": organ,
                    "loading_standardized": float(pooled["loadings"][index]),
                    "shared_variance_fraction": float(pooled["loadings"][index] ** 2),
                    "residual_variance_standardized": float(pooled["residual_variances"][index]),
                    "fold_loading_min": float(
                        min(result["loadings"][index] for result in fold_results)
                    ),
                    "fold_loading_max": float(
                        max(result["loadings"][index] for result in fold_results)
                    ),
                }
            )

        pooled_matrix = pooled["residual"]
        for left in range(len(ORGANS)):
            for right in range(left + 1, len(ORGANS)):
                pooled_value = float(pooled_matrix[left, right])
                fold_values = fold_matrices[:, left, right]
                pooled_sign = np.sign(pooled_value)
                sign_consistency = float(np.mean(np.sign(fold_values) == pooled_sign))
                residual_rows.append(
                    {
                        "sample": sample,
                        "group_label": label,
                        "membership_group": "Full cohort" if membership_group is None else membership_group,
                        "n": int(len(frame)),
                        "organ_1": ORGANS[left],
                        "organ_2": ORGANS[right],
                        "pooled_standardized_covariance_residual": pooled_value,
                        "fold_0": float(fold_values[0]),
                        "fold_1": float(fold_values[1]),
                        "fold_2": float(fold_values[2]),
                        "fold_3": float(fold_values[3]),
                        "fold_4": float(fold_values[4]),
                        "fold_min": float(fold_values.min()),
                        "fold_max": float(fold_values.max()),
                        "fold_sign_consistency": sign_consistency,
                        "pooled_abs_rank": 0,
                        "display_edge": False,
                        "display_edge_reason": "",
                    }
                )

    residuals = pd.DataFrame(residual_rows)
    for sample in residuals["sample"].unique():
        mask = residuals["sample"].eq(sample)
        ranks = residuals.loc[mask, "pooled_standardized_covariance_residual"].abs().rank(
            method="first", ascending=False
        )
        residuals.loc[mask, "pooled_abs_rank"] = ranks.astype(int)
        stable_top = (ranks <= 9) & (
            residuals.loc[mask, "fold_sign_consistency"].to_numpy(float) >= 0.8
        )
        selected = residuals.loc[mask].index[stable_top]
        residuals.loc[selected, "display_edge"] = True
        residuals.loc[selected, "display_edge_reason"] = "top_9_fold_stable"

        # Keep every organ visible using an observed incident edge. Stable edges
        # are preferred; a pooled strongest edge is used only if a group has no
        # fold-stable incident edge for that organ.
        for organ in ORGANS:
            group_rows = residuals.loc[mask]
            incident = group_rows.loc[
                group_rows["organ_1"].eq(organ) | group_rows["organ_2"].eq(organ)
            ]
            if incident["display_edge"].any():
                continue
            stable = incident.loc[incident["fold_sign_consistency"] >= 0.8]
            candidates = stable if not stable.empty else incident
            selected_index = candidates[
                "pooled_standardized_covariance_residual"
            ].abs().idxmax()
            residuals.loc[selected_index, "display_edge"] = True
            residuals.loc[selected_index, "display_edge_reason"] = (
                "strongest_fold_stable_incident"
                if not stable.empty
                else "strongest_observed_incident"
            )

    return residuals, pd.DataFrame(fit_rows), pd.DataFrame(parameter_rows)


def edge_endpoints(
    first: tuple[float, float], second: tuple[float, float], radius: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    vector = np.asarray(second) - np.asarray(first)
    direction = vector / np.linalg.norm(vector)
    return (
        tuple(np.asarray(first) + radius * direction),
        tuple(np.asarray(second) - radius * direction),
    )


def edge_curvature(organ_1: str, organ_2: str) -> float:
    pair = tuple(sorted((organ_1, organ_2)))
    curvature = {
        ("cardiovascular", "hepatic"): 0.10,
        ("cardiovascular", "immune"): 0.08,
        ("cardiovascular", "metabolic"): -0.12,
        ("cardiovascular", "musculoskeletal"): 0.10,
        ("cardiovascular", "pulmonary"): -0.16,
        ("cardiovascular", "renal"): -0.10,
        ("hepatic", "immune"): -0.08,
        ("hepatic", "metabolic"): -0.10,
        ("hepatic", "musculoskeletal"): 0.09,
        ("hepatic", "pulmonary"): 0.12,
        ("hepatic", "renal"): -0.10,
        ("immune", "metabolic"): 0.13,
        ("immune", "musculoskeletal"): -0.12,
        ("immune", "pulmonary"): -0.10,
        ("immune", "renal"): 0.10,
        ("metabolic", "musculoskeletal"): 0.10,
        ("metabolic", "pulmonary"): -0.08,
        ("metabolic", "renal"): 0.10,
        ("musculoskeletal", "pulmonary"): -0.12,
        ("musculoskeletal", "renal"): 0.10,
        ("pulmonary", "renal"): -0.12,
    }
    return curvature[pair]


def draw_network(
    axis: plt.Axes,
    sample: str,
    title: str,
    residuals: pd.DataFrame,
    fits: pd.DataFrame,
    global_max: float,
    top_row: bool,
) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-1.32, 1.32)
    axis.set_ylim(-1.31, 1.18)
    axis.axis("off")
    title_size = 6.8
    node_size = 5.2
    node_radius = 0.285
    title_y = 1.005 if top_row else 1.085
    n_y = 0.950 if top_row else 1.015
    axis.text(
        0.5,
        title_y,
        title,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=title_size,
        fontweight="bold",
        color=DARK,
        linespacing=0.90,
    )
    n = int(fits.loc[fits["sample"].eq(sample), "n"].iloc[0])
    axis.text(
        0.5,
        n_y,
        f"n={n:,}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )

    displayed = residuals.loc[
        residuals["sample"].eq(sample) & residuals["display_edge"]
    ].copy()
    for _, row in displayed.sort_values("pooled_abs_rank", ascending=False).iterrows():
        first = POSITIONS[row["organ_1"]]
        second = POSITIONS[row["organ_2"]]
        start, end = edge_endpoints(first, second, node_radius)
        value = float(row["pooled_standardized_covariance_residual"])
        width = 0.58 + 4.0 * abs(value) / global_max
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-",
                connectionstyle=(
                    f"arc3,rad={edge_curvature(row['organ_1'], row['organ_2']):.3f}"
                ),
                color=RED if value > 0 else BLUE,
                linewidth=width,
                alpha=0.96,
                capstyle="round",
                zorder=1,
            )
        )

    for organ in ORGANS:
        axis.add_patch(
            Circle(
                POSITIONS[organ],
                radius=node_radius,
                facecolor=NEUTRAL,
                edgecolor=DARK,
                linewidth=0.82,
                zorder=5,
            )
        )
        axis.text(
            *POSITIONS[organ],
            NETWORK_LABELS[organ],
            ha="center",
            va="center",
            fontsize=node_size,
            color=DARK,
            zorder=6,
        )


def heatmap_matrix(residuals: pd.DataFrame) -> tuple[list[tuple[str, str]], np.ndarray]:
    pairs = [
        (ORGANS[left], ORGANS[right])
        for left in range(len(ORGANS))
        for right in range(left + 1, len(ORGANS))
    ]
    matrix = np.zeros((len(pairs), len(HEATMAP_GROUPS)), dtype=float)
    for row_index, (organ_1, organ_2) in enumerate(pairs):
        for column_index, (sample, _, _, _) in enumerate(HEATMAP_GROUPS):
            match = residuals.loc[
                residuals["sample"].eq(sample)
                & residuals["organ_1"].eq(organ_1)
                & residuals["organ_2"].eq(organ_2),
                "pooled_standardized_covariance_residual",
            ]
            if len(match) != 1:
                raise RuntimeError(f"residual matrix lookup failed for {sample}/{organ_1}/{organ_2}")
            matrix[row_index, column_index] = float(match.iloc[0])
    return pairs, matrix


def draw_heatmap(
    axis: plt.Axes, color_axis: plt.Axes, residuals: pd.DataFrame
) -> float:
    pairs, matrix = heatmap_matrix(residuals)
    limit = float(np.max(np.abs(matrix)))
    colormap = LinearSegmentedColormap.from_list(
        "residual_red_blue", [BLUE, NEUTRAL, RED]
    )
    image = axis.imshow(
        matrix,
        cmap=colormap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(HEATMAP_GROUPS)))
    axis.set_xticklabels(
        [
            "Full\ncohort",
            "Baseline\nfree\nHealth",
            "Cardio.\nDis.",
            "Pulm.\nDis.",
            "Musculo.\nDis.",
            "Immune\nDis.",
            "Renal\nDis.",
            "Hepatic\nDis.",
            "Metab.\nDis.",
            "Cardio.\nHealth",
            "Pulm.\nHealth",
            "Musculo.\nHealth",
            "Immune\nHealth",
            "Renal\nHealth",
            "Hepatic\nHealth",
            "Metab.\nHealth",
            "Other\nrecorded\ndisease",
        ],
        rotation=0,
        ha="center",
        va="top",
        fontsize=5.0,
        fontstretch="condensed",
        linespacing=0.92,
    )
    axis.set_yticks(np.arange(len(pairs)))
    axis.set_yticklabels(
        [f"{ORGAN_LABELS[left]} - {ORGAN_LABELS[right]}" for left, right in pairs],
        fontsize=5.2,
    )
    axis.set_title(
        "Residual covariance in the Full cohort and 16 clinical groups",
        fontsize=8.3,
        fontweight="bold",
        pad=7,
        loc="center",
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color=WHITE if abs(value) > 0.58 * limit else DARK,
            )

    axis.set_xticks(np.arange(-0.5, len(HEATMAP_GROUPS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(pairs), 1), minor=True)
    axis.grid(which="minor", color=WHITE, linewidth=0.42)
    for boundary in (0.5, 1.5, 8.5, 15.5):
        axis.axvline(boundary, color="#9F9F9F", linewidth=0.72, zorder=5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(which="major", length=0, pad=2.0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    colorbar = axis.figure.colorbar(image, cax=color_axis, orientation="vertical")
    colorbar.set_label(
        "Standardized covariance residual",
        fontsize=5.3,
        labelpad=4,
    )
    colorbar.ax.tick_params(labelsize=5.0, length=2.0, width=0.5)
    colorbar.outline.set_linewidth(0.5)
    return limit


def render_figure(
    residuals: pd.DataFrame, fits: pd.DataFrame
) -> list[Path]:
    figure = plt.figure(figsize=(7.204724, 5.905512))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.72, 1.00],
        left=0.075,
        right=0.987,
        bottom=0.160,
        top=0.935,
        wspace=0.135,
    )
    left = outer[0].subgridspec(1, 2, width_ratios=[28, 1], wspace=0.08)
    heat_axis = figure.add_subplot(left[0, 0])
    color_axis = figure.add_subplot(left[0, 1])
    draw_heatmap(heat_axis, color_axis, residuals)

    right = outer[1].subgridspec(
        2,
        2,
        wspace=0.04,
        hspace=0.22,
    )
    network_axes = [
        figure.add_subplot(right[0, 0]),
        figure.add_subplot(right[0, 1]),
        figure.add_subplot(right[1, 0]),
        figure.add_subplot(right[1, 1]),
    ]
    figure.text(
        0.79,
        0.978,
        "Representative residual covariance networks",
        ha="center",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color=DARK,
    )
    selected_samples = {sample for sample, _, _ in NETWORK_GROUPS}
    global_max = float(
        residuals.loc[
            residuals["sample"].isin(selected_samples) & residuals["display_edge"],
            "pooled_standardized_covariance_residual",
        ]
        .abs()
        .max()
    )
    for index, (axis, (sample, title, _)) in enumerate(
        zip(network_axes, NETWORK_GROUPS)
    ):
        draw_network(
            axis=axis,
            sample=sample,
            title=title,
            residuals=residuals,
            fits=fits,
            global_max=global_max,
            top_row=index < 2,
        )

    legend = [
        Line2D([0], [0], color=RED, linewidth=2.3, label="Positive residual covariance"),
        Line2D([0], [0], color=BLUE, linewidth=2.3, label="Negative residual covariance"),
        Line2D([0], [0], color=MID, linewidth=0.8, label="Line width: absolute magnitude"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.79, 0.037),
        ncol=1,
        frameon=False,
        fontsize=5.0,
        handlelength=3.2,
        labelspacing=0.35,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "Figure6_Shared_Component_Residual_Atlas_preview"
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    tiff_path = stem.with_suffix(".tiff")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white")
    paths = [svg_path, pdf_path, png_path, tiff_path]
    plt.close(figure)
    return paths


def main() -> int:
    set_style()
    SOURCE.mkdir(parents=True, exist_ok=True)
    data, membership = load_inputs()
    residuals, fits, parameters = analyse_groups(data, membership)

    residual_path = SOURCE / "sem_16group_residual_covariance.csv"
    fit_path = SOURCE / "sem_16group_fit_indices.csv"
    parameter_path = SOURCE / "sem_16group_standardized_parameters.csv"
    residuals.to_csv(residual_path, index=False)
    fits.to_csv(fit_path, index=False)
    parameters.to_csv(parameter_path, index=False)
    outputs = render_figure(residuals, fits)

    group_counts = {
        sample: int(fits.loc[fits["sample"].eq(sample), "n"].iloc[0])
        for sample, _, _, _ in CLINICAL_GROUPS
    }
    network_samples = [sample for sample, _, _ in NETWORK_GROUPS]
    report = {
        "status": "PREVIEW",
        "figure_number": 6,
        "model": MODEL,
        "estimator": "DWLS",
        "residual_definition": (
            "observed standardized covariance minus one-factor-implied common covariance"
        ),
        "clinical_group_count": len(CLINICAL_GROUPS),
        "heatmap_column_count": len(HEATMAP_GROUPS),
        "clinical_group_counts": group_counts,
        "heatmap_pair_count": int(len(ORGANS) * (len(ORGANS) - 1) / 2),
        "network_samples": network_samples,
        "edge_selection": (
            "nine largest absolute pooled residuals with at least four-of-five fold sign "
            "agreement, plus the strongest observed incident edge only when needed to keep "
            "an organ visible"
        ),
        "directional_paths": 0,
        "subsampling": False,
        "source_data": [residual_path.name, fit_path.name, parameter_path.name],
        "outputs": [path.name for path in outputs],
    }
    (OUTPUT / "PREVIEW_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
