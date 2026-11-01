#!/usr/bin/env python3
"""Export independently assembleable Figure 6 components from validated source data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import pandas as pd

from export_from_png import create_derived_exports

from build_figure6_shared_component_residual_atlas import (
    BLUE,
    DARK,
    MID,
    NEUTRAL,
    ORGAN_LABELS,
    ORGANS,
    POSITIONS,
    RED,
    draw_heatmap,
    edge_curvature,
    edge_endpoints,
)


BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE / "source_data"
OUTPUT = BASE / "outputs" / "Figure6_split_exports"

NETWORK_SPECS = (
    ("full_cohort", "Full cohort", "full_cohort"),
    ("health", "Baseline-free Health", "baseline_free_health"),
    ("cardiovascular_health", "Cardiovascular Health", "cardiovascular_health"),
    ("cardiovascular_disease", "Cardiovascular disease", "cardiovascular_disease"),
)

NETWORK_POSITIONS = {
    organ: (position[0] * 1.98, position[1] * 1.925)
    for organ, position in POSITIONS.items()
}


def set_component_style() -> None:
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


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residuals = pd.read_csv(SOURCE / "sem_16group_residual_covariance.csv")
    fits = pd.read_csv(SOURCE / "sem_16group_fit_indices.csv")
    parameters = pd.read_csv(SOURCE / "sem_16group_standardized_parameters.csv")
    if residuals["sample"].nunique() != 17:
        raise RuntimeError("expected Full cohort plus 16 clinical groups")
    if set(residuals.groupby("sample").size().unique()) != {21}:
        raise RuntimeError("each heatmap sample must contain all 21 organ pairs")
    if set(parameters.groupby("sample").size().unique()) != {7}:
        raise RuntimeError("each SEM sample must contain seven organ parameters")
    requested = {sample for sample, _, _ in NETWORK_SPECS}
    if not requested.issubset(set(residuals["sample"])):
        raise RuntimeError("one or more requested network samples are missing")
    return residuals, fits, parameters


def save_component(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    tiff_path = stem.with_suffix(".tiff")
    export_options = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.015,
    }
    figure.savefig(png_path, dpi=600, **export_options)
    create_derived_exports(
        png_path=png_path,
        svg_path=svg_path,
        pdf_path=pdf_path,
        tiff_path=tiff_path,
    )
    return [svg_path, pdf_path, png_path, tiff_path]


def render_heatmap(residuals: pd.DataFrame) -> list[Path]:
    figure = plt.figure(figsize=(7.204724, 4.921260))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[34, 1],
        left=0.105,
        right=0.965,
        bottom=0.205,
        top=0.920,
        wspace=0.075,
    )
    heat_axis = figure.add_subplot(grid[0, 0])
    color_axis = figure.add_subplot(grid[0, 1])
    draw_heatmap(heat_axis, color_axis, residuals)
    paths = save_component(
        figure, OUTPUT / "Figure6_heatmap_full_cohort_and_16_groups"
    )
    plt.close(figure)
    return paths


def fit_summary(
    sample: str, fits: pd.DataFrame, parameters: pd.DataFrame
) -> tuple[int, float, float, float]:
    fit = fits.loc[fits["sample"].eq(sample)].iloc[0]
    params = parameters.loc[parameters["sample"].eq(sample)]
    mean_shared = 100.0 * float(params["shared_variance_fraction"].mean())
    return int(fit["n"]), mean_shared, float(fit["cfi"]), float(fit["rmsea"])


def draw_network_component(
    axis: plt.Axes,
    sample: str,
    title: str,
    residuals: pd.DataFrame,
    fits: pd.DataFrame,
    parameters: pd.DataFrame,
    global_max: float,
) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-2.36, 2.36)
    axis.set_ylim(-2.20, 1.94)
    axis.axis("off")
    axis.text(
        0.5,
        1.075,
        title,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.3,
        fontweight="bold",
        color=DARK,
    )
    n, mean_shared, cfi, rmsea = fit_summary(sample, fits, parameters)
    axis.text(
        0.5,
        1.012,
        (
            f"n={n:,}  |  mean shared variance={mean_shared:.1f}%  |  "
            f"CFI={cfi:.3f}, RMSEA={rmsea:.3f}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )

    node_radius = 0.320
    displayed = residuals.loc[
        residuals["sample"].eq(sample) & residuals["display_edge"]
    ].copy()
    for _, row in displayed.sort_values("pooled_abs_rank", ascending=False).iterrows():
        first = NETWORK_POSITIONS[row["organ_1"]]
        second = NETWORK_POSITIONS[row["organ_2"]]
        start, end = edge_endpoints(first, second, node_radius)
        value = float(row["pooled_standardized_covariance_residual"])
        width = 0.72 + 6.2 * abs(value) / global_max
        curvature = edge_curvature(row["organ_1"], row["organ_2"])
        connection_style = f"arc3,rad={1.55 * curvature:.3f}"
        for color, line_width, zorder in (
            ("white", width + 2.40, 1),
            (RED if value > 0 else BLUE, width, 2),
        ):
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-",
                    connectionstyle=connection_style,
                    color=color,
                    linewidth=line_width,
                    alpha=1.0 if color == "white" else 0.96,
                    capstyle="round",
                    zorder=zorder,
                )
            )

    for organ in ORGANS:
        axis.add_patch(
            Circle(
                NETWORK_POSITIONS[organ],
                radius=node_radius,
                facecolor=NEUTRAL,
                edgecolor=DARK,
                linewidth=1.10,
                zorder=5,
            )
        )
        axis.text(
            *NETWORK_POSITIONS[organ],
            ORGAN_LABELS[organ],
            ha="center",
            va="center",
            fontsize=7.0,
            color=DARK,
            zorder=6,
        )


def network_global_max(residuals: pd.DataFrame) -> float:
    samples = {sample for sample, _, _ in NETWORK_SPECS}
    return float(
        residuals.loc[
            residuals["sample"].isin(samples) & residuals["display_edge"],
            "pooled_standardized_covariance_residual",
        ]
        .abs()
        .max()
    )


def render_network_sheet(
    residuals: pd.DataFrame, fits: pd.DataFrame, parameters: pd.DataFrame
) -> list[Path]:
    global_max = network_global_max(residuals)
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 8.40))
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.015,
        top=0.940,
        wspace=-0.020,
        hspace=0.03125,
    )
    for axis, (sample, title, _) in zip(axes.flat, NETWORK_SPECS):
        draw_network_component(
            axis, sample, title, residuals, fits, parameters, global_max
        )
    handles = [
        Line2D([0], [0], color=RED, linewidth=3.0, label="Positive residual covariance"),
        Line2D([0], [0], color=BLUE, linewidth=3.0, label="Negative residual covariance"),
    ]
    figure.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.518),
        ncol=1,
        frameon=False,
        fontsize=6.5,
        handlelength=3.2,
        labelspacing=0.45,
    )
    paths = save_component(figure, OUTPUT / "Figure6_networks_2x2")
    plt.close(figure)
    return paths


def render_individual_networks(
    residuals: pd.DataFrame, fits: pd.DataFrame, parameters: pd.DataFrame
) -> list[Path]:
    global_max = network_global_max(residuals)
    outputs: list[Path] = []
    for sample, title, slug in NETWORK_SPECS:
        figure = plt.figure(figsize=(4.55, 4.10))
        axis = figure.add_axes([0.045, 0.030, 0.910, 0.840])
        draw_network_component(
            axis, sample, title, residuals, fits, parameters, global_max
        )
        outputs.extend(
            save_component(figure, OUTPUT / f"Figure6_network_{slug}")
        )
        plt.close(figure)
    return outputs


def render_legend() -> list[Path]:
    figure = plt.figure(figsize=(2.834646, 0.787402))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    handles = [
        Line2D([0], [0], color=RED, linewidth=3.0, label="Positive residual covariance"),
        Line2D([0], [0], color=BLUE, linewidth=3.0, label="Negative residual covariance"),
    ]
    axis.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=6.2,
        handlelength=3.2,
        labelspacing=0.50,
    )
    paths = save_component(figure, OUTPUT / "Figure6_network_legend")
    plt.close(figure)
    return paths


def main() -> int:
    set_component_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    residuals, fits, parameters = load_sources()
    outputs: list[Path] = []
    outputs.extend(render_heatmap(residuals))
    outputs.extend(render_network_sheet(residuals, fits, parameters))
    outputs.extend(render_individual_networks(residuals, fits, parameters))
    outputs.extend(render_legend())
    report = {
        "status": "READY_FOR_MANUAL_ASSEMBLY",
        "heatmap_columns": 17,
        "heatmap_pairs": 21,
        "network_samples": [sample for sample, _, _ in NETWORK_SPECS],
        "global_network_scale": True,
        "formats": ["svg", "pdf", "png_600dpi", "tiff_600dpi"],
        "outputs": [path.name for path in outputs],
    }
    (OUTPUT / "SPLIT_EXPORT_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
