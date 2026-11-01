#!/usr/bin/env python3
"""Redraw Figure 7 with evaluation stages and age performance shown first."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Revision-0730" / "4-Data"
OUT = Path(__file__).resolve().parent / "outputs" / "combined"
STEM = "Figure7_external_transportability"

ORGANS = [
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
]
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
    "cardiovascular": "#D55E00",
    "hepatic": "#009E73",
    "immune": "#CC79A7",
    "metabolic": "#E69F00",
    "musculoskeletal": "#E76F51",
    "pulmonary": "#0072B2",
    "renal": "#2A9D8F",
}

INK = "#252525"
MID = "#6B7280"
GRID = "#DCE2E8"
DIRECT = "#6B7280"
ALIGNED = "#0072B2"
LOCAL = "#E69F00"
RELEASES = ["P_2017_MARCH2020_PREPANDEMIC", "L_AUG2021_AUG2023"]
RELEASE_LABELS = {
    RELEASES[0]: "2017–Mar 2020",
    RELEASES[1]: "Aug 2021–2023",
}
RELEASE_COLORS = {RELEASES[0]: "#0072B2", RELEASES[1]: "#009E73"}
STAGES = ["Direct transfer", "Support aligned", "Local recalibration"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 5.1,
            "axes.titlesize": 5.2,
            "axes.labelsize": 5.1,
            "xtick.labelsize": 5.1,
            "ytick.labelsize": 5.1,
            "legend.fontsize": 5.1,
            "legend.frameon": False,
            "axes.linewidth": 0.70,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.1, width=0.65)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.48, alpha=0.85)
        ax.set_axisbelow(True)


def panel_body(fig: plt.Figure, spec, label: str, title: str):
    grid = spec.subgridspec(2, 1, height_ratios=(0.12, 0.88), hspace=0.01)
    header = fig.add_subplot(grid[0, 0])
    header.set_axis_off()
    header.text(0.00, 0.58, label, fontsize=7.0, fontweight="bold", ha="left", va="center")
    header.text(0.085, 0.58, title, fontsize=6.1, fontweight="bold", ha="left", va="center")
    return grid[1, 0]


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "mimic_age": pd.read_csv(DATA / "MIMIC_age_transportability.csv"),
        "mimic_age_macro": pd.read_csv(DATA / "MIMIC_age_transportability_macro.csv"),
        "mimic_risk": pd.read_csv(DATA / "MIMIC_12m_delta_c.csv"),
        "mimic_calibration": pd.read_csv(DATA / "MIMIC_12m_risk_calibration.csv"),
        "mimic_hr": pd.read_csv(DATA / "MIMIC_12m_cause_specific_hr.csv"),
        "nhanes_age": pd.read_csv(DATA / "NHANES_age_transportability_stages.csv"),
        "nhanes_or": pd.read_csv(DATA / "NHANES_survey_weighted_questionnaire_or.csv"),
    }


def draw_mimic_age(fig: plt.Figure, spec, data: dict[str, pd.DataFrame]) -> None:
    body = panel_body(fig, spec, "A", "MIMIC-IV age performance")
    grid = body.subgridspec(2, 2, wspace=0.36, hspace=0.51)
    age = data["mimic_age"]
    macro = data["mimic_age_macro"]
    metrics = [
        ("mae", "MAE (years)", None),
        ("pearson", "Pearson r", None),
        ("r2", "R²", None),
        ("calibration_slope", "Age calibration slope", 1.0),
    ]
    stages = ["Direct transfer", "Support aligned"]
    stage_colors = [DIRECT, ALIGNED]
    for index, (metric, title, reference) in enumerate(metrics):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        macro_values = []
        for x, (stage, color) in enumerate(zip(stages, stage_colors)):
            group = age.loc[(age.stage == stage) & (age.metric == metric)]
            values = group.mean5_estimate.to_numpy(float)
            ax.vlines(x, values.min(), values.max(), color=color, lw=0.9, alpha=0.42, zorder=1)
            ax.scatter(np.full(len(values), x), values, s=8, color=color, alpha=0.30, edgecolor="none", zorder=2)
            value = float(macro.loc[(macro.stage == stage) & (macro.metric == metric), "mean5_estimate"].iloc[0])
            macro_values.append(value)
            ax.scatter(x, value, s=28, marker="D", color=color, edgecolor="white", linewidth=0.5, zorder=4)
            ax.annotate(
                f"{value:.3f}",
                (x, value),
                xytext=(5 if x == 0 else -5, 7 if x == 0 else -9),
                textcoords="offset points",
                ha="left" if x == 0 else "right",
                va="bottom" if x == 0 else "top",
                fontsize=5.0,
            )
        ax.plot([0, 1], macro_values, color=INK, lw=0.85, zorder=3)
        if reference is not None:
            ax.axhline(reference, color="#9AA3AB", ls="--", lw=0.7)
        labels = ["Frozen\ndirect", "Input-support\nadaptation"] if index >= 2 else ["", ""]
        ax.set_xticks([0, 1], labels)
        ax.set_title(title, fontsize=5.2, fontweight="bold", pad=2.2)
        ax.tick_params(axis="x", pad=3.0)
        ax.margins(x=0.10, y=0.25)
        clean_axis(ax, "y")
def draw_nhanes_age(fig: plt.Figure, spec, data: dict[str, pd.DataFrame]) -> None:
    body = panel_body(fig, spec, "D", "NHANES age performance")
    grid = body.subgridspec(3, 1, hspace=0.28)
    stage = data["nhanes_age"]
    metrics = [
        ("mae", "MAE (years)"),
        ("pearson", "Pearson r"),
        ("r2", "R²"),
    ]
    for index, (metric, ylabel) in enumerate(metrics):
        ax = fig.add_subplot(grid[index, 0])
        for release in RELEASES:
            group = stage.loc[(stage.release_id == release) & (stage.metric == metric)].set_index("stage").loc[STAGES]
            ax.plot(
                np.arange(3),
                group.estimate.to_numpy(float),
                marker="o",
                ms=3.3,
                lw=1.2,
                color=RELEASE_COLORS[release],
                label=RELEASE_LABELS[release],
            )
        labels = ["Frozen\ndirect", "Input-support\nadaptation", "Local\nrecalibration"] if index == 2 else ["", "", ""]
        ax.set_xticks(np.arange(3), labels)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", pad=3.0)
        ax.margins(x=0.05, y=0.18)
        clean_axis(ax, "y")
        if index == 0:
            ax.legend(loc="upper right", fontsize=5.1, handlelength=1.6)


def draw_nhanes_calibration(fig: plt.Figure, spec, data: dict[str, pd.DataFrame]) -> None:
    body = panel_body(fig, spec, "E", "NHANES age calibration")
    ax = fig.add_subplot(body)
    stage = data["nhanes_age"]
    for release in RELEASES:
        group = stage.loc[
            (stage.release_id == release) & (stage.metric == "calibration_slope")
        ].set_index("stage").loc[STAGES]
        ax.plot(
            np.arange(3),
            group.estimate.to_numpy(float),
            marker="o",
            ms=4.2,
            lw=1.35,
            color=RELEASE_COLORS[release],
            label=RELEASE_LABELS[release],
        )
        for x, value in enumerate(group.estimate.to_numpy(float)):
            ax.annotate(
                f"{value:.3f}",
                (x, value),
                xytext=(0, -12 if release == RELEASES[0] else 8),
                textcoords="offset points",
                ha="center",
                fontsize=5.0,
                color=RELEASE_COLORS[release],
            )
    ax.axhline(1.0, color="#7E8790", ls="--", lw=0.8)
    ax.set_xticks(
        np.arange(3),
        ["Frozen\ndirect", "Input-support\nadaptation", "Local\nrecalibration"],
    )
    ax.set_ylabel("Calibration slope")
    ax.tick_params(axis="x", pad=3.0)
    ax.margins(x=0.08, y=0.22)
    ax.legend(loc="lower right", fontsize=5.2, handlelength=1.6)
    clean_axis(ax, "y")


def draw_mimic_hr(fig: plt.Figure, spec, data: dict[str, pd.DataFrame]) -> None:
    body = panel_body(fig, spec, "C", "MIMIC-IV organ associations")
    ax = fig.add_subplot(body)
    ax.text(
        0,
        1.025,
        "Input-support adaptation; 12-month outcomes",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )
    hr = data["mimic_hr"].copy()
    hr["order"] = hr.organ_system.map({organ: i for i, organ in enumerate(ORGANS)})
    hr = hr.sort_values("order")
    y = np.arange(len(hr))[::-1]
    for yy, (_, row) in zip(y, hr.iterrows()):
        organ = row.organ_system
        point = float(row.hr_per_1sd_gap)
        low = float(row.hr_per_1sd_ci_lower)
        high = float(row.hr_per_1sd_ci_upper)
        ax.errorbar(point, yy, xerr=[[point - low], [high - point]], fmt="o", ms=4.0, color=ORGAN_COLORS[organ], ecolor=ORGAN_COLORS[organ], elinewidth=1.05, capsize=2.0)
        ax.text(high + 0.012, yy, f"{point:.2f} ({low:.2f}–{high:.2f})", va="center", fontsize=5.0)
    ax.axvline(1, color="#7E8790", ls="--", lw=0.8)
    ax.set_yticks(y, [ORGAN_LABELS[o] for o in hr.organ_system])
    ax.set_xlim(0.98, float(hr.hr_per_1sd_ci_upper.max()) + 0.20)
    ax.set_xticks(np.arange(1.0, 1.81, 0.2))
    ax.set_xlabel("HR per 1-SD corresponding OSAG")
    clean_axis(ax, "x")


def draw_mimic_prediction(fig: plt.Figure, spec, data: dict[str, pd.DataFrame]) -> None:
    body = panel_body(fig, spec, "B", "MIMIC-IV 12-month prediction")
    grid = body.subgridspec(2, 1, height_ratios=(0.94, 1.06), hspace=0.72)
    risk = data["mimic_risk"]
    calibration = data["mimic_calibration"]

    ax = fig.add_subplot(grid[0, 0])
    comparisons = ["age_sex_plus_gap", "age_sex_density_plus_gap"]
    labels = ["Age + sex", "+ record\ndensity"]
    ypos = np.array([1.0, 0.0])
    for stage_index, (stage, color, marker) in enumerate(
        [("Direct transfer", DIRECT, "s"), ("Support aligned", ALIGNED, "o")]
    ):
        for comp_index, comp in enumerate(comparisons):
            group = risk.loc[(risk.stage == stage) & (risk.comparison == comp)]
            y = ypos[comp_index] + (0.10 if stage_index else -0.10)
            ax.scatter(group.delta, np.full(len(group), y), s=7, color=color, alpha=0.34, edgecolor="none")
            mean = float(group.delta.mean())
            ax.scatter(mean, y, s=30, color=color, marker=marker, edgecolor="white", linewidth=0.5, zorder=4)
            ax.text(mean, y + (0.15 if stage_index else -0.15), f"{mean:+.4f}", color=color, ha="center", va="center", fontsize=5.0)
    ax.axvline(0, color="#7E8790", ls="--", lw=0.75)
    ax.set_xlim(-0.012, 0.046)
    ax.set_ylim(-0.42, 1.42)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("Macro ΔHarrell C")
    ax.set_title("12-month discrimination", loc="left", fontsize=5.2, fontweight="bold", pad=2.2)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=DIRECT, label="Frozen direct"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ALIGNED, label="Input-support adaptation"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=5.1, handletextpad=0.3)
    clean_axis(ax, "x")

    ax = fig.add_subplot(grid[1, 0])
    for stage, color, label in [
        ("Direct transfer", DIRECT, "Frozen direct"),
        ("Support aligned", ALIGNED, "Input-support adaptation"),
    ]:
        group = calibration.loc[calibration.stage == stage]
        ax.plot(group.predicted, group.observed, marker="o", ms=2.3, lw=1.05, color=color, label=label)
    limit = [0, max(calibration.predicted.max(), calibration.observed.max()) * 1.08]
    ax.plot(limit, limit, color="#7E8790", ls="--", lw=0.7)
    ax.set_xlim(limit)
    ax.set_ylim(limit)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed risk")
    ax.set_title("12-month risk calibration", loc="left", fontsize=5.2, fontweight="bold", pad=2.2)
    clean_axis(ax, "both")


def draw_nhanes_or(fig: plt.Figure, spec, data: dict[str, pd.DataFrame]) -> None:
    body = panel_body(fig, spec, "F", "NHANES questionnaire associations")
    ax = fig.add_subplot(body)
    ax.text(
        0,
        1.025,
        "Input-support adaptation; no external labels used",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MID,
    )
    ors = data["nhanes_or"]
    estimable = ors.loc[ors.estimability_status == "estimable"]
    log_scale_values = estimable[["estimate", "ci_lower", "ci_upper"]].astype(float).to_numpy()
    if np.any(log_scale_values <= 0):
        raise ValueError("Log-scaled NHANES association estimates and intervals must be positive")
    ybase = np.arange(len(ORGANS))[::-1]
    for release_index, release in enumerate(RELEASES):
        group = ors.loc[ors.release_id == release].set_index("organ_system")
        offset = 0.11 if release_index == 0 else -0.11
        for yy, organ in zip(ybase, ORGANS):
            row = group.loc[organ]
            if row.estimability_status != "estimable":
                continue
            point = float(row.estimate)
            low = float(row.ci_lower)
            high = float(row.ci_upper)
            ax.errorbar(point, yy + offset, xerr=[[point - low], [high - point]], fmt="o", ms=3.8, color=RELEASE_COLORS[release], ecolor=RELEASE_COLORS[release], elinewidth=1.0, capsize=1.9)
    ax.axvline(1, color="#7E8790", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.75, 4.2)
    ax.set_xticks([0.8, 1.0, 2.0, 3.0, 4.0], ["0.8", "1", "2", "3", "4"])
    ax.minorticks_off()
    ax.set_yticks(ybase, [ORGAN_LABELS[o] for o in ORGANS])
    ax.set_xlabel("Survey-weighted OR per 1-SD corresponding OSAG")
    ax.text(1.02, ybase[ORGANS.index("immune")], "not estimable", ha="left", va="center", fontsize=5.0, color=MID)
    handles = [
        Line2D([0], [0], marker="o", color=RELEASE_COLORS[r], lw=1.1, ms=3.7, label=RELEASE_LABELS[r])
        for r in RELEASES
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.2)
    clean_axis(ax, "x")


def draw_stage_contract(fig: plt.Figure, spec) -> None:
    body = panel_body(fig, spec, "F", "External evaluation stages")
    ax = fig.add_subplot(body)
    ax.set_axis_off()
    rows = [
        (
            DIRECT,
            "Frozen direct transfer",
            "Frozen UKB model applied unchanged",
            "No target labels used for adaptation",
            "Strict external\nvalidation",
        ),
        (
            ALIGNED,
            "Input-support adaptation",
            "UKB model trained on target-matched inputs",
            "No target labels used for adaptation",
            "Label-free input\nadaptation",
        ),
        (
            LOCAL,
            "Local age recalibration",
            "Cross-fitted mapping using target-cohort ages",
            "Target disease labels not used",
            "Labelled adaptation\n(NHANES only)",
        ),
    ]
    y_positions = [0.78, 0.47, 0.16]
    for (color, title, operation, labels, interpretation), y in zip(rows, y_positions):
        patch = FancyBboxPatch(
            (0.02, y - 0.105),
            0.96,
            0.22,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=color,
            alpha=0.10,
            edgecolor=color,
            linewidth=0.9,
        )
        ax.add_patch(patch)
        ax.add_patch(
            FancyBboxPatch(
                (0.035, y - 0.088),
                0.018,
                0.176,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
        )
        ax.text(0.072, y + 0.060, title, transform=ax.transAxes, ha="left", va="center", fontsize=6.9, fontweight="bold", color=color)
        ax.text(0.072, y + 0.010, operation, transform=ax.transAxes, ha="left", va="center", fontsize=5.8)
        ax.text(0.072, y - 0.040, labels, transform=ax.transAxes, ha="left", va="center", fontsize=5.8)
        ax.text(0.965, y, interpretation, transform=ax.transAxes, ha="right", va="center", fontsize=5.7, fontweight="bold", color=color)
    ax.text(0.02, 0.98, "Only frozen direct transfer is the strict external validation stage.", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, fontweight="bold")


def main() -> None:
    configure_style()
    data = load_data()
    OUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.20, 4.22))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.992,
        top=0.975,
        bottom=0.075,
        wspace=0.26,
        hspace=0.16,
    )
    draw_mimic_age(fig, grid[0, 0], data)
    draw_mimic_prediction(fig, grid[0, 1], data)
    draw_mimic_hr(fig, grid[0, 2], data)
    draw_nhanes_age(fig, grid[1, 0], data)
    draw_nhanes_calibration(fig, grid[1, 1], data)
    draw_nhanes_or(fig, grid[1, 2], data)

    fig.savefig(OUT / f"{STEM}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{STEM}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{STEM}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{STEM}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(OUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
