from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[4]
FINAL = ROOT / "Revision-0710/Final_Fig_0716"
FIG13 = FINAL / "Fig13"
FIG14 = FINAL / "Fig14"
WORK = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/04_external_validation/core_0716"
)
ORGANS = [
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
]
LABEL = {
    "cardiovascular": "Cardiovascular",
    "hepatic": "Hepatic",
    "immune": "Immune",
    "metabolic": "Metabolic",
    "musculoskeletal": "Musculoskeletal",
    "pulmonary": "Pulmonary",
    "renal": "Renal",
}
SHORT = {
    "cardiovascular": "Cardio.",
    "hepatic": "Hepatic",
    "immune": "Immune",
    "metabolic": "Metabolic",
    "musculoskeletal": "Musculo.",
    "pulmonary": "Pulm.",
    "renal": "Renal",
}
COLORS = {
    "cardiovascular": "#E64B35",
    "hepatic": "#00A087",
    "immune": "#F39B7F",
    "metabolic": "#8491B4",
    "musculoskeletal": "#7E6148",
    "pulmonary": "#4DBBD5",
    "renal": "#3C5488",
}
BLUE = "#4E79A7"
ORANGE = "#E69F00"
TEAL = "#2A9D8F"
GRAY = "#BDBDBD"
TEXT = "#222222"
RELEASE_LABEL = {
    "P_2017_MARCH2020_PREPANDEMIC": "P release",
    "L_AUG2021_AUG2023": "L release",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def setup_style() -> None:
    sns.set_theme(style="ticks", font="Times New Roman", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )


def format_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.tick_params(labelsize=8, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle=":", linewidth=0.6, alpha=0.45)
    sns.despine(ax=ax)


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    return outputs


def save_panels(
    fig: plt.Figure,
    axes: np.ndarray,
    directory: Path,
    prefix: str,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outputs: list[Path] = []
    for label, ax in zip("ABCDEF", axes.ravel()):
        boxes = [ax.get_tightbbox(renderer)]
        boxes.extend(child.get_tightbbox(renderer) for child in ax.child_axes)
        bbox = matplotlib.transforms.Bbox.union(boxes)
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.06, 1.08)
        for extension in ("png", "pdf", "svg"):
            path = directory / f"{prefix}{label}.{extension}"
            fig.savefig(
                path,
                dpi=300,
                bbox_inches=bbox,
                facecolor="white",
            )
            outputs.append(path)
    return outputs


def mimic_display_flow() -> pd.DataFrame:
    flow = pd.read_csv(FIG13 / "Source_Data/Fig13A_mimic_cohort_flow.csv")
    primary = flow.loc[flow["step_order"].isin([1, 5, 7])].drop_duplicates(
        "step_order"
    )
    age_metrics = pd.read_csv(
        FIG13 / "Source_Data/Fig13B_raw_age_metrics_checkpoint_range.csv"
    )
    age_n = int(age_metrics["n_participants"].max())
    receipt = json.loads(
        (
            WORK
            / "shared/05_outcomes_after_prediction_seal/OUTCOME_READ_RECEIPT.json"
        ).read_text(encoding="utf-8")
    )
    events = pd.read_csv(
        FIG13 / "Source_Data/MIMIC_incident_recurrent_event_counts.csv"
    )
    incident = events.loc[
        events["analysis_role"].eq("incident_primary")
        & events["horizon_months"].eq(12)
    ]
    rows = [
        ("Raw MIMIC-IV subjects", int(primary.loc[primary.step_order.eq(1), "n_subjects"].iloc[0]), "construction"),
        ("Valid adult survived discharge", int(primary.loc[primary.step_order.eq(5), "n_subjects"].iloc[0]), "construction"),
        ("Second-admission landmark", int(primary.loc[primary.step_order.eq(7), "n_subjects"].iloc[0]), "shared_landmark"),
        ("Age analysis: 40-73 years", age_n, "age_analysis"),
        ("Risk analysis: endpoint eligible", int(receipt["endpoint_eligible_subjects"]), "risk_analysis"),
        ("12-month incident cohorts (max)", int(incident["n"].max()), "risk_analysis"),
    ]
    result = pd.DataFrame(rows, columns=["flow_step", "n_subjects", "population_role"])
    result["display_order"] = np.arange(1, len(result) + 1)
    write_csv(FIG13 / "Source_Data/Fig13A_display_flow.csv", result)
    return result


def plot_mimic_flow(ax: plt.Axes, flow: pd.DataFrame) -> None:
    plot = flow.iloc[::-1]
    colors = [TEAL, TEAL, BLUE, BLUE, ORANGE, ORANGE][::-1]
    ax.barh(plot["flow_step"], plot["n_subjects"], color=colors, height=0.65)
    ax.set_xlabel("Participants (n)", fontsize=9)
    ax.set_title("Cohort construction and analysis populations", loc="left", fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    for y, value in zip(plot["flow_step"], plot["n_subjects"]):
        ax.text(value + 6000, y, f"{int(value):,}", va="center", fontsize=7.5)
    ax.set_xlim(0, max(flow["n_subjects"]) * 1.18)
    ax.text(
        0.0,
        -0.18,
        "Age and risk analyses are parallel branches after the shared landmark.",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    format_axis(ax, "x")


def pivot_age_metrics(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    point = data.pivot_table(
        index=["release_id", "organ_system"],
        columns="metric",
        values="mean5_estimate",
        aggfunc="first",
    ).reset_index()
    mae_range = data.loc[data["metric"].eq("mae")][
        ["release_id", "organ_system", "checkpoint_min", "checkpoint_max"]
    ]
    return point.merge(
        mae_range,
        on=["release_id", "organ_system"],
        how="left",
        validate="one_to_one",
    )


def plot_mimic_age_metrics(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    data = metrics.set_index("organ_system").reindex(ORGANS).reset_index()
    y = np.arange(len(data))
    data_min = min(data["mae"].min(), data["checkpoint_min"].min())
    data_max = max(data["mae"].max(), data["checkpoint_max"].max())
    span = data_max - data_min
    left = data_min - 0.08 * span
    right = data_max + 0.20 * span
    label_x = data_max + 0.16 * span
    for index, row in data.iterrows():
        ax.hlines(
            index,
            row["checkpoint_min"],
            row["checkpoint_max"],
            color=COLORS[row["organ_system"]],
            linewidth=1.2,
        )
        ax.plot(
            row["mae"],
            index,
            marker="o",
            color=COLORS[row["organ_system"]],
            markersize=5,
        )
        ax.text(
            label_x,
            index,
            f"r={row['pearson']:.2f}; $R^2$={row['r2']:.2f}",
            fontsize=7,
            va="center",
            ha="right",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[organ] for organ in data["organ_system"]])
    ax.invert_yaxis()
    ax.set_xlabel("Raw zero-shot MAE (years)", fontsize=9)
    ax.set_xlim(left, right)
    ax.set_title("Age transportability (40-73 years)", loc="left", fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        -0.18,
        "Horizontal lines: range across five fixed checkpoints; points: mean-5 ensemble",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    format_axis(ax, "x")


def macro_calibration(path: Path, partition_column: str = "partition") -> pd.DataFrame:
    data = pd.read_csv(path)
    summary = (
        data.groupby([partition_column, "age_bin"], as_index=False)
        .agg(
            chronological_age_mean=("chronological_age_mean", "mean"),
            predicted_age_mean=("predicted_age_mean5_mean", "mean"),
            predicted_min=("predicted_age_mean5_mean", "min"),
            predicted_max=("predicted_age_mean5_mean", "max"),
            n=("n", "sum"),
        )
        .sort_values([partition_column, "chronological_age_mean"])
    )
    return summary


def plot_mimic_calibration(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = summary["chronological_age_mean"].to_numpy(dtype=float)
    y = summary["predicted_age_mean"].to_numpy(dtype=float)
    ax.fill_between(
        x,
        summary["predicted_min"],
        summary["predicted_max"],
        color=BLUE,
        alpha=0.16,
        linewidth=0,
        label="Organ range",
    )
    ax.plot(x, y, marker="o", color=BLUE, linewidth=1.8, markersize=4, label="Macro mean")
    ax.plot([40, 73], [40, 73], linestyle="--", color="#777777", linewidth=1.0, label="Ideal")
    ax.set_xlim(40, 73)
    ax.set_ylim(40, 73)
    ax.set_xlabel("Chronological age (years)", fontsize=9)
    ax.set_ylabel("Predicted age (years)", fontsize=9)
    ax.set_title("Raw age calibration", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    format_axis(ax, "both")


def plot_incremental_risk(ax: plt.Axes, data: pd.DataFrame) -> None:
    ax.axis("off")
    selected = data.loc[data["horizon_months"].eq(12)].copy()
    metric_info = [
        ("harrell_c", "$\\Delta$C"),
        ("ipcw_auc", "$\\Delta$AUC"),
        ("ipcw_brier", "Brier imp."),
    ]
    for column, (metric, title) in enumerate(metric_info):
        inset = ax.inset_axes([0.01 + column * 0.33, 0.05, 0.31, 0.86])
        metric_rows = selected.loc[selected["metric"].eq(metric)]
        for comparison, color, marker, offset in (
            ("age_sex_plus_gap", BLUE, "o", -0.10),
            ("age_sex_density_plus_gap", ORANGE, "s", 0.10),
        ):
            values = metric_rows.loc[metric_rows["comparison"].eq(comparison)].set_index(
                "organ_system"
            ).reindex(ORGANS)
            y = np.arange(len(ORGANS)) + offset
            lower = values["delta"] - values["delta_ci_lower"]
            upper = values["delta_ci_upper"] - values["delta"]
            inset.errorbar(
                values["delta"],
                y,
                xerr=np.vstack([lower, upper]),
                fmt=marker,
                color=color,
                ecolor=color,
                markersize=3.5,
                capsize=1.5,
                linewidth=0.8,
                label=("Age+sex" if comparison == "age_sex_plus_gap" else "Age+sex+density"),
            )
        inset.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        inset.set_title(title, fontsize=8.5, fontweight="bold")
        inset.set_ylim(-0.6, len(ORGANS) - 0.4)
        inset.set_yticks(np.arange(len(ORGANS)))
        if column == 0:
            inset.set_yticklabels([SHORT[o] for o in ORGANS], fontsize=6.8)
        else:
            inset.set_yticklabels([])
        inset.invert_yaxis()
        inset.tick_params(axis="x", labelsize=6.5)
        if metric == "ipcw_brier":
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            inset.xaxis.set_major_formatter(formatter)
        format_axis(inset, "x")
        if column == 1:
            inset.legend(
                frameon=False,
                fontsize=6.2,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.25),
            )
    ax.set_title(
        "12-month paired OOF incremental utility", loc="left", fontsize=10, fontweight="bold", pad=2
    )


def plot_mimic_hr(ax: plt.Axes, data: pd.DataFrame) -> None:
    selected = (
        data.loc[data["horizon_months"].eq(12)]
        .set_index("organ_system")
        .reindex(ORGANS)
        .reset_index()
    )
    y = np.arange(len(selected))
    for index, row in selected.iterrows():
        lower = row["hr_per_5y_gap"] - row["hr_per_5y_ci_lower"]
        upper = row["hr_per_5y_ci_upper"] - row["hr_per_5y_gap"]
        ax.errorbar(
            row["hr_per_5y_gap"],
            index,
            xerr=np.array([[lower], [upper]]),
            fmt="o",
            color=COLORS[row["organ_system"]],
            ecolor=COLORS[row["organ_system"]],
            markersize=4.5,
            capsize=2,
        )
    ax.axvline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[o] for o in selected["organ_system"]])
    ax.invert_yaxis()
    ax.set_xlabel("Cause-specific HR per +5 years gap", fontsize=9)
    ax.set_title("12-month incident associations", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlim(0.86, 1.26)
    format_axis(ax, "x")


def plot_mimic_risk_calibration(ax: plt.Axes, data: pd.DataFrame) -> None:
    macro = (
        data.groupby("risk_decile", as_index=False)
        .agg(
            predicted=("predicted_risk_mean", "mean"),
            observed=("ipcw_observed_risk", "mean"),
            observed_min=("ipcw_observed_risk", "min"),
            observed_max=("ipcw_observed_risk", "max"),
        )
        .sort_values("risk_decile")
    )
    write_csv(FIG13 / "Source_Data/Fig13F_macro_risk_calibration.csv", macro)
    ax.fill_between(
        macro["predicted"],
        macro["observed_min"],
        macro["observed_max"],
        color=TEAL,
        alpha=0.16,
        linewidth=0,
        label="Organ range",
    )
    ax.plot(
        macro["predicted"],
        macro["observed"],
        marker="o",
        color=TEAL,
        linewidth=1.8,
        markersize=4,
        label="Macro mean",
    )
    upper = max(macro["predicted"].max(), macro["observed_max"].max()) * 1.08
    ax.plot([0, upper], [0, upper], color="#777777", linestyle="--", linewidth=1.0, label="Ideal")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Predicted 12-month risk", fontsize=9)
    ax.set_ylabel("IPCW-observed risk", fontsize=9)
    ax.set_title("OOF risk calibration", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    format_axis(ax, "both")


def render_fig13() -> list[Path]:
    source = FIG13 / "Source_Data"
    flow = mimic_display_flow()
    metrics = pivot_age_metrics(source / "Fig13B_raw_age_metrics_checkpoint_range.csv")
    calibration = macro_calibration(source / "Fig13C_raw_age_calibration_bins.csv")
    write_csv(source / "Fig13C_macro_age_calibration.csv", calibration)
    risk = pd.read_csv(source / "Fig13D_paired_incremental_risk_metrics.csv")
    hr = pd.read_csv(source / "Fig13E_cause_specific_hr.csv")
    risk_calibration = pd.read_csv(source / "Fig13F_risk_calibration_deciles.csv")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.8), dpi=300)
    plot_mimic_flow(axes[0, 0], flow)
    plot_mimic_age_metrics(axes[0, 1], metrics)
    plot_mimic_calibration(axes[0, 2], calibration)
    plot_incremental_risk(axes[1, 0], risk)
    plot_mimic_hr(axes[1, 1], hr)
    plot_mimic_risk_calibration(axes[1, 2], risk_calibration)
    for label, ax in zip("ABCDEF", axes.ravel()):
        panel_label(ax, label)
    fig.suptitle(
        "MIMIC-IV external transportability and incident clinical utility",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    outputs = save_figure(fig, FIG13 / "Main_Figure", "Fig13_MIMIC_external_validation")
    outputs.extend(save_panels(fig, axes, FIG13 / "Panels", "Fig13"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), dpi=300)
    for panel_index, (ax, (metric, title)) in enumerate(zip(
        axes.ravel(),
        [
            ("harrell_c", "6-month $\\Delta$C"),
            ("ipcw_auc", "6-month $\\Delta$AUC"),
            ("ipcw_brier", "6-month Brier improvement"),
            ("harrell_c", "12-month $\\Delta$C"),
            ("ipcw_auc", "12-month $\\Delta$AUC"),
            ("ipcw_brier", "12-month Brier improvement"),
        ],
    )):
        horizon = 6 if panel_index < 3 else 12
        values = risk.loc[
            risk["horizon_months"].eq(horizon)
            & risk["metric"].eq(metric)
            & risk["comparison"].eq("age_sex_density_plus_gap")
        ].set_index("organ_system").reindex(ORGANS)
        y = np.arange(len(ORGANS))
        ax.errorbar(
            values["delta"],
            y,
            xerr=np.vstack(
                [values["delta"] - values["delta_ci_lower"], values["delta_ci_upper"] - values["delta"]]
            ),
            fmt="o",
            color=ORANGE,
            ecolor=ORANGE,
            capsize=2,
            markersize=4,
        )
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[o] for o in ORGANS], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, fontweight="bold")
        format_axis(ax, "x")
    fig.tight_layout()
    outputs.extend(
        save_figure(fig, FIG13 / "Supplementary", "Fig13S1_paired_risk_deltas")
    )
    plt.close(fig)

    aj = pd.read_csv(source / "Fig13F_aalen_johansen_overall.csv")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300, sharex=True)
    for ax, organ in zip(axes.ravel(), ORGANS):
        values = aj.loc[aj["organ_system"].eq(organ)]
        ax.plot(values["time_days"] / 30.4375, values["target_cif"], color=COLORS[organ], linewidth=1.8)
        ax.plot(values["time_days"] / 30.4375, values["death_cif"], color="#777777", linewidth=1.2, linestyle="--")
        ax.set_title(SHORT[organ], fontsize=9, fontweight="bold")
        ax.set_xlabel("Months", fontsize=8)
        ax.set_ylabel("Cumulative incidence", fontsize=8)
        format_axis(ax, "both")
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    outputs.extend(
        save_figure(fig, FIG13 / "Supplementary", "Fig13S2_Aalen_Johansen")
    )
    plt.close(fig)
    return outputs


def nhanes_flow_and_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    flow = pd.read_csv(FIG14 / "Source_Data/Fig14A_release_cohort_flow.csv")
    qc = pd.read_csv(FIG14 / "Source_Data/Fig14B_release_slot_qc.csv")
    measured = (
        qc.assign(coverage_fraction=qc["n_observed"] / qc["n_eligible"])
        .groupby("release_id", as_index=False)
        .agg(
            slots_measured=("n_observed", lambda value: int((value > 0).sum())),
            median_slot_coverage=("coverage_fraction", "median"),
        )
    )
    flow = flow.merge(measured, on="release_id", how="left", validate="one_to_one")
    write_csv(FIG14 / "Source_Data/Fig14A_display_flow.csv", flow)
    heatmap = qc[
        ["release_id", "slot_index", "feature_name", "n_eligible", "n_observed", "qc_status"]
    ].copy()
    heatmap["coverage_fraction"] = heatmap["n_observed"] / heatmap["n_eligible"]
    write_csv(FIG14 / "Source_Data/Fig14B_coverage_heatmap.csv", heatmap)
    return flow, heatmap


def plot_nhanes_flow(ax: plt.Axes, flow: pd.DataFrame) -> None:
    release_order = ["P_2017_MARCH2020_PREPANDEMIC", "L_AUG2021_AUG2023"]
    values = flow.set_index("release_id").reindex(release_order).reset_index()
    bars = ax.bar(
        [RELEASE_LABEL[r] for r in values["release_id"]],
        values["n"],
        color=[BLUE, TEAL],
        width=0.58,
    )
    for bar, row in zip(bars, values.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 90,
            f"n={int(row.n):,}\n{int(row.slots_measured)}/78 slots",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("MEC examined age 40-73 (n)", fontsize=9)
    ax.set_ylim(0, max(values["n"]) * 1.24)
    ax.set_title("Release-specific external cohorts", loc="left", fontsize=10, fontweight="bold")
    format_axis(ax, "y")


def plot_nhanes_heatmap(ax: plt.Axes, heatmap: pd.DataFrame) -> None:
    release_order = ["P_2017_MARCH2020_PREPANDEMIC", "L_AUG2021_AUG2023"]
    matrix = (
        heatmap.pivot(index="release_id", columns="slot_index", values="coverage_fraction")
        .reindex(release_order)
        .reindex(columns=range(78))
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Observed fraction", "shrink": 0.75},
        xticklabels=10,
        yticklabels=["P", "L"],
    )
    ax.set_xlabel("Frozen UKB phenotype slot index", fontsize=9)
    ax.set_ylabel("Release", fontsize=9)
    ax.set_title("78-slot measurement coverage", loc="left", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)


def plot_nhanes_age_metrics(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    release_order = ["P_2017_MARCH2020_PREPANDEMIC", "L_AUG2021_AUG2023"]
    y = np.arange(len(ORGANS))
    for release, color, marker, offset in (
        (release_order[0], BLUE, "o", -0.11),
        (release_order[1], TEAL, "s", 0.11),
    ):
        values = metrics.loc[metrics["release_id"].eq(release)].set_index(
            "organ_system"
        ).reindex(ORGANS)
        ax.hlines(
            y + offset,
            values["checkpoint_min"],
            values["checkpoint_max"],
            color=color,
            linewidth=1.1,
        )
        ax.plot(
            values["mae"],
            y + offset,
            linestyle="none",
            marker=marker,
            color=color,
            markersize=4.5,
            label=RELEASE_LABEL[release],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[o] for o in ORGANS])
    ax.invert_yaxis()
    ax.set_xlabel("Raw zero-shot MAE (years)", fontsize=9)
    data_min = min(metrics["mae"].min(), metrics["checkpoint_min"].min())
    data_max = max(metrics["mae"].max(), metrics["checkpoint_max"].max())
    span = data_max - data_min
    ax.set_xlim(data_min - 0.10 * span, data_max + 0.10 * span)
    ax.set_title("Age transportability by release", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    format_axis(ax, "x")


def plot_nhanes_calibration(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for release, color, marker in (
        ("P_2017_MARCH2020_PREPANDEMIC", BLUE, "o"),
        ("L_AUG2021_AUG2023", TEAL, "s"),
    ):
        values = summary.loc[summary["partition"].eq(release)].sort_values(
            "chronological_age_mean"
        )
        ax.plot(
            values["chronological_age_mean"],
            values["predicted_age_mean"],
            marker=marker,
            color=color,
            linewidth=1.7,
            markersize=4,
            label=RELEASE_LABEL[release],
        )
    ax.plot([40, 73], [40, 73], color="#777777", linestyle="--", linewidth=1.0, label="Ideal")
    ax.set_xlim(40, 73)
    ax.set_ylim(40, 73)
    ax.set_xlabel("Chronological age (years)", fontsize=9)
    ax.set_ylabel("Predicted age (years)", fontsize=9)
    ax.set_title("Raw release-specific calibration", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    format_axis(ax, "both")


def local_recalibration_summary(local: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    recal = (
        local.loc[
            local["record_type"].eq("heldout_metric")
            & local["metric"].isin(["mae", "pearson", "r2"])
        ]
        .groupby(["release_id", "organ_system", "metric"], as_index=False)
        .agg(
            recalibrated_mean=("estimate", "mean"),
            recalibrated_min=("estimate", "min"),
            recalibrated_max=("estimate", "max"),
        )
    )
    raw_long = raw.melt(
        id_vars=["release_id", "organ_system"],
        value_vars=["mae", "pearson", "r2"],
        var_name="metric",
        value_name="raw_mean5",
    )
    result = recal.merge(
        raw_long,
        on=["release_id", "organ_system", "metric"],
        how="left",
        validate="one_to_one",
    )
    result["uncertainty_label"] = "range across five held-out calibration folds"
    write_csv(FIG14 / "Source_Data/Fig14E_recalibration_summary.csv", result)
    return result


def plot_nhanes_recalibration(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.axis("off")
    for column, release in enumerate(
        ["P_2017_MARCH2020_PREPANDEMIC", "L_AUG2021_AUG2023"]
    ):
        inset = ax.inset_axes([0.02 + column * 0.49, 0.05, 0.46, 0.86])
        values = summary.loc[
            summary["release_id"].eq(release) & summary["metric"].eq("mae")
        ].set_index("organ_system").reindex(ORGANS)
        y = np.arange(len(ORGANS))
        for index in range(len(ORGANS)):
            inset.plot(
                [values.iloc[index]["raw_mean5"], values.iloc[index]["recalibrated_mean"]],
                [index, index],
                color="#AAAAAA",
                linewidth=0.8,
            )
        inset.scatter(values["raw_mean5"], y, color=GRAY, s=18, label="Raw", zorder=3)
        inset.scatter(
            values["recalibrated_mean"],
            y,
            color=BLUE if column == 0 else TEAL,
            marker="s",
            s=20,
            label="Cross-fit recal.",
            zorder=3,
        )
        inset.set_yticks(y)
        if column == 0:
            inset.set_yticklabels([SHORT[o] for o in ORGANS], fontsize=7)
        else:
            inset.set_yticklabels([])
        inset.invert_yaxis()
        inset.set_title(RELEASE_LABEL[release], fontsize=8.5, fontweight="bold")
        inset.set_xlabel("MAE (years)", fontsize=7.5)
        format_axis(inset, "x")
        if column == 1:
            inset.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("Release-local cross-fit recalibration sensitivity", loc="left", fontsize=10, fontweight="bold")


def plot_questionnaire_or(ax: plt.Axes, data: pd.DataFrame) -> None:
    selected = data.loc[data["estimability_status"].eq("estimable")].copy()
    y = np.arange(len(ORGANS))
    for release, color, marker, offset in (
        ("P_2017_MARCH2020_PREPANDEMIC", BLUE, "o", -0.11),
        ("L_AUG2021_AUG2023", TEAL, "s", 0.11),
    ):
        values = selected.loc[selected["release_id"].eq(release)].set_index(
            "organ_system"
        ).reindex(ORGANS)
        valid = values["estimate"].notna().to_numpy()
        lower = values.loc[valid, "estimate"] - values.loc[valid, "ci_lower"]
        upper = values.loc[valid, "ci_upper"] - values.loc[valid, "estimate"]
        ax.errorbar(
            values.loc[valid, "estimate"],
            y[valid] + offset,
            xerr=np.vstack([lower, upper]),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4,
            label=RELEASE_LABEL[release],
        )
    ax.axvline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[o] for o in ORGANS])
    ax.invert_yaxis()
    immune_y = ORGANS.index("immune")
    ax.text(
        1.0,
        immune_y,
        "Not estimable by design",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#666666",
    )
    ax.set_xlabel("Survey-weighted OR per 1 SD gap", fontsize=9)
    ax.set_title("Questionnaire associations", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    format_axis(ax, "x")


def render_fig14() -> list[Path]:
    source = FIG14 / "Source_Data"
    flow, heatmap = nhanes_flow_and_coverage()
    metrics = pivot_age_metrics(source / "Fig14C_raw_age_metrics_checkpoint_range.csv")
    calibration = macro_calibration(source / "Fig14D_raw_age_calibration_bins.csv")
    write_csv(source / "Fig14D_macro_age_calibration.csv", calibration)
    local = pd.read_csv(source / "Fig14E_local_crossfit_recalibration.csv")
    recal = local_recalibration_summary(local, metrics)
    questionnaire = pd.read_csv(source / "Fig14F_questionnaire_or.csv")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.8), dpi=300)
    plot_nhanes_flow(axes[0, 0], flow)
    plot_nhanes_heatmap(axes[0, 1], heatmap)
    plot_nhanes_age_metrics(axes[0, 2], metrics)
    plot_nhanes_calibration(axes[1, 0], calibration)
    plot_nhanes_recalibration(axes[1, 1], recal)
    plot_questionnaire_or(axes[1, 2], questionnaire)
    for label, ax in zip("ABCDEF", axes.ravel()):
        panel_label(ax, label)
    fig.suptitle(
        "NHANES release-specific phenotype transportability",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(w_pad=2.5, h_pad=2.8)
    outputs = save_figure(fig, FIG14 / "Main_Figure", "Fig14_NHANES_external_validation")
    outputs.extend(save_panels(fig, axes, FIG14 / "Panels", "Fig14"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    for column, metric in enumerate(("mae", "pearson", "r2")):
        for row, release in enumerate(
            ["P_2017_MARCH2020_PREPANDEMIC", "L_AUG2021_AUG2023"]
        ):
            ax = axes[row, column]
            values = recal.loc[
                recal["release_id"].eq(release) & recal["metric"].eq(metric)
            ].set_index("organ_system").reindex(ORGANS)
            y = np.arange(len(ORGANS))
            for index in range(len(ORGANS)):
                ax.plot(
                    [values.iloc[index]["raw_mean5"], values.iloc[index]["recalibrated_mean"]],
                    [index, index],
                    color="#AAAAAA",
                    linewidth=0.8,
                )
            ax.scatter(values["raw_mean5"], y, color=GRAY, s=18, label="Raw")
            ax.scatter(
                values["recalibrated_mean"],
                y,
                color=BLUE if row == 0 else TEAL,
                marker="s",
                s=20,
                label="Cross-fit recal.",
            )
            ax.set_yticks(y)
            ax.set_yticklabels([SHORT[o] for o in ORGANS], fontsize=7)
            ax.invert_yaxis()
            ax.set_title(
                f"{RELEASE_LABEL[release]}: {metric.upper() if metric != 'pearson' else 'Pearson r'}",
                fontsize=9,
                fontweight="bold",
            )
            format_axis(ax, "x")
            if row == 0 and column == 2:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    outputs.extend(
        save_figure(fig, FIG14 / "Supplementary", "Fig14S1_raw_vs_recalibrated_metrics")
    )
    plt.close(fig)
    return outputs


def main() -> int:
    setup_style()
    for directory in (FIG13, FIG14):
        for child in ("Main_Figure", "Supplementary", "Panels", "Source_Data"):
            (directory / child).mkdir(parents=True, exist_ok=True)
    outputs = [*render_fig13(), *render_fig14()]
    manifest = {
        "schema_version": "revision0710.external_validation.fig13_fig14.render.v1",
        "status": "FIG13_FIG14_RENDER_COMPLETE",
        "created_at": now_iso(),
        "style_reference": "Revision-0626 Final_Figure_Data_0708: Times New Roman, organ colors, white background, thin axes",
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    write_json(WORK / "shared/07_render/RENDER_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
