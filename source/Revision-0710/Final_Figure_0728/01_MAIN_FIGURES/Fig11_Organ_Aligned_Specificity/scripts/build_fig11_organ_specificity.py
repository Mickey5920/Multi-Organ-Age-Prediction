#!/usr/bin/env python3
"""Build the concise main-text candidate for organ-aligned specificity evidence.

This script only redraws previously audited statistics. It does not train, refit,
or rescore the neural network or survival models.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT = Path(__file__).resolve()


def find_repo_root() -> Path:
    for candidate in (SCRIPT.parent, *SCRIPT.parents):
        if (candidate / "Revision-0710").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


ROOT = find_repo_root()
OUTPUT_ROOT = SCRIPT.parent.parent
SOURCE_DIR = OUTPUT_ROOT / "source_data"
OUTPUT_DIR = OUTPUT_ROOT / "outputs"
QA_DIR = OUTPUT_ROOT / "qa"

CROSS_HR_INPUT = SOURCE_DIR / "panel_a_fullrisk_cross_organ_hr_49.csv"
HR_SUMMARY_INPUT = SOURCE_DIR / "panel_a_organ_alignment_summary.csv"
FIRST_EVENT_SUMMARY_INPUT = SOURCE_DIR / "panel_b_unique_first_event_summary.csv"
FIRST_EVENT_PERMUTATIONS_INPUT = (
    SOURCE_DIR / "panel_b_unique_first_event_label_permutations_1000.csv"
)
MAPPING_INPUT = SOURCE_DIR / "panel_c_exact_7factorial_label_mappings_5040.csv"
CLAIM_SUMMARY_INPUT = SOURCE_DIR / "figure_claim_summary.json"

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

CORAL = "#E76F51"
CORAL_DARK = "#C84D36"
SLATE = "#7F8C8D"
SLATE_LIGHT = "#C4CDD4"
GRID = "#E4E7E9"
TEXT = "#202020"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    sns.set_theme(style="white", font="Times New Roman", font_scale=1.0)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_label(axis: plt.Axes, label: str, x: float = -0.11, y: float = 1.075) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
    )


def summarize_hr(cross_hr: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for target in ORGANS:
        subset = cross_hr.loc[cross_hr["target_outcome"].eq(target)].copy()
        matched = subset.loc[subset["source_organ"].eq(target)].iloc[0]
        off_target = subset.loc[subset["source_organ"].ne(target)]
        rank = int(
            subset["coef_per_sd"].rank(method="min", ascending=False).loc[
                subset["source_organ"].eq(target)
            ].iloc[0]
        )
        rows.append(
            {
                "target_outcome": target,
                "target_outcome_label": ORGAN_LABELS[target],
                "matched_hr": float(matched["hr_per_sd"]),
                "matched_rank_among_7": rank,
                "matched_is_strongest": bool(rank == 1),
                "matched_log_hr": float(matched["coef_per_sd"]),
                "mean_off_target_log_hr": float(off_target["coef_per_sd"].mean()),
                "matched_above_off_target_mean": bool(
                    float(matched["coef_per_sd"])
                    > float(off_target["coef_per_sd"].mean())
                ),
            }
        )
    summary = pd.DataFrame(rows)
    return (
        int(summary["matched_is_strongest"].sum()),
        int(summary["matched_above_off_target_mean"].sum()),
        summary,
    )


def draw_hr_heatmap(
    axis: plt.Axes,
    cross_hr: pd.DataFrame,
    matched_strongest: int,
    matched_above_mean: int,
) -> None:
    panel_label(axis, "a", x=-0.16, y=1.205)
    axis.text(
        0.5,
        1.18,
        "Organ-aligned prospective associations",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT,
    )
    axis.text(
        0.5,
        1.105,
        (
            f"Matched offset strongest in {matched_strongest}/7 outcomes; "
            f"above off-target mean in {matched_above_mean}/7"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#4B4B4B",
    )

    coefficients = cross_hr.pivot(
        index="source_organ", columns="target_outcome", values="coef_per_sd"
    ).loc[list(ORGANS), list(ORGANS)]
    hrs = cross_hr.pivot(
        index="source_organ", columns="target_outcome", values="hr_per_sd"
    ).loc[list(ORGANS), list(ORGANS)]
    max_abs = float(np.abs(coefficients.to_numpy(float)).max())
    cmap = LinearSegmentedColormap.from_list(
        "organ_alignment", ["#3D6FA8", "#F7F7F7", "#C83C3C"], N=256
    )
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    sns.heatmap(
        coefficients,
        ax=axis,
        cmap=cmap,
        norm=norm,
        square=True,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        annot=False,
    )
    for row in range(len(ORGANS)):
        for column in range(len(ORGANS)):
            value = float(hrs.iloc[row, column])
            background = float(coefficients.iloc[row, column])
            color = "white" if abs(background) >= 0.105 else "#303030"
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold" if row == column else "normal",
                color=color,
            )
    for index in range(len(ORGANS)):
        axis.add_patch(
            Rectangle(
                (index, index),
                1,
                1,
                fill=False,
                edgecolor="#171717",
                linewidth=1.8,
            )
        )

    labels = [ORGAN_LABELS[organ] for organ in ORGANS]
    axis.set_xticklabels(labels, rotation=43, ha="right", fontsize=7.8)
    axis.set_yticklabels(labels, rotation=0, fontsize=7.8)
    axis.set_xlabel("Future organ event", labelpad=5)
    axis.set_ylabel("Source relative organ offset", labelpad=7)
    axis.tick_params(length=0)

def unique_first_event_summary(
    summary: dict[str, object], permutations: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = (
        (
            "Top-1",
            "unique_first_observed_top1_accuracy",
            "unique_first_bootstrap_top1_ci95",
            "unique_first_top1_accuracy",
        ),
        (
            "Top-2",
            "unique_first_observed_top2_accuracy",
            "unique_first_bootstrap_top2_ci95",
            "unique_first_top2_accuracy",
        ),
    )
    for label, observed_key, ci_key, permutation_column in definitions:
        values = permutations[permutation_column].to_numpy(float)
        rows.append(
            {
                "metric": label,
                "participants_unique_first_event": int(
                    summary["participants_with_unique_first_event"]
                ),
                "observed_accuracy": float(summary[observed_key]),
                "bootstrap_ci95_low": float(summary[ci_key][0]),
                "bootstrap_ci95_high": float(summary[ci_key][1]),
                "permutation_mean": float(values.mean()),
                "permutation_ci95_low": float(np.quantile(values, 0.025)),
                "permutation_ci95_high": float(np.quantile(values, 0.975)),
                "permutation_replicates": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def draw_first_event(
    axis: plt.Axes,
    summary_frame: pd.DataFrame,
    permutations: pd.DataFrame,
) -> None:
    panel_label(axis, "b", x=-0.13, y=1.32)
    n_unique = int(summary_frame["participants_unique_first_event"].iloc[0])
    axis.text(
        0.5,
        1.29,
        "First-organ-event identification",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT,
    )
    axis.text(
        0.5,
        1.205,
        f"Unique first events, N={n_unique:,}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#4B4B4B",
    )
    axis.text(
        0.5,
        1.105,
        "Coral: observed (bootstrap 95% CI)   |   Grey: 1,000 label permutations",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#586366",
    )

    positions = np.asarray([1.0, 2.0])
    distributions = [
        permutations["unique_first_top1_accuracy"].to_numpy(float),
        permutations["unique_first_top2_accuracy"].to_numpy(float),
    ]
    violin = axis.violinplot(
        distributions,
        positions=positions,
        widths=0.52,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(SLATE_LIGHT)
        body.set_edgecolor("white")
        body.set_alpha(0.85)
        body.set_linewidth(0.7)

    for index, (_, row) in enumerate(summary_frame.iterrows()):
        x = positions[index]
        low = float(row["permutation_ci95_low"])
        high = float(row["permutation_ci95_high"])
        mean = float(row["permutation_mean"])
        observed = float(row["observed_accuracy"])
        observed_low = float(row["bootstrap_ci95_low"])
        observed_high = float(row["bootstrap_ci95_high"])
        axis.vlines(x, low, high, color=SLATE, linewidth=1.1, zorder=3)
        axis.scatter(
            x,
            mean,
            marker="D",
            s=27,
            color=SLATE,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.errorbar(
            x,
            observed,
            yerr=np.asarray([[observed - observed_low], [observed_high - observed]]),
            fmt="o",
            markersize=6.6,
            color=CORAL,
            ecolor=CORAL_DARK,
            elinewidth=1.5,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=5,
        )
        axis.text(
            x,
            observed + 0.034,
            f"{100 * observed:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=CORAL_DARK,
            fontweight="bold",
        )
        axis.text(
            x + 0.17,
            mean,
            f"{100 * mean:.1f}%",
            ha="left",
            va="center",
            fontsize=6.9,
            color="#586366",
        )

    axis.set_xticks(positions, summary_frame["metric"], fontsize=8.5)
    axis.set_xlim(0.55, 2.45)
    axis.set_ylim(0.0, 0.68)
    axis.set_ylabel("Identification accuracy")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", length=0)
def draw_mapping(
    axis: plt.Axes,
    mappings: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    panel_label(axis, "c", x=-0.13, y=1.30)
    axis.text(
        0.5,
        1.27,
        "Organ-label mapping negative control",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT,
    )
    alternative = mappings.loc[
        ~mappings["is_identity_mapping"].astype(bool),
        "macro_mean_incremental_delta_c",
    ].to_numpy(float)
    axis.hist(
        alternative,
        bins=38,
        color=SLATE_LIGHT,
        edgecolor="white",
        linewidth=0.45,
    )
    identity = float(summary["observed_identity_macro_delta_c"])
    axis.axvline(identity, color=CORAL, linewidth=2.1, zorder=4)
    axis.text(
        0.5,
        1.155,
        (
            rf"Identity mapping: $\Delta$C={identity:.4f}; "
            f"rank {int(summary['identity_rank_descending'])}/5,040"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.3,
        color=CORAL_DARK,
        fontweight="bold",
    )
    axis.text(
        0.5,
        1.075,
        (
            f"Only {int(round(float(summary['exact_one_sided_probability_mapping_ge_identity']) * 5040))} "
            "of 5,040 mappings were at or above identity"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#586366",
    )
    axis.set_xlabel(r"Macro held-out incremental $\Delta$C")
    axis.set_ylabel("Number of mappings")
    axis.grid(axis="y", color=GRID, linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)


def save_all_formats(figure: plt.Figure, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for extension in ("svg", "pdf", "png"):
        path = OUTPUT_DIR / f"{stem}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    return outputs


def render(
    cross_hr: pd.DataFrame,
    matched_strongest: int,
    matched_above_mean: int,
    first_event_frame: pd.DataFrame,
    first_event_permutations: pd.DataFrame,
    mappings: pd.DataFrame,
    mapping_summary: dict[str, object],
) -> list[Path]:
    setup_style()
    figure = plt.figure(figsize=(9.4, 7.0))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.25, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.075,
        right=0.975,
        bottom=0.08,
        top=0.89,
        wspace=0.42,
        hspace=0.72,
    )
    heatmap_axis = figure.add_subplot(grid[:, 0])
    first_event_axis = figure.add_subplot(grid[0, 1])
    mapping_axis = figure.add_subplot(grid[1, 1])

    draw_hr_heatmap(
        heatmap_axis,
        cross_hr,
        matched_strongest=matched_strongest,
        matched_above_mean=matched_above_mean,
    )
    draw_first_event(first_event_axis, first_event_frame, first_event_permutations)
    draw_mapping(mapping_axis, mappings, mapping_summary)

    outputs = save_all_formats(figure, "Fig11_organ_aligned_specificity")
    plt.close(figure)

    panel_a_figure, panel_a_axis = plt.subplots(figsize=(5.8, 5.7))
    panel_a_figure.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.80)
    draw_hr_heatmap(
        panel_a_axis,
        cross_hr,
        matched_strongest=matched_strongest,
        matched_above_mean=matched_above_mean,
    )
    outputs.extend(save_all_formats(panel_a_figure, "Fig11a_organ_aligned_hr_matrix"))
    plt.close(panel_a_figure)

    panel_b_figure, panel_b_axis = plt.subplots(figsize=(5.0, 4.35))
    panel_b_figure.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.69)
    draw_first_event(panel_b_axis, first_event_frame, first_event_permutations)
    outputs.extend(
        save_all_formats(panel_b_figure, "Fig11b_first_organ_event_identification")
    )
    plt.close(panel_b_figure)

    panel_c_figure, panel_c_axis = plt.subplots(figsize=(5.0, 4.35))
    panel_c_figure.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.70)
    draw_mapping(panel_c_axis, mappings, mapping_summary)
    outputs.extend(
        save_all_formats(panel_c_figure, "Fig11c_organ_label_mapping_negative_control")
    )
    plt.close(panel_c_figure)
    return outputs


def main() -> int:
    for directory in (OUTPUT_ROOT, SOURCE_DIR, OUTPUT_DIR, QA_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    cross_hr = pd.read_csv(CROSS_HR_INPUT)
    frozen_hr_summary = pd.read_csv(HR_SUMMARY_INPUT)
    frozen_first_event = pd.read_csv(FIRST_EVENT_SUMMARY_INPUT)
    frozen_claims = json.loads(CLAIM_SUMMARY_INPUT.read_text(encoding="utf-8"))
    first_event_permutations = pd.read_csv(FIRST_EVENT_PERMUTATIONS_INPUT)
    mappings = pd.read_csv(MAPPING_INPUT)

    top1 = frozen_first_event.loc[frozen_first_event["metric"].eq("Top-1")].iloc[0]
    top2 = frozen_first_event.loc[frozen_first_event["metric"].eq("Top-2")].iloc[0]
    first_event_claim = frozen_claims["panel_b"]
    first_event_summary = {
        "participants_with_unique_first_event": int(
            first_event_claim["unique_first_event_participants"]
        ),
        "unique_first_observed_top1_accuracy": float(top1["observed_accuracy"]),
        "unique_first_bootstrap_top1_ci95": [
            float(top1["bootstrap_ci95_low"]),
            float(top1["bootstrap_ci95_high"]),
        ],
        "unique_first_permutation_top1_mean": float(top1["permutation_mean"]),
        "unique_first_observed_top2_accuracy": float(top2["observed_accuracy"]),
        "unique_first_bootstrap_top2_ci95": [
            float(top2["bootstrap_ci95_low"]),
            float(top2["bootstrap_ci95_high"]),
        ],
        "unique_first_permutation_top2_mean": float(top2["permutation_mean"]),
        "bootstrap_replicates": int(first_event_claim["bootstrap_replicates"]),
        "label_permutation_replicates": int(
            first_event_claim["label_permutation_replicates"]
        ),
    }
    mapping_claim = frozen_claims["panel_c"]
    mapping_summary = {
        "permutations": int(mapping_claim["mapping_permutations"]),
        "identity_rank_descending": int(mapping_claim["identity_rank_descending"]),
        "observed_identity_macro_delta_c": float(
            mapping_claim["identity_macro_delta_c"]
        ),
        "exact_one_sided_probability_mapping_ge_identity": float(
            mapping_claim["mapping_tail_fraction_at_or_above_identity"]
        ),
    }

    matched_strongest, matched_above_mean, hr_summary = summarize_hr(cross_hr)
    pd.testing.assert_frame_equal(
        hr_summary.reset_index(drop=True),
        frozen_hr_summary[hr_summary.columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )
    first_event_frame = unique_first_event_summary(
        first_event_summary, first_event_permutations
    )

    identity_mask = mappings["is_identity_mapping"].map(
        lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
    )
    identity_score = float(
        mappings.loc[identity_mask, "macro_mean_incremental_delta_c"].iloc[0]
    )
    identity_rank = int(
        1
        + (
            mappings["macro_mean_incremental_delta_c"].astype(float)
            > identity_score
        ).sum()
    )
    identity_tail = float(
        (
            mappings["macro_mean_incremental_delta_c"].astype(float)
            >= identity_score
        ).mean()
    )
    if not (
        int(identity_mask.sum()) == 1
        and len(mappings) == mapping_summary["permutations"] == 5040
        and identity_rank == mapping_summary["identity_rank_descending"]
        and np.isclose(identity_score, mapping_summary["observed_identity_macro_delta_c"])
        and np.isclose(
            identity_tail,
            mapping_summary["exact_one_sided_probability_mapping_ge_identity"],
        )
    ):
        raise AssertionError("Frozen organ-label mapping summary changed")

    cross_hr.to_csv(SOURCE_DIR / "panel_a_fullrisk_cross_organ_hr_49.csv", index=False)
    hr_summary.to_csv(SOURCE_DIR / "panel_a_organ_alignment_summary.csv", index=False)
    first_event_frame.to_csv(
        SOURCE_DIR / "panel_b_unique_first_event_summary.csv", index=False
    )
    first_event_permutations[
        [
            "replicate",
            "unique_first_top1_accuracy",
            "unique_first_top2_accuracy",
            "seed",
        ]
    ].to_csv(
        SOURCE_DIR / "panel_b_unique_first_event_label_permutations_1000.csv",
        index=False,
    )
    mappings.to_csv(
        SOURCE_DIR / "panel_c_exact_7factorial_label_mappings_5040.csv", index=False
    )

    claim_summary = {
        "schema_version": "m11.maintext.organ_specificity.claims.v1",
        "generated_at_utc": utc_now(),
        "neural_model_refit": False,
        "survival_model_refit": False,
        "panel_a": {
            "matched_offset_strongest_outcomes": matched_strongest,
            "matched_offset_above_offtarget_mean_outcomes": matched_above_mean,
            "total_outcomes": 7,
            "interpretation": "organ-aligned prospective association, not complete organ independence",
        },
        "panel_b": {
            "unique_first_event_participants": int(
                first_event_summary["participants_with_unique_first_event"]
            ),
            "observed_top1": float(
                first_event_summary["unique_first_observed_top1_accuracy"]
            ),
            "permutation_mean_top1": float(
                first_event_summary["unique_first_permutation_top1_mean"]
            ),
            "observed_top2": float(
                first_event_summary["unique_first_observed_top2_accuracy"]
            ),
            "permutation_mean_top2": float(
                first_event_summary["unique_first_permutation_top2_mean"]
            ),
            "bootstrap_replicates": int(first_event_summary["bootstrap_replicates"]),
            "label_permutation_replicates": int(
                first_event_summary["label_permutation_replicates"]
            ),
        },
        "panel_c": {
            "mapping_permutations": int(mapping_summary["permutations"]),
            "identity_rank_descending": int(
                mapping_summary["identity_rank_descending"]
            ),
            "identity_macro_delta_c": float(
                mapping_summary["observed_identity_macro_delta_c"]
            ),
            "mapping_tail_fraction_at_or_above_identity": float(
                mapping_summary["exact_one_sided_probability_mapping_ge_identity"]
            ),
            "interpretation": "post-hoc output-label mapping negative control, not neural-query masking",
        },
    }
    (SOURCE_DIR / "figure_claim_summary.json").write_text(
        json.dumps(claim_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    outputs = render(
        cross_hr,
        matched_strongest,
        matched_above_mean,
        first_event_frame,
        first_event_permutations,
        mappings,
        mapping_summary,
    )

    svg_paths = [path for path in outputs if path.suffix.lower() == ".svg"]
    svg_checks = {
        path.stem: {
            "editable_text": "<text" in path.read_text(encoding="utf-8"),
            "embedded_raster": (
                "<image" in path.read_text(encoding="utf-8")
                or "data:image" in path.read_text(encoding="utf-8")
            ),
        }
        for path in svg_paths
    }
    qa = {
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "confirmed_current_main_fig11": True,
        "self_contained_within_current_v1": True,
        "neural_model_refit": False,
        "survival_model_refit": False,
        "cross_organ_hr_cells": int(len(cross_hr)),
        "cross_organ_matrix_complete": bool(
            len(cross_hr) == 49
            and cross_hr[["source_organ", "target_outcome"]]
            .drop_duplicates()
            .shape[0]
            == 49
        ),
        "matched_offset_strongest_outcomes": matched_strongest,
        "matched_offset_above_offtarget_mean_outcomes": matched_above_mean,
        "unique_first_event_participants": int(
            first_event_summary["participants_with_unique_first_event"]
        ),
        "first_event_label_permutations": int(len(first_event_permutations)),
        "observed_top1_gt_permutation_mean": bool(
            first_event_frame.loc[first_event_frame["metric"].eq("Top-1"), "observed_accuracy"].iloc[0]
            > first_event_frame.loc[first_event_frame["metric"].eq("Top-1"), "permutation_mean"].iloc[0]
        ),
        "observed_top2_gt_permutation_mean": bool(
            first_event_frame.loc[first_event_frame["metric"].eq("Top-2"), "observed_accuracy"].iloc[0]
            > first_event_frame.loc[first_event_frame["metric"].eq("Top-2"), "permutation_mean"].iloc[0]
        ),
        "exact_label_mappings": int(len(mappings)),
        "identity_mapping_rank": int(mapping_summary["identity_rank_descending"]),
        "outputs_present": all(path.exists() and path.stat().st_size > 0 for path in outputs),
        "exported_figures": int(len(outputs)),
        "individual_panels_exported": all(
            (OUTPUT_DIR / f"{stem}.{extension}").exists()
            for stem in (
                "Fig11a_organ_aligned_hr_matrix",
                "Fig11b_first_organ_event_identification",
                "Fig11c_organ_label_mapping_negative_control",
            )
            for extension in ("svg", "pdf", "png")
        ),
        "svg_checks": svg_checks,
        "all_svg_editable_text": all(
            item["editable_text"] for item in svg_checks.values()
        ),
        "all_svg_without_embedded_raster": all(
            not item["embedded_raster"] for item in svg_checks.values()
        ),
    }
    if not all(
        [
            qa["cross_organ_matrix_complete"],
            qa["observed_top1_gt_permutation_mean"],
            qa["observed_top2_gt_permutation_mean"],
            qa["outputs_present"],
            qa["individual_panels_exported"],
            qa["all_svg_editable_text"],
            qa["all_svg_without_embedded_raster"],
            qa["first_event_label_permutations"] == 1000,
            qa["exact_label_mappings"] == 5040,
        ]
    ):
        qa["status"] = "FAIL"
        raise RuntimeError(f"Confirmed Fig.11 QA failed: {qa}")
    (QA_DIR / "QA_REPORT.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# Fig.11：器官对齐的部分特异性证据

这是作者确认的唯一正文 Fig.11 交付包，已在本目录内自包含，不依赖候选或开发版本目录。图件只复用已核查统计，不重新训练神经网络，也不重新拟合生存模型。

- a：7×7 完整调整 HR 矩阵。对应器官 offset 在 {matched_strongest}/7 类结局中关联最强，且 7/7 均高于非靶器官平均关联。
- b：唯一首次器官事件人群（N={int(first_event_summary['participants_with_unique_first_event']):,}）的个体内器官排序。Top-1 为 {100*float(first_event_summary['unique_first_observed_top1_accuracy']):.2f}%（置换均值 {100*float(first_event_summary['unique_first_permutation_top1_mean']):.2f}%）；Top-2 为 {100*float(first_event_summary['unique_first_observed_top2_accuracy']):.2f}%（置换均值 {100*float(first_event_summary['unique_first_permutation_top2_mean']):.2f}%）。
- c：七个输出标签的全部 5,040 种映射负对照。真实器官映射排名第 {int(mapping_summary['identity_rank_descending'])}/5,040。

`outputs` 提供合并图，以及 a、b、c 三个独立子图的 SVG、PDF 和 PNG；`source_data`、`scripts`、`qa` 和 `MANIFEST.json`用于复核和重绘。

允许结论：器官偏移具有“器官对齐的部分特异性”或“器官倾向性”。

不支持：七个器官信号完全独立、器官映射达到唯一全局最优、或者该标签置换等同于神经网络内部 query 遮蔽。
"""
    (OUTPUT_ROOT / "README_CN.md").write_text(readme, encoding="utf-8")

    inputs = [
        CROSS_HR_INPUT,
        HR_SUMMARY_INPUT,
        FIRST_EVENT_SUMMARY_INPUT,
        FIRST_EVENT_PERMUTATIONS_INPUT,
        MAPPING_INPUT,
        CLAIM_SUMMARY_INPUT,
    ]
    artifacts = sorted(
        [
            *[path for path in OUTPUT_DIR.iterdir() if path.is_file()],
            QA_DIR / "QA_REPORT.json",
            OUTPUT_ROOT / "README_CN.md",
        ]
    )
    manifest = {
        "schema_version": "final_figure_0725.fig11.organ_specificity.current.v2",
        "status": qa["status"],
        "generated_at_utc": utc_now(),
        "role": "confirmed current main Fig.11",
        "confirmed_by_author": True,
        "self_contained": True,
        "rendered_from_frozen_statistics_only": True,
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

    print(json.dumps(qa, ensure_ascii=False, indent=2))
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
