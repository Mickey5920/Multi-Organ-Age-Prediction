"""Render Figure 7 from aggregate tables; no participant data or model fitting.

Default: reconstruct all six panels using the retained study renderer.
--base-figure: preserve the submitted vector figure and add only the four new
joint paired-bootstrap macro C-index intervals. The original PDF is optional
and is not redistributed as a code dependency.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "source/Revision-0730/0-Others/Fig12_MIMIC_IV_External_Transportability/redraw_figure6_external_transportability.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("retained_figure7", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA = PACKAGE / "reference/figure7"
    return module


def macro_intervals():
    table = pd.read_csv(PACKAGE / "reference/r1_3_macro_paired_bootstrap_ci.csv")
    table = table.loc[table.metric == "delta_harrell_c"].copy()
    assert len(table) == 4 and not table.duplicated(["stage", "comparison"]).any()
    assert (table.ci95_low <= table.point_estimate).all()
    assert (table.point_estimate <= table.ci95_high).all()
    assert (table.replicates == 1000).all()
    return table.set_index(["stage", "comparison"])


def draw_prediction(fig, spec, data, renderer, intervals):
    r = renderer
    body = r.panel_body(fig, spec, "B", "MIMIC-IV 12-month prediction")
    grid = body.subgridspec(2, 1, height_ratios=(0.94, 1.06), hspace=0.72)
    risk, calibration = data["mimic_risk"], data["mimic_calibration"]
    ax = fig.add_subplot(grid[0, 0])
    comparisons = ["age_sex_plus_gap", "age_sex_density_plus_gap"]
    for stage_index, (stage, key, color, marker) in enumerate([
        ("Direct transfer", "direct", r.DIRECT, "s"),
        ("Support aligned", "adapted", r.ALIGNED, "o"),
    ]):
        for comp_index, comp in enumerate(comparisons):
            group = risk.loc[(risk.stage == stage) & (risk.comparison == comp)]
            assert len(group) == 7
            row = intervals.loc[(key, comp)]
            mean = float(group.delta.mean())
            assert np.isclose(mean, row.point_estimate, rtol=0, atol=1e-12)
            y = [1.0, 0.0][comp_index] + (0.10 if stage_index else -0.10)
            ax.scatter(group.delta, np.full(len(group), y), s=7, color=color, alpha=0.34, edgecolor="none")
            ax.errorbar(mean, y, xerr=[[mean-row.ci95_low], [row.ci95_high-mean]],
                        fmt="none", ecolor=color, elinewidth=0.65, capsize=1.6,
                        capthick=0.65, zorder=3)
            ax.scatter(mean, y, s=30, color=color, marker=marker, edgecolor="white", linewidth=0.5, zorder=4)
            ax.text(mean, y + (0.15 if stage_index else -0.15), f"{mean:+.4f}", color=color,
                    ha="center", va="center", fontsize=5.0)
    ax.axvline(0, color="#7E8790", ls="--", lw=0.75)
    ax.set_xlim(-0.012, 0.046)
    ax.set_ylim(-0.42, 1.42)
    ax.set_yticks([1, 0], ["Age + sex", "+ record\ndensity"])
    ax.set_xlabel("Macro ΔHarrell C")
    ax.set_title("12-month discrimination", loc="left", fontsize=5.2, fontweight="bold", pad=2.2)
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=r.DIRECT, label="Frozen direct"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=r.ALIGNED, label="Input-support aligned"),
    ], loc="lower right", fontsize=5.1, handletextpad=0.3)
    r.clean_axis(ax, "x")
    ax = fig.add_subplot(grid[1, 0])
    for stage, color in [("Direct transfer", r.DIRECT), ("Support aligned", r.ALIGNED)]:
        group = calibration.loc[calibration.stage == stage]
        ax.plot(group.predicted, group.observed, marker="o", ms=2.3, lw=1.05, color=color)
    limit = [0, max(calibration.predicted.max(), calibration.observed.max()) * 1.08]
    ax.plot(limit, limit, color="#7E8790", ls="--", lw=0.7)
    ax.set_xlim(limit)
    ax.set_ylim(limit)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed risk")
    ax.set_title("12-month risk calibration", loc="left", fontsize=5.2, fontweight="bold", pad=2.2)
    r.clean_axis(ax, "both")


def full_figure(renderer, intervals, output, alignment_helper=None):
    r = renderer
    data = r.load_data()
    fig = plt.figure(figsize=(7.20, 4.22))
    grid = fig.add_gridspec(2, 3, left=0.045, right=0.992, top=0.975, bottom=0.075,
                          wspace=0.26, hspace=0.16)
    r.draw_mimic_age(fig, grid[0, 0], data)
    draw_prediction(fig, grid[0, 1], data, r, intervals)
    r.draw_mimic_hr(fig, grid[0, 2], data)
    r.draw_nhanes_age(fig, grid[1, 0], data)
    r.draw_nhanes_calibration(fig, grid[1, 1], data)
    r.draw_nhanes_or(fig, grid[1, 2], data)
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text(text.get_text().replace("Input-support adaptation", "Input-support aligned")
                      .replace("Input-support\nadaptation", "Input-support\naligned"))
    if alignment_helper:
        spec = importlib.util.spec_from_file_location("alignment", alignment_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.require_matplotlib_panel_alignment(fig, json_out=output / "alignment.json")
    for extension in ("pdf", "svg"):
        fig.savefig(output / f"Figure 7_reconstructed.{extension}", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def preserve_vector_layout(base, intervals, output, renderer):
    import fitz
    original = fitz.open(base)
    assert len(original) == 1 and not original[0].get_images(), "Expected original one-page vector Figure 7"
    page = original[0]
    points = []
    for drawing in page.get_drawings():
        box = drawing["rect"]
        x, y = (box.x0+box.x1)/2, (box.y0+box.y1)/2
        # Exclude legend symbols (black outline); macro symbols have white edges.
        if (170 < x < 345 and y < 70 and 4.8 < box.width < 6.2 and
                4.8 < box.height < 6.2 and drawing.get("color") == (1., 1., 1.)):
            points.append((x, y, drawing["fill"]))
    assert len(points) == 4, "Macro-marker geometry differs from the archived layout"
    keyed = []
    for key, blue in [("direct", False), ("adapted", True)]:
        stage_points = sorted([p for p in points if (p[2][0] < .1) == blue], key=lambda p:p[1])
        assert len(stage_points) == 2
        for comp, point in zip(["age_sex_plus_gap", "age_sex_density_plus_gap"], stage_points):
            keyed.append((key, point, intervals.loc[(key, comp)]))
    values = np.array([row.point_estimate for _, _, row in keyed])
    centers = np.array([point[0] for _, point, _ in keyed])
    slope, intercept = np.polyfit(values, centers, 1)
    assert slope > 0 and np.max(np.abs(slope*values+intercept-centers)) < .01
    width, height = page.rect.width, page.rect.height
    fig = plt.figure(figsize=(width/72, height/72))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    for key, (x, y, _), row in keyed:
        color = renderer.DIRECT if key == "direct" else renderer.ALIGNED
        lo, hi = slope*row.ci95_low+intercept, slope*row.ci95_high+intercept
        ax.plot([lo, hi], [y, y], color=color, lw=.65, zorder=3)
        ax.plot([lo, lo], [y-1.6, y+1.6], color=color, lw=.65, zorder=3)
        ax.plot([hi, hi], [y-1.6, y+1.6], color=color, lw=.65, zorder=3)
        ax.scatter(x, y, s=30, marker="s" if key == "direct" else "o", color=color,
                   edgecolor="white", linewidth=.5, zorder=4)
    layer_path = output / "Figure 7_interval_layer.pdf"
    fig.savefig(layer_path, transparent=True, pad_inches=0)
    plt.close(fig)
    with fitz.open(layer_path) as layer:
        page.show_pdf_page(page.rect, layer, 0, overlay=True)
    original.save(output / "Figure 7.pdf", garbage=0, deflate=True)
    original.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-figure", type=Path)
    parser.add_argument("--alignment-helper", type=Path, help="Optional local Matplotlib alignment audit module")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = load_renderer()
    renderer.configure_style()
    intervals = macro_intervals()
    full_figure(renderer, intervals, args.output_dir, args.alignment_helper)
    if args.base_figure:
        preserve_vector_layout(args.base_figure, intervals, args.output_dir, renderer)
    print("Aggregate-only Figure 7 export complete; no fitting or participant input.")


if __name__ == "__main__":
    main()
