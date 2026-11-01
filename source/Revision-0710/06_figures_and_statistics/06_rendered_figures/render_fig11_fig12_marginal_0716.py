#!/usr/bin/env python3
"""Render Fig11/Fig12 in the 0626 style from single-gap marginal associations."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


SCRIPT = Path(__file__).resolve()
ROOT = next(parent for parent in SCRIPT.parents if (parent / "Revision-0710").is_dir())
INPUT_ROOT = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/03_reviewer_statistics"
    / "05_fig11_fig12_marginal_and_review_closure/02_single_gap_associations"
)
COEFFICIENTS = INPUT_ROOT / "Fig12/marginal_association_coefficients.csv"
MATRIX = INPUT_ROOT / "Fig11/AllGroups_Marginal_Association_Matrix.csv"
LEGACY_SCRIPT = (
    ROOT
    / "Revision-0626/18_final_ehr_age_downstream_gp_medication_20260707"
    / "source_scripts_snapshot/render_legacy_style_figures.py"
)
OUTPUT_ROOT = ROOT / "Revision-0710/Final_Fig_0716"
FIG11 = OUTPUT_ROOT / "Fig11"
FIG12 = OUTPUT_ROOT / "Fig12"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy() -> Any:
    spec = importlib.util.spec_from_file_location("legacy_0626_marginal_0716", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy renderer: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original_layout = module.nx.circular_layout
    reference_order = ["Pulm.", "Immune", "Musculo.", "Hepatic", "Cardio.", "Metabolic", "Renal"]

    def reference_circular_layout(graph: Any, *args: Any, **kwargs: Any) -> dict[Any, Any]:
        present = list(graph.nodes)
        ordered = [node for node in reference_order if node in present]
        ordered.extend(node for node in present if node not in ordered)
        layout_graph = module.nx.Graph()
        layout_graph.add_nodes_from(ordered)
        return original_layout(layout_graph, *args, **kwargs)

    module.nx.circular_layout = reference_circular_layout
    module.setup_legacy_style()
    return module


def image_qa(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        std = float(np.mean(ImageStat.Stat(rgb).stddev))
        return {
            "relative_path": str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"),
            "width": rgb.width,
            "height": rgb.height,
            "channel_std_mean": std,
            "nonblank": bool(rgb.width > 100 and rgb.height > 100 and std > 1.0),
        }


def load_matrix(legacy: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = pd.read_csv(COEFFICIENTS)
    expected_rows = 16 * 7 * 6
    if len(coefficients) != expected_rows:
        raise RuntimeError(f"unexpected coefficient rows: {len(coefficients)}")
    if coefficients.duplicated(["analysis_group", "source_organ", "target_organ"]).any():
        raise RuntimeError("duplicate marginal association key")
    if not coefficients["other_age_gaps_in_model"].eq(0).all():
        raise RuntimeError("another age gap entered a marginal model")
    coefficients["Path"] = coefficients["association"].map(legacy.title_path)
    matrix = coefficients.pivot(index="Path", columns="analysis_group", values="beta_standardized")
    matrix = matrix.reindex(columns=list(GROUP_ORDER))
    if matrix.shape != (42, 16) or matrix.isna().any().any():
        raise RuntimeError(f"unexpected marginal matrix shape/completeness: {matrix.shape}")
    return coefficients, matrix


def render_heatmap(legacy: Any, matrix: pd.DataFrame) -> list[Path]:
    FIG11.mkdir(parents=True, exist_ok=False)
    matrix.to_csv(FIG11 / "AllGroups_Marginal_Association_Matrix.csv", lineterminator="\n")
    display = matrix.copy()
    display.index = [legacy.short_label(path) for path in display.index]
    display.columns = [legacy.short_label(column) for column in display.columns]
    legacy.plt.figure(figsize=(15.2, max(8, len(display) * 0.35)))
    legacy.sns.heatmap(
        display.astype(float),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={
            "label": "Standardized adjusted marginal association",
            "fraction": 0.035,
            "pad": 0.02,
        },
    )
    legacy.plt.title("Adjusted Marginal Association Across All Groups")
    legacy.plt.ylabel("Directed association (source -> target)")
    legacy.plt.xlabel("Group")
    legacy.plt.tight_layout()
    outputs: list[Path] = []
    for extension, dpi in (("png", 300), ("pdf", 600), ("svg", 600)):
        path = FIG11 / f"Fig-11-Marginal-Association-Heatmap.{extension}"
        legacy.plt.savefig(path, dpi=dpi, format=extension)
        outputs.append(path)
    legacy.plt.close()
    return outputs


def plot_network_panel(
    legacy: Any,
    edge_frames: list[pd.DataFrame],
    panel_titles: list[str],
) -> list[Path]:
    combined = pd.concat(edge_frames, ignore_index=True)
    max_coef = float(np.max(np.abs(combined["coef"].to_numpy(dtype=float))))
    positive = np.abs(combined.loc[combined["coef"] > 0, "coef"].to_numpy(dtype=float))
    negative = np.abs(combined.loc[combined["coef"] < 0, "coef"].to_numpy(dtype=float))
    fig, axes = legacy.plt.subplots(2, 2, figsize=(14, 14), dpi=600)
    fig.subplots_adjust(wspace=0.20, hspace=0.30)
    for index, edge_frame in enumerate(edge_frames):
        axis = axes.flatten()[index]
        legacy.draw_network_from_df(edge_frame, panel_titles[index], axis, max_coef)
        axis.set_xlim(-1.32, 1.32)
        axis.set_ylim(-1.32, 1.32)

    handles: list[Any] = []
    if len(positive):
        weights = [float(np.min(positive)), float(np.median(positive)), float(np.max(positive))]
        handles.extend(
            [legacy.Patch(facecolor="none", edgecolor="none", label="Positive association (Red)"),
             *legacy.make_group_lines(weights, max_coef, "#FF0000")]
        )
    if len(negative):
        weights = [float(np.min(negative)), float(np.median(negative)), float(np.max(negative))]
        handles.extend(
            [legacy.Patch(facecolor="none", edgecolor="none", label="Negative association (Blue)"),
             *legacy.make_group_lines(weights, max_coef, "#4472C4")]
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.45),
        ncol=2,
        fontsize=11,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.8,
    )
    legacy.plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    outputs: list[Path] = []
    for extension, dpi in (("png", 300), ("pdf", 600), ("svg", 600)):
        path = FIG12 / f"Fig-12-Marginal-Association-Network.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        outputs.append(path)
    legacy.plt.close(fig)
    return outputs


def render_network(
    legacy: Any,
    coefficients: pd.DataFrame,
    matrix: pd.DataFrame,
) -> list[Path]:
    FIG12.mkdir(parents=True, exist_ok=False)
    coefficients.to_csv(
        FIG12 / "marginal_association_coefficients.csv",
        index=False,
        lineterminator="\n",
    )
    group_plot_map = {
        "Health": "Health",
        "cardiovascular_Health": "cardiovascular_Health",
        "cardiovascular": "cardiovascular_Disease",
        "Other_Disease": "Other_Disease",
    }
    panel_titles = [
        "a) Health Group",
        "b) Cardiovascular Health Group",
        "c) Cardiovascular Disease Group",
        "d) Other Disease Group",
    ]
    edge_frames: list[pd.DataFrame] = []
    for group, tag in group_plot_map.items():
        top = legacy.get_top_paths(matrix, "Health", group, n_top=10)
        pd.Series(top, name="association").to_csv(
            FIG12 / f"Top17Associations_health_vs_{tag}.csv",
            index=False,
            lineterminator="\n",
        )
        edge_frames.append(
            legacy.save_edges_for_group(
                matrix,
                group,
                top,
                FIG12 / f"{tag}_marginal_association_edges.csv",
            )
        )
    return plot_network_panel(legacy, edge_frames, panel_titles)


def main() -> int:
    if FIG11.exists() or FIG12.exists():
        raise FileExistsError("create-once Fig11/Fig12 final directory already exists")
    for path in (COEFFICIENTS, MATRIX, LEGACY_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy()
    coefficients, matrix = load_matrix(legacy)
    outputs = [*render_heatmap(legacy, matrix), *render_network(legacy, coefficients, matrix)]
    qa = [image_qa(path) for path in outputs if path.suffix.lower() == ".png"]
    if not all(item["nonblank"] for item in qa):
        raise RuntimeError(f"blank output detected: {qa}")
    manifest = {
        "schema_version": "revision0710.fig11_fig12.single_gap_marginal.0626_style.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calculation": "one standardized source-organ gap plus seven Fig8 clinical covariates; no other gaps",
        "interpretation": "adjusted marginal association; not causal, mechanistic, or SEM",
        "style_source": str(LEGACY_SCRIPT.relative_to(ROOT)).replace("\\", "/"),
        "inputs": {
            str(COEFFICIENTS.relative_to(ROOT)).replace("\\", "/"): sha256(COEFFICIENTS),
            str(MATRIX.relative_to(ROOT)).replace("\\", "/"): sha256(MATRIX),
            str(LEGACY_SCRIPT.relative_to(ROOT)).replace("\\", "/"): sha256(LEGACY_SCRIPT),
        },
        "rows": len(coefficients),
        "matrix_shape": list(matrix.shape),
        "coefficient_range": [
            float(coefficients["beta_standardized"].min()),
            float(coefficients["beta_standardized"].max()),
        ],
        "outputs": [
            {
                "relative_path": str(path.relative_to(OUTPUT_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
        "image_qa": qa,
    }
    (FIG11 / "FIG11_FIG12_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (FIG12 / "FIG11_FIG12_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "rows": len(coefficients), "qa": qa}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
