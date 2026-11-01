#!/usr/bin/env python3
"""Build Fig11/Fig12 source data from single-gap adjusted associations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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
DISPLAY_ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)
GROUPS = (
    "Health",
    *DISPLAY_ORGANS,
    *(f"{organ}_Health" for organ in DISPLAY_ORGANS),
    "Other_Disease",
)
GAP_COLUMNS = {organ: f"{organ}_linear_corrected_age_gap" for organ in ORGANS}
CONTROL_COLUMNS = (
    "sex_male",
    "employment_employed",
    "sleep_duration",
    "smoking_current",
    "sensitivity_pm25_2010",
    "education_higher",
    "townsend",
)
BINARY_CONTROLS = {
    "sex_male",
    "employment_employed",
    "smoking_current",
    "education_higher",
}


def root_from_script() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = root_from_script()
ADAPTER = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/05_source_data_by_figure"
    / "corrected_canonical_0626_adapter_v1/formal_source_v1"
)
DEFAULT_WIDE = ADAPTER / "03_fig11_fig12/canonical_age_gap_wide.parquet"
DEFAULT_MEMBERSHIP = ADAPTER / "03_fig11_fig12/corrected_display_group_membership.parquet"
DEFAULT_CONTEXT = (
    ROOT
    / "Revision-0710/05_downstream/06_metric_v2/01_contextual_covariates"
    / "participant_context_covariates.parquet"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "02_single_gap_associations"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def zscore(values: np.ndarray) -> np.ndarray:
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("zero or invalid standard deviation")
    return (values - center) / scale


def residualize_gaps(gaps: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(controls), dtype=float), controls])
    coefficients, _, rank, _ = np.linalg.lstsq(design, gaps, rcond=None)
    if rank != design.shape[1]:
        raise RuntimeError(f"control design rank deficient: {rank}/{design.shape[1]}")
    return gaps - design @ coefficients


def directional_association(
    source: np.ndarray,
    target: np.ndarray,
    *,
    nuisance_df: int,
) -> dict[str, float]:
    denominator = float(np.dot(source, source))
    if denominator <= 0.0:
        raise RuntimeError("source residual has zero variance")
    beta = float(np.dot(source, target) / denominator)
    residual = target - beta * source
    n = len(source)
    dof = n - nuisance_df - 1
    if dof <= 2:
        raise RuntimeError("insufficient residual degrees of freedom")
    # HC1 sandwich variance after Frisch-Waugh-Lovell residualization.
    meat = float(np.dot(np.square(source), np.square(residual)))
    variance_hc1 = (n / dof) * meat / (denominator * denominator)
    se = float(np.sqrt(max(variance_hc1, 0.0)))
    statistic = beta / se if se > 0.0 else np.inf
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=dof))
    critical = float(stats.t.ppf(0.975, df=dof))
    return {
        "beta_standardized": beta,
        "se_hc1": se,
        "ci95_low": beta - critical * se,
        "ci95_high": beta + critical * se,
        "t_statistic": statistic,
        "p_value": p_value,
        "residual_df": float(dof),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, default=DEFAULT_WIDE)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [args.wide.resolve(), args.membership.resolve(), args.context.resolve()]
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
    if args.output_dir.exists():
        raise FileExistsError(f"create-once output already exists: {args.output_dir}")

    wide = pd.read_parquet(
        args.wide,
        columns=["participant_id", "outer_fold", *GAP_COLUMNS.values()],
    )
    context = pd.read_parquet(
        args.context,
        columns=["participant_id", *CONTROL_COLUMNS],
    )
    membership = pd.read_parquet(
        args.membership,
        columns=["participant_id", "analysis_group"],
    )
    for frame in (wide, context, membership):
        frame["participant_id"] = frame["participant_id"].astype(str)
    if wide["participant_id"].duplicated().any() or context["participant_id"].duplicated().any():
        raise RuntimeError("participant-level inputs must have unique keys")
    if membership.duplicated(["participant_id", "analysis_group"]).any():
        raise RuntimeError("duplicate group membership")
    if set(membership["analysis_group"].astype(str).unique()) != set(GROUPS):
        raise RuntimeError("membership groups do not match the frozen 16-group contract")

    base = wide.merge(context, on="participant_id", how="inner", validate="one_to_one")
    membership_by_group = {
        group: set(part["participant_id"])
        for group, part in membership.groupby("analysis_group", observed=True, sort=False)
    }
    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    gap_names = list(GAP_COLUMNS)
    required_numeric = [*GAP_COLUMNS.values(), *CONTROL_COLUMNS]

    for group in GROUPS:
        selected = base.loc[base["participant_id"].isin(membership_by_group[group])].copy()
        complete = selected[["participant_id", "outer_fold", *required_numeric]].copy()
        complete[required_numeric] = complete[required_numeric].apply(pd.to_numeric, errors="coerce")
        complete = complete.replace([np.inf, -np.inf], np.nan).dropna()
        if len(complete) < 100:
            raise RuntimeError(f"group {group} has fewer than 100 complete cases")
        gaps = zscore(complete[[GAP_COLUMNS[item] for item in gap_names]].to_numpy(float))
        control_frame = complete[list(CONTROL_COLUMNS)].copy()
        continuous = [column for column in CONTROL_COLUMNS if column not in BINARY_CONTROLS]
        control_frame[continuous] = zscore(control_frame[continuous].to_numpy(float))
        controls = control_frame.to_numpy(float)
        residual_gaps = residualize_gaps(gaps, controls)
        group_rows.append(
            {
                "analysis_group": group,
                "n_membership": len(membership_by_group[group]),
                "n_complete": len(complete),
                "complete_fraction": len(complete) / len(membership_by_group[group]),
                "outer_folds": ",".join(str(int(x)) for x in sorted(complete["outer_fold"].unique())),
            }
        )
        group_start = len(rows)
        for source_index, source in enumerate(gap_names):
            for target_index, target in enumerate(gap_names):
                if source == target:
                    continue
                estimate = directional_association(
                    residual_gaps[:, source_index],
                    residual_gaps[:, target_index],
                    nuisance_df=1 + len(CONTROL_COLUMNS),
                )
                raw_r = float(np.corrcoef(gaps[:, source_index], gaps[:, target_index])[0, 1])
                partial_r = float(
                    np.corrcoef(residual_gaps[:, source_index], residual_gaps[:, target_index])[0, 1]
                )
                rows.append(
                    {
                        "analysis_group": group,
                        "source_organ": source,
                        "target_organ": target,
                        "association": f"{source}->{target}",
                        "n_complete": len(complete),
                        "model": "target_gap_z ~ source_gap_z + seven_Fig8_clinical_covariates",
                        "other_age_gaps_in_model": 0,
                        "raw_pearson_r": raw_r,
                        "partial_pearson_r": partial_r,
                        **estimate,
                    }
                )
        p_values = np.asarray([row["p_value"] for row in rows[group_start:]], dtype=float)
        q_values = bh(p_values)
        for row, q_value in zip(rows[group_start:], q_values):
            row["q_value_bh_42_within_group"] = float(q_value)

    estimates = pd.DataFrame(rows)
    if len(estimates) != len(GROUPS) * len(ORGANS) * (len(ORGANS) - 1):
        raise RuntimeError(f"unexpected association row count: {len(estimates)}")
    matrix = estimates.pivot(
        index="association", columns="analysis_group", values="beta_standardized"
    ).reindex(columns=list(GROUPS))
    edge_rows = estimates.loc[
        estimates["source_organ"].map(ORGANS.index)
        < estimates["target_organ"].map(ORGANS.index)
    ].copy()
    edge_rows["edge"] = edge_rows.apply(
        lambda row: "--".join(sorted([str(row["source_organ"]), str(row["target_organ"])])),
        axis=1,
    )
    edge_rows = edge_rows.rename(columns={"partial_pearson_r": "undirected_partial_r"})

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig11 = args.output_dir / "Fig11"
    fig12 = args.output_dir / "Fig12"
    fig11.mkdir()
    fig12.mkdir()
    matrix.to_csv(fig11 / "AllGroups_Marginal_Association_Matrix.csv", lineterminator="\n")
    estimates.to_csv(fig12 / "marginal_association_coefficients.csv", index=False, lineterminator="\n")
    edge_rows.to_csv(fig12 / "marginal_association_unique_edges.csv", index=False, lineterminator="\n")
    pd.DataFrame(group_rows).to_csv(
        args.output_dir / "group_complete_case_counts.csv", index=False, lineterminator="\n"
    )
    manifest = {
        "schema_version": "revision0710.fig11_fig12.single_gap_adjusted_association.v1",
        "status": "COMPLETE_SOURCE_DATA_NO_RENDERING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": [
            {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
        "groups": list(GROUPS),
        "organs": list(ORGANS),
        "controls": list(CONTROL_COLUMNS),
        "estimator": "standardized OLS coefficient with HC1 SE after FWL residualization",
        "interpretation": "adjusted marginal association; not causal, mechanistic, or SEM path",
        "row_counts": {
            "participants_wide": len(wide),
            "association_rows": len(estimates),
            "matrix_rows": len(matrix),
            "matrix_columns": len(matrix.columns),
            "unique_undirected_edge_rows": len(edge_rows),
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["row_counts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

