#!/usr/bin/env python3
"""Run reviewer-requested diagnostics for the revised Fig. 11-12 associations.

This script is deliberately read-only with respect to the frozen analysis inputs.  It
reconstructs the exact complete-case, standardized, single-source adjusted regressions
used by the final Fig. 11-12 source package and adds diagnostics appropriate for those
ordinary least-squares models.  It does not fit or diagnose a structural equation model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


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
EXPECTED_INPUT_HASHES = {
    "canonical_age_gap_wide.parquet": (
        "2f0c0859a4d21b8887168c0c77c852576375ec675a2a30e9fd46074df4f04bd9"
    ),
    "display_group_membership.parquet": (
        "53bba3cd1e478b374097680eb008309b5527a489a0313cd5aac1b7251b7c462c"
    ),
    "participant_context_covariates.parquet": (
        "2b4d46d69b1c5c2c9629b27c12dce2d4c3460028fc16a0baffcdfdcbd44fb0bc"
    ),
    "marginal_association_coefficients.csv": (
        "d88fd6e57869866802dc679d7d01fa73905000fa576f296d1652ebd972c85e6f"
    ),
}


def workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
REVISION = ROOT / "Revision-0710"
DEFAULT_INPUT_ROOT = (
    REVISION / "Final_Fig_0716/Codebook_v2_Huber1/00_inputs/fig11_fig12"
)
DEFAULT_WIDE = DEFAULT_INPUT_ROOT / "canonical_age_gap_wide.parquet"
DEFAULT_MEMBERSHIP = DEFAULT_INPUT_ROOT / "display_group_membership.parquet"
DEFAULT_CONTEXT = (
    REVISION
    / "05_downstream/06_metric_v2/01_contextual_covariates/"
    "participant_context_covariates.parquet"
)
DEFAULT_COEFFICIENTS = (
    REVISION
    / "Final_Fig_0716/Codebook_v2_Huber1/Fig11_Fig12_Source/Fig12/"
    "marginal_association_coefficients.csv"
)
DEFAULT_OUTPUT = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/"
    "17_r3_5_pairwise_association_diagnostics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, default=DEFAULT_WIDE)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("zero or invalid standard deviation")
    return (values - center) / scale


def residualize(values: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, int]:
    design = np.column_stack([np.ones(len(controls), dtype=float), controls])
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != design.shape[1]:
        raise RuntimeError(f"control design rank deficient: {rank}/{design.shape[1]}")
    return values - design @ coefficients, int(rank)


def directional_beta(source: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.dot(source, source))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("source residual has zero or invalid variance")
    return float(np.dot(source, target) / denominator)


def moment_diagnostics(values: np.ndarray) -> tuple[float, float]:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    second = float(np.mean(np.square(centered)))
    if second <= 0.0:
        raise RuntimeError("residual variance is zero")
    skewness = float(np.mean(np.power(centered, 3)) / second ** 1.5)
    excess_kurtosis = float(np.mean(np.power(centered, 4)) / second**2 - 3.0)
    return skewness, excess_kurtosis


def numeric_summary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise RuntimeError("cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def main() -> int:
    args = parse_args()
    inputs = [
        args.wide.resolve(),
        args.membership.resolve(),
        args.context.resolve(),
        args.coefficients.resolve(),
    ]
    missing = [path.as_posix() for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise FileExistsError(f"create-once output already exists: {args.output_dir}")

    input_records: list[dict[str, object]] = []
    for path in inputs:
        observed = sha256_file(path)
        expected = EXPECTED_INPUT_HASHES[path.name]
        if observed.lower() != expected.lower():
            raise RuntimeError(
                f"frozen input hash mismatch for {path.name}: {observed} != {expected}"
            )
        input_records.append(
            {
                "path": path.as_posix(),
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
        )

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
    coefficients = pd.read_csv(args.coefficients)

    for frame in (wide, context, membership):
        frame["participant_id"] = frame["participant_id"].astype(str)
    if wide["participant_id"].duplicated().any():
        raise RuntimeError("wide input must contain one row per participant")
    if context["participant_id"].duplicated().any():
        raise RuntimeError("context input must contain one row per participant")
    if membership.duplicated(["participant_id", "analysis_group"]).any():
        raise RuntimeError("duplicate participant-group membership")
    if set(membership["analysis_group"].astype(str).unique()) != set(GROUPS):
        raise RuntimeError("membership groups do not match the 16-group contract")
    if len(coefficients) != len(GROUPS) * len(ORGANS) * (len(ORGANS) - 1):
        raise RuntimeError("final coefficient file does not contain 672 rows")
    if coefficients.duplicated(
        ["analysis_group", "source_organ", "target_organ"]
    ).any():
        raise RuntimeError("duplicate final association key")

    coefficient_lookup = coefficients.set_index(
        ["analysis_group", "source_organ", "target_organ"]
    )
    base = wide.merge(context, on="participant_id", how="inner", validate="one_to_one")
    membership_by_group = {
        str(group): set(part["participant_id"])
        for group, part in membership.groupby(
            "analysis_group", observed=True, sort=False
        )
    }

    required_numeric = [*GAP_COLUMNS.values(), *CONTROL_COLUMNS]
    continuous_controls = [
        column for column in CONTROL_COLUMNS if column not in BINARY_CONTROLS
    ]
    association_rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for group in GROUPS:
        selected = base.loc[
            base["participant_id"].isin(membership_by_group[group])
        ].copy()
        complete = selected[
            ["participant_id", "outer_fold", *required_numeric]
        ].copy()
        complete[required_numeric] = complete[required_numeric].apply(
            pd.to_numeric, errors="coerce"
        )
        complete = complete.replace([np.inf, -np.inf], np.nan).dropna()
        n = len(complete)
        if n < 100:
            raise RuntimeError(f"group {group} has fewer than 100 complete cases")

        raw_gaps = complete[list(GAP_COLUMNS.values())].to_numpy(float)
        gaps = zscore(raw_gaps)
        control_frame = complete[list(CONTROL_COLUMNS)].copy()
        control_frame[continuous_controls] = zscore(
            control_frame[continuous_controls].to_numpy(float)
        )
        controls = control_frame.to_numpy(float)
        residual_gaps, control_rank = residualize(gaps, controls)

        lower = np.quantile(raw_gaps, 0.01, axis=0)
        upper = np.quantile(raw_gaps, 0.99, axis=0)
        winsorized_gaps = zscore(np.clip(raw_gaps, lower, upper))
        residual_winsorized_gaps, winsor_control_rank = residualize(
            winsorized_gaps, controls
        )
        if winsor_control_rank != control_rank:
            raise RuntimeError("unexpected control-rank change after gap winsorization")

        group_rows.append(
            {
                "analysis_group": group,
                "n_membership": len(membership_by_group[group]),
                "n_complete": n,
                "complete_fraction": n / len(membership_by_group[group]),
                "residual_df_per_model": n - 9,
                "outer_folds": ",".join(
                    str(int(value))
                    for value in sorted(complete["outer_fold"].unique())
                ),
            }
        )

        for source_index, source in enumerate(ORGANS):
            source_gap = gaps[:, source_index]
            source_residual = residual_gaps[:, source_index]

            full_design = np.column_stack(
                [np.ones(n, dtype=float), source_gap, controls]
            )
            full_rank = int(np.linalg.matrix_rank(full_design))
            if full_rank != full_design.shape[1]:
                raise RuntimeError(
                    f"rank-deficient full design for {group}/{source}: "
                    f"{full_rank}/{full_design.shape[1]}"
                )
            cross_product = full_design.T @ full_design
            cross_product_inverse = np.linalg.inv(cross_product)
            leverage = np.einsum(
                "ij,jk,ik->i",
                full_design,
                cross_product_inverse,
                full_design,
                optimize=True,
            )
            if np.any(leverage < -1e-10) or np.any(leverage >= 1.0):
                raise RuntimeError(f"invalid leverage for {group}/{source}")
            leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)

            predictors = np.column_stack([source_gap, controls])
            standardized_predictors = zscore(predictors)
            correlation = (
                standardized_predictors.T @ standardized_predictors / n
            )
            eigenvalues = np.linalg.eigvalsh(correlation)
            if float(np.min(eigenvalues)) <= 0.0:
                raise RuntimeError(f"invalid correlation eigenvalue for {group}/{source}")
            inverse_correlation = np.linalg.inv(correlation)
            vifs = np.diag(inverse_correlation)
            condition_index = float(
                np.sqrt(float(np.max(eigenvalues)) / float(np.min(eigenvalues)))
            )
            design_rows.append(
                {
                    "analysis_group": group,
                    "source_organ": source,
                    "n_complete": n,
                    "design_rank": full_rank,
                    "design_columns": full_design.shape[1],
                    "source_gap_vif": float(vifs[0]),
                    "maximum_predictor_vif": float(np.max(vifs)),
                    "condition_index_standardized_predictors": condition_index,
                    "maximum_leverage": float(np.max(leverage)),
                    "p999_leverage": float(np.quantile(leverage, 0.999)),
                }
            )

            quadratic_design = np.column_stack(
                [
                    np.ones(n, dtype=float),
                    source_gap,
                    np.square(source_gap),
                    controls,
                ]
            )
            quadratic_rank = int(np.linalg.matrix_rank(quadratic_design))
            if quadratic_rank != quadratic_design.shape[1]:
                raise RuntimeError(
                    f"rank-deficient quadratic design for {group}/{source}"
                )
            quadratic_coefficients = np.linalg.solve(
                quadratic_design.T @ quadratic_design,
                quadratic_design.T @ gaps,
            )
            quadratic_residuals = gaps - quadratic_design @ quadratic_coefficients
            quadratic_sse = np.sum(np.square(quadratic_residuals), axis=0)

            for target_index, target in enumerate(ORGANS):
                if source == target:
                    continue
                target_residual = residual_gaps[:, target_index]
                beta = directional_beta(source_residual, target_residual)
                model_residual = target_residual - beta * source_residual
                sse = float(np.dot(model_residual, model_residual))
                sst = float(
                    np.dot(
                        gaps[:, target_index] - gaps[:, target_index].mean(),
                        gaps[:, target_index] - gaps[:, target_index].mean(),
                    )
                )
                r_squared = 1.0 - sse / sst
                adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - 9)
                partial_r = float(
                    np.corrcoef(source_residual, target_residual)[0, 1]
                )
                quadratic_delta_r_squared = float(
                    max(0.0, (sse - float(quadratic_sse[target_index])) / sst)
                )

                mse = sse / (n - 9)
                cooks_distance = (
                    np.square(model_residual)
                    / (9.0 * mse)
                    * leverage
                    / np.square(1.0 - leverage)
                )
                skewness, excess_kurtosis = moment_diagnostics(model_residual)
                winsorized_beta = directional_beta(
                    residual_winsorized_gaps[:, source_index],
                    residual_winsorized_gaps[:, target_index],
                )

                final_row = coefficient_lookup.loc[(group, source, target)]
                final_beta = float(final_row["beta_standardized"])
                final_n = int(final_row["n_complete"])
                if final_n != n:
                    raise RuntimeError(
                        f"complete-case mismatch for {group}/{source}/{target}"
                    )

                association_rows.append(
                    {
                        "analysis_group": group,
                        "source_organ": source,
                        "target_organ": target,
                        "n_complete": n,
                        "residual_df": n - 9,
                        "beta_recomputed": beta,
                        "beta_final_source": final_beta,
                        "absolute_beta_parity_error": abs(beta - final_beta),
                        "full_model_r_squared": r_squared,
                        "full_model_adjusted_r_squared": adjusted_r_squared,
                        "partial_pearson_r": partial_r,
                        "partial_r_squared": partial_r**2,
                        "quadratic_delta_r_squared": quadratic_delta_r_squared,
                        "residual_skewness": skewness,
                        "residual_excess_kurtosis": excess_kurtosis,
                        "maximum_cooks_distance": float(np.max(cooks_distance)),
                        "p999_cooks_distance": float(
                            np.quantile(cooks_distance, 0.999)
                        ),
                        "fraction_cooks_distance_gt_4_over_n": float(
                            np.mean(cooks_distance > (4.0 / n))
                        ),
                        "n_cooks_distance_gt_1": int(np.sum(cooks_distance > 1.0)),
                        "winsorized_1_99_beta": winsorized_beta,
                        "absolute_beta_change_after_winsorization": abs(
                            winsorized_beta - beta
                        ),
                    }
                )

    associations = pd.DataFrame(association_rows)
    designs = pd.DataFrame(design_rows)
    groups = pd.DataFrame(group_rows)

    expected_associations = len(GROUPS) * len(ORGANS) * (len(ORGANS) - 1)
    expected_designs = len(GROUPS) * len(ORGANS)
    max_parity_error = float(associations["absolute_beta_parity_error"].max())
    qa = {
        "status": "PASS",
        "expected_association_rows": expected_associations,
        "observed_association_rows": len(associations),
        "expected_design_rows": expected_designs,
        "observed_design_rows": len(designs),
        "expected_group_rows": len(GROUPS),
        "observed_group_rows": len(groups),
        "all_outputs_finite": bool(
            np.isfinite(
                associations.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(
                designs.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(groups["complete_fraction"].to_numpy(float)).all()
        ),
        "maximum_beta_parity_error": max_parity_error,
        "beta_parity_tolerance": 1e-10,
        "all_full_designs_rank_complete": bool(
            (designs["design_rank"] == designs["design_columns"]).all()
        ),
        "all_672_models_estimable": len(associations) == expected_associations,
        "all_cooks_distance_below_1": bool(
            (associations["n_cooks_distance_gt_1"] == 0).all()
        ),
        "input_hashes_match_frozen_contract": True,
    }
    if (
        len(associations) != expected_associations
        or len(designs) != expected_designs
        or len(groups) != len(GROUPS)
        or not qa["all_outputs_finite"]
        or max_parity_error > qa["beta_parity_tolerance"]
        or not qa["all_full_designs_rank_complete"]
    ):
        qa["status"] = "FAIL"
        raise RuntimeError(json.dumps(qa, indent=2))

    summary = {
        "schema_version": "r3_5_pairwise_association_diagnostics.v1",
        "data_use": {
            "groups": len(groups),
            "association_models": len(associations),
            "design_matrices": len(designs),
            "n_complete": numeric_summary(groups["n_complete"]),
            "complete_fraction": numeric_summary(groups["complete_fraction"]),
            "residual_df": numeric_summary(groups["residual_df_per_model"]),
        },
        "estimates_and_fit": {
            "beta": numeric_summary(associations["beta_recomputed"]),
            "full_model_adjusted_r_squared": numeric_summary(
                associations["full_model_adjusted_r_squared"]
            ),
            "partial_r_squared": numeric_summary(
                associations["partial_r_squared"]
            ),
        },
        "linearity": {
            "quadratic_delta_r_squared": numeric_summary(
                associations["quadratic_delta_r_squared"]
            )
        },
        "residual_shape": {
            "residual_skewness": numeric_summary(
                associations["residual_skewness"]
            ),
            "absolute_residual_skewness": numeric_summary(
                associations["residual_skewness"].abs()
            ),
            "residual_excess_kurtosis": numeric_summary(
                associations["residual_excess_kurtosis"]
            ),
        },
        "influence_and_outlier_sensitivity": {
            "maximum_cooks_distance_per_model": numeric_summary(
                associations["maximum_cooks_distance"]
            ),
            "fraction_cooks_distance_gt_4_over_n": numeric_summary(
                associations["fraction_cooks_distance_gt_4_over_n"]
            ),
            "absolute_beta_change_after_1_99_winsorization": numeric_summary(
                associations["absolute_beta_change_after_winsorization"]
            ),
            "models_with_any_cooks_distance_gt_1": int(
                (associations["n_cooks_distance_gt_1"] > 0).sum()
            ),
        },
        "collinearity": {
            "source_gap_vif": numeric_summary(designs["source_gap_vif"]),
            "maximum_predictor_vif": numeric_summary(
                designs["maximum_predictor_vif"]
            ),
            "condition_index_standardized_predictors": numeric_summary(
                designs["condition_index_standardized_predictors"]
            ),
        },
        "qa": qa,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_data = args.output_dir / "Source_Data"
    source_data.mkdir()
    association_path = source_data / "association_diagnostics.csv"
    design_path = source_data / "design_collinearity_diagnostics.csv"
    group_path = source_data / "group_data_use_and_degrees_of_freedom.csv"
    associations.to_csv(association_path, index=False, lineterminator="\n")
    designs.to_csv(design_path, index=False, lineterminator="\n")
    groups.to_csv(group_path, index=False, lineterminator="\n")
    summary_path = args.output_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    qa_path = args.output_dir / "QA_REPORT.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    readme = f"""# R3.5 配对调整后关联诊断

本目录基于最终 Fig. 11-12 的同一参与者级输入和完整病例口径，重建 16 个人群中的 672 项单来源器官 gap 调整后线性关联，并补充与该回归 estimand 相匹配的模型诊断。未重新训练模型，也未拟合结构方程模型。

## 核心结果

- 完整病例：{int(groups['n_complete'].min()):,}-{int(groups['n_complete'].max()):,}；占各组成员的 {100 * groups['complete_fraction'].min():.1f}%-{100 * groups['complete_fraction'].max():.1f}%。
- 每个模型残差自由度：{int(groups['residual_df_per_model'].min()):,}-{int(groups['residual_df_per_model'].max()):,}；672 项关联均可估计。
- 调整后线性模型 adjusted R2：中位数 {associations['full_model_adjusted_r_squared'].median():.6f}，范围 {associations['full_model_adjusted_r_squared'].min():.6f}-{associations['full_model_adjusted_r_squared'].max():.6f}。
- 加入来源 gap 二次项后的 Delta R2：中位数 {associations['quadratic_delta_r_squared'].median():.6f}，最大值 {associations['quadratic_delta_r_squared'].max():.6f}。
- 1%/99% 组内缩尾后，beta 绝对变化：中位数 {associations['absolute_beta_change_after_winsorization'].median():.6f}，最大值 {associations['absolute_beta_change_after_winsorization'].max():.6f}。
- 最大 predictor VIF：中位数 {designs['maximum_predictor_vif'].median():.3f}，全局最大 {designs['maximum_predictor_vif'].max():.3f}；标准化 predictor 条件指数最大 {designs['condition_index_standardized_predictors'].max():.3f}。
- 任何 Cook's distance > 1 的模型数：{int((associations['n_cooks_distance_gt_1'] > 0).sum())}/672。
- 与最终系数表的最大绝对复算误差：{max_parity_error:.3e}。

## 文件说明

- `Source_Data/association_diagnostics.csv`：逐人群、逐有序器官对的拟合、线性、残差、影响点和缩尾敏感性结果。
- `Source_Data/design_collinearity_diagnostics.csv`：112 个“人群 x 来源器官”设计矩阵的秩、VIF、条件指数和 leverage。
- `Source_Data/group_data_use_and_degrees_of_freedom.csv`：16 个人群的成员数、完整病例数/比例和逐模型残差自由度。
- `SUMMARY.json`：全部诊断的汇总分布。
- `QA_REPORT.json`：行数、有限值、矩阵秩、输入哈希和最终系数复算一致性检查。
- `MANIFEST.json`：输入、输出、环境和 SHA-256 记录。

## 解释边界

这些诊断用于评价修订后的配对调整线性关联是否可估计、近似线性、是否受单个高影响观测或明显设计矩阵共线性主导。它们不能把观测性关联转化为器官间因果方向或机制路径，也不能排除未测量疾病负担、医疗接触或生物因素造成的残余混杂。
"""
    readme_path = args.output_dir / "README_CN.md"
    readme_path.write_text(readme, encoding="utf-8")

    output_paths = [
        association_path,
        design_path,
        group_path,
        summary_path,
        qa_path,
        readme_path,
    ]
    manifest = {
        "schema_version": "r3_5_pairwise_association_diagnostics.manifest.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "inputs": input_records,
        "analysis_contract": {
            "groups": list(GROUPS),
            "organs": list(ORGANS),
            "controls": list(CONTROL_COLUMNS),
            "primary_model": (
                "target gap z ~ one source gap z + seven clinical covariates"
            ),
            "estimator": "OLS after FWL residualization",
            "inference_in_final_source": "HC1 heteroskedasticity-robust SE",
            "multiple_testing_in_final_source": (
                "Benjamini-Hochberg correction over 42 associations within each group"
            ),
            "linearity_diagnostic": (
                "incremental R2 after adding a within-group squared source-gap term"
            ),
            "outlier_sensitivity": (
                "repeat the full standardization/residualization contract after "
                "within-group 1st/99th percentile winsorization of all seven gaps"
            ),
            "influence_diagnostic": "OLS leverage and Cook's distance",
            "collinearity_diagnostic": (
                "VIF and condition index for standardized source gap plus seven covariates"
            ),
        },
        "row_counts": {
            "association_diagnostics": len(associations),
            "design_diagnostics": len(designs),
            "group_data_use": len(groups),
        },
        "qa": qa,
        "outputs": [
            {
                "path": path.relative_to(args.output_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
