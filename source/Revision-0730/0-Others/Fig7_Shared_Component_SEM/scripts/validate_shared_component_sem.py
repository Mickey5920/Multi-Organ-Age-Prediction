#!/usr/bin/env python3
"""Validate the EHR-AGE shared-component SEM on current participant-disjoint OSAGs.

The prespecified measurement model contains one shared factor measured by the
seven corrected organ-specific age gaps. It is fitted with diagonally weighted
least squares. Development folds 0-2 and validation folds 3-4 are separated,
and fivefold, clinical-state, distributional and influence sensitivities are
reported. No organ-to-organ directional or causal path is specified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import semopy
from scipy import optimize, stats
from sklearn.covariance import MinCovDet


ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
MODEL_SYNTAX = "Shared =~ " + " + ".join(ORGANS)


def find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "Revision-0730").is_dir() and (candidate / "Revision-0710").is_dir():
            return candidate
    raise RuntimeError("repository root not found")


ROOT = find_root()
WIDE = ROOT / "Revision-0710/06_figures_and_statistics/02_canonical_analysis_data/02_oof_wide/canonical_age_gap_wide.parquet"
OUTCOMES = ROOT / "Revision-0710/01_data_rebuild/02_outcomes_60m/organ_outcomes_60m.parquet"
DISPLAY_MEMBERSHIP = ROOT / (
    "Revision-0710/Final_Fig_0716/Codebook_v2_Huber1/00_inputs/"
    "fig11_fig12/display_group_membership.parquet"
)
CONTEXT = ROOT / (
    "Revision-0710/05_downstream/06_metric_v2/01_contextual_covariates/"
    "participant_context_covariates.parquet"
)
BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE / "source_data"
QA = BASE / "qa"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_wide() -> pd.DataFrame:
    columns = [
        "participant_id",
        "outer_fold",
        "actual_age",
        "sex",
        "strict_healthy_reference_60m",
        "phenotype_missing_count",
        *[f"{organ}__corrected_age_gap" for organ in ORGANS],
    ]
    frame = pl.read_parquet(WIDE).select(columns).rename(
        {f"{organ}__corrected_age_gap": organ for organ in ORGANS}
    )
    data = pd.DataFrame(frame.to_dict(as_series=False))
    data["outer_fold"] = data["outer_fold"].astype(int)
    if data["participant_id"].duplicated().any():
        raise RuntimeError("participant IDs are not unique")
    if not np.isfinite(data[list(ORGANS)].to_numpy(float)).all():
        raise RuntimeError("corrected OSAG matrix contains missing or non-finite values")
    return data


def clinical_memberships() -> dict[str, set[str]]:
    frame = pl.read_parquet(OUTCOMES).filter(pl.col("organ_system") == "cardiovascular")
    prevalent = set(
        frame.filter(pl.col("has_prevalent_target") == 1)["participant_id"].to_list()
    )
    other_system = set(
        frame.filter(pl.col("cohort_state_60m") == "target_disease_free_other_system_disease")[
            "participant_id"
        ].to_list()
    )
    return {
        "baseline_cardiovascular_disease": prevalent,
        "cardiovascular_free_other_organ_disease": other_system,
    }


def figure7_group_memberships() -> dict[str, set[str]]:
    group_map = {
        "figure7_health": "Health",
        "figure7_cardiovascular_health": "cardiovascular_Health",
        "figure7_cardiovascular_disease": "cardiovascular",
        "figure7_other_disease": "Other_Disease",
    }
    frame = pl.read_parquet(DISPLAY_MEMBERSHIP).filter(
        pl.col("analysis_group").is_in(list(group_map.values()))
    )
    if frame.select(pl.struct(["participant_id", "analysis_group"]).n_unique()).item() != frame.height:
        raise RuntimeError("duplicate participant-group memberships")
    memberships: dict[str, set[str]] = {}
    for sample, group in group_map.items():
        ids = set(
            frame.filter(pl.col("analysis_group") == group)["participant_id"].to_list()
        )
        if len(ids) < 1000:
            raise RuntimeError(f"insufficient Figure 7 membership for {group}")
        memberships[sample] = ids
    context = pl.read_parquet(CONTEXT, columns=["participant_id", "has_preindex_icd"])
    valid_followup = pl.read_parquet(OUTCOMES).filter(
        (pl.col("organ_system") == "cardiovascular") & (pl.col("at_risk") == 1)
    ).select("participant_id")
    baseline_health = context.filter(pl.col("has_preindex_icd") == 0).join(
        valid_followup, on="participant_id", how="inner"
    )
    memberships["figure7_baseline_health_risk"] = set(
        baseline_health["participant_id"].to_list()
    )
    if len(memberships["figure7_baseline_health_risk"]) < 1000:
        raise RuntimeError("insufficient Baseline Health risk-group membership")
    return memberships


def standardize(frame: pd.DataFrame, mean: pd.Series | None = None, sd: pd.Series | None = None):
    values = frame[list(ORGANS)].astype(float)
    if mean is None:
        mean = values.mean(axis=0)
    if sd is None:
        sd = values.std(axis=0, ddof=0)
    if (sd <= 0).any():
        raise RuntimeError("non-positive OSAG scale")
    return (values - mean) / sd, mean, sd


def rmsea_confidence_interval(chi_square: float, df: int, n: int, alpha: float = 0.10):
    if df <= 0 or n <= 1:
        return np.nan, np.nan

    def ncp_for_cdf(target: float) -> float:
        at_zero = stats.ncx2.cdf(chi_square, df, 0.0)
        if target >= at_zero:
            return 0.0
        upper = max(chi_square, 1.0)
        while stats.ncx2.cdf(chi_square, df, upper) > target and upper < 1e10:
            upper *= 2.0
        return optimize.brentq(
            lambda value: stats.ncx2.cdf(chi_square, df, value) - target,
            0.0,
            upper,
        )

    lower_ncp = ncp_for_cdf(1.0 - alpha / 2.0)
    upper_ncp = ncp_for_cdf(alpha / 2.0)
    return (
        float(np.sqrt(max(lower_ncp, 0.0) / (df * (n - 1)))),
        float(np.sqrt(max(upper_ncp, 0.0) / (df * (n - 1)))),
    )


def fit_sem(frame: pd.DataFrame, sample: str, transform=None):
    data, _, _ = standardize(frame)
    if transform == "winsorized_1_99":
        data = data.clip(lower=data.quantile(0.01), upper=data.quantile(0.99), axis=1)
        data, _, _ = standardize(data.rename(columns={organ: organ for organ in ORGANS}))
    model = semopy.Model(MODEL_SYNTAX)
    optimizer = model.fit(data=data, obj="DWLS", solver="SLSQP")
    fit = semopy.calc_stats(model).iloc[0]
    estimates = model.inspect(std_est=True)
    loadings = estimates.loc[(estimates["op"] == "~") & (estimates["rval"] == "Shared")].copy()
    residuals = estimates.loc[
        (estimates["op"] == "~~")
        & (estimates["lval"] == estimates["rval"])
        & estimates["lval"].isin(ORGANS)
    ].copy()
    order = list(model.vars["observed"])
    observed = data[order].cov().to_numpy(float)
    implied = model.calc_sigma()[0]
    scale = np.sqrt(np.outer(np.diag(observed), np.diag(observed)))
    standardized_residual = (observed - implied) / scale
    off_diagonal = ~np.eye(len(ORGANS), dtype=bool)
    srmr = float(np.sqrt(np.mean(np.square(standardized_residual[off_diagonal]))))
    chi_square = float(fit["chi2"])
    df = int(fit["DoF"])
    rmsea_low, rmsea_high = rmsea_confidence_interval(chi_square, df, len(data))
    fit_row = {
        "sample": sample,
        "n": int(len(data)),
        "estimator": "DWLS",
        "optimizer_success": bool(optimizer.success),
        "df": df,
        "chi_square": chi_square,
        "cfi": float(fit["CFI"]),
        "tli": float(fit["TLI"]),
        "rmsea": float(fit["RMSEA"]),
        "rmsea_ci90_low": rmsea_low,
        "rmsea_ci90_high": rmsea_high,
        "srmr": srmr,
        "max_abs_standardized_covariance_residual": float(
            np.max(np.abs(standardized_residual[off_diagonal]))
        ),
        "loading_min": float(loadings["Est. Std"].min()),
        "loading_max": float(loadings["Est. Std"].max()),
        "residual_variance_min": float(residuals["Est. Std"].min()),
        "residual_variance_max": float(residuals["Est. Std"].max()),
        "admissible": bool(
            optimizer.success
            and (loadings["Est. Std"].astype(float) > 0).all()
            and (residuals["Est. Std"].astype(float) > 0).all()
        ),
    }
    parameter_rows = []
    for organ in ORGANS:
        loading = loadings.loc[loadings["lval"] == organ].iloc[0]
        residual = residuals.loc[residuals["lval"] == organ].iloc[0]
        parameter_rows.append(
            {
                "sample": sample,
                "organ": organ,
                "loading_unstandardized": float(loading["Estimate"]),
                "loading_standardized": float(loading["Est. Std"]),
                "loading_se": np.nan if loading["Std. Err"] == "-" else float(loading["Std. Err"]),
                "residual_variance_unstandardized": float(residual["Estimate"]),
                "residual_variance_standardized": float(residual["Est. Std"]),
                "shared_variance_fraction": float(loading["Est. Std"]) ** 2,
            }
        )
    return model, data, fit_row, parameter_rows


def correlation_diagnostics(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    data, _, _ = standardize(frame)
    correlation = data.corr().to_numpy(float)
    eigenvalues = np.linalg.eigvalsh(correlation)
    off_diagonal = correlation[~np.eye(len(ORGANS), dtype=bool)]
    diagnostics = {
        "pairwise_correlation_min": float(np.min(off_diagonal)),
        "pairwise_correlation_max": float(np.max(off_diagonal)),
        "correlation_eigenvalue_min": float(np.min(eigenvalues)),
        "correlation_eigenvalue_max": float(np.max(eigenvalues)),
        "correlation_condition_number": float(np.max(eigenvalues) / np.min(eigenvalues)),
        "correlation_determinant": float(np.linalg.det(correlation)),
        "positive_definite": bool(np.min(eigenvalues) > 0),
    }
    matrix = pd.DataFrame(correlation, index=ORGANS, columns=ORGANS)
    return diagnostics, matrix


def distribution_diagnostics(data: pd.DataFrame, model: semopy.Model, standardized: pd.DataFrame):
    factor = model.predict_factors(standardized)["Shared"].to_numpy(float)
    factor = (factor - factor.mean()) / factor.std(ddof=0)
    estimates = model.inspect(std_est=True)
    loading_map = {
        row["lval"]: float(row["Est. Std"])
        for _, row in estimates.loc[(estimates["op"] == "~") & (estimates["rval"] == "Shared")].iterrows()
    }
    rows = []
    for organ in ORGANS:
        observed = standardized[organ].to_numpy(float)
        residual = observed - loading_map[organ] * factor
        rows.append(
            {
                "organ": organ,
                "observed_skewness": float(stats.skew(observed, bias=False)),
                "observed_excess_kurtosis": float(stats.kurtosis(observed, fisher=True, bias=False)),
                "factor_score_residual_skewness": float(stats.skew(residual, bias=False)),
                "factor_score_residual_excess_kurtosis": float(
                    stats.kurtosis(residual, fisher=True, bias=False)
                ),
            }
        )
    correlation = standardized.corr().to_numpy(float)
    squared_distance = np.einsum(
        "ij,jk,ik->i",
        standardized.to_numpy(float),
        np.linalg.inv(correlation),
        standardized.to_numpy(float),
    )
    global_diagnostics = {
        "mardia_kurtosis": float(np.mean(np.square(squared_distance))),
        "mardia_kurtosis_normal_expectation": float(len(ORGANS) * (len(ORGANS) + 2)),
        "mahalanobis_distance_squared_median": float(np.median(squared_distance)),
        "mahalanobis_distance_squared_p99": float(np.quantile(squared_distance, 0.99)),
        "complete_osag_rows": int(len(data)),
        "missing_osag_cells": int(data[list(ORGANS)].isna().sum().sum()),
    }
    return pd.DataFrame(rows), global_diagnostics


def linearity_diagnostics(standardized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left_index, left in enumerate(ORGANS):
        x = standardized[left].to_numpy(float)
        x2 = np.square(x)
        design = np.column_stack([np.ones(len(x)), x, x2])
        for right in ORGANS[left_index + 1 :]:
            y = standardized[right].to_numpy(float)
            linear_r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
            coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            quadratic_r2 = float(1.0 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2))
            rows.append(
                {
                    "organ_1": left,
                    "organ_2": right,
                    "linear_r2": linear_r2,
                    "quadratic_r2": quadratic_r2,
                    "quadratic_incremental_r2": quadratic_r2 - linear_r2,
                }
            )
    return pd.DataFrame(rows)


def robust_outlier_subset(frame: pd.DataFrame):
    standardized, _, _ = standardize(frame)
    rng = np.random.default_rng(20260803)
    reference_index = rng.choice(len(standardized), size=min(20000, len(standardized)), replace=False)
    estimator = MinCovDet(random_state=20260803, support_fraction=0.8).fit(
        standardized.iloc[reference_index].to_numpy(float)
    )
    distances = estimator.mahalanobis(standardized.to_numpy(float))
    threshold = float(np.quantile(distances, 0.99))
    keep = distances <= threshold
    diagnostics = {
        "robust_mahalanobis_reference_n": int(len(reference_index)),
        "robust_mahalanobis_exclusion_threshold_p99": threshold,
        "excluded_n": int((~keep).sum()),
        "retained_n": int(keep.sum()),
    }
    return frame.loc[keep].copy(), diagnostics


def cross_fitted_factor_scores(frame: pd.DataFrame):
    score = pd.Series(np.nan, index=frame.index, dtype=float)
    loading_rows = []
    for fold in range(5):
        train = frame.loc[frame["outer_fold"] != fold]
        test = frame.loc[frame["outer_fold"] == fold]
        train_z, train_mean, train_sd = standardize(train)
        test_z, _, _ = standardize(test, train_mean, train_sd)
        model = semopy.Model(MODEL_SYNTAX)
        optimizer = model.fit(data=train_z, obj="DWLS", solver="SLSQP")
        if not optimizer.success:
            raise RuntimeError(f"cross-fitted SEM failed for held-out fold {fold}")
        fold_score = model.predict_factors(test_z)["Shared"].to_numpy(float)
        if np.corrcoef(fold_score, test_z.mean(axis=1).to_numpy(float))[0, 1] < 0:
            fold_score *= -1.0
        score.loc[test.index] = fold_score
        estimates = model.inspect(std_est=True)
        for organ in ORGANS:
            loading = estimates.loc[
                (estimates["op"] == "~")
                & (estimates["rval"] == "Shared")
                & (estimates["lval"] == organ),
                "Est. Std",
            ].iloc[0]
            loading_rows.append(
                {
                    "held_out_fold": fold,
                    "training_n": len(train),
                    "held_out_n": len(test),
                    "organ": organ,
                    "loading_standardized": float(loading),
                }
            )
    if score.isna().any():
        raise RuntimeError("cross-fitted factor scores are incomplete")
    result = frame[["participant_id", "outer_fold"]].copy()
    result["shared_factor_score_cross_fitted"] = score
    result["shared_factor_score_cross_fitted_z"] = (
        score - score.mean()
    ) / score.std(ddof=0)
    return result, pd.DataFrame(loading_rows)


def main() -> int:
    SOURCE.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    data = load_wide()
    memberships = clinical_memberships()
    figure7_memberships = figure7_group_memberships()
    samples = {
        "development_folds_0_1_2": data.loc[data["outer_fold"].isin([0, 1, 2])],
        "validation_folds_3_4": data.loc[data["outer_fold"].isin([3, 4])],
        "full_cohort": data,
        **{f"outer_fold_{fold}": data.loc[data["outer_fold"] == fold] for fold in range(5)},
        "baseline_cardiovascular_disease": data.loc[
            data["participant_id"].isin(memberships["baseline_cardiovascular_disease"])
        ],
        "cardiovascular_free_other_organ_disease": data.loc[
            data["participant_id"].isin(memberships["cardiovascular_free_other_organ_disease"])
        ],
        **{
            sample: data.loc[data["participant_id"].isin(ids)]
            for sample, ids in figure7_memberships.items()
        },
    }
    fit_rows = []
    parameter_rows = []
    fitted_models = {}
    standardized_samples = {}
    for sample, frame in samples.items():
        print(f"Fitting SEM sample: {sample} (n={len(frame)})", flush=True)
        model, standardized, fit, parameters = fit_sem(frame, sample)
        fitted_models[sample] = model
        standardized_samples[sample] = standardized
        fit_rows.append(fit)
        parameter_rows.extend(parameters)

    _, _, win_fit, win_parameters = fit_sem(data, "full_cohort_winsorized_1_99", "winsorized_1_99")
    fit_rows.append(win_fit)
    parameter_rows.extend(win_parameters)
    no_outliers, outlier_diagnostics = robust_outlier_subset(data)
    _, _, outlier_fit, outlier_parameters = fit_sem(
        no_outliers, "full_cohort_excluding_top_1pct_robust_mahalanobis"
    )
    fit_rows.append(outlier_fit)
    parameter_rows.extend(outlier_parameters)

    fit_table = pd.DataFrame(fit_rows)
    parameter_table = pd.DataFrame(parameter_rows)
    correlation_summary, correlation_matrix = correlation_diagnostics(data)
    distribution_table, distribution_summary = distribution_diagnostics(
        data, fitted_models["full_cohort"], standardized_samples["full_cohort"]
    )
    linearity_table = linearity_diagnostics(standardized_samples["full_cohort"])
    scores, crossfit_loadings = cross_fitted_factor_scores(data)

    full_parameters = parameter_table.loc[parameter_table["sample"] == "full_cohort"]
    loadings = full_parameters["loading_standardized"].to_numpy(float)
    residuals = full_parameters["residual_variance_standardized"].to_numpy(float)
    reliability = {
        "mean_shared_variance_fraction": float(np.mean(np.square(loadings))),
        "composite_reliability": float(np.sum(loadings) ** 2 / (np.sum(loadings) ** 2 + np.sum(residuals))),
        "average_variance_extracted": float(np.sum(np.square(loadings)) / (np.sum(np.square(loadings)) + np.sum(residuals))),
    }
    fold_matrix = parameter_table.loc[
        parameter_table["sample"].str.startswith("outer_fold_"),
        ["sample", "organ", "loading_standardized"],
    ].pivot(index="sample", columns="organ", values="loading_standardized")
    congruences = []
    for left_index, left in enumerate(fold_matrix.index):
        for right in fold_matrix.index[left_index + 1 :]:
            a = fold_matrix.loc[left].to_numpy(float)
            b = fold_matrix.loc[right].to_numpy(float)
            congruences.append(float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))))
    stability = {
        "fivefold_tucker_congruence_min": float(np.min(congruences)),
        "fivefold_tucker_congruence_max": float(np.max(congruences)),
        "fivefold_loading_absolute_range_max": float(
            (fold_matrix.max(axis=0) - fold_matrix.min(axis=0)).max()
        ),
        "development_validation_loading_max_abs_difference": float(
            np.max(
                np.abs(
                    parameter_table.loc[
                        parameter_table["sample"] == "development_folds_0_1_2"
                    ].set_index("organ")["loading_standardized"]
                    - parameter_table.loc[
                        parameter_table["sample"] == "validation_folds_3_4"
                    ].set_index("organ")["loading_standardized"]
                )
            )
        ),
    }

    fit_table.to_csv(SOURCE / "sem_fit_indices.csv", index=False)
    parameter_table.to_csv(SOURCE / "sem_standardized_parameters.csv", index=False)
    correlation_matrix.to_csv(SOURCE / "osag_correlation_matrix.csv")
    distribution_table.to_csv(SOURCE / "sem_distribution_diagnostics.csv", index=False)
    linearity_table.to_csv(SOURCE / "sem_pairwise_linearity_diagnostics.csv", index=False)
    pl.from_pandas(scores).write_parquet(
        SOURCE / "participant_cross_fitted_shared_factor_scores.parquet"
    )
    crossfit_loadings.to_csv(SOURCE / "cross_fitted_sem_loadings.csv", index=False)

    summary = {
        "analysis": "prespecified one-factor covariance-based SEM",
        "estimator": "diagonally weighted least squares",
        "model_syntax": MODEL_SYNTAX,
        "observed_variables": list(ORGANS),
        "latent_variables": ["Shared"],
        "free_parameters": 14,
        "sample_moments": 28,
        "degrees_of_freedom": 14,
        "identification": "overidentified marker-variable model",
        "directional_organ_paths": 0,
        "organ_to_organ_causal_claim": False,
        "input_rows": len(data),
        "input_hashes": {
            str(WIDE.relative_to(ROOT)).replace("\\", "/"): sha256(WIDE),
            str(OUTCOMES.relative_to(ROOT)).replace("\\", "/"): sha256(OUTCOMES),
            str(DISPLAY_MEMBERSHIP.relative_to(ROOT)).replace("\\", "/"): sha256(DISPLAY_MEMBERSHIP),
            str(CONTEXT.relative_to(ROOT)).replace("\\", "/"): sha256(CONTEXT),
        },
        "correlation_diagnostics": correlation_summary,
        "distribution_diagnostics": distribution_summary,
        "outlier_sensitivity": outlier_diagnostics,
        "measurement_reliability": reliability,
        "stability": stability,
        "linearity": {
            "quadratic_incremental_r2_median": float(linearity_table["quadratic_incremental_r2"].median()),
            "quadratic_incremental_r2_max": float(linearity_table["quadratic_incremental_r2"].max()),
        },
        "risk_evidence_linkage": {
            "source": "Supplementary Table ST6, rows 206-213",
            "role": "existing fully adjusted prospective evidence for the seven target-organ OSAG risk markers",
            "not_refitted_in_sem_validation": True,
        },
        "strict_health_reference_role": (
            "calibration reference only; not used as an SEM validation subgroup because "
            "its seven corrected OSAGs are zero-referenced by design"
        ),
    }
    (SOURCE / "sem_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (SOURCE / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
                "script_sha256": sha256(Path(__file__).resolve()),
                "outputs": {
                    path.name: sha256(path)
                    for path in sorted(SOURCE.iterdir())
                    if path.is_file() and path.name != "analysis_manifest.json"
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(fit_table.to_string(index=False))
    print(parameter_table.loc[parameter_table["sample"] == "full_cohort"].to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
