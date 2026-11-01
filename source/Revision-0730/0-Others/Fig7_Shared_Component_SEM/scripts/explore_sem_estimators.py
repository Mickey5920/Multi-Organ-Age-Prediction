#!/usr/bin/env python3
"""Compare covariance-based SEM estimators for the prespecified shared factor."""

from __future__ import annotations

from pathlib import Path

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


def find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "Revision-0730").is_dir() and (candidate / "Revision-0710").is_dir():
            return candidate
    raise RuntimeError("repository root not found")


ROOT = find_root()
WIDE = ROOT / "Revision-0710/06_figures_and_statistics/02_canonical_analysis_data/02_oof_wide/canonical_age_gap_wide.parquet"
OUTPUT = Path(__file__).resolve().parent.parent / "development"


def load_data() -> pd.DataFrame:
    columns = ["outer_fold", *[f"{organ}__corrected_age_gap" for organ in ORGANS]]
    frame = pl.read_parquet(WIDE).select(columns).rename(
        {f"{organ}__corrected_age_gap": organ for organ in ORGANS}
    )
    return pd.DataFrame(frame.to_dict(as_series=False))


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[list(ORGANS)].astype(float)
    return (values - values.mean()) / values.std(ddof=0)


def fit(data: pd.DataFrame, sample: str, estimator: str) -> dict[str, object]:
    model = semopy.Model("Shared =~ " + " + ".join(ORGANS))
    if estimator in {"DWLS", "WLS"}:
        optimizer = model.fit(data=data, obj=estimator, solver="SLSQP")
    else:
        optimizer = model.fit(cov=data.cov(), n_samples=len(data), obj=estimator, solver="SLSQP")
    stats = semopy.calc_stats(model).iloc[0]
    estimates = model.inspect(std_est=True)
    loadings = estimates.loc[(estimates["op"] == "~") & (estimates["rval"] == "Shared")]
    residuals = estimates.loc[
        (estimates["op"] == "~~")
        & (estimates["lval"] == estimates["rval"])
        & estimates["lval"].isin(ORGANS)
    ]
    order = list(model.vars["observed"])
    observed = data[order].cov().to_numpy(float)
    implied = model.calc_sigma()[0]
    scale = np.sqrt(np.outer(np.diag(observed), np.diag(observed)))
    matrix_residual = (observed - implied) / scale
    off_diagonal = ~np.eye(len(ORGANS), dtype=bool)
    return {
        "sample": sample,
        "estimator": estimator,
        "n": len(data),
        "success": bool(optimizer.success),
        "df": int(stats["DoF"]),
        "chi_square": float(stats["chi2"]),
        "cfi": float(stats["CFI"]),
        "tli": float(stats["TLI"]),
        "rmsea": float(stats["RMSEA"]),
        "srmr": float(np.sqrt(np.mean(np.square(matrix_residual[off_diagonal])))),
        "loading_min": float(loadings["Est. Std"].min()),
        "loading_max": float(loadings["Est. Std"].max()),
        "residual_variance_min": float(residuals["Est. Std"].min()),
        "residual_variance_max": float(residuals["Est. Std"].max()),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_data = load_data()
    samples = {
        "development_folds_0_1_2": all_data.loc[all_data["outer_fold"].isin([0, 1, 2])],
        "validation_folds_3_4": all_data.loc[all_data["outer_fold"].isin([3, 4])],
        "full_cohort": all_data,
    }
    rows = []
    for sample, frame in samples.items():
        data = standardize(frame)
        for estimator in ("MLW", "ULS", "GLS", "DWLS"):
            try:
                rows.append(fit(data, sample, estimator))
            except Exception as exc:
                rows.append({"sample": sample, "estimator": estimator, "error": repr(exc)})
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "sem_estimator_comparison.csv", index=False)
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
