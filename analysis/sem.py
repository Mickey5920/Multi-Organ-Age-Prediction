from __future__ import annotations

import numpy as np

import pandas as pd

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

MODEL = "Shared =~ " + " + ".join(ORGANS)

def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[list(ORGANS)].astype(float)
    scale = values.std(axis=0, ddof=0)
    if (scale <= 0).any():
        raise RuntimeError("non-positive within-group OSAG scale")
    return (values - values.mean(axis=0)) / scale

def fit_one_factor(frame: pd.DataFrame) -> dict[str, object]:
    standardized = standardize(frame)
    model = semopy.Model(MODEL)
    optimizer = model.fit(data=standardized, obj="DWLS", solver="SLSQP")
    if not optimizer.success:
        raise RuntimeError("one-factor DWLS SEM did not converge")
    # Standard errors are not required for this descriptive residual atlas.
    # Skipping them also avoids an information-matrix inversion when a tightly
    # aligned subgroup lies on a residual-variance boundary.
    estimates = model.inspect(std_est=True, information=None)
    loadings = np.array(
        [
            float(
                estimates.loc[
                    (estimates["op"] == "~")
                    & (estimates["rval"] == "Shared")
                    & (estimates["lval"] == organ),
                    "Est. Std",
                ].iloc[0]
            )
            for organ in ORGANS
        ]
    )
    residual_variances = np.array(
        [
            float(
                estimates.loc[
                    (estimates["op"] == "~~")
                    & (estimates["lval"] == organ)
                    & (estimates["rval"] == organ),
                    "Est. Std",
                ].iloc[0]
            )
            for organ in ORGANS
        ]
    )
    observed = standardized.corr().to_numpy(float)
    residual = observed - np.outer(loadings, loadings)
    np.fill_diagonal(residual, 0.0)
    fit = semopy.calc_stats(model).iloc[0]
    off_diagonal = ~np.eye(len(ORGANS), dtype=bool)
    srmr = float(np.sqrt(np.mean(np.square(residual[off_diagonal]))))
    return {
        "loadings": loadings,
        "residual_variances": residual_variances,
        "residual": residual,
        "optimizer_success": bool(optimizer.success),
        "df": int(fit["DoF"]),
        "chi_square": float(fit["chi2"]),
        "cfi": float(fit["CFI"]),
        "tli": float(fit["TLI"]),
        "rmsea": float(fit["RMSEA"]),
        "srmr": srmr,
    }

def sample_frame(
    data: pd.DataFrame, membership: pd.DataFrame, membership_group: str | None
) -> pd.DataFrame:
    if membership_group is None:
        return data.copy()
    ids = membership.loc[
        membership["analysis_group"].eq(membership_group), "participant_id"
    ]
    frame = data.loc[data["participant_id"].isin(ids)].copy()
    if len(frame) != len(ids):
        raise RuntimeError(f"membership join lost rows for {membership_group}")
    return frame
