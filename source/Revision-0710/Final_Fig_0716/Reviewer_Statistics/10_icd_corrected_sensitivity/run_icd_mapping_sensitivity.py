from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[2]
REPO_ROOT = REVISION_ROOT.parent
CORE_ROOT = (
    REVISION_ROOT
    / "06_figures_and_statistics"
    / "03_reviewer_statistics"
    / "04_cox_diagnostics_and_competing_risk"
)
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from survival_analysis_core import crossfit_exposure_transform, fit_cox_breslow  # noqa: E402


OOF_PATH = (
    REVISION_ROOT
    / "03_ehr_age"
    / "04_fixed_replication"
    / "02_published_fivefold"
    / "02_oof_hr_km_inputs"
    / "ehr_age_d07_003_e55_oof.parquet"
)
OUTCOME_PATHS = {
    "runtime_v0": REVISION_ROOT / "01_data_rebuild" / "02_outcomes_60m" / "organ_outcomes_60m.parquet",
    "corrected_primary": (
        REVISION_ROOT
        / "01_data_rebuild"
        / "07_corrected_primary_v1"
        / "02_outcomes_60m"
        / "organ_outcomes_60m.parquet"
    ),
    "single_assignment": HERE
    / "single_assignment_rebuild"
    / "02_outcomes_60m"
    / "organ_outcomes_60m.parquet",
}
SOURCE_DIR = HERE / "Source_Data"
FIGURE_DIR = HERE / "Figures"
ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
DISPLAY = {
    "cardiovascular": "Cardiovascular",
    "hepatic": "Hepatic",
    "immune": "Immune",
    "metabolic": "Metabolic",
    "musculoskeletal": "Musculoskeletal",
    "pulmonary": "Pulmonary",
    "renal": "Renal",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_oof_covariates() -> pd.DataFrame:
    columns = [
        "participant_id",
        "organ_system",
        "outer_fold",
        "actual_age",
        "sex",
        "strict_healthy_reference_60m",
        "predicted_age_raw",
        "corrected_age_gap",
    ]
    frame = pd.read_parquet(OOF_PATH, columns=columns)
    frame["participant_id"] = frame["participant_id"].astype(str)
    if len(frame) != 456_773 * 7 or frame.duplicated(["participant_id", "organ_system"]).any():
        raise ValueError("frozen EHR-AGE OOF coverage changed")
    if set(frame["organ_system"].unique()) != set(ORGANS):
        raise ValueError("frozen EHR-AGE organ set changed")
    return frame


def _strict_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    strict = frame.loc[frame["strict_healthy_reference_60m"].eq(1)]
    actual = strict["actual_age"].to_numpy(float)
    predicted = strict["predicted_age_raw"].to_numpy(float)
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n_participant_organ_rows": len(strict),
        "n_participants": strict["participant_id"].nunique(),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "r2": float(1.0 - np.sum((actual - predicted) ** 2) / denominator),
    }


def _foldwise_c_index(
    frame: pd.DataFrame, base_beta: np.ndarray, augmented_beta: np.ndarray
) -> tuple[float, float, float, list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    for fold in range(5):
        part = frame.loc[frame["outer_fold"].eq(fold)]
        base_x = part[["age_per_10y", "sex"]].to_numpy(float)
        augmented_x = part[["age_per_10y", "sex", "gap_z_crossfit"]].to_numpy(float)
        base = float(concordance_index(part["duration_months"], -(base_x @ base_beta), part["event"]))
        augmented = float(
            concordance_index(part["duration_months"], -(augmented_x @ augmented_beta), part["event"])
        )
        rows.append(
            {
                "outer_fold": fold,
                "n": len(part),
                "events": int(part["event"].sum()),
                "base_c_index": base,
                "augmented_c_index": augmented,
                "delta_c_index": augmented - base,
            }
        )
    result = pd.DataFrame(rows)
    weights = result["events"].to_numpy(float)
    base = float(np.average(result["base_c_index"], weights=weights))
    augmented = float(np.average(result["augmented_c_index"], weights=weights))
    return base, augmented, augmented - base, rows


def _evaluate_mapping(
    covariates: pd.DataFrame, mapping_id: str, outcome_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = pd.read_parquet(
        outcome_path,
        columns=[
            "participant_id",
            "organ_system",
            "event",
            "duration_months",
            "competing_death",
            "at_risk",
            "has_prevalent_target",
            "strict_healthy_reference_60m",
            "cohort_state_60m",
        ],
    )
    outcomes["participant_id"] = outcomes["participant_id"].astype(str)
    if len(outcomes) != 456_773 * 7 or outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise ValueError(f"{mapping_id}: outcome coverage changed")

    joined = covariates.merge(
        outcomes,
        on=["participant_id", "organ_system"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_oof", ""),
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{mapping_id}: OOF/outcome keys differ")
    if joined["strict_healthy_reference_60m_oof"].ne(joined["strict_healthy_reference_60m"]).any():
        raise ValueError(f"{mapping_id}: strict Health changed")

    count_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for organ in ORGANS:
        organ_all = joined.loc[joined["organ_system"].eq(organ)].copy()
        count_rows.append(
            {
                "mapping_id": mapping_id,
                "organ_system": organ,
                "participant_n": len(organ_all),
                "prevalent_n": int(organ_all["has_prevalent_target"].sum()),
                "at_risk_n": int(organ_all["at_risk"].sum()),
                "event_n": int(organ_all["event"].sum()),
                "competing_death_n": int(organ_all["competing_death"].sum()),
                "strict_healthy_n": int(organ_all["strict_healthy_reference_60m"].sum()),
            }
        )
        eligible, _ = crossfit_exposure_transform(
            organ_all,
            exposure_col="corrected_age_gap",
            fold_col="outer_fold",
            eligibility_col="at_risk",
        )
        eligible["age_per_10y"] = (eligible["actual_age"] - 60.0) / 10.0
        duration = eligible["duration_months"].to_numpy(float)
        event = eligible["event"].to_numpy(np.int8)
        strata = eligible["outer_fold"].to_numpy(int)
        base_x = eligible[["age_per_10y", "sex"]].to_numpy(float)
        augmented_x = eligible[["age_per_10y", "sex", "gap_z_crossfit"]].to_numpy(float)
        base_fit = fit_cox_breslow(base_x, duration, event, strata=strata)
        augmented_fit = fit_cox_breslow(augmented_x, duration, event, strata=strata)
        if not base_fit.converged or not augmented_fit.converged:
            raise RuntimeError(f"{mapping_id}/{organ}: Cox model did not converge")
        base_c, augmented_c, delta_c, folds = _foldwise_c_index(
            eligible, base_fit.beta, augmented_fit.beta
        )
        index = 2
        beta = float(augmented_fit.beta[index])
        se = float(augmented_fit.standard_error[index])
        metric_rows.append(
            {
                "mapping_id": mapping_id,
                "organ_system": organ,
                "n": len(eligible),
                "events": int(event.sum()),
                "gap_beta_per_crossfit_sd": beta,
                "gap_hr_per_crossfit_sd": math.exp(beta),
                "gap_hr_ci95_low": math.exp(beta - 1.959963984540054 * se),
                "gap_hr_ci95_high": math.exp(beta + 1.959963984540054 * se),
                "gap_p_value": float(augmented_fit.p_value[index]),
                "base_c_index": base_c,
                "augmented_c_index": augmented_c,
                "delta_c_index": delta_c,
            }
        )
        for row in folds:
            fold_rows.append({"mapping_id": mapping_id, "organ_system": organ, **row})
    return pd.DataFrame(count_rows), pd.DataFrame(metric_rows), pd.DataFrame(fold_rows)


def _comparison(metrics: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    reference = metrics.loc[metrics["mapping_id"].eq("runtime_v0")].set_index("organ_system")
    rows: list[dict[str, object]] = []
    for mapping_id in ("corrected_primary", "single_assignment"):
        candidate = metrics.loc[metrics["mapping_id"].eq(mapping_id)].set_index("organ_system")
        candidate_counts = counts.loc[counts["mapping_id"].eq(mapping_id)].set_index("organ_system")
        reference_counts = counts.loc[counts["mapping_id"].eq("runtime_v0")].set_index("organ_system")
        for organ in ORGANS:
            old = reference.loc[organ]
            new = candidate.loc[organ]
            rows.append(
                {
                    "mapping_id": mapping_id,
                    "organ_system": organ,
                    "event_n_delta": int(candidate_counts.loc[organ, "event_n"] - reference_counts.loc[organ, "event_n"]),
                    "prevalent_n_delta": int(candidate_counts.loc[organ, "prevalent_n"] - reference_counts.loc[organ, "prevalent_n"]),
                    "hr_delta": float(new["gap_hr_per_crossfit_sd"] - old["gap_hr_per_crossfit_sd"]),
                    "hr_relative_change": float(
                        new["gap_hr_per_crossfit_sd"] / old["gap_hr_per_crossfit_sd"] - 1.0
                    ),
                    "delta_c_change": float(new["delta_c_index"] - old["delta_c_index"]),
                    "hr_material_5pct": bool(
                        abs(new["gap_hr_per_crossfit_sd"] / old["gap_hr_per_crossfit_sd"] - 1.0) >= 0.05
                    ),
                    "delta_c_material_0p002": bool(abs(new["delta_c_index"] - old["delta_c_index"]) >= 0.002),
                }
            )
    return pd.DataFrame(rows)


def _plot(comparison: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    y = np.arange(len(ORGANS))
    for mapping_id, offset, color, marker, label in (
        ("corrected_primary", -0.10, "#0072B2", "o", "Corrected primary"),
        ("single_assignment", 0.10, "#D55E00", "s", "Single assignment"),
    ):
        part = comparison.loc[comparison["mapping_id"].eq(mapping_id)].set_index("organ_system").loc[list(ORGANS)]
        axes[0].scatter(part["hr_relative_change"] * 100.0, y + offset, color=color, marker=marker, s=25, label=label)
        axes[1].scatter(part["delta_c_change"], y + offset, color=color, marker=marker, s=25)
    axes[0].set_yticks(y, [DISPLAY[organ] for organ in ORGANS])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Relative HR change vs runtime v0 (%)")
    axes[1].set_xlabel("Change in age+sex incremental C-index")
    axes[0].set_title("A  HR sensitivity", loc="left", fontweight="bold")
    axes[1].set_title("B  Incremental discrimination sensitivity", loc="left", fontweight="bold")
    for axis in axes:
        axis.axvline(0, color="#555555", lw=0.8, ls="--")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURE_DIR / f"Supplementary_ICD_Mapping_Sensitivity.{extension}",
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    covariates = _load_oof_covariates()
    strict_metrics = _strict_metrics(covariates)
    count_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    for mapping_id, path in OUTCOME_PATHS.items():
        counts, metrics, folds = _evaluate_mapping(covariates, mapping_id, path)
        count_frames.append(counts)
        metric_frames.append(metrics)
        fold_frames.append(folds)
    counts = pd.concat(count_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    comparison = _comparison(metrics, counts)
    counts.to_csv(SOURCE_DIR / "cohort_counts_by_mapping.csv", index=False)
    metrics.to_csv(SOURCE_DIR / "age_sex_cox_and_delta_c_by_mapping.csv", index=False)
    folds.to_csv(SOURCE_DIR / "age_sex_delta_c_by_mapping_and_fold.csv", index=False)
    comparison.to_csv(SOURCE_DIR / "mapping_deltas_vs_runtime.csv", index=False)
    pd.DataFrame([strict_metrics]).assign(mapping_invariant=True).to_csv(
        SOURCE_DIR / "strict_age_metrics_mapping_invariant.csv", index=False
    )
    _plot(comparison)

    corrected = comparison.loc[comparison["mapping_id"].eq("corrected_primary")]
    single = comparison.loc[comparison["mapping_id"].eq("single_assignment")]
    decision = {
        "status": "PASS",
        "strict_age_metrics": strict_metrics,
        "corrected_primary_vs_runtime": {
            "max_abs_hr_relative_change": float(corrected["hr_relative_change"].abs().max()),
            "max_abs_delta_c_change": float(corrected["delta_c_change"].abs().max()),
            "material_cells": int((corrected["hr_material_5pct"] | corrected["delta_c_material_0p002"]).sum()),
        },
        "single_assignment_vs_runtime": {
            "max_abs_hr_relative_change": float(single["hr_relative_change"].abs().max()),
            "max_abs_delta_c_change": float(single["delta_c_change"].abs().max()),
            "material_cells": int((single["hr_material_5pct"] | single["delta_c_material_0p002"]).sum()),
        },
        "retraining_gate": {
            "hr_relative_change_threshold": 0.05,
            "delta_c_absolute_change_threshold": 0.002,
            "requires_corrected_label_retraining": bool(
                (corrected["hr_material_5pct"] | corrected["delta_c_material_0p002"]).any()
            ),
            "note": "The frozen prediction cache is mapping-independent; this gate concerns outcome-definition sensitivity only.",
        },
        "inputs": {mapping_id: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)} for mapping_id, path in OUTCOME_PATHS.items()},
        "oof_prediction_sha256": sha256_file(OOF_PATH),
    }
    (HERE / "DECISION_SUMMARY.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
