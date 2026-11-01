#!/usr/bin/env python3
"""Run Revision-0710 primary age+sex HR/KM/competing-risk analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lifelines
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
from lifelines.utils import concordance_index

from survival_analysis_core import (
    benjamini_hochberg,
    crossfit_covariates,
    crossfit_exposure_transform,
    fit_cox_breslow,
    fixed_effect_meta,
    logrank_test,
    nonparametric_curves,
    schoenfeld_ph_test,
    stratified_aj_bootstrap,
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
HORIZONS = (24.0, 60.0)
SCHEMA_VERSION = "revision0710.primary_survival.age_sex.v1"
ANALYSIS_SCOPE = "new_downstream_association_estimation"
DISCRIMINATION_SCOPE = "apparent_discrimination_of_new_downstream_association_model"
DEFAULT_PUBLICATION_TAG = "primary_survival_age_sex_v1"
LOCKED_INPUT_SHA256 = {
    "oof": "35a37a39f555f3d5706da5049a00498f3d30c21aca265cf807b49f2561d43a69",
    "oof_manifest": "5868d135911dd6624cc0d541921e31612c0edb69fdaaa696785610a694dbb945",
    "outcomes": "3ce09bca650758a4044e81c312dd781c71ca823f1d4567d3d0473158ec1248ee",
    "outcome_audit": "116aad032da303785ecc05b0fc04d32a01131bb02b818e79041954ae3b8c5a6c",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=root
        / "Revision-0710/03_ehr_age/04_fixed_replication/02_published_fivefold/02_oof_hr_km_inputs/ehr_age_d07_003_e55_oof.parquet",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=root / "Revision-0710/01_data_rebuild/02_outcomes_60m/organ_outcomes_60m.parquet",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--figure-source-root",
        type=Path,
        default=root / "Revision-0710/06_figures_and_statistics/05_source_data_by_figure",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument("--publication-tag", default=DEFAULT_PUBLICATION_TAG)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--publish-pointers-only", action="store_true")
    parser.add_argument("--extended-covariates", type=Path)
    parser.add_argument("--extended-covariate-columns", default="")
    parser.add_argument("--extended-binary-columns", default="")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bundle_staging_path(final_output: Path, pid: int) -> Path:
    return final_output.parent / f".b.{pid}"


def _pointer_staging_path(pointer_directory: Path, pid_token: str) -> Path:
    return pointer_directory.parent / f".p.{pid_token}"


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _validate_canonical_publication(
    *,
    is_default_canonical: bool,
    bootstrap_replicates: int,
    seed: int,
    has_extended_covariates: bool,
    observed_hashes: dict[str, str] | None = None,
) -> None:
    if not is_default_canonical:
        return
    if bootstrap_replicates != 500 or seed != 710:
        raise ValueError("default canonical publication requires --bootstrap-replicates 500 --seed 710")
    if has_extended_covariates:
        raise ValueError(
            "default canonical publication is the age+sex primary model; use a new tag/output for extended covariates"
        )
    if observed_hashes is not None and observed_hashes != LOCKED_INPUT_SHA256:
        raise RuntimeError(
            f"default canonical locked-input hash gate failed: observed={observed_hashes} expected={LOCKED_INPUT_SHA256}"
        )


def _read_organ(oof_path: Path, outcomes_path: Path, organ: str) -> tuple[pd.DataFrame, dict[str, object]]:
    oof_columns = [
        "participant_id",
        "organ_system",
        "outer_fold",
        "actual_age",
        "sex",
        "corrected_age_gap",
        "strict_healthy_reference_60m",
        "at_risk",
        "duration_months",
        "event",
    ]
    outcome_columns = [
        "participant_id",
        "organ_system",
        "strict_healthy_reference_60m",
        "at_risk",
        "duration_months",
        "event",
        "competing_death",
        "censor_reason",
        "has_prevalent_target",
        "full_incident_risk_set_60m",
        "cohort_state_60m",
    ]
    filters = [("organ_system", "==", organ)]
    oof = pd.read_parquet(oof_path, columns=oof_columns, filters=filters)
    outcome = pd.read_parquet(outcomes_path, columns=outcome_columns, filters=filters)
    keys = ["participant_id", "organ_system"]
    if oof.duplicated(keys).any() or outcome.duplicated(keys).any():
        raise ValueError(f"{organ}: duplicate participant-organ keys")
    joined = oof.merge(outcome, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_oof", ""))
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{organ}: OOF/outcome key mismatch")
    joined = joined.drop(columns="_merge")
    mismatch: dict[str, int] = {}
    for column in ("strict_healthy_reference_60m", "at_risk", "event"):
        mismatch[column] = int(joined[f"{column}_oof"].ne(joined[column]).sum())
    paired_missing = joined["duration_months_oof"].isna() & joined["duration_months"].isna()
    duration_equal = np.isclose(
        joined["duration_months_oof"], joined["duration_months"], rtol=0.0, atol=1e-12
    )
    nonrisk_sentinel = (
        joined["at_risk"].eq(0)
        & joined["duration_months"].isna()
        & joined["duration_months_oof"].fillna(0.0).eq(0.0)
    )
    mismatch["duration_months"] = int((~paired_missing & ~duration_equal & ~nonrisk_sentinel).sum())
    if any(mismatch.values()):
        raise ValueError(f"{organ}: authoritative outcome mismatch {mismatch}")
    joined = joined.drop(
        columns=[
            "strict_healthy_reference_60m_oof",
            "at_risk_oof",
            "event_oof",
            "duration_months_oof",
        ]
    )
    if joined["participant_id"].nunique() != len(joined):
        raise ValueError(f"{organ}: participant is not unique")
    if not set(pd.unique(joined["outer_fold"])) == {0, 1, 2, 3, 4}:
        raise ValueError(f"{organ}: outer folds are incomplete")
    if joined[["actual_age", "sex", "corrected_age_gap"]].isna().any().any():
        raise ValueError(f"{organ}: missing primary covariates")
    if not set(pd.unique(joined["sex"])).issubset({0.0, 1.0}):
        raise ValueError(f"{organ}: sex must use binary 0/1 coding")
    if ((joined["event"] + joined["competing_death"]) > 1).any():
        raise ValueError(f"{organ}: target event and death overlap")
    if (joined.loc[joined["at_risk"].eq(0), ["event", "competing_death"]].sum(axis=1) > 0).any():
        raise ValueError(f"{organ}: non-risk rows carry an event")
    audit = {
        "organ_system": organ,
        "joined_rows": int(len(joined)),
        "participants": int(joined["participant_id"].nunique()),
        "key_mismatch_rows": 0,
        "overlap_mismatch": mismatch,
        "oof_nonrisk_zero_to_authoritative_nan_duration_rows": int(nonrisk_sentinel.sum()),
        "duration_equality_rule": "exact within at_risk; OOF zero sentinel must map to authoritative NaN outside risk set",
        "at_risk_n": int(joined["at_risk"].sum()),
        "target_events": int(joined["event"].sum()),
        "competing_deaths": int(joined["competing_death"].sum()),
        "prevalent_target_n": int(joined["has_prevalent_target"].sum()),
        "sex_coding": "0/1 as carried by frozen OOF table",
        "status": "PASS",
    }
    return joined, audit


def _cox_design(frame: pd.DataFrame, extra: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    columns = ["age_per_10y", "sex", "gap_z_crossfit"] + list(extra or [])
    return frame[columns].to_numpy(dtype=float), columns


def _foldwise_concordance(
    frame: pd.DataFrame,
    base_beta: np.ndarray,
    augmented_beta: np.ndarray,
    augmented_columns: list[str],
) -> tuple[float, float, float, list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    for fold in sorted(pd.unique(frame["outer_fold"])):
        part = frame.loc[frame["outer_fold"].eq(fold)]
        base_x = part[["age_per_10y", "sex"]].to_numpy(float)
        augmented_x = part[augmented_columns].to_numpy(float)
        base = float(
            concordance_index(part["duration_months"], -(base_x @ base_beta), part["event"])
        )
        augmented = float(
            concordance_index(
                part["duration_months"], -(augmented_x @ augmented_beta), part["event"]
            )
        )
        rows.append(
            {
                "outer_fold": int(fold),
                "n": int(len(part)),
                "events": int(part["event"].sum()),
                "base_c_index": base,
                "augmented_c_index": augmented,
                "delta_c_index": augmented - base,
            }
        )
    fold = pd.DataFrame(rows)
    weights = fold["events"].to_numpy(float)
    base = float(np.average(fold["base_c_index"], weights=weights))
    augmented = float(np.average(fold["augmented_c_index"], weights=weights))
    return base, augmented, augmented - base, rows


def _run_cox(
    frame: pd.DataFrame,
    organ: str,
    extra_covariates: list[str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    frame = frame.copy()
    frame["age_per_10y"] = (frame["actual_age"] - 60.0) / 10.0
    duration = frame["duration_months"].to_numpy(float)
    event = frame["event"].to_numpy(np.int8)
    strata = frame["outer_fold"].to_numpy()
    base_columns = ["age_per_10y", "sex"]
    base_x = frame[base_columns].to_numpy(float)
    augmented_x, augmented_columns = _cox_design(frame, extra_covariates)
    base_fit = fit_cox_breslow(base_x, duration, event, strata=strata)
    augmented_fit = fit_cox_breslow(augmented_x, duration, event, strata=strata)
    if not base_fit.converged or not augmented_fit.converged:
        raise RuntimeError(f"{organ}: primary Cox model did not converge")
    c_base, c_augmented, delta_c, c_by_fold = _foldwise_concordance(
        frame, base_fit.beta, augmented_fit.beta, augmented_columns
    )
    coefficient_rows: list[dict[str, object]] = []
    z95 = 1.959963984540054
    for i, covariate in enumerate(augmented_columns):
        coefficient_rows.append(
            {
                "organ_system": organ,
                "model": "age_sex_plus_continuous_corrected_age_gap",
                "covariate": covariate,
                "unit": {
                    "age_per_10y": "per_10_years",
                    "sex": "1_vs_0",
                    "gap_z_crossfit": "per_1_non_target_fold_SD",
                }.get(covariate, "crossfit_transformed_unit"),
                "beta": float(augmented_fit.beta[i]),
                "se_model_based": float(augmented_fit.standard_error[i]),
                "hr": float(np.exp(augmented_fit.beta[i])),
                "hr_ci95_low": float(np.exp(augmented_fit.beta[i] - z95 * augmented_fit.standard_error[i])),
                "hr_ci95_high": float(np.exp(augmented_fit.beta[i] + z95 * augmented_fit.standard_error[i])),
                "p_value": float(augmented_fit.p_value[i]),
                "ties": augmented_fit.ties,
                "outer_fold_baseline_hazard": "stratified",
                "participant_cluster_note": "one row per participant within organ; no repeated within-organ cluster",
            }
        )
    coefficients = pd.DataFrame(coefficient_rows)
    ph = schoenfeld_ph_test(
        augmented_fit,
        augmented_x,
        duration,
        event,
        strata=strata,
        covariate_names=augmented_columns,
    )
    ph.insert(0, "organ_system", organ)
    fold_rows: list[dict[str, object]] = []
    for fold in sorted(pd.unique(frame["outer_fold"])):
        part = frame.loc[frame["outer_fold"].eq(fold)]
        x = part[augmented_columns].to_numpy(float)
        fit = fit_cox_breslow(x, part["duration_months"], part["event"])
        if not fit.converged:
            raise RuntimeError(f"{organ}: fold {fold} Cox model did not converge")
        index = augmented_columns.index("gap_z_crossfit")
        fold_rows.append(
            {
                "organ_system": organ,
                "outer_fold": int(fold),
                "n": int(len(part)),
                "events": int(part["event"].sum()),
                "beta_gap": float(fit.beta[index]),
                "se_gap": float(fit.standard_error[index]),
                "hr_gap": float(np.exp(fit.beta[index])),
                "hr_ci95_low": float(np.exp(fit.beta[index] - z95 * fit.standard_error[index])),
                "hr_ci95_high": float(np.exp(fit.beta[index] + z95 * fit.standard_error[index])),
                "p_value": float(fit.p_value[index]),
                "converged": bool(fit.converged),
                "iterations": int(fit.iterations),
            }
        )
    folds = pd.DataFrame(fold_rows)
    meta = {"organ_system": organ, **fixed_effect_meta(folds["beta_gap"], folds["se_gap"])}
    gap = coefficients.loc[coefficients["covariate"].eq("gap_z_crossfit")].iloc[0]
    summary = {
        "organ_system": organ,
        "n": int(len(frame)),
        "events": int(frame["event"].sum()),
        "competing_deaths": int(frame["competing_death"].sum()),
        "gap_hr_per_crossfit_sd": float(gap["hr"]),
        "gap_hr_ci95_low": float(gap["hr_ci95_low"]),
        "gap_hr_ci95_high": float(gap["hr_ci95_high"]),
        "gap_p_value": float(gap["p_value"]),
        "base_c_index": c_base,
        "augmented_c_index": c_augmented,
        "delta_c_index": delta_c,
        "c_index_aggregation": "event_weighted_mean_of_within_outer_fold_Harrell_C",
        "discrimination_scope": DISCRIMINATION_SCOPE,
        "is_frozen_prediction_metric": False,
        "base_converged": bool(base_fit.converged),
        "augmented_converged": bool(augmented_fit.converged),
        "augmented_iterations": int(augmented_fit.iterations),
        "augmented_score_max_abs": float(augmented_fit.score_max_abs),
        "ties": "breslow",
    }
    c_fold = pd.DataFrame(c_by_fold)
    c_fold.insert(0, "organ_system", organ)
    ph["record_type"] = "ph_test"
    return summary, coefficients, folds, meta, pd.concat([ph, c_fold.assign(record_type="c_index")], ignore_index=True, sort=False)


def _run_time_window_cox(frame: pd.DataFrame, organ: str) -> pd.DataFrame:
    """Estimate prespecified 0-24m and 24-60m landmark Cox sensitivities."""
    rows: list[dict[str, object]] = []
    definitions = [
        (
            "0_to_24_months",
            frame.assign(
                window_duration=np.minimum(frame["duration_months"], 24.0),
                window_event=(frame["event"].eq(1) & frame["duration_months"].le(24.0)).astype(np.int8),
            ),
            "all baseline at-risk participants; administratively censored at 24 months",
        ),
        (
            "24_to_60_months_landmark",
            frame.loc[frame["duration_months"].gt(24.0)].assign(
                window_duration=lambda x: x["duration_months"] - 24.0,
                window_event=lambda x: x["event"].astype(np.int8),
            ),
            "participants alive and target-event-free beyond 24 months; time origin reset to month 24",
        ),
    ]
    z95 = 1.959963984540054
    for window, part, definition in definitions:
        part = part.copy()
        part["age_per_10y"] = (part["actual_age"] - 60.0) / 10.0
        columns = ["age_per_10y", "sex", "gap_z_crossfit"]
        fit = fit_cox_breslow(
            part[columns].to_numpy(float),
            part["window_duration"].to_numpy(float),
            part["window_event"].to_numpy(np.int8),
            strata=part["outer_fold"].to_numpy(),
        )
        if not fit.converged:
            raise RuntimeError(f"{organ} {window} Cox did not converge")
        index = columns.index("gap_z_crossfit")
        rows.append(
            {
                "organ_system": organ,
                "time_window": window,
                "window_definition": definition,
                "n_at_window_start": int(len(part)),
                "events_in_window": int(part["window_event"].sum()),
                "gap_beta": float(fit.beta[index]),
                "gap_se_model_based": float(fit.standard_error[index]),
                "gap_hr_per_crossfit_sd": float(np.exp(fit.beta[index])),
                "gap_hr_ci95_low": float(np.exp(fit.beta[index] - z95 * fit.standard_error[index])),
                "gap_hr_ci95_high": float(np.exp(fit.beta[index] + z95 * fit.standard_error[index])),
                "gap_p_value": float(fit.p_value[index]),
                "outer_fold_baseline_hazard": "stratified",
                "ties": "breslow",
                "converged": bool(fit.converged),
                "iterations": int(fit.iterations),
            }
        )
    return pd.DataFrame(rows)


def _run_nonparametric(
    frame: pd.DataFrame,
    organ: str,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    risks: list[pd.DataFrame] = []
    risk_table_rows: list[dict[str, object]] = []
    for quartile in (1, 2, 3, 4):
        part = frame.loc[frame["gap_quartile_crossfit"].eq(quartile)]
        curve, risk = nonparametric_curves(
            part["duration_months"], part["event"], part["competing_death"], horizons=HORIZONS
        )
        curve.insert(0, "gap_quartile_crossfit", quartile)
        curve.insert(0, "organ_system", organ)
        risk.insert(0, "gap_quartile_crossfit", quartile)
        risk.insert(0, "organ_system", organ)
        curves.append(curve)
        risks.append(risk)
        for requested in (0.0, 12.0, 24.0, 36.0, 48.0, 60.0):
            max_duration = float(part["duration_months"].max())
            evaluated = max_duration if requested == 60.0 and 0 <= 60.0 - max_duration <= 0.1 else requested
            risk_table_rows.append(
                {
                    "organ_system": organ,
                    "gap_quartile_crossfit": quartile,
                    "requested_time_months": requested,
                    "evaluated_time_months": evaluated,
                    "at_risk_before": int((part["duration_months"] >= evaluated - 1e-12).sum()),
                    "n_total": int(len(part)),
                    "target_events_total": int(part["event"].sum()),
                    "competing_deaths_total": int(part["competing_death"].sum()),
                }
            )
    all_test = logrank_test(
        frame["duration_months"], frame["event"], frame["gap_quartile_crossfit"]
    )
    extreme = frame.loc[frame["gap_quartile_crossfit"].isin([1, 4])]
    extreme_test = logrank_test(
        extreme["duration_months"], extreme["event"], extreme["gap_quartile_crossfit"]
    )
    logrank = pd.DataFrame(
        [
            {"organ_system": organ, "contrast": "Q1-Q4_global", **all_test},
            {"organ_system": organ, "contrast": "Q4_vs_Q1", **extreme_test},
        ]
    )
    bootstrap_risk, bootstrap_difference = stratified_aj_bootstrap(
        frame,
        horizons=HORIZONS,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    bootstrap_risk.insert(0, "organ_system", organ)
    bootstrap_difference.insert(0, "organ_system", organ)
    risk = pd.concat(risks, ignore_index=True).merge(
        bootstrap_risk,
        on=["organ_system", "gap_quartile_crossfit", "requested_horizon_months"],
        how="left",
        validate="one_to_one",
    )
    point = risk.pivot(
        index="requested_horizon_months", columns="gap_quartile_crossfit", values="aj_target_cif"
    )
    difference = bootstrap_difference.copy()
    difference["aj_target_cif_q1"] = difference["requested_horizon_months"].map(point[1])
    difference["aj_target_cif_q4"] = difference["requested_horizon_months"].map(point[4])
    difference["aj_target_cif_risk_difference_q4_minus_q1"] = (
        difference["aj_target_cif_q4"] - difference["aj_target_cif_q1"]
    )
    return (
        pd.concat(curves, ignore_index=True),
        risk,
        difference,
        pd.DataFrame(risk_table_rows),
        logrank,
    )


def _write_table(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }


def _write_json(payload: object, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def _public_record(record: dict[str, object], stage_root: Path, public_root: Path) -> dict[str, object]:
    updated = dict(record)
    relative = Path(str(record["path"])).resolve().relative_to(stage_root.resolve())
    updated["path"] = (public_root.resolve() / relative).as_posix()
    return updated


def _verify_existing(output: Path) -> dict[str, object]:
    manifest_path = output / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"existing canonical directory lacks MANIFEST.json: {output}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    if payload.get("status") != "COMPLETE":
        failures.append("manifest_status_not_complete")
    records_to_check = [
        *payload.get("locked_inputs", []),
        *payload.get("code_and_contract_lineage", []),
        *payload.get("outputs", []),
    ]
    for record in records_to_check:
        path = Path(str(record["path"]))
        checked += 1
        if not path.is_file():
            failures.append(f"missing:{path}")
        elif record.get("sha256") and sha256_file(path) != record["sha256"]:
            failures.append(f"sha256:{path}")
    for records in payload.get("figure_source_tables", {}).values():
        for record in records:
            path = Path(str(record["path"]))
            checked += 1
            if not path.is_file():
                failures.append(f"missing:{path}")
            elif sha256_file(path) != record["sha256"]:
                failures.append(f"sha256:{path}")
    for figure in ("Fig08", "Fig09", "Fig10"):
        source_manifest = output / "figure_sources" / figure / "MANIFEST.json"
        checked += 1
        if not source_manifest.is_file():
            failures.append(f"missing:{source_manifest}")
            continue
        source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        binding = source_payload.get("analysis_manifest", {})
        if binding.get("sha256") != sha256_file(manifest_path):
            failures.append(f"analysis_manifest_binding:{source_manifest}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "checked_files": checked,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def _source_manifest(
    directory: Path,
    public_directory: Path,
    figure: str,
    files: list[Path],
    analysis_manifest_path: Path,
    analysis_manifest_sha256: str,
    inputs: list[dict[str, object]],
    command: str,
) -> None:
    row_grain = {
        "cox_primary_summary.csv": "one organ",
        "cox_coefficients.csv": "one organ-covariate coefficient",
        "cox_fold_estimates.csv": "one organ-outer-fold gap coefficient",
        "cox_fold_meta.csv": "one organ fixed-effect meta-analysis",
        "cox_time_window_sensitivity.csv": "one organ-time-window gap coefficient",
        "cox_ph_and_cindex_diagnostics.csv": "one organ-diagnostic or organ-fold C-index",
        "km_aj_curves.parquet": "one organ-quartile event time",
        "quartile_counts_and_risk_table.csv": "one organ-quartile requested risk-table time",
        "absolute_risk_24_60m.csv": "one organ-quartile-horizon",
        "logrank_tests.csv": "one organ-prespecified contrast",
        "organ_quartile_absolute_risk.csv": "one organ-quartile-horizon",
        "q4_minus_q1_absolute_risk_difference.csv": "one organ-horizon Q4-minus-Q1 contrast",
    }

    def describe(file: Path) -> dict[str, object]:
        if file.suffix == ".parquet":
            parquet = pq.ParquetFile(file)
            rows = int(parquet.metadata.num_rows)
            columns = parquet.schema.names
        else:
            table = pd.read_csv(file)
            rows = int(len(table))
            columns = list(table.columns)
        return {
            "path": (public_directory / file.relative_to(directory)).as_posix(),
            "sha256": sha256_file(file),
            "bytes": file.stat().st_size,
            "rows": rows,
            "columns": columns,
            "row_grain": row_grain.get(file.name, "see table columns"),
        }

    payload = {
        "schema_version": f"revision0710.figure_source.{figure.lower()}.v1",
        "created_utc": utc_now(),
        "figure": figure,
        "row_grain": "declared separately for every source table",
        "cohort": "organ-specific at_risk == 1",
        "generation_command": command,
        "analysis_manifest": {
            "path": analysis_manifest_path.as_posix(),
            "sha256": analysis_manifest_sha256,
        },
        "locked_inputs": inputs,
        "source_files": [describe(file) for file in files],
        "status": "COMPLETE",
    }
    _write_json(payload, directory / "MANIFEST.json")


def _publish_figure_pointer(
    pointer_directory: Path,
    canonical_source_directory: Path,
    canonical_manifest: Path,
    *,
    token: str,
) -> None:
    if pointer_directory.exists():
        raise FileExistsError(f"refusing to overwrite figure-source pointer: {pointer_directory}")
    staging = _pointer_staging_path(pointer_directory, token)
    staging.mkdir(parents=True, exist_ok=False)
    source_manifest = canonical_source_directory / "MANIFEST.json"
    payload = {
        "schema_version": "revision0710.figure_source.pointer.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE_POINTER_TO_ATOMIC_CANONICAL_BUNDLE",
        "canonical_bundle_manifest": {
            "path": canonical_manifest.as_posix(),
            "sha256": sha256_file(canonical_manifest),
        },
        "canonical_figure_source_directory": canonical_source_directory.as_posix(),
        "canonical_figure_source_manifest": {
            "path": source_manifest.as_posix(),
            "sha256": sha256_file(source_manifest),
        },
        "note": "Source tables live inside the single atomically promoted analysis bundle; this directory is an index only.",
    }
    _write_json(payload, staging / "SOURCE_POINTER.json")
    staging.replace(pointer_directory)


def main() -> int:
    args = parse_args()
    default_canonical_output = (
        Path(__file__).resolve().parent / "outputs" / DEFAULT_PUBLICATION_TAG
    ).resolve()
    final_output = (
        args.output_dir
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / args.publication_tag
    ).resolve()
    final_sources = {
        figure: final_output / "figure_sources" / figure
        for figure in ("Fig08", "Fig09", "Fig10")
    }
    pointer_directories = {
        figure: args.figure_source_root.resolve() / figure / args.publication_tag
        for figure in ("Fig08", "Fig09", "Fig10")
    }
    if args.verify_only:
        print(json.dumps(_verify_existing(final_output), indent=2), flush=True)
        return 0
    if args.publish_pointers_only:
        if not final_output.is_dir():
            raise FileNotFoundError(f"canonical bundle is missing: {final_output}")
        verification = _verify_existing(final_output)
        token = str(os.getpid())
        published: list[str] = []
        for figure in ("Fig08", "Fig09", "Fig10"):
            if pointer_directories[figure].exists():
                continue
            _publish_figure_pointer(
                pointer_directories[figure],
                final_sources[figure],
                final_output / "MANIFEST.json",
                token=token,
            )
            published.append(figure)
        print(
            json.dumps(
                {"status": "PASS", "canonical_verification": verification, "published": published},
                indent=2,
            ),
            flush=True,
        )
        return 0
    is_default_canonical = (
        final_output == default_canonical_output and args.publication_tag == DEFAULT_PUBLICATION_TAG
    )
    existing = [path for path in [final_output, *pointer_directories.values()] if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite canonical publication artifacts; use --verify-only: "
            + ", ".join(path.as_posix() for path in existing)
        )
    if not args.oof.is_file() or not args.outcomes.is_file():
        raise FileNotFoundError("locked OOF or authoritative outcome input is missing")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    extended_columns = [x.strip() for x in args.extended_covariate_columns.split(",") if x.strip()]
    extended_binary = [x.strip() for x in args.extended_binary_columns.split(",") if x.strip()]
    if args.extended_covariates and not extended_columns:
        raise ValueError("--extended-covariate-columns is required with --extended-covariates")
    if extended_columns and not args.extended_covariates:
        raise ValueError("extended columns require --extended-covariates")
    extended = None
    if args.extended_covariates:
        extended = pd.read_parquet(args.extended_covariates)
        if "participant_id" not in extended or extended["participant_id"].duplicated().any():
            raise ValueError("extended covariates require unique participant_id")
        extended["participant_id"] = extended["participant_id"].astype(str)
    _validate_canonical_publication(
        is_default_canonical=is_default_canonical,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        has_extended_covariates=extended is not None,
    )
    oof_manifest = args.oof.resolve().parent / "MANIFEST.json"
    outcome_audit = args.outcomes.resolve().parent / "outcome_rebuild_audit.json"
    if not oof_manifest.is_file() or not outcome_audit.is_file():
        raise FileNotFoundError("locked OOF manifest or authoritative outcome audit is missing")
    oof_registry = json.loads(oof_manifest.read_text(encoding="utf-8"))
    outcome_registry = json.loads(outcome_audit.read_text(encoding="utf-8"))
    observed_oof_sha256 = sha256_file(args.oof)
    observed_hashes = {
        "oof": observed_oof_sha256,
        "oof_manifest": sha256_file(oof_manifest),
        "outcomes": sha256_file(args.outcomes),
        "outcome_audit": sha256_file(outcome_audit),
    }
    _validate_canonical_publication(
        is_default_canonical=is_default_canonical,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        has_extended_covariates=extended is not None,
        observed_hashes=observed_hashes,
    )
    if oof_registry.get("oof_sha256") != observed_oof_sha256:
        raise RuntimeError("OOF parquet does not match its locked MANIFEST")
    if outcome_registry.get("status") != "PASS":
        raise RuntimeError("authoritative outcome rebuild audit is not PASS")
    input_records = [
        {
            "role": "frozen_fivefold_oof_predictions",
            "path": args.oof.resolve().as_posix(),
            "sha256": observed_oof_sha256,
        },
        {
            "role": "authoritative_organ_outcomes_60m",
            "path": args.outcomes.resolve().as_posix(),
            "sha256": observed_hashes["outcomes"],
        },
        {
            "role": "locked_fivefold_oof_registry",
            "path": oof_manifest.as_posix(),
            "sha256": observed_hashes["oof_manifest"],
        },
        {
            "role": "authoritative_outcome_rebuild_audit",
            "path": outcome_audit.as_posix(),
            "sha256": observed_hashes["outcome_audit"],
        },
    ]
    if args.extended_covariates:
        input_records.append(
            {
                "role": "optional_extended_covariates",
                "path": args.extended_covariates.resolve().as_posix(),
                "sha256": sha256_file(args.extended_covariates),
            }
        )
    token = str(os.getpid())
    output = _bundle_staging_path(final_output, os.getpid())
    source_staging = {
        figure: output / "figure_sources" / figure for figure in ("Fig08", "Fig09", "Fig10")
    }
    output.mkdir(parents=True, exist_ok=False)
    analysis_output = output / "analysis"
    analysis_output.mkdir(parents=True, exist_ok=False)
    for directory in source_staging.values():
        directory.mkdir(parents=True, exist_ok=False)
    log = RunLog(output / "generation.log")
    started = time.time()
    log.write(f"start schema={SCHEMA_VERSION} bootstrap={args.bootstrap_replicates} seed={args.seed}")
    audits: list[dict[str, object]] = []
    references: list[pd.DataFrame] = []
    cox_summary: list[dict[str, object]] = []
    cox_coefficients: list[pd.DataFrame] = []
    cox_folds: list[pd.DataFrame] = []
    cox_meta: list[dict[str, object]] = []
    cox_windows: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    absolute_risk: list[pd.DataFrame] = []
    risk_difference: list[pd.DataFrame] = []
    risk_tables: list[pd.DataFrame] = []
    logranks: list[pd.DataFrame] = []
    extended_audits: list[pd.DataFrame] = []
    for organ_index, organ in enumerate(ORGANS):
        organ_started = time.time()
        joined, audit = _read_organ(args.oof, args.outcomes, organ)
        audits.append(audit)
        risk, reference = crossfit_exposure_transform(joined)
        reference.insert(0, "organ_system", organ)
        references.append(reference)
        extra_transformed: list[str] = []
        if extended is not None:
            risk = risk.merge(
                extended[["participant_id", *extended_columns]],
                on="participant_id",
                how="left",
                validate="many_to_one",
            )
            risk, extended_audit, extra_transformed = crossfit_covariates(
                risk, extended_columns, binary_columns=extended_binary
            )
            extended_audit.insert(0, "organ_system", organ)
            extended_audits.append(extended_audit)
        summary, coefficients, folds, meta, diagnostic = _run_cox(
            risk, organ, extra_covariates=extra_transformed
        )
        cox_summary.append(summary)
        cox_coefficients.append(coefficients)
        cox_folds.append(folds)
        cox_meta.append(meta)
        cox_windows.append(_run_time_window_cox(risk, organ))
        diagnostics.append(diagnostic)
        curve, risk_result, difference, risk_table, logrank = _run_nonparametric(
            risk,
            organ,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed + 1009 * organ_index,
        )
        curves.append(curve)
        absolute_risk.append(risk_result)
        risk_difference.append(difference)
        risk_tables.append(risk_table)
        logranks.append(logrank)
        log.write(
            f"organ_complete organ={organ} n={len(risk)} events={int(risk.event.sum())} "
            f"deaths={int(risk.competing_death.sum())} seconds={time.time() - organ_started:.2f}"
        )
    audit_frame = pd.json_normalize(audits, sep="__")
    observed_counts = {
        "joined_participant_organ": int(audit_frame["joined_rows"].sum()),
        "at_risk_participant_organ": int(audit_frame["at_risk_n"].sum()),
        "events": int(audit_frame["target_events"].sum()),
        "competing_deaths": int(audit_frame["competing_deaths"].sum()),
    }
    expected_counts = {
        "joined_participant_organ": int(oof_registry["row_count"]),
        "at_risk_participant_organ": int(outcome_registry["full_incident_risk_rows"]),
        "events": int(outcome_registry["incident_event_rows"]),
        "competing_deaths": int(outcome_registry["competing_death_rows"]),
    }
    if observed_counts != expected_counts:
        raise RuntimeError(f"locked row-count gate failed: observed={observed_counts} expected={expected_counts}")
    reference_frame = pd.concat(references, ignore_index=True)
    cox_summary_frame = pd.DataFrame(cox_summary)
    coefficient_frame = pd.concat(cox_coefficients, ignore_index=True)
    fold_frame = pd.concat(cox_folds, ignore_index=True)
    meta_frame = pd.DataFrame(cox_meta)
    window_frame = pd.concat(cox_windows, ignore_index=True)
    diagnostic_frame = pd.concat(diagnostics, ignore_index=True, sort=False)
    curve_frame = pd.concat(curves, ignore_index=True)
    absolute_risk_frame = pd.concat(absolute_risk, ignore_index=True)
    difference_frame = pd.concat(risk_difference, ignore_index=True)
    risk_table_frame = pd.concat(risk_tables, ignore_index=True)
    logrank_frame = pd.concat(logranks, ignore_index=True)
    coefficient_frame["q_value_bh_primary_gap_7_organs"] = np.nan
    gap_mask = coefficient_frame["covariate"].eq("gap_z_crossfit")
    coefficient_frame.loc[gap_mask, "q_value_bh_primary_gap_7_organs"] = benjamini_hochberg(
        coefficient_frame.loc[gap_mask, "p_value"]
    )
    cox_summary_frame["gap_q_value_bh_7_organs"] = benjamini_hochberg(
        cox_summary_frame["gap_p_value"]
    )
    for window in window_frame["time_window"].unique():
        mask = window_frame["time_window"].eq(window)
        window_frame.loc[mask, "gap_q_value_bh_7_organs_within_window"] = benjamini_hochberg(
            window_frame.loc[mask, "gap_p_value"]
        )
    ph_mask = diagnostic_frame["record_type"].eq("ph_test") & diagnostic_frame["covariate"].eq("gap_z_crossfit")
    diagnostic_frame.loc[ph_mask, "q_value_bh_gap_ph_14_tests"] = benjamini_hochberg(
        diagnostic_frame.loc[ph_mask, "p_value"]
    )
    for contrast in logrank_frame["contrast"].unique():
        mask = logrank_frame["contrast"].eq(contrast)
        logrank_frame.loc[mask, "q_value_bh_7_organs_within_contrast"] = benjamini_hochberg(
            logrank_frame.loc[mask, "p_value"]
        )
    output_records: list[dict[str, object]] = []
    def record(record_value: dict[str, object]) -> None:
        output_records.append(_public_record(record_value, output, final_output))

    record(_write_table(audit_frame, analysis_output / "00_input_and_join_audit.csv"))
    record(_write_table(reference_frame, analysis_output / "01_crossfit_exposure_reference.csv"))
    record(_write_table(cox_summary_frame, analysis_output / "02_cox_primary_summary.csv"))
    record(_write_table(coefficient_frame, analysis_output / "03_cox_coefficients.csv"))
    record(_write_table(fold_frame, analysis_output / "04_cox_fold_estimates.csv"))
    record(_write_table(meta_frame, analysis_output / "05_cox_fold_meta.csv"))
    record(_write_table(window_frame, analysis_output / "06_cox_time_window_sensitivity.csv"))
    record(_write_table(diagnostic_frame, analysis_output / "07_ph_and_cindex_diagnostics.csv"))
    record(_write_table(risk_table_frame, analysis_output / "08_quartile_counts_and_risk_table.csv"))
    record(_write_table(logrank_frame, analysis_output / "09_logrank_tests.csv"))
    record(_write_table(curve_frame, analysis_output / "10_km_aj_curves.parquet"))
    record(_write_table(absolute_risk_frame, analysis_output / "11_absolute_risk_24_60m.csv"))
    record(_write_table(difference_frame, analysis_output / "12_q4_minus_q1_risk_difference.csv"))
    if extended_audits:
        record(
            _write_table(pd.concat(extended_audits, ignore_index=True), analysis_output / "13_extended_crossfit_audit.csv")
        )
    extended_status = {
        "status": "RUN" if extended is not None else "NOT_RUN_NO_COVARIATE_INPUT",
        "interface": "optional participant-level parquet plus explicit primary/binary columns",
        "imputation_and_scaling": "non-target-four-fold only",
        "input": input_records[-1] if extended is not None else None,
        "columns": extended_columns,
        "binary_columns": extended_binary,
    }
    record(_write_json(extended_status, analysis_output / "13_extended_covariate_status.json"))
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_scope": ANALYSIS_SCOPE,
        "primary_population": "each organ, at_risk == 1",
        "primary_exposure": "continuous corrected_age_gap, standardized from other four OOF folds",
        "primary_model": "cause-specific Cox: age per 10y + sex + gap per crossfit SD; outer-fold-stratified baseline",
        "competing_death_in_primary_cox": "censored at death",
        "competing_risk_sensitivity": "nonparametric Aalen-Johansen target CIF with death as competing event",
        "quartiles": "cut points from other four OOF folds; visualization/sensitivity only",
        "c_index_boundary": {
            "scope": DISCRIMINATION_SCOPE,
            "is_frozen_prediction_metric": False,
            "warning": "Do not replace or relabel the sealed outer-fold EHR-AGE Delta C-index with these values.",
        },
        "multiple_testing": "BH within prespecified 7-organ families",
        "organs": cox_summary_frame.to_dict(orient="records"),
        "status": "COMPLETE",
    }
    record(_write_json(summary_payload, analysis_output / "14_primary_summary.json"))
    source_files: dict[str, list[Path]] = {"Fig08": [], "Fig09": [], "Fig10": []}
    source_map = {
        "Fig08": [
            (cox_summary_frame, "cox_primary_summary.csv"),
            (coefficient_frame, "cox_coefficients.csv"),
            (fold_frame, "cox_fold_estimates.csv"),
            (meta_frame, "cox_fold_meta.csv"),
            (window_frame, "cox_time_window_sensitivity.csv"),
            (diagnostic_frame, "cox_ph_and_cindex_diagnostics.csv"),
        ],
        "Fig09": [
            (curve_frame, "km_aj_curves.parquet"),
            (risk_table_frame, "quartile_counts_and_risk_table.csv"),
            (absolute_risk_frame, "absolute_risk_24_60m.csv"),
            (logrank_frame, "logrank_tests.csv"),
        ],
        "Fig10": [
            (absolute_risk_frame, "organ_quartile_absolute_risk.csv"),
            (difference_frame, "q4_minus_q1_absolute_risk_difference.csv"),
        ],
    }
    for figure, tables in source_map.items():
        directory = source_staging[figure]
        for table, filename in tables:
            path = directory / filename
            _write_table(table, path)
            source_files[figure].append(path)
    command = " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "generation_command": command,
        "default_canonical_contract_enforced": is_default_canonical,
        "default_canonical_locked_sha256": LOCKED_INPUT_SHA256 if is_default_canonical else None,
        "locked_inputs": input_records,
        "code_and_contract_lineage": [
            {
                "role": "runner",
                "path": Path(__file__).resolve().as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "role": "statistical_core",
                "path": (Path(__file__).resolve().parent / "survival_analysis_core.py").as_posix(),
                "sha256": sha256_file(Path(__file__).resolve().parent / "survival_analysis_core.py"),
            },
            {
                "role": "analysis_contract",
                "path": (Path(__file__).resolve().parent / "ANALYSIS_CONTRACT.json").as_posix(),
                "sha256": sha256_file(Path(__file__).resolve().parent / "ANALYSIS_CONTRACT.json"),
            },
        ],
        "analysis_contract": {
            "population": "organ-specific at_risk == 1",
            "exposure": "continuous corrected_age_gap primary",
            "standardization": "other four OOF folds, eligible rows, ddof=0",
            "quartiles": "other four OOF folds, eligible rows",
            "cox": "unpenalized Breslow cause-specific, outer-fold-stratified",
            "inference": "model-based pooled CI plus five-fold fixed-effect meta and heterogeneity",
            "ph": "rank_lifelines_cumsum_event and log-time scaled Schoenfeld tests",
            "km": "death censored; descriptive",
            "competing_risk": "ties-safe Aalen-Johansen target/death CIF",
            "bootstrap": "participant resampling stratified by outer fold and crossfit quartile",
            "horizons_months": list(HORIZONS),
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "prediction_evaluation_boundary": summary_payload["c_index_boundary"],
        "row_counts": observed_counts,
        "row_count_gate_expected": expected_counts,
        "outputs": output_records,
        "figure_source_tables": {
            figure: [
                {
                    "path": (final_sources[figure] / path.relative_to(source_staging[figure])).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in paths
            ]
            for figure, paths in source_files.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "lifelines": lifelines.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "elapsed_seconds_before_manifest": time.time() - started,
    }
    manifest_path = output / "MANIFEST.json"
    _write_json(manifest_payload, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    for figure, paths in source_files.items():
        _source_manifest(
            source_staging[figure],
            final_sources[figure],
            figure,
            paths,
            final_output / "MANIFEST.json",
            manifest_sha256,
            input_records,
            command,
        )
    log.write(f"staging_complete seconds={time.time() - started:.2f} manifest_sha256={manifest_sha256}")
    output.replace(final_output)
    log.path = final_output / "generation.log"
    log.write(f"promoted status=COMPLETE manifest_sha256={manifest_sha256}")
    verification = _verify_existing(final_output)
    log.write(f"verify_after_promotion status={verification['status']} checked_files={verification['checked_files']}")
    try:
        for figure in ("Fig08", "Fig09", "Fig10"):
            _publish_figure_pointer(
                pointer_directories[figure],
                final_sources[figure],
                final_output / "MANIFEST.json",
                token=token,
            )
    except Exception as exc:
        log.write(f"figure_source_pointers status=INCOMPLETE error={type(exc).__name__}:{exc}")
        raise RuntimeError(
            "canonical bundle is complete and verified, but a convenience pointer failed; "
            "rerun with --publish-pointers-only"
        ) from exc
    log.write("figure_source_pointers status=COMPLETE figures=Fig08,Fig09,Fig10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
