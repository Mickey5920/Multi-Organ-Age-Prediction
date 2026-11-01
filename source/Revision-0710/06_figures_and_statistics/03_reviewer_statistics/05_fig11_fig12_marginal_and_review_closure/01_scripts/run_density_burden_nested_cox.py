#!/usr/bin/env python3
"""Run separate density and baseline-burden Cox adjustments for seven organs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
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
CONTEXT_RAW = (
    "smoking_current",
    "education_higher",
    "townsend",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "employment_other",
    "employment_missing",
    "sleep_duration",
)
DENSITY_RAW = (
    "phenotype_missing_fraction",
    "log1p_preindex_icd_count",
    "log1p_preindex_gp_count",
    "has_preindex_icd",
    "has_preindex_gp",
)
BURDEN_RAW = ("prevalent_organ_count",)
BINARY_RAW = (
    "smoking_current",
    "education_higher",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "employment_other",
    "employment_missing",
    "has_preindex_icd",
    "has_preindex_gp",
)
MODEL_ORDER = ("M0_age_sex", "M1_context", "M2D_density", "M2B_burden", "M3_combined")


def root_from_script() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = root_from_script()
SURVIVAL_ROOT = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/03_reviewer_statistics"
    / "04_cox_diagnostics_and_competing_risk"
)
if str(SURVIVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_ROOT))

from survival_analysis_core import (  # noqa: E402
    benjamini_hochberg,
    crossfit_covariates,
    crossfit_exposure_transform,
    fit_cox_breslow,
    schoenfeld_ph_test,
)


DEFAULT_OOF = (
    ROOT
    / "Revision-0710/03_ehr_age/04_fixed_replication/02_published_fivefold"
    / "02_oof_hr_km_inputs/ehr_age_d07_003_e55_oof.parquet"
)
DEFAULT_CONTEXT = (
    ROOT
    / "Revision-0710/05_downstream/06_metric_v2/01_contextual_covariates"
    / "participant_context_covariates.parquet"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "03_density_burden_nested_cox"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def transformed_for(raw_columns: tuple[str, ...], transformed: list[str]) -> list[str]:
    allowed: list[str] = []
    for raw in raw_columns:
        for candidate in (f"{raw}__missing", f"{raw}__xf"):
            if candidate in transformed:
                allowed.append(candidate)
    return allowed


def fit_model(
    frame: pd.DataFrame,
    organ: str,
    model: str,
    extras: list[str],
) -> tuple[dict[str, object], object, list[str], np.ndarray]:
    columns = ["age_per_10y", "sex", "gap_z_crossfit", *extras]
    x = frame[columns].to_numpy(dtype=np.float64)
    duration = frame["duration_months"].to_numpy(dtype=np.float64)
    event = frame["event"].to_numpy(dtype=np.int8)
    strata = frame["outer_fold"].to_numpy(dtype=np.int8)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{organ} {model}: non-finite design")
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise RuntimeError(f"{organ} {model}: design rank {rank}/{x.shape[1]}")
    gram = (x.T @ x) / len(x)
    condition = float(np.linalg.cond(gram))
    fitted = fit_cox_breslow(x, duration, event, strata=strata)
    if not fitted.converged:
        raise RuntimeError(f"{organ} {model}: Cox did not converge")
    gap_index = columns.index("gap_z_crossfit")
    beta = float(fitted.beta[gap_index])
    se = float(fitted.standard_error[gap_index])
    z95 = 1.959963984540054
    row = {
        "organ_system": organ,
        "model": model,
        "n": len(frame),
        "events": int(event.sum()),
        "n_covariates": len(columns),
        "design_rank": rank,
        "gram_condition_number": condition,
        "gap_beta_per_crossfit_sd": beta,
        "gap_se_model_based": se,
        "gap_hr_per_crossfit_sd": float(np.exp(beta)),
        "gap_hr_ci95_low": float(np.exp(beta - z95 * se)),
        "gap_hr_ci95_high": float(np.exp(beta + z95 * se)),
        "gap_p_value": float(fitted.p_value[gap_index]),
        "converged": bool(fitted.converged),
        "iterations": int(fitted.iterations),
        "ties": fitted.ties,
        "outer_fold_baseline_hazard": "stratified",
        "other_organ_age_gaps_in_model": 0,
    }
    return row, fitted, columns, x


def main() -> int:
    args = parse_args()
    inputs = [args.oof.resolve(), args.context.resolve()]
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
    if args.output_dir.exists():
        raise FileExistsError(f"create-once output already exists: {args.output_dir}")
    context_columns = ["participant_id", "fold_id", *CONTEXT_RAW, *DENSITY_RAW, *BURDEN_RAW]
    context = pd.read_parquet(args.context, columns=context_columns)
    context["participant_id"] = context["participant_id"].astype(str)
    if context["participant_id"].duplicated().any():
        raise RuntimeError("context participant IDs are not unique")

    rows: list[dict[str, object]] = []
    transform_audits: list[pd.DataFrame] = []
    exposure_audits: list[pd.DataFrame] = []
    ph_rows: list[pd.DataFrame] = []
    elapsed_rows: list[dict[str, object]] = []
    raw_all = (*CONTEXT_RAW, *DENSITY_RAW, *BURDEN_RAW)

    for organ in ORGANS:
        started = time.time()
        frame = pd.read_parquet(
            args.oof,
            columns=[
                "participant_id",
                "organ_system",
                "outer_fold",
                "actual_age",
                "sex",
                "corrected_age_gap",
                "at_risk",
                "duration_months",
                "event",
            ],
            filters=[("organ_system", "==", organ)],
        )
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame = frame.loc[frame["at_risk"].eq(1)].copy()
        frame = frame.merge(context, on="participant_id", how="left", validate="one_to_one")
        if not frame["fold_id"].eq(frame["outer_fold"]).all():
            raise RuntimeError(f"{organ}: context fold mismatch")
        if frame[["duration_months", "event", "actual_age", "sex"]].isna().any().any():
            raise RuntimeError(f"{organ}: missing primary analysis value")
        transformed_frame, exposure_audit = crossfit_exposure_transform(frame)
        exposure_audit.insert(0, "organ_system", organ)
        exposure_audits.append(exposure_audit)
        transformed_frame["age_per_10y"] = (transformed_frame["actual_age"] - 60.0) / 10.0
        transformed_frame, transform_audit, transformed_names = crossfit_covariates(
            transformed_frame,
            raw_all,
            binary_columns=BINARY_RAW,
        )
        transform_audit.insert(0, "organ_system", organ)
        transform_audits.append(transform_audit)
        context_names = transformed_for(CONTEXT_RAW, transformed_names)
        density_names = transformed_for(DENSITY_RAW, transformed_names)
        burden_names = transformed_for(BURDEN_RAW, transformed_names)
        models = {
            "M0_age_sex": [],
            "M1_context": context_names,
            "M2D_density": [*context_names, *density_names],
            "M2B_burden": [*context_names, *burden_names],
            "M3_combined": [*context_names, *density_names, *burden_names],
        }
        for model in MODEL_ORDER:
            row, fitted, columns, x = fit_model(transformed_frame, organ, model, models[model])
            rows.append(row)
            if model == "M3_combined":
                ph = schoenfeld_ph_test(
                    fitted,
                    x,
                    transformed_frame["duration_months"].to_numpy(float),
                    transformed_frame["event"].to_numpy(np.int8),
                    strata=transformed_frame["outer_fold"].to_numpy(np.int8),
                    covariate_names=columns,
                )
                ph = ph.loc[ph["covariate"].eq("gap_z_crossfit")].copy()
                ph.insert(0, "model", model)
                ph.insert(0, "organ_system", organ)
                ph_rows.append(ph)
        elapsed_rows.append(
            {
                "organ_system": organ,
                "n": len(transformed_frame),
                "events": int(transformed_frame["event"].sum()),
                "elapsed_seconds": time.time() - started,
            }
        )
        print(
            f"{organ}: n={len(transformed_frame):,} events={int(transformed_frame.event.sum()):,} "
            f"seconds={time.time() - started:.1f}",
            flush=True,
        )

    estimates = pd.DataFrame(rows)
    if len(estimates) != len(ORGANS) * len(MODEL_ORDER):
        raise RuntimeError(f"unexpected nested-model row count: {len(estimates)}")
    estimates["gap_q_value_bh_7_organs_within_model"] = np.nan
    for model in MODEL_ORDER:
        mask = estimates["model"].eq(model)
        estimates.loc[mask, "gap_q_value_bh_7_organs_within_model"] = benjamini_hochberg(
            estimates.loc[mask, "gap_p_value"]
        )

    beta = estimates.pivot(index="organ_system", columns="model", values="gap_beta_per_crossfit_sd")
    hr = estimates.pivot(index="organ_system", columns="model", values="gap_hr_per_crossfit_sd")
    attenuation_rows: list[dict[str, object]] = []
    for organ in ORGANS:
        m0 = float(beta.loc[organ, "M0_age_sex"])
        m1 = float(beta.loc[organ, "M1_context"])
        for model in MODEL_ORDER:
            current = float(beta.loc[organ, model])
            attenuation_rows.append(
                {
                    "organ_system": organ,
                    "model": model,
                    "gap_beta": current,
                    "gap_hr": float(hr.loc[organ, model]),
                    "beta_retained_vs_M0_percent": 100.0 * current / m0,
                    "beta_attenuation_vs_M0_percent": 100.0 * (m0 - current) / m0,
                    "beta_retained_vs_M1_percent": 100.0 * current / m1,
                    "beta_attenuation_vs_M1_percent": 100.0 * (m1 - current) / m1,
                }
            )
    attenuation = pd.DataFrame(attenuation_rows)
    ph = pd.concat(ph_rows, ignore_index=True)
    ph["q_value_bh_14_gap_ph_tests"] = benjamini_hochberg(ph["p_value"])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    estimates.to_csv(args.output_dir / "nested_cox_gap_estimates.csv", index=False, lineterminator="\n")
    attenuation.to_csv(args.output_dir / "density_burden_attenuation.csv", index=False, lineterminator="\n")
    ph.to_csv(args.output_dir / "m3_gap_ph_diagnostics.csv", index=False, lineterminator="\n")
    pd.concat(exposure_audits, ignore_index=True).to_csv(
        args.output_dir / "crossfit_gap_reference.csv", index=False, lineterminator="\n"
    )
    pd.concat(transform_audits, ignore_index=True).to_csv(
        args.output_dir / "crossfit_covariate_audit.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(elapsed_rows).to_csv(
        args.output_dir / "runtime_by_organ.csv", index=False, lineterminator="\n"
    )
    model_contract = {
        "M0_age_sex": "age + sex + target gap",
        "M1_context": "M0 + lifestyle/socioeconomic/sleep context",
        "M2D_density": "M1 + phenotype availability and pre-index ICD/GP density",
        "M2B_burden": "M1 + baseline seven-organ prevalence count",
        "M3_combined": "M1 + density/data availability + baseline prevalence burden",
    }
    (args.output_dir / "MODEL_CONTRACT.json").write_text(
        json.dumps(model_contract, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "revision0710.density_burden.nested_target_gap_cox.v1",
        "status": "COMPLETE",
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
        "models": model_contract,
        "population": "organ-specific at_risk == 1",
        "exposure": "target corrected age gap per other-four-fold SD",
        "outcome": "60-month cause-specific target-organ incidence; death censored",
        "cox": "unpenalized Breslow with outer-fold-stratified baseline hazards",
        "rows": len(estimates),
        "organs": list(ORGANS),
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

