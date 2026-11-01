#!/usr/bin/env python3
"""Recalculate R3.2 Cox diagnostics and competing-risk absolute risks.

This runner is deliberately downstream-only: it reads the final five-fold OOF
predictions, never retrains EHR-AGE, and writes an independent evidence bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

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
EXPECTED = {
    "participants": 456_773,
    "joined_participant_organ": 3_197_411,
    "at_risk_participant_organ": 2_914_947,
    "events": 162_125,
    "competing_deaths": 13_972,
}
EXPECTED_SHA256 = {
    "oof": "bfc9e0de88121bb4b520bd6c878ac4d655d00b7d1dfa1f64b5f7a4de0b633239",
    "outcomes": "3a9021f00080c25fd523debcdf075a5be8281c9658db261af99a8bdc11f4e735",
    "context": "2b4d46d69b1c5c2c9629b27c12dce2d4c3460028fc16a0baffcdfdcbd44fb0bc",
    "v2_labels": "ce8dbbd8a149a872655cb6d37c0f2f7928b1471d4a54857c7d1189ecebcaaa27",
}
HORIZONS = (24.0, 60.0)


def workspace_root() -> Path:
    for parent in (Path.cwd(), *Path(__file__).resolve().parents):
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
REVISION = ROOT / "Revision-0710"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Source_Data"
OOF = (
    REVISION
    / "03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1/02_summary"
    / "oof_predictions.parquet"
)
OUTCOMES = REVISION / "01_data_rebuild/09_v2_labels/organ_outcomes_60m.parquet"
CONTEXT = (
    REVISION
    / "05_downstream/06_metric_v2/01_contextual_covariates"
    / "participant_context_covariates.parquet"
)
V2_LABELS = REVISION / "01_data_rebuild/09_v2_labels/icd_participant_labels_v2.parquet"
BURDEN_RUNNER = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/12_e47_disease_burden_adjustment"
    / "run_e47_disease_burden_adjustment.py"
)
BURDEN_ESTIMATES = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/12_e47_disease_burden_adjustment"
    / "Source_Data/cox_gap_estimates.csv"
)
SURVIVAL_ROOT = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics"
    / "04_cox_diagnostics_and_competing_risk"
)
SURVIVAL_CORE = SURVIVAL_ROOT / "survival_analysis_core.py"
PRIMARY_RUNNER = SURVIVAL_ROOT / "run_primary_survival.py"

if str(SURVIVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_ROOT))

from survival_analysis_core import (  # noqa: E402
    benjamini_hochberg,
    crossfit_covariates,
    crossfit_exposure_transform,
    fit_cox_breslow,
    schoenfeld_ph_test,
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_table(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return {
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }


def validate_hashes() -> list[dict[str, object]]:
    paths = {"oof": OOF, "outcomes": OUTCOMES, "context": CONTEXT, "v2_labels": V2_LABELS}
    rows: list[dict[str, object]] = []
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        expected = EXPECTED_SHA256[role]
        if observed != expected:
            raise RuntimeError(f"{role} SHA-256 mismatch: {observed} != {expected}")
        rows.append(
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "sha256": observed,
                "bytes": int(path.stat().st_size),
            }
        )
    return rows


def load_joined_organ(
    burden: ModuleType,
    context: pd.DataFrame,
    organ: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    oof = burden.load_organ_frame(OOF, context, organ)
    outcome = pd.read_parquet(
        OUTCOMES,
        columns=[
            "participant_id",
            "organ_system",
            "at_risk",
            "event",
            "duration_months",
            "competing_death",
            "has_prevalent_target",
        ],
        filters=[("organ_system", "==", organ)],
    )
    outcome["participant_id"] = outcome["participant_id"].astype(str)
    keys = ["participant_id", "organ_system"]
    if outcome.duplicated(keys).any():
        raise RuntimeError(f"{organ}: duplicate outcome keys")
    joined = oof.merge(
        outcome,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_oof", ""),
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(f"{organ}: incomplete OOF/outcome join")
    mismatch = {
        "at_risk": int(joined["at_risk_oof"].ne(joined["at_risk"]).sum()),
        "event": int(joined["event_oof"].ne(joined["event"]).sum()),
    }
    risk = joined["at_risk"].eq(1)
    duration_mismatch = ~np.isclose(
        joined.loc[risk, "duration_months_oof"].to_numpy(float),
        joined.loc[risk, "duration_months"].to_numpy(float),
        rtol=0.0,
        # OOF duration passed through a float32 prediction export, whereas the
        # authoritative outcome table retains float64 precision.
        atol=1e-5,
    )
    mismatch["duration_months_at_risk"] = int(duration_mismatch.sum())
    if any(mismatch.values()):
        raise RuntimeError(f"{organ}: OOF/outcome mismatch {mismatch}")
    if ((joined["event"] + joined["competing_death"]) > 1).any():
        raise RuntimeError(f"{organ}: target event overlaps competing death")
    if joined.loc[risk, "duration_months"].isna().any() or joined.loc[
        risk, "duration_months"
    ].le(0).any():
        raise RuntimeError(f"{organ}: invalid authoritative at-risk duration")
    if (risk & joined["has_prevalent_target"].eq(1)).any():
        raise RuntimeError(f"{organ}: prevalent target remains at risk")
    joined = joined.drop(
        columns=[
            "_merge",
            "at_risk_oof",
            "event_oof",
            "duration",
            "duration_months_oof",
        ]
    )
    audit = {
        "organ_system": organ,
        "joined_rows": int(len(joined)),
        "unique_participants": int(joined["participant_id"].nunique()),
        "at_risk_n": int(joined["at_risk"].sum()),
        "target_events": int(joined["event"].sum()),
        "competing_deaths": int(joined["competing_death"].sum()),
        "key_mismatch_rows": 0,
        **{f"field_mismatch_{key}": value for key, value in mismatch.items()},
        "status": "PASS",
    }
    return joined, audit


def prepare_design(
    burden: ModuleType,
    joined: pd.DataFrame,
    organ: str,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    risk, gap_reference = crossfit_exposure_transform(joined)
    risk["age_per_10y"] = (risk["actual_age"] - 60.0) / 10.0
    detail_raw = burden.detail_raw_for(organ, include_target=False)
    raw = (*burden.CONTEXT_RAW, *burden.DENSITY_RAW, *detail_raw)
    risk, covariate_audit, transformed = crossfit_covariates(
        risk,
        raw,
        binary_columns=burden.BINARY_RAW,
    )
    extras = burden.transformed_for(raw, transformed)
    columns = ["age_per_10y", "sex", "gap_z_crossfit", *extras]
    x = risk[columns].to_numpy(np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{organ}: nonfinite Cox design")
    rank = int(np.linalg.matrix_rank(x))
    if rank != len(columns):
        raise RuntimeError(f"{organ}: Cox design rank {rank}/{len(columns)}")
    gap_reference.insert(0, "organ_system", organ)
    covariate_audit.insert(0, "organ_system", organ)
    return risk, columns, gap_reference, covariate_audit


def fit_full_and_ph(
    risk: pd.DataFrame,
    columns: Sequence[str],
    organ: str,
    reference: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    x = risk[list(columns)].to_numpy(np.float64)
    duration = risk["duration_months"].to_numpy(np.float64)
    event = risk["event"].to_numpy(np.int8)
    strata = risk["outer_fold"].to_numpy(np.int8)
    fit = fit_cox_breslow(x, duration, event, strata=strata)
    if not fit.converged:
        raise RuntimeError(f"{organ}: full adjusted Cox did not converge")
    gap_index = list(columns).index("gap_z_crossfit")
    beta = float(fit.beta[gap_index])
    # The prior adjustment table used the float32 duration copied through the
    # OOF export; this closure uses the matching authoritative float64 outcome
    # duration. Require numerical agreement to 1e-6 and retain the exact delta.
    if not np.isclose(beta, float(reference["gap_beta_per_crossfit_sd"]), rtol=0.0, atol=1e-6):
        raise RuntimeError(
            f"{organ}: full-model gap beta parity failed: {beta} vs "
            f"{reference['gap_beta_per_crossfit_sd']}"
        )
    z95 = 1.959963984540054
    summary = {
        "organ_system": organ,
        "n": int(len(risk)),
        "events": int(event.sum()),
        "competing_deaths": int(risk["competing_death"].sum()),
        "n_covariates": int(len(columns)),
        "design_rank": int(np.linalg.matrix_rank(x)),
        "gap_beta_per_crossfit_sd": beta,
        "gap_se_model_based": float(fit.standard_error[gap_index]),
        "gap_hr_per_crossfit_sd": float(np.exp(beta)),
        "gap_hr_ci95_low": float(np.exp(beta - z95 * fit.standard_error[gap_index])),
        "gap_hr_ci95_high": float(np.exp(beta + z95 * fit.standard_error[gap_index])),
        "gap_p_value": float(fit.p_value[gap_index]),
        "log_partial_likelihood": float(fit.log_likelihood),
        "aic_partial": float(-2.0 * fit.log_likelihood + 2.0 * len(columns)),
        "converged": bool(fit.converged),
        "iterations": int(fit.iterations),
        "score_max_abs": float(fit.score_max_abs),
        "ties": fit.ties,
        "outer_fold_baseline_hazard": "stratified",
        "reference_beta_absolute_difference": float(
            abs(beta - float(reference["gap_beta_per_crossfit_sd"]))
        ),
    }
    coefficients = pd.DataFrame(
        {
            "organ_system": organ,
            "covariate": list(columns),
            "beta": fit.beta,
            "se_model_based": fit.standard_error,
            "hr": np.exp(fit.beta),
            "hr_ci95_low": np.exp(fit.beta - z95 * fit.standard_error),
            "hr_ci95_high": np.exp(fit.beta + z95 * fit.standard_error),
            "p_value": fit.p_value,
        }
    )
    ph = schoenfeld_ph_test(
        fit,
        x,
        duration,
        event,
        strata=strata,
        covariate_names=columns,
    )
    ph.insert(0, "organ_system", organ)
    return summary, coefficients, ph


def fit_windows(
    risk: pd.DataFrame,
    columns: Sequence[str],
    organ: str,
) -> pd.DataFrame:
    definitions = (
        (
            "0_to_24_months",
            risk.assign(
                window_duration=np.minimum(risk["duration_months"], 24.0),
                window_event=(risk["event"].eq(1) & risk["duration_months"].le(24.0)).astype(
                    np.int8
                ),
            ),
            "all baseline at-risk participants; administratively censored at 24 months",
        ),
        (
            "24_to_60_months_landmark",
            risk.loc[risk["duration_months"].gt(24.0)].assign(
                window_duration=lambda x: x["duration_months"] - 24.0,
                window_event=lambda x: x["event"].astype(np.int8),
            ),
            "alive and target-event-free beyond 24 months; time origin reset to month 24",
        ),
    )
    rows: list[dict[str, object]] = []
    z95 = 1.959963984540054
    gap_index = list(columns).index("gap_z_crossfit")
    for window, part, definition in definitions:
        fit = fit_cox_breslow(
            part[list(columns)].to_numpy(np.float64),
            part["window_duration"].to_numpy(np.float64),
            part["window_event"].to_numpy(np.int8),
            strata=part["outer_fold"].to_numpy(np.int8),
        )
        if not fit.converged:
            raise RuntimeError(f"{organ} {window}: Cox did not converge")
        beta = float(fit.beta[gap_index])
        se = float(fit.standard_error[gap_index])
        rows.append(
            {
                "organ_system": organ,
                "time_window": window,
                "window_definition": definition,
                "n_at_window_start": int(len(part)),
                "events_in_window": int(part["window_event"].sum()),
                "n_covariates": int(len(columns)),
                "gap_beta_per_crossfit_sd": beta,
                "gap_se_model_based": se,
                "gap_hr_per_crossfit_sd": float(np.exp(beta)),
                "gap_hr_ci95_low": float(np.exp(beta - z95 * se)),
                "gap_hr_ci95_high": float(np.exp(beta + z95 * se)),
                "gap_p_value": float(fit.p_value[gap_index]),
                "log_partial_likelihood": float(fit.log_likelihood),
                "aic_partial": float(-2.0 * fit.log_likelihood + 2.0 * len(columns)),
                "converged": bool(fit.converged),
                "iterations": int(fit.iterations),
                "ties": fit.ties,
                "outer_fold_baseline_hazard": "stratified",
            }
        )
    return pd.DataFrame(rows)


def add_bh(frame: pd.DataFrame, p_column: str, q_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[q_column] = benjamini_hochberg(result[p_column].to_numpy(float))
    return result


def environment_receipt() -> dict[str, str]:
    import scipy

    versions = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import pyarrow

        versions["pyarrow"] = pyarrow.__version__
    except ImportError:
        versions["pyarrow"] = "not installed"
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def verify_existing() -> dict[str, object]:
    receipt_path = HERE / "SURVIVAL_RUN_RECEIPT.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in [*receipt["inputs"], *receipt["lineage"], *receipt["outputs"]]:
        path = Path(record["path"])
        if not path.is_file():
            failures.append(f"missing:{path}")
        elif sha256_file(path) != record["sha256"]:
            failures.append(f"sha256:{path}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checked": int(len(receipt["inputs"]) + len(receipt["lineage"]) + len(receipt["outputs"])),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> int:
    args = parse_args()
    if args.verify_only:
        print(json.dumps(verify_existing(), indent=2), flush=True)
        return 0
    if args.bootstrap_replicates != 500 or args.seed != 710:
        raise ValueError("formal run requires 500 bootstrap replicates and seed 710")
    receipt_path = HERE / "SURVIVAL_RUN_RECEIPT.json"
    if receipt_path.exists():
        raise FileExistsError("survival run already sealed; use --verify-only")
    SOURCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    input_records = validate_hashes()
    burden = load_module("r3_2_burden_support", BURDEN_RUNNER)
    primary = load_module("r3_2_primary_support", PRIMARY_RUNNER)
    context, context_audit = burden.load_context(CONTEXT, V2_LABELS)
    reference = pd.read_csv(BURDEN_ESTIMATES)
    reference = reference.loc[reference["model"].eq("M3_detail")].set_index("organ_system")
    if set(reference.index) != set(ORGANS):
        raise RuntimeError("full-adjustment reference does not cover seven organs")

    audit_rows: list[dict[str, object]] = []
    gap_references: list[pd.DataFrame] = []
    covariate_audits: list[pd.DataFrame] = []
    full_rows: list[dict[str, object]] = []
    coefficients: list[pd.DataFrame] = []
    ph_tables: list[pd.DataFrame] = []
    windows: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    risks: list[pd.DataFrame] = []
    differences: list[pd.DataFrame] = []
    risk_tables: list[pd.DataFrame] = []
    logranks: list[pd.DataFrame] = []
    log_lines = [f"{utc_now()} start bootstrap=500 seed=710"]

    for organ_index, organ in enumerate(ORGANS):
        organ_started = time.time()
        joined, audit = load_joined_organ(burden, context, organ)
        risk, columns, gap_reference, covariate_audit = prepare_design(burden, joined, organ)
        full, coefficient, ph = fit_full_and_ph(risk, columns, organ, reference.loc[organ])
        window = fit_windows(risk, columns, organ)
        curve, absolute, difference, risk_table, logrank = primary._run_nonparametric(
            risk,
            organ,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed + 1009 * organ_index,
        )
        audit_rows.append(audit)
        gap_references.append(gap_reference)
        covariate_audits.append(covariate_audit)
        full_rows.append(full)
        coefficients.append(coefficient)
        ph_tables.append(ph)
        windows.append(window)
        curves.append(curve)
        risks.append(absolute)
        differences.append(difference)
        risk_tables.append(risk_table)
        logranks.append(logrank)
        log_lines.append(
            f"{utc_now()} organ_complete organ={organ} n={len(risk)} "
            f"events={int(risk['event'].sum())} deaths={int(risk['competing_death'].sum())} "
            f"seconds={time.time() - organ_started:.3f}"
        )

    audit_frame = pd.DataFrame(audit_rows)
    observed = {
        "participants": int(audit_frame["unique_participants"].min()),
        "joined_participant_organ": int(audit_frame["joined_rows"].sum()),
        "at_risk_participant_organ": int(audit_frame["at_risk_n"].sum()),
        "events": int(audit_frame["target_events"].sum()),
        "competing_deaths": int(audit_frame["competing_deaths"].sum()),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"row-count gate failed: observed={observed} expected={EXPECTED}")

    full_frame = add_bh(
        pd.DataFrame(full_rows),
        "gap_p_value",
        "gap_q_value_bh_7_organs",
    )
    ph_frame = pd.concat(ph_tables, ignore_index=True)
    ph_frame = add_bh(ph_frame, "p_value", "q_value_bh_all_covariate_tests")
    focal_mask = ph_frame["covariate"].eq("gap_z_crossfit")
    focal = add_bh(
        ph_frame.loc[focal_mask].copy(),
        "p_value",
        "gap_q_value_bh_14_tests",
    )
    ph_frame["gap_q_value_bh_14_tests"] = np.nan
    ph_frame.loc[focal.index, "gap_q_value_bh_14_tests"] = focal[
        "gap_q_value_bh_14_tests"
    ].to_numpy()
    window_frame = pd.concat(windows, ignore_index=True)
    window_frame["gap_q_value_bh_7_organs_within_window"] = np.nan
    for _, indices in window_frame.groupby("time_window", sort=False).groups.items():
        idx = list(indices)
        window_frame.loc[idx, "gap_q_value_bh_7_organs_within_window"] = benjamini_hochberg(
            window_frame.loc[idx, "gap_p_value"].to_numpy(float)
        )

    outputs = [
        write_table(audit_frame, SOURCE / "input_join_audit.csv"),
        write_table(pd.concat(gap_references, ignore_index=True), SOURCE / "crossfit_gap_reference.csv"),
        write_table(
            pd.concat(covariate_audits, ignore_index=True),
            SOURCE / "crossfit_covariate_audit.csv",
        ),
        write_table(full_frame, SOURCE / "full_adjusted_cox_gap.csv"),
        write_table(pd.concat(coefficients, ignore_index=True), SOURCE / "full_adjusted_cox_coefficients.csv"),
        write_table(ph_frame, SOURCE / "full_adjusted_ph_diagnostics.csv"),
        write_table(focal, SOURCE / "target_gap_ph_diagnostics.csv"),
        write_table(window_frame, SOURCE / "full_adjusted_time_window_cox.csv"),
        write_table(pd.concat(curves, ignore_index=True), SOURCE / "km_aj_curves.parquet"),
        write_table(pd.concat(risks, ignore_index=True), SOURCE / "absolute_risk_24_60m.csv"),
        write_table(pd.concat(differences, ignore_index=True), SOURCE / "q4_minus_q1_risk_difference.csv"),
        write_table(pd.concat(risk_tables, ignore_index=True), SOURCE / "quartile_numbers_at_risk.csv"),
        write_table(pd.concat(logranks, ignore_index=True), SOURCE / "logrank_tests.csv"),
    ]
    elapsed = time.time() - started
    log_lines.append(f"{utc_now()} complete seconds={elapsed:.3f}")
    log_path = HERE / "analysis_log.md"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    outputs.append(
        {
            "path": log_path.resolve().as_posix(),
            "sha256": sha256_file(log_path),
            "bytes": int(log_path.stat().st_size),
        }
    )
    lineage = []
    for role, path in (
        ("runner", Path(__file__).resolve()),
        ("statistical_core", SURVIVAL_CORE),
        ("competing_risk_support_runner", PRIMARY_RUNNER),
        ("burden_covariate_support_runner", BURDEN_RUNNER),
        ("full_adjustment_parity_reference", BURDEN_ESTIMATES),
        ("analysis_plan", HERE / "ANALYSIS_PLAN.md"),
    ):
        lineage.append(
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        )
    receipt = {
        "schema_version": "r3_2.final_model_survival_recalculation.v1",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "analysis_role": "final_fivefold_oof_ph_windows_and_competing_risk",
        "model": {
            "exposure": "continuous corrected target-organ gap per other-four-fold SD",
            "cox": "unpenalized cause-specific Cox; Breslow ties; outer-fold-stratified baseline",
            "adjustment": "age, sex, context, phenotype missingness, pre-index ICD/GP density, and non-target recorded prevalent burden",
            "ph": "scaled Schoenfeld tests with rank and log-time transforms",
            "windows_months": ["0-24", "24-60 landmark"],
            "death_in_cox": "censored at death",
            "absolute_risk": "Aalen-Johansen with death retained as competing event",
            "fine_gray_performed": False,
        },
        "bootstrap": {
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "strata": "outer fold x cross-fitted gap quartile",
        },
        "context_audit": context_audit,
        "row_counts": observed,
        "inputs": input_records,
        "lineage": lineage,
        "outputs": outputs,
        "environment": environment_receipt(),
        "elapsed_seconds": float(elapsed),
    }
    json_write(receipt_path, receipt)
    print(json.dumps({"status": "COMPLETE", "elapsed_seconds": elapsed, "row_counts": observed}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
