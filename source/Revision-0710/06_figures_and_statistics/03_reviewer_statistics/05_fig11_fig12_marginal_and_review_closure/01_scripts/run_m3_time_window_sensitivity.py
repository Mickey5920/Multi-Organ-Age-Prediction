#!/usr/bin/env python3
"""Estimate prespecified M3 target-gap HRs within (0,24] and (24,60] months."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_density_burden_nested_cox import (
    BINARY_RAW,
    BURDEN_RAW,
    CONTEXT_RAW,
    DEFAULT_CONTEXT,
    DEFAULT_OOF,
    DENSITY_RAW,
    ORGANS,
    benjamini_hochberg,
    crossfit_covariates,
    crossfit_exposure_transform,
    fit_cox_breslow,
    sha256_file,
    transformed_for,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "05_m3_time_window_sensitivity"
WINDOWS = (
    (
        "early_0_24_closed_right",
        "(0,24] months; all baseline at-risk participants; administrative censoring at 24 months",
    ),
    (
        "late_24_60_closed_right",
        "(24,60] months; alive and target-event-free beyond 24 months; time origin reset to month 24",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def window_frame(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "early_0_24_closed_right":
        part = frame.copy()
        part["window_duration"] = np.minimum(part["duration_months"], 24.0)
        part["window_event"] = (
            part["event"].eq(1) & part["duration_months"].le(24.0)
        ).astype(np.int8)
        return part
    if window == "late_24_60_closed_right":
        part = frame.loc[frame["duration_months"].gt(24.0)].copy()
        part["window_duration"] = part["duration_months"] - 24.0
        part["window_event"] = part["event"].astype(np.int8)
        return part
    raise ValueError(f"unknown window: {window}")


def main() -> int:
    args = parse_args()
    inputs = [args.oof.resolve(), args.context.resolve()]
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
    if args.output_dir.exists():
        raise FileExistsError(f"create-once output already exists: {args.output_dir}")

    raw_all = (*CONTEXT_RAW, *DENSITY_RAW, *BURDEN_RAW)
    context_columns = ["participant_id", "fold_id", *raw_all]
    context = pd.read_parquet(args.context, columns=context_columns)
    context["participant_id"] = context["participant_id"].astype(str)
    if context["participant_id"].duplicated().any():
        raise RuntimeError("context participant IDs are not unique")

    rows: list[dict[str, object]] = []
    started = time.time()
    z95 = 1.959963984540054
    for organ in ORGANS:
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
        frame, _ = crossfit_exposure_transform(frame)
        frame["age_per_10y"] = (frame["actual_age"] - 60.0) / 10.0
        frame, _, transformed_names = crossfit_covariates(
            frame,
            raw_all,
            binary_columns=BINARY_RAW,
        )
        extras = [
            *transformed_for(CONTEXT_RAW, transformed_names),
            *transformed_for(DENSITY_RAW, transformed_names),
            *transformed_for(BURDEN_RAW, transformed_names),
        ]
        columns = ["age_per_10y", "sex", "gap_z_crossfit", *extras]
        gap_index = columns.index("gap_z_crossfit")
        for window, definition in WINDOWS:
            part = window_frame(frame, window)
            x = part[columns].to_numpy(dtype=np.float64)
            duration = part["window_duration"].to_numpy(dtype=np.float64)
            event = part["window_event"].to_numpy(dtype=np.int8)
            strata = part["outer_fold"].to_numpy(dtype=np.int8)
            rank = int(np.linalg.matrix_rank(x))
            if rank != x.shape[1]:
                raise RuntimeError(f"{organ} {window}: design rank {rank}/{x.shape[1]}")
            fitted = fit_cox_breslow(x, duration, event, strata=strata)
            if not fitted.converged:
                raise RuntimeError(f"{organ} {window}: Cox did not converge")
            beta = float(fitted.beta[gap_index])
            se = float(fitted.standard_error[gap_index])
            rows.append(
                {
                    "organ_system": organ,
                    "model": "M3_combined",
                    "time_window": window,
                    "window_definition": definition,
                    "n_at_window_start": len(part),
                    "events_in_window": int(event.sum()),
                    "n_covariates": len(columns),
                    "design_rank": rank,
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
            )
        print(f"{organ}: both windows complete", flush=True)

    results = pd.DataFrame(rows)
    if len(results) != len(ORGANS) * len(WINDOWS):
        raise RuntimeError(f"unexpected result rows: {len(results)}")
    results["gap_q_value_bh_7_organs_within_window"] = np.nan
    for window, _ in WINDOWS:
        mask = results["time_window"].eq(window)
        results.loc[mask, "gap_q_value_bh_7_organs_within_window"] = benjamini_hochberg(
            results.loc[mask, "gap_p_value"]
        )
    if not np.isfinite(
        results[
            [
                "gap_beta_per_crossfit_sd",
                "gap_se_model_based",
                "gap_hr_per_crossfit_sd",
                "gap_hr_ci95_low",
                "gap_hr_ci95_high",
                "gap_p_value",
                "gap_q_value_bh_7_organs_within_window",
            ]
        ].to_numpy(float)
    ).all():
        raise RuntimeError("non-finite time-window result")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(
        args.output_dir / "m3_time_window_gap_hr.csv",
        index=False,
        lineterminator="\n",
    )
    manifest = {
        "schema_version": "revision0710.density_burden.m3_time_window_sensitivity.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "model": "M3_combined",
        "windows": [window for window, _ in WINDOWS],
        "interpretation": "within-window average association; constant-effect wording forbidden when PH diagnostics reject",
        "inputs": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        ],
        "rows": len(results),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
