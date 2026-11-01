#!/usr/bin/env python
"""Run a dependency-level metric v2 smoke test without parquet or pytest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_DIR = Path(__file__).resolve().parents[1]
if str(METRIC_DIR) not in sys.path:
    sys.path.insert(0, str(METRIC_DIR))

from common.cox_scoring import fit_lifelines_cox_scores  # noqa: E402
from common.survival_metrics import (  # noqa: E402
    build_metric_time_grid,
    evaluate_censoring_aware_metrics,
)


def make_frame(n: int, seed: int, role: str, offset: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    event_time = rng.exponential(1.0 / (0.018 * np.exp(0.8 * x)))
    censor_time = rng.uniform(18.0, 90.0, size=n)
    duration = np.minimum(np.minimum(event_time, censor_time), 60.0)
    event = ((event_time <= censor_time) & (event_time < 60.0)).astype(np.int8)
    duration[:4] = [0.2, 60.0, 60.0, 60.0]
    event[:4] = [1, 0, 0, 0]
    return pd.DataFrame(
        {
            "participant_id": [f"p{offset + index}" for index in range(n)],
            "duration": duration,
            "event": event,
            "split_role": role,
            "outer_fold": 0,
            "x": x,
            "sex": rng.integers(0, 2, size=n),
        }
    )


def main() -> None:
    import lifelines
    import sksurv

    fit = make_frame(600, 710, "validation", 0)
    score = make_frame(400, 711, "test", 10_000)
    grid = build_metric_time_grid(
        fit,
        score,
        ibs_times=(1.0, 6.0, 12.0, 24.0, 36.0, 48.0, 60.0),
    )
    bundle = fit_lifelines_cox_scores(
        fit,
        score,
        covariates=("x", "sex"),
        time_grid=grid,
        phase="formal",
    )
    report, calibration = evaluate_censoring_aware_metrics(
        fit,
        score,
        risk_score=bundle.risk_score,
        survival_probability=bundle.survival_probability,
        time_grid=grid,
        calibration_bins=5,
    )
    if not 0.0 <= float(report["uno_c_60m"]) <= 1.0:
        raise RuntimeError("invalid Uno C")
    if len(calibration) != 15:
        raise RuntimeError("unexpected calibration row count")
    print(
        json.dumps(
            {
                "status": "PASS",
                "python": sys.version.split()[0],
                "sksurv": sksurv.__version__,
                "lifelines": lifelines.__version__,
                "uno_c_60m": report["uno_c_60m"],
                "auc_60m": report["dynamic_auc"]["60m"],
                "ibs": report["integrated_brier_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
