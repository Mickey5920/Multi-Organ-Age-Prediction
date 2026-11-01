from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "bootstrap_objective_matched_v2.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_objective_matched_v2", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load bootstrap module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    largest_error = 0.0
    for trial in range(args.trials):
        n = int(rng.integers(20, 400))
        duration = rng.integers(1, 20, size=n).astype(np.float64)
        event = rng.random(n) < 0.35
        if not event.any():
            event[0] = True
        score = np.round(rng.normal(size=n), 2)
        multiplicity = rng.integers(0, 5, size=n, dtype=np.int64)
        expanded = np.repeat(np.arange(n), multiplicity)
        if len(expanded) < 3 or not event[expanded].any():
            continue
        expected = MODULE.harrell_c(
            event[expanded], duration[expanded], score[expanded]
        )
        context = MODULE.prepare_weighted_cindex(event, duration, score)
        observed = MODULE.weighted_concordance_from_context(multiplicity, *context)
        error = abs(expected - observed)
        largest_error = max(largest_error, error)
        if error > 1e-15:
            raise RuntimeError(
                f"trial {trial} mismatch: expected={expected} observed={observed} error={error}"
            )
    print(
        f"PASS trials={args.trials} largest_absolute_error={largest_error:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
