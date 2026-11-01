from __future__ import annotations

import argparse
import os
from dataclasses import replace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from baseline_training import MODEL_NAMES, TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one leakage-safe Revision-0710 baseline")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(model_name=args.model, outer_fold=args.outer_fold)
    overrides = {
        "cache_dir": args.cache_dir,
        "output_root": args.output_root,
        "max_batches": args.max_batches,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    if args.no_amp:
        config = replace(config, amp=False)
    print(run_training(config))


if __name__ == "__main__":
    main()
