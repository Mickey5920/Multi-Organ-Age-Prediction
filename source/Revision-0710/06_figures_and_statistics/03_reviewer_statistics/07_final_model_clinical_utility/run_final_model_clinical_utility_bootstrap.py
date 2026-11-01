#!/usr/bin/env python3
"""Paired participant bootstrap for the frozen final-model clinical-utility predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from sksurv.nonparametric import CensoringDistributionEstimator


def workspace_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "Revision-0710").is_dir():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
REVISION = ROOT / "Revision-0710"
ESTABLISHED_BOOTSTRAP = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics"
    / "05_fig11_fig12_marginal_and_review_closure/01_scripts"
    / "run_clinical_utility_paired_bootstrap.py"
)
ANALYSIS_ROOT = (
    REVISION / "Final_Fig_0716/Reviewer_Statistics/13_final_model_clinical_utility"
)
INPUT_CACHE = ANALYSIS_ROOT / "00_input_cache"
POINT_ROOT = ANALYSIS_ROOT / "01_point"
WORK_ROOT = ANALYSIS_ROOT / "02_bootstrap_work"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "03_bootstrap_1000"
FAST_INPUT_ROOT = ANALYSIS_ROOT / "02_fast_bootstrap_input"
CONTEXT_PICKLE = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics/03_clinical_utility"
    / "01_formal_run/02_input_cache/participant_context_covariates.pkl"
)


def load_established_bootstrap() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "established_clinical_utility_bootstrap", ESTABLISHED_BOOTSTRAP
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import established bootstrap: {ESTABLISHED_BOOTSTRAP}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_module(module: ModuleType) -> None:
    module.CACHE_ROOT = POINT_ROOT / "folds"
    module.CONTEXT_PICKLE = CONTEXT_PICKLE
    module.POINT_CELL = POINT_ROOT / "cell_incremental_metrics.csv"
    module.POINT_ORGAN = POINT_ROOT / "organ_incremental_metrics.csv"
    module.POINT_MACRO = POINT_ROOT / "macro_incremental_metrics.csv"


def load_fold_cells(
    module: ModuleType,
    fold: int,
    context: pd.DataFrame,
) -> tuple[dict[tuple[int, str], object], np.ndarray]:
    ids = pd.to_numeric(
        context.loc[context["fold_id"].eq(fold), "participant_id"], errors="raise"
    ).to_numpy(dtype=np.int64)
    universe = np.sort(ids)
    cells: dict[tuple[int, str], object] = {}
    fit_all = pd.read_pickle(INPUT_CACHE / f"outer_fold_{fold}__calibration.pkl")
    fit_all["participant_id"] = fit_all["participant_id"].astype(str)
    for organ in module.ORGANS:
        npz_path = module.CACHE_ROOT / f"outer_fold_{fold}/prediction_cache/{organ}.npz"
        with np.load(npz_path) as cache:
            participant_id = cache["participant_id"].astype(np.int64)
            duration = cache["duration"].astype(np.float64)
            event = cache["event"].astype(np.int8)
            base_prefix = f"{module.TIER}__base"
            base_score = cache[f"{base_prefix}__risk_score"].astype(np.float64)
            base_risk = cache[f"{base_prefix}__risk_60m"].astype(np.float64)
            augmented_score = {
                augmentation: cache[
                    f"{module.TIER}__{augmentation}__risk_score"
                ].astype(np.float64)
                for augmentation in module.AUGMENTATIONS
            }
            augmented_risk = {
                augmentation: cache[
                    f"{module.TIER}__{augmentation}__risk_60m"
                ].astype(np.float64)
                for augmentation in module.AUGMENTATIONS
            }
        fit = fit_all.loc[
            fit_all["organ_system"].eq(organ) & fit_all["at_risk"].eq(1),
            ["duration", "event"],
        ].copy()
        fit_duration = pd.to_numeric(fit["duration"], errors="raise").to_numpy(float)
        fit_event = pd.to_numeric(fit["event"], errors="raise").to_numpy(np.int8)
        boundary = float(min(fit_duration.max(), duration.max()))
        fit_boundary_event = (
            np.isclose(fit_duration, boundary, atol=1e-6, rtol=0.0)
            & (fit_event == 1)
        )
        fit_event[fit_boundary_event] = 0
        fit_survival = module.survival(fit_duration, fit_event)
        censoring = CensoringDistributionEstimator().fit(fit_survival)
        effective = float(np.nextafter(boundary, 0.0))
        score_survival = module.survival(duration, event)
        uno_event_weight = np.square(censoring.predict_ipcw(score_survival))
        uno_event_weight[duration >= effective] = 0.0
        case_weight, control_weight = module.ipcw_components(
            duration,
            event,
            censoring,
            effective,
        )
        positions = np.searchsorted(universe, participant_id)
        if np.any(positions >= len(universe)) or not np.array_equal(
            universe[positions], participant_id
        ):
            raise RuntimeError(f"fold {fold} {organ}: cache IDs outside fold universe")
        cells[(fold, organ)] = module.Cell(
            fold=fold,
            organ=organ,
            participant_id=participant_id,
            duration=duration,
            event=event,
            fit_survival=fit_survival,
            censoring=censoring,
            effective_horizon=effective,
            uno_event_weight=uno_event_weight,
            case_weight=case_weight,
            control_weight=control_weight,
            base_score=base_score,
            base_risk=base_risk,
            augmented_score=augmented_score,
            augmented_risk=augmented_risk,
            universe_positions=positions,
        )
    return cells, universe


def explicit_incremental_metrics(
    module: ModuleType,
    cell: object,
    indices: np.ndarray,
    augmentation: str,
    base: dict[str, float],
) -> dict[str, float]:
    augmented = module.explicit_model_metrics(
        cell,
        indices,
        cell.augmented_score[augmentation],
        cell.augmented_risk[augmentation],
    )
    duration = cell.duration[indices]
    event = cell.event[indices]
    old = cell.base_risk[indices]
    new = cell.augmented_risk[augmentation][indices]
    case_weight, control_weight = module.ipcw_components(
        duration,
        event,
        cell.censoring,
        cell.effective_horizon,
    )
    idi = (
        module.weighted_mean(new, case_weight)
        - module.weighted_mean(old, case_weight)
        - module.weighted_mean(new, control_weight)
        + module.weighted_mean(old, control_weight)
    )
    up = (new > old).astype(float)
    down = (new < old).astype(float)
    nri = module.weighted_mean(up - down, case_weight) + module.weighted_mean(
        down - up, control_weight
    )
    delta_nb = []
    n = float(len(indices))
    for threshold in module.THRESHOLDS:
        odds = float(threshold / (1.0 - threshold))
        selected_old = old >= threshold
        selected_new = new >= threshold
        old_nb = float(
            case_weight[selected_old].sum() / n
            - control_weight[selected_old].sum() / n * odds
        )
        new_nb = float(
            case_weight[selected_new].sum() / n
            - control_weight[selected_new].sum() / n * odds
        )
        delta_nb.append(new_nb - old_nb)
    return {
        "delta_harrell_c": augmented["harrell"] - base["harrell"],
        "delta_uno_c_60m": augmented["uno"] - base["uno"],
        "delta_auc_60m": augmented["auc"] - base["auc"],
        "brier_improvement_60m": base["brier"] - augmented["brier"],
        "ipcw_idi_60m": float(idi),
        "ipcw_continuous_nri_60m": float(nri),
        "mean_delta_net_benefit": float(np.mean(delta_nb)),
    }


def run_fold_explicit(
    module: ModuleType,
    fold: int,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    context = pd.read_pickle(CONTEXT_PICKLE)
    context["participant_id"] = context["participant_id"].astype(str)
    cells, universe = load_fold_cells(module, fold, context)
    rng = np.random.default_rng(seed + 10007 * fold)
    probability = np.full(len(universe), 1.0 / len(universe), dtype=float)
    rows = []
    started = time.time()
    for replicate in range(replicates):
        universe_counts = rng.multinomial(len(universe), probability)
        for organ in module.ORGANS:
            cell = cells[(fold, organ)]
            counts = universe_counts[cell.universe_positions]
            indices = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
            if len(indices) < 100 or int(cell.event[indices].sum()) < 2:
                raise RuntimeError(
                    f"invalid bootstrap risk set: b={replicate} fold={fold} organ={organ}"
                )
            base = module.explicit_model_metrics(
                cell,
                indices,
                cell.base_score,
                cell.base_risk,
            )
            for augmentation in module.AUGMENTATIONS:
                result = explicit_incremental_metrics(
                    module,
                    cell,
                    indices,
                    augmentation,
                    base,
                )
                rows.append(
                    {
                        "bootstrap": replicate,
                        "outer_fold": fold,
                        "organ_system": organ,
                        "augmentation": augmentation,
                        "n_resampled_at_risk": len(indices),
                        "events_resampled": int(cell.event[indices].sum()),
                        **result,
                    }
                )
        if (replicate + 1) % 10 == 0 or replicate == 0:
            print(
                f"fold {fold} bootstrap {replicate + 1}/{replicates} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def benchmark(module: ModuleType) -> None:
    context = pd.read_pickle(CONTEXT_PICKLE)
    context["participant_id"] = context["participant_id"].astype(str)
    cells, universe = load_fold_cells(module, 0, context)
    cell = cells[(0, "cardiovascular")]
    rng = np.random.default_rng(7110)
    counts = rng.multinomial(
        len(universe), np.full(len(universe), 1.0 / len(universe), dtype=float)
    )[cell.universe_positions]
    indices = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    started = time.perf_counter()
    weighted = module.incremental_metrics(cell, counts, "all_gaps")
    weighted_seconds = time.perf_counter() - started
    started = time.perf_counter()
    base = module.explicit_model_metrics(cell, indices, cell.base_score, cell.base_risk)
    explicit = explicit_incremental_metrics(module, cell, indices, "all_gaps", base)
    explicit_seconds = time.perf_counter() - started
    differences = {metric: explicit[metric] - weighted[metric] for metric in module.METRICS}
    print(
        json.dumps(
            {
                "weighted_seconds": weighted_seconds,
                "explicit_seconds": explicit_seconds,
                "speedup": weighted_seconds / explicit_seconds,
                "max_abs_difference": max(abs(value) for value in differences.values()),
                "differences": differences,
            },
            indent=2,
        ),
        flush=True,
    )


def prepare_fast_input(module: ModuleType, seed: int) -> None:
    if FAST_INPUT_ROOT.exists():
        raise FileExistsError(f"create-once output already exists: {FAST_INPUT_ROOT}")
    FAST_INPUT_ROOT.mkdir(parents=True, exist_ok=False)
    context = pd.read_pickle(CONTEXT_PICKLE)
    context["participant_id"] = context["participant_id"].astype(str)
    files = []
    parity = []
    for fold in module.FOLDS:
        cells, universe = load_fold_cells(module, fold, context)
        rng = np.random.default_rng(seed + 10007 * fold)
        parity_universe_counts = rng.multinomial(
            len(universe), np.full(len(universe), 1.0 / len(universe), dtype=float)
        )
        fold_dir = FAST_INPUT_ROOT / f"outer_fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        for organ in module.ORGANS:
            cell = cells[(fold, organ)]
            output = fold_dir / f"{organ}.npz"
            np.savez_compressed(
                output,
                universe_size=np.asarray([len(universe)], dtype=np.int64),
                universe_positions=cell.universe_positions.astype(np.int64),
                duration=cell.duration.astype(np.float64),
                event=cell.event.astype(np.int8),
                uno_event_weight=cell.uno_event_weight.astype(np.float64),
                case_weight=cell.case_weight.astype(np.float64),
                control_weight=cell.control_weight.astype(np.float64),
                base_score=cell.base_score.astype(np.float64),
                base_risk=cell.base_risk.astype(np.float64),
                target_gap_score=cell.augmented_score["target_gap"].astype(np.float64),
                target_gap_risk=cell.augmented_risk["target_gap"].astype(np.float64),
                all_gaps_score=cell.augmented_score["all_gaps"].astype(np.float64),
                all_gaps_risk=cell.augmented_risk["all_gaps"].astype(np.float64),
            )
            files.append(
                {
                    "outer_fold": fold,
                    "organ_system": organ,
                    "path": output.as_posix(),
                    "bytes": output.stat().st_size,
                    "sha256": module.sha256_file(output),
                    "rows": len(cell.duration),
                    "events": int(cell.event.sum()),
                    "universe_size": len(universe),
                }
            )
            if (fold, organ) in ((0, "cardiovascular"), (4, "immune")):
                counts = parity_universe_counts[cell.universe_positions]
                count_path = fold_dir / f"{organ}__parity_counts.npy"
                np.save(count_path, counts.astype(np.int64), allow_pickle=False)
                for augmentation in module.AUGMENTATIONS:
                    result = module.incremental_metrics(cell, counts, augmentation)
                    parity.append(
                        {
                            "outer_fold": fold,
                            "organ_system": organ,
                            "augmentation": augmentation,
                            "counts_path": count_path.as_posix(),
                            "counts_sha256": module.sha256_file(count_path),
                            "metrics": result,
                        }
                    )
    manifest = {
        "schema_version": "revision0710.final_model_clinical_utility.fast_bootstrap_input.v1",
        "status": "PASS",
        "seed_for_parity": seed,
        "metrics_contract": {
            "path": ESTABLISHED_BOOTSTRAP.as_posix(),
            "sha256": module.sha256_file(ESTABLISHED_BOOTSTRAP),
        },
        "files": files,
        "parity_reference": parity,
    }
    (FAST_INPUT_ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def combine(module: ModuleType, replicates: int, seed: int, output_dir: Path) -> None:
    fold_files = [
        WORK_ROOT / f"seed_{seed}" / f"fold_{fold}__replicates_{replicates}.csv.gz"
        for fold in module.FOLDS
    ]
    missing = [path for path in fold_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fold bootstrap files: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"create-once output already exists: {output_dir}")
    cell_bootstrap = pd.concat([pd.read_csv(path) for path in fold_files], ignore_index=True)
    organ_bootstrap = cell_bootstrap.groupby(
        ["bootstrap", "organ_system", "augmentation"], as_index=False
    )[list(module.METRICS)].mean()
    macro_bootstrap = organ_bootstrap.groupby(
        ["bootstrap", "augmentation"], as_index=False
    )[list(module.METRICS)].mean()
    point_cell = pd.read_csv(module.POINT_CELL)
    point_organ = pd.read_csv(module.POINT_ORGAN)
    point_macro = pd.read_csv(module.POINT_MACRO)
    context = pd.read_pickle(CONTEXT_PICKLE)
    context["participant_id"] = context["participant_id"].astype(str)
    audit_cells = {}
    audit_universes = {}
    for fold in (0, 4):
        cells, universe = load_fold_cells(module, fold, context)
        audit_cells.update(cells)
        audit_universes[fold] = universe
    parity = module.point_parity(
        audit_cells,
        point_cell,
        selected_cells=((0, "cardiovascular"), (4, "immune")),
    )
    bootstrap_parity = module.bootstrap_weight_parity(
        audit_cells,
        audit_universes,
        selected_cells=((0, "cardiovascular"), (4, "immune")),
        seed=seed,
    )
    point_organ = point_organ.loc[point_organ["tier"].eq(module.TIER)].copy()
    point_organ["augmentation"] = point_organ["augmented_model"].str.replace(
        f"{module.TIER}__", "", regex=False
    )
    point_macro = point_macro.loc[point_macro["tier"].eq(module.TIER)].copy()
    point_macro["augmentation"] = point_macro["augmented_model"].str.replace(
        f"{module.TIER}__", "", regex=False
    )
    organ_ci = module.summarize_ci(
        organ_bootstrap,
        ["organ_system", "augmentation"],
        point_organ,
        ["organ_system", "augmentation"],
    )
    macro_ci = module.summarize_ci(
        macro_bootstrap,
        ["augmentation"],
        point_macro,
        ["augmentation"],
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    parity.to_csv(output_dir / "point_metric_parity.csv", index=False, lineterminator="\n")
    bootstrap_parity.to_csv(
        output_dir / "bootstrap_weight_metric_parity.csv", index=False, lineterminator="\n"
    )
    cell_bootstrap.to_csv(
        output_dir / "cell_bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
        lineterminator="\n",
    )
    organ_bootstrap.to_csv(
        output_dir / "organ_bootstrap_replicates.csv", index=False, lineterminator="\n"
    )
    macro_bootstrap.to_csv(
        output_dir / "macro_bootstrap_replicates.csv", index=False, lineterminator="\n"
    )
    organ_ci.to_csv(output_dir / "organ_paired_bootstrap_ci.csv", index=False, lineterminator="\n")
    macro_ci.to_csv(output_dir / "macro_paired_bootstrap_ci.csv", index=False, lineterminator="\n")
    manifest = {
        "schema_version": "revision0710.final_model_clinical_utility.bootstrap.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "replicates": replicates,
        "seed": seed,
        "tier": module.TIER,
        "augmentations": list(module.AUGMENTATIONS),
        "metrics": list(module.METRICS),
        "bootstrap_unit": "participant within outer fold",
        "dependence": "same fold-level resampling weights shared across organs and paired models",
        "model_refit": False,
        "fit_censoring_distribution_refit": False,
        "death_handling": "cause-specific censoring",
        "point_parity_max_abs_difference": float(parity["difference"].abs().max()),
        "bootstrap_weight_parity_max_abs_difference": float(
            bootstrap_parity["difference"].abs().max()
        ),
        "established_bootstrap": {
            "path": ESTABLISHED_BOOTSTRAP.as_posix(),
            "sha256": module.sha256_file(ESTABLISHED_BOOTSTRAP),
        },
        "fold_inputs": [
            {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": module.sha256_file(path)}
            for path in fold_files
        ],
        "point_inputs": [
            {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": module.sha256_file(path)}
            for path in (module.POINT_CELL, module.POINT_ORGAN, module.POINT_MACRO)
        ],
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("benchmark", "prepare-fast-input", "run-fold", "combine")
    )
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7110)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    module = load_established_bootstrap()
    configure_module(module)
    if args.command == "benchmark":
        benchmark(module)
        return
    if args.command == "prepare-fast-input":
        prepare_fast_input(module, args.seed)
        return
    if args.command == "run-fold":
        if args.fold is None:
            raise ValueError("--fold is required for run-fold")
        output = WORK_ROOT / f"seed_{args.seed}" / f"fold_{args.fold}__replicates_{args.replicates}.csv.gz"
        if output.exists():
            raise FileExistsError(f"create-once output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = run_fold_explicit(module, args.fold, args.replicates, args.seed)
        frame.to_csv(output, index=False, compression="gzip", lineterminator="\n")
        return
    combine(module, args.replicates, args.seed, args.output_dir.resolve())


if __name__ == "__main__":
    main()
