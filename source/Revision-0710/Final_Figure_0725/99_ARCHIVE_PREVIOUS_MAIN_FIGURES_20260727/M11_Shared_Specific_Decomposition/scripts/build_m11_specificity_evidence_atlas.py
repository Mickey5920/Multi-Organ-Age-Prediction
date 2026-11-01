#!/usr/bin/env python3
"""Build an organ-specificity evidence atlas from frozen predictions.

Outputs
-------
1. Four-panel evidence atlas:
   a) 7x7 held-out incremental Harrell C-index matrix;
   b) first-organ-event Top-1/Top-2 ranking versus label permutations;
   c) exact 7! output-label mapping permutation distribution;
   d) matched minus strongest off-target held-out delta-C contrasts.
2. Seven-organ quartile dose-response atlas comparing the matched relative
   offset with the strongest off-target relative offset.

The output-label permutation is a post-hoc negative control and is not called
an internal neural-query masking experiment. No neural network is retrained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path
import platform
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import build_m11_shared_specific_candidate as shared
import build_m11_direct_specificity_candidate as direct


ROOT = shared.ROOT
SCRIPT = Path(__file__).resolve()
FIGURE_ROOT = SCRIPT.parent.parent
OUTPUT_ROOT = FIGURE_ROOT / "CANDIDATE_v3_SPECIFICITY_EVIDENCE_ATLAS"
SOURCE_DIR = OUTPUT_ROOT / "source_data"
OUTPUT_DIR = OUTPUT_ROOT / "outputs"
QA_DIR = OUTPUT_ROOT / "qa"

ORGANS = shared.ORGANS
ORGAN_LABELS = shared.ORGAN_LABELS
ORGAN_COLORS = shared.ORGAN_COLORS
FULL_RISK_COVARIATES = shared.FULL_RISK_COVARIATES
COX_PENALIZER = shared.COX_PENALIZER
SEED = 20260727
N_RESAMPLES = 1000

V2_ROOT = FIGURE_ROOT / "CANDIDATE_v2_DIRECT_EVIDENCE/source_data"
DIAGONAL_DELTA_INPUT = V2_ROOT / "target_relative_offset_heldout_delta_c_35cells.csv"
CROSS_HR_INPUT = V2_ROOT / "cross_organ_relative_offset_fullrisk_cox_49.csv"
SPECIFICITY_INPUT = V2_ROOT / "matched_vs_offtarget_specificity_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--penalizer", type=float, default=COX_PENALIZER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--reuse-statistics", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_directories() -> None:
    for path in (OUTPUT_ROOT, SOURCE_DIR, OUTPUT_DIR, QA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.075,
        label,
        transform=axis.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
    )


def compute_delta_c_matrix(
    *,
    penalizer: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_cells = SOURCE_DIR / "cross_organ_heldout_delta_c_245cells.csv"
    final_summary = SOURCE_DIR / "cross_organ_heldout_delta_c_49summary.csv"
    if final_cells.exists() and final_summary.exists():
        cells = pd.read_csv(final_cells)
        summary = pd.read_csv(final_summary)
        valid = (
            len(cells) == 245
            and len(summary) == 49
            and np.allclose(cells["penalizer"].to_numpy(float), penalizer)
            and set(cells["evaluation_partition"].astype(str))
            == {"same-outer-model outer-test participants"}
        )
        if valid:
            return cells, summary

    diagonal = pd.read_csv(DIAGONAL_DELTA_INPUT)
    checkpoint = SOURCE_DIR / "_checkpoint_cross_organ_delta_c.csv"
    if checkpoint.exists():
        rows: list[dict[str, object]] = pd.read_csv(checkpoint).to_dict(orient="records")
    else:
        rows = []
    completed = {
        (int(row["outer_fold"]), str(row["source_organ"]), str(row["target_outcome"]))
        for row in rows
    }
    context = direct.load_context_table()
    base_features = [*FULL_RISK_COVARIATES, "base_gap", "mean_offset"]
    for fold in range(5):
        calibration_path, test_path = direct.outer_prediction_paths(fold)
        calibration_participant, calibration_values = direct.load_prediction_components(
            calibration_path
        )
        test_participant, test_values = direct.load_prediction_components(test_path)
        calibration_analysis = direct.attach_analysis(calibration_participant, context)
        test_analysis = direct.attach_analysis(test_participant, context)
        for target in ORGANS:
            train = direct.outcome_frame(
                target,
                calibration_participant,
                calibration_values,
                calibration_analysis,
            )
            test = direct.outcome_frame(
                target,
                test_participant,
                test_values,
                test_analysis,
            )
            diagonal_cell = diagonal.loc[
                diagonal["target_organ"].eq(target)
                & diagonal["outer_fold"].eq(fold)
            ].iloc[0]
            c_base = float(diagonal_cell["c_shared_baseline"])
            for source in ORGANS:
                key = (fold, source, target)
                if key in completed:
                    print(
                        f"Delta-C matrix fold={fold} target={target} source={source} [checkpoint]",
                        flush=True,
                    )
                    continue
                if source == target:
                    c_augmented = float(
                        diagonal_cell["c_plus_target_relative_offset"]
                    )
                    fit_source = "reused matched target cell"
                else:
                    print(
                        f"Delta-C matrix fold={fold} target={target} source={source}",
                        flush=True,
                    )
                    relative = f"relative_offset_{source}"
                    score = direct.fit_and_score_cox(
                        train,
                        test,
                        [*base_features, relative],
                        penalizer=penalizer,
                    )
                    c_augmented = direct.harrell_c(test, score)
                    fit_source = "new off-target augmented Cox fit"
                rows.append(
                    {
                        "outer_fold": fold,
                        "source_organ": source,
                        "source_organ_label": ORGAN_LABELS[source],
                        "target_outcome": target,
                        "target_outcome_label": ORGAN_LABELS[target],
                        "is_matched_target": bool(source == target),
                        "n_calibration": int(len(train)),
                        "events_calibration": int(train["event"].sum()),
                        "n_test": int(len(test)),
                        "events_test": int(test["event"].sum()),
                        "c_shared_baseline": c_base,
                        "c_plus_source_relative_offset": c_augmented,
                        "incremental_delta_c": float(c_augmented - c_base),
                        "penalizer": penalizer,
                        "fit_source": fit_source,
                        "fit_partition": "same-outer-model calibration participants",
                        "evaluation_partition": "same-outer-model outer-test participants",
                    }
                )
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
    cells = pd.DataFrame(rows).sort_values(
        ["outer_fold", "target_outcome", "source_organ"]
    ).reset_index(drop=True)
    if len(cells) != 245:
        raise RuntimeError(f"Expected 245 delta-C cells, observed {len(cells)}")
    summaries: list[dict[str, object]] = []
    for target in ORGANS:
        for source in ORGANS:
            values = cells.loc[
                cells["target_outcome"].eq(target)
                & cells["source_organ"].eq(source),
                "incremental_delta_c",
            ].to_numpy(float)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            critical = float(stats.t.ppf(0.975, df=4))
            summaries.append(
                {
                    "source_organ": source,
                    "source_organ_label": ORGAN_LABELS[source],
                    "target_outcome": target,
                    "target_outcome_label": ORGAN_LABELS[target],
                    "is_matched_target": bool(source == target),
                    "outer_folds": 5,
                    "mean_incremental_delta_c": mean,
                    "min_fold_incremental_delta_c": float(values.min()),
                    "max_fold_incremental_delta_c": float(values.max()),
                    "paired_fold_t_ci95_low": mean - critical * se,
                    "paired_fold_t_ci95_high": mean + critical * se,
                    "folds_positive": int(np.sum(values > 0.0)),
                }
            )
    summary = pd.DataFrame(summaries)
    cells.to_csv(final_cells, index=False)
    summary.to_csv(final_summary, index=False)
    checkpoint.unlink(missing_ok=True)
    return cells, summary


def compute_mapping_permutations(
    delta_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    matrix = delta_summary.pivot(
        index="source_organ",
        columns="target_outcome",
        values="mean_incremental_delta_c",
    ).loc[list(ORGANS), list(ORGANS)].to_numpy(float)
    rows: list[dict[str, object]] = []
    identity = tuple(range(len(ORGANS)))
    for index, mapping in enumerate(permutations(range(len(ORGANS)))):
        values = np.asarray([matrix[mapping[target], target] for target in range(7)])
        rows.append(
            {
                "permutation_index": index,
                "mapping": "|".join(ORGANS[source] for source in mapping),
                "is_identity_mapping": bool(mapping == identity),
                "macro_mean_incremental_delta_c": float(values.mean()),
                "organs_positive": int(np.sum(values > 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    observed = float(
        frame.loc[frame["is_identity_mapping"], "macro_mean_incremental_delta_c"].iloc[0]
    )
    rank = int(
        frame["macro_mean_incremental_delta_c"].rank(
            method="min", ascending=False
        ).loc[frame["is_identity_mapping"]].iloc[0]
    )
    summary = {
        "permutations": int(len(frame)),
        "observed_identity_macro_delta_c": observed,
        "identity_rank_descending": rank,
        "exact_one_sided_probability_mapping_ge_identity": float(
            np.mean(frame["macro_mean_incremental_delta_c"] >= observed - 1e-15)
        ),
        "permutation_mean": float(frame["macro_mean_incremental_delta_c"].mean()),
        "permutation_ci95_low": float(
            frame["macro_mean_incremental_delta_c"].quantile(0.025)
        ),
        "permutation_ci95_high": float(
            frame["macro_mean_incremental_delta_c"].quantile(0.975)
        ),
        "masked_all_relative_offsets_delta_c": 0.0,
    }
    frame.to_csv(SOURCE_DIR / "exact_7factorial_offset_mapping_permutations.csv", index=False)
    (SOURCE_DIR / "offset_mapping_permutation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame, summary


def compute_first_event_ranking(
    participant: pd.DataFrame,
    values: dict[str, pd.DataFrame],
    *,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    at_risk = values["at_risk"].loc[participant.index, list(ORGANS)].to_numpy(int).astype(bool)
    event = values["event"].loc[participant.index, list(ORGANS)].to_numpy(int).astype(bool)
    duration = values["duration"].loc[participant.index, list(ORGANS)].to_numpy(float)
    eligible = at_risk.all(axis=1) & event.any(axis=1)
    selected_event = event[eligible]
    selected_duration = duration[eligible]
    event_time = np.where(selected_event, selected_duration, np.inf)
    first_time = event_time.min(axis=1)
    first_event = selected_event & np.isclose(
        selected_duration,
        first_time[:, None],
        rtol=0.0,
        atol=1e-6,
    )
    offsets = np.column_stack(
        [participant.loc[eligible, f"relative_offset_{organ}"].to_numpy(float) for organ in ORGANS]
    )
    ranking = np.argsort(-offsets, axis=1, kind="stable")
    rows_index = np.arange(len(ranking))
    top1 = first_event[rows_index, ranking[:, 0]]
    top2 = top1 | first_event[rows_index, ranking[:, 1]]
    tie_count = first_event.sum(axis=1)
    unique_first = tie_count == 1
    folds = participant.loc[eligible, "outer_fold"].to_numpy(int)
    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, object]] = []
    fold_indices = [np.flatnonzero(folds == fold) for fold in range(5)]
    unique_fold_indices = [np.flatnonzero((folds == fold) & unique_first) for fold in range(5)]
    for replicate in range(resamples):
        sampled = np.concatenate(
            [indices[rng.integers(0, len(indices), size=len(indices))] for indices in fold_indices]
        )
        sampled_unique = np.concatenate(
            [indices[rng.integers(0, len(indices), size=len(indices))] for indices in unique_fold_indices]
        )
        bootstrap_rows.append(
            {
                "replicate": replicate + 1,
                "top1_accuracy": float(top1[sampled].mean()),
                "top2_accuracy": float(top2[sampled].mean()),
                "unique_first_top1_accuracy": float(top1[sampled_unique].mean()),
                "unique_first_top2_accuracy": float(top2[sampled_unique].mean()),
                "seed": seed,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    permutation_rows: list[dict[str, object]] = []
    identity = np.arange(7)
    for replicate in range(resamples):
        mapping = rng.permutation(7)
        while np.array_equal(mapping, identity):
            mapping = rng.permutation(7)
        mapped_top1 = mapping[ranking[:, 0]]
        mapped_top2 = mapping[ranking[:, :2]]
        hit1 = first_event[rows_index, mapped_top1]
        hit2 = hit1 | first_event[rows_index, mapped_top2[:, 1]]
        permutation_rows.append(
            {
                "replicate": replicate + 1,
                "mapping": "|".join(ORGANS[index] for index in mapping),
                "top1_accuracy": float(hit1.mean()),
                "top2_accuracy": float(hit2.mean()),
                "unique_first_top1_accuracy": float(hit1[unique_first].mean()),
                "unique_first_top2_accuracy": float(hit2[unique_first].mean()),
                "seed": seed,
            }
        )
    permutation_frame = pd.DataFrame(permutation_rows)
    chance_top1 = float(np.mean(tie_count / 7.0))
    chance_top2 = float(
        np.mean(1.0 - ((7 - tie_count) * (6 - tie_count)) / (7.0 * 6.0))
    )
    summary = {
        "eligibility": "at risk for all seven target organs and at least one 60-month event",
        "participants_at_risk_all_seven": int(at_risk.all(axis=1).sum()),
        "participants_with_first_event": int(eligible.sum()),
        "participants_with_tied_first_event": int(np.sum(tie_count > 1)),
        "participants_with_unique_first_event": int(np.sum(unique_first)),
        "observed_top1_accuracy": float(top1.mean()),
        "observed_top2_accuracy": float(top2.mean()),
        "unique_first_observed_top1_accuracy": float(top1[unique_first].mean()),
        "unique_first_observed_top2_accuracy": float(top2[unique_first].mean()),
        "unique_first_bootstrap_top1_ci95": [
            float(bootstrap["unique_first_top1_accuracy"].quantile(0.025)),
            float(bootstrap["unique_first_top1_accuracy"].quantile(0.975)),
        ],
        "unique_first_bootstrap_top2_ci95": [
            float(bootstrap["unique_first_top2_accuracy"].quantile(0.025)),
            float(bootstrap["unique_first_top2_accuracy"].quantile(0.975)),
        ],
        "unique_first_permutation_top1_mean": float(
            permutation_frame["unique_first_top1_accuracy"].mean()
        ),
        "unique_first_permutation_top2_mean": float(
            permutation_frame["unique_first_top2_accuracy"].mean()
        ),
        "bootstrap_top1_ci95": [
            float(bootstrap["top1_accuracy"].quantile(0.025)),
            float(bootstrap["top1_accuracy"].quantile(0.975)),
        ],
        "bootstrap_top2_ci95": [
            float(bootstrap["top2_accuracy"].quantile(0.025)),
            float(bootstrap["top2_accuracy"].quantile(0.975)),
        ],
        "permutation_top1_mean": float(permutation_frame["top1_accuracy"].mean()),
        "permutation_top1_ci95": [
            float(permutation_frame["top1_accuracy"].quantile(0.025)),
            float(permutation_frame["top1_accuracy"].quantile(0.975)),
        ],
        "permutation_top2_mean": float(permutation_frame["top2_accuracy"].mean()),
        "permutation_top2_ci95": [
            float(permutation_frame["top2_accuracy"].quantile(0.025)),
            float(permutation_frame["top2_accuracy"].quantile(0.975)),
        ],
        "tie_aware_random_chance_top1": chance_top1,
        "tie_aware_random_chance_top2": chance_top2,
        "bootstrap_replicates": resamples,
        "label_permutation_replicates": resamples,
        "seed": seed,
    }
    bootstrap.to_csv(SOURCE_DIR / "first_event_ranking_bootstrap_1000.csv", index=False)
    permutation_frame.to_csv(
        SOURCE_DIR / "first_event_ranking_label_permutations_1000.csv", index=False
    )
    (SOURCE_DIR / "first_event_ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bootstrap, permutation_frame, summary


def compute_best_offtarget_contrasts(
    delta_cells: pd.DataFrame,
    delta_summary: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 101)
    rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for target in ORGANS:
        summary_target = delta_summary.loc[delta_summary["target_outcome"].eq(target)]
        matched = summary_target.loc[summary_target["source_organ"].eq(target)].iloc[0]
        best_off = (
            summary_target.loc[summary_target["source_organ"].ne(target)]
            .sort_values("mean_incremental_delta_c", ascending=False)
            .iloc[0]
        )
        source_off = str(best_off["source_organ"])
        matched_folds = (
            delta_cells.loc[
                delta_cells["target_outcome"].eq(target)
                & delta_cells["source_organ"].eq(target)
            ]
            .sort_values("outer_fold")["incremental_delta_c"]
            .to_numpy(float)
        )
        off_folds = (
            delta_cells.loc[
                delta_cells["target_outcome"].eq(target)
                & delta_cells["source_organ"].eq(source_off)
            ]
            .sort_values("outer_fold")["incremental_delta_c"]
            .to_numpy(float)
        )
        differences = matched_folds - off_folds
        boot = np.empty(resamples, dtype=float)
        for replicate in range(resamples):
            sampled = rng.integers(0, 5, size=5)
            boot[replicate] = float(differences[sampled].mean())
            replicate_rows.append(
                {
                    "target_outcome": target,
                    "replicate": replicate + 1,
                    "mean_matched_minus_best_offtarget_delta_c": boot[replicate],
                    "seed": seed + 101,
                }
            )
        rows.append(
            {
                "target_outcome": target,
                "target_outcome_label": ORGAN_LABELS[target],
                "best_offtarget_source": source_off,
                "best_offtarget_source_label": ORGAN_LABELS[source_off],
                "matched_mean_delta_c": float(matched["mean_incremental_delta_c"]),
                "best_offtarget_mean_delta_c": float(best_off["mean_incremental_delta_c"]),
                "matched_minus_best_offtarget_delta_c": float(differences.mean()),
                "outer_fold_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                "outer_fold_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                "folds_matched_greater": int(np.sum(differences > 0.0)),
                "selection_note": "best off-target selected by the same five-fold mean; exploratory",
            }
        )
    summary = pd.DataFrame(rows)
    replicates = pd.DataFrame(replicate_rows)
    summary.to_csv(SOURCE_DIR / "matched_vs_best_offtarget_delta_c_contrasts.csv", index=False)
    replicates.to_csv(
        SOURCE_DIR / "matched_vs_best_offtarget_outerfold_bootstrap_1000.csv", index=False
    )
    return summary, replicates


def fit_quartile_dose_response(
    participant: pd.DataFrame,
    values: dict[str, pd.DataFrame],
    analysis: pd.DataFrame,
    *,
    penalizer: float,
) -> pd.DataFrame:
    from lifelines import CoxPHFitter

    final_path = SOURCE_DIR / "relative_offset_quartile_fullrisk_cox_49x3.csv"
    if final_path.exists():
        cached = pd.read_csv(final_path)
        if len(cached) == 147 and np.allclose(cached["penalizer"], penalizer):
            return cached
    checkpoint = SOURCE_DIR / "_checkpoint_quartile_dose_response.csv"
    if checkpoint.exists():
        rows: list[dict[str, object]] = pd.read_csv(checkpoint).to_dict(orient="records")
    else:
        rows = []
    completed = {
        (str(row["source_organ"]), str(row["target_outcome"]), int(row["quartile"]))
        for row in rows
    }
    base_features = [*FULL_RISK_COVARIATES, "base_gap", "mean_offset"]
    for target in ORGANS:
        target_frame = direct.outcome_frame(target, participant, values, analysis)
        for source in ORGANS:
            if all((source, target, quartile) in completed for quartile in (2, 3, 4)):
                print(
                    f"Quartile Cox target={target} source={source} [checkpoint]",
                    flush=True,
                )
                continue
            print(f"Quartile Cox target={target} source={source}", flush=True)
            relative = f"relative_offset_{source}"
            quartile_code, boundaries = pd.qcut(
                target_frame[relative],
                q=4,
                labels=False,
                retbins=True,
                duplicates="raise",
            )
            augmented = target_frame.copy()
            dummy_names: list[str] = []
            for quartile in (2, 3, 4):
                name = f"quartile_{quartile}"
                augmented[name] = (quartile_code.to_numpy(int) == quartile - 1).astype(float)
                dummy_names.append(name)
            standardized, _, _, scales = direct.numeric_imputed(
                augmented,
                augmented,
                [*base_features, *dummy_names],
            )
            retained_base = [column for column in base_features if column in standardized]
            model_frame = standardized[[*retained_base, *dummy_names]].copy()
            model_frame["duration"] = augmented["duration"].to_numpy(float)
            model_frame["event"] = augmented["event"].to_numpy(int)
            model = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    model_frame,
                    duration_col="duration",
                    event_col="event",
                    show_progress=False,
                    batch_mode=True,
                    fit_options={"max_steps": 100},
                )
            counts = np.bincount(quartile_code.to_numpy(int), minlength=4)
            for quartile, name in zip((2, 3, 4), dummy_names):
                item = model.summary.loc[name]
                raw_coef = float(item["coef"]) / float(scales[name])
                raw_se = float(item["se(coef)"]) / float(scales[name])
                rows.append(
                    {
                        "source_organ": source,
                        "source_organ_label": ORGAN_LABELS[source],
                        "target_outcome": target,
                        "target_outcome_label": ORGAN_LABELS[target],
                        "is_matched_target": bool(source == target),
                        "quartile": quartile,
                        "reference_quartile": 1,
                        "n_at_risk": int(len(model_frame)),
                        "events": int(model_frame["event"].sum()),
                        "quartile_n": int(counts[quartile - 1]),
                        "q1_upper": float(boundaries[1]),
                        "q2_upper": float(boundaries[2]),
                        "q3_upper": float(boundaries[3]),
                        "log_hr_vs_q1": raw_coef,
                        "hr_vs_q1": float(np.exp(raw_coef)),
                        "hr_ci95_low": float(np.exp(raw_coef - 1.96 * raw_se)),
                        "hr_ci95_high": float(np.exp(raw_coef + 1.96 * raw_se)),
                        "se_log_hr": raw_se,
                        "z": float(raw_coef / raw_se),
                        "p_value": float(2.0 * stats.norm.sf(abs(raw_coef / raw_se))),
                        "penalizer": penalizer,
                    }
                )
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
    result = pd.DataFrame(rows)
    result["q_value_bh_147"] = shared.bh_adjust(result["p_value"].to_numpy(float))
    result.to_csv(final_path, index=False)
    checkpoint.unlink(missing_ok=True)
    return result


def draw_delta_matrix(axis: plt.Axes, delta_summary: pd.DataFrame) -> None:
    panel_label(axis, "a")
    axis.set_title(
        "Held-out incremental discrimination matrix",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    matrix = delta_summary.pivot(
        index="source_organ",
        columns="target_outcome",
        values="mean_incremental_delta_c",
    ).loc[list(ORGANS), list(ORGANS)]
    max_abs = max(0.001, float(np.abs(matrix.to_numpy(float)).max()))
    annotation = matrix.map(lambda value: f"{value:+.4f}").to_numpy(object)
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        square=True,
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": 6.6},
        linewidths=0.8,
        linecolor="white",
        cbar=False,
    )
    labels = [ORGAN_LABELS[organ] for organ in ORGANS]
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(labels, rotation=0, fontsize=8)
    axis.set_xlabel("Future organ event", fontsize=9.5)
    axis.set_ylabel("Added relative organ offset", fontsize=9.5)
    for index in range(7):
        axis.add_patch(
            Rectangle((index, index), 1, 1, fill=False, edgecolor="#202020", linewidth=2.0)
        )
    axis.text(
        0,
        -0.21,
        "Mean across five calibration-to-outer-test folds; black boxes mark matched organs.",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#555555",
        va="top",
    )


def draw_first_event(axis: plt.Axes, summary: dict[str, object]) -> None:
    panel_label(axis, "b")
    axis.set_title(
        "First-organ-event ranking",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    x = np.asarray([0, 1], dtype=float)
    observed = np.asarray(
        [summary["observed_top1_accuracy"], summary["observed_top2_accuracy"]],
        dtype=float,
    )
    observed_low = np.asarray(
        [summary["bootstrap_top1_ci95"][0], summary["bootstrap_top2_ci95"][0]],
        dtype=float,
    )
    observed_high = np.asarray(
        [summary["bootstrap_top1_ci95"][1], summary["bootstrap_top2_ci95"][1]],
        dtype=float,
    )
    permuted = np.asarray(
        [summary["permutation_top1_mean"], summary["permutation_top2_mean"]],
        dtype=float,
    )
    permuted_low = np.asarray(
        [summary["permutation_top1_ci95"][0], summary["permutation_top2_ci95"][0]],
        dtype=float,
    )
    permuted_high = np.asarray(
        [summary["permutation_top1_ci95"][1], summary["permutation_top2_ci95"][1]],
        dtype=float,
    )
    axis.errorbar(
        x - 0.08,
        observed,
        yerr=[observed - observed_low, observed_high - observed],
        fmt="o",
        color="#E76F51",
        markersize=8,
        capsize=4,
        linewidth=1.7,
        label="Observed mapping",
    )
    axis.errorbar(
        x + 0.08,
        permuted,
        yerr=[permuted - permuted_low, permuted_high - permuted],
        fmt="s",
        color="#7F8C8D",
        markersize=7,
        capsize=4,
        linewidth=1.5,
        label="1,000 label permutations",
    )
    chance = [summary["tie_aware_random_chance_top1"], summary["tie_aware_random_chance_top2"]]
    axis.scatter(x, chance, marker="_", s=160, color="#202020", linewidth=1.5, label="Tie-aware chance")
    for index, value in enumerate(observed):
        axis.text(index - 0.08, value + 0.012, f"{100*value:.1f}%", ha="center", fontsize=8)
    axis.set_xticks(x, ["Top-1", "Top-2"], fontsize=9)
    axis.set_ylabel("Accuracy", fontsize=9.5)
    axis.set_ylim(0.0, min(1.0, float(observed.max()) * 1.22))
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", frameon=False, fontsize=7.5)
    axis.text(
        0,
        -0.16,
        f"N={int(summary['participants_with_first_event']):,} with ≥1 first event; tie-aware scoring.",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#555555",
        va="top",
    )


def draw_mapping_permutation(
    axis: plt.Axes,
    mapping: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    panel_label(axis, "c")
    axis.set_title(
        "Organ-label mapping negative control",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    values = mapping.loc[
        ~mapping["is_identity_mapping"], "macro_mean_incremental_delta_c"
    ].to_numpy(float)
    axis.hist(values, bins=38, color="#B9C3CC", edgecolor="white", linewidth=0.5)
    observed = float(summary["observed_identity_macro_delta_c"])
    axis.axvline(observed, color="#E76F51", linewidth=2.2, label="Observed identity mapping")
    axis.axvline(0.0, color="#202020", linestyle="--", linewidth=1.2, label="All offsets masked")
    axis.set_xlabel(r"Macro held-out incremental $\Delta$C", fontsize=9.5)
    axis.set_ylabel("Number of mappings", fontsize=9.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7.5, loc="upper left")
    axis.text(
        0.98,
        0.96,
        f"Identity rank: {summary['identity_rank_descending']}/5,040\nTail fraction={summary['exact_one_sided_probability_mapping_ge_identity']:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    axis.text(
        0,
        -0.16,
        "Post-hoc output-label permutation; this is not internal neural-query masking.",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#555555",
        va="top",
    )


def draw_best_offtarget(axis: plt.Axes, contrasts: pd.DataFrame) -> None:
    panel_label(axis, "d")
    axis.set_title(
        "Matched versus strongest off-target offset",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    display = contrasts.set_index("target_outcome").loc[list(reversed(ORGANS))].reset_index()
    y = np.arange(len(display))
    for index, row in display.iterrows():
        organ = str(row["target_outcome"])
        point = float(row["matched_minus_best_offtarget_delta_c"])
        low = float(row["outer_fold_bootstrap_ci95_low"])
        high = float(row["outer_fold_bootstrap_ci95_high"])
        axis.errorbar(
            point,
            index,
            xerr=np.asarray([[point - low], [high - point]]),
            fmt="o",
            color=ORGAN_COLORS[organ],
            markersize=6.5,
            capsize=3,
            linewidth=1.5,
        )
        axis.text(
            high + 0.0003,
            index,
            f"{point:+.4f}",
            va="center",
            fontsize=7.5,
        )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.1)
    axis.set_yticks(y, display["target_outcome_label"], fontsize=8.5)
    axis.set_xlabel(r"Matched minus best off-target $\Delta$C", fontsize=9.5)
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    lower = min(-0.001, float(display["outer_fold_bootstrap_ci95_low"].min()) - 0.001)
    upper = float(display["outer_fold_bootstrap_ci95_high"].max()) + 0.004
    axis.set_xlim(lower, upper)
    axis.text(
        0,
        -0.16,
        "Exploratory: strongest off-target selected by the same five-fold mean; 1,000 outer-fold resamples.",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#555555",
        va="top",
    )


def render_atlas(
    delta_summary: pd.DataFrame,
    first_event_summary: dict[str, object],
    mapping: pd.DataFrame,
    mapping_summary: dict[str, object],
    contrasts: pd.DataFrame,
) -> list[Path]:
    shared.setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 10.0))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.30, hspace=0.40)
    draw_delta_matrix(axes[0, 0], delta_summary)
    draw_first_event(axes[0, 1], first_event_summary)
    draw_mapping_permutation(axes[1, 0], mapping, mapping_summary)
    draw_best_offtarget(axes[1, 1], contrasts)
    outputs: list[Path] = []
    for extension in ("svg", "pdf", "png"):
        path = OUTPUT_DIR / f"M11_specificity_evidence_atlas.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def render_dose_response(
    quartiles: pd.DataFrame,
    specificity: pd.DataFrame,
) -> list[Path]:
    shared.setup_style()
    figure, axes = plt.subplots(2, 4, figsize=(13.6, 6.8), sharex=True)
    axes_flat = axes.ravel()
    strongest = specificity.set_index("target_outcome")["strongest_off_target_source"].to_dict()
    for axis, target in zip(axes_flat[:7], ORGANS):
        off_source = str(strongest[target])
        for source, color, linestyle, label in (
            (target, ORGAN_COLORS[target], "-", "Matched offset"),
            (off_source, "#7F8C8D", "--", f"Best off-target ({ORGAN_LABELS[off_source]})"),
        ):
            part = quartiles.loc[
                quartiles["target_outcome"].eq(target)
                & quartiles["source_organ"].eq(source)
            ].sort_values("quartile")
            x = np.asarray([1, 2, 3, 4])
            hr = np.concatenate([[1.0], part["hr_vs_q1"].to_numpy(float)])
            low = np.concatenate([[1.0], part["hr_ci95_low"].to_numpy(float)])
            high = np.concatenate([[1.0], part["hr_ci95_high"].to_numpy(float)])
            axis.plot(x, hr, marker="o", color=color, linestyle=linestyle, linewidth=1.7, markersize=4, label=label)
            axis.fill_between(x, low, high, color=color, alpha=0.10)
        axis.axhline(1.0, color="#777777", linestyle=":", linewidth=0.9)
        axis.set_title(ORGAN_LABELS[target], fontsize=11, fontweight="bold")
        axis.text(
            0.98,
            0.95,
            f"Off: {ORGAN_LABELS[off_source]}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#6F7B7C",
        )
        axis.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        if target in ("cardiovascular", "musculoskeletal"):
            axis.set_ylabel("Adjusted HR vs Q1", fontsize=9)
        if target == "cardiovascular":
            axis.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes_flat[7].axis("off")
    figure.suptitle(
        "Matched and strongest off-target relative-offset quartile associations",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        "Full clinical and record-density adjustment plus shared base gap and mean offset; strongest off-target selected by continuous adjusted HR; shaded bands are Wald 95% CIs.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.96))
    outputs: list[Path] = []
    for extension in ("svg", "pdf", "png"):
        (OUTPUT_DIR / f"M11_relative_offset_quartile_dose_response.{extension}").unlink(
            missing_ok=True
        )
    for extension in ("svg", "pdf", "png"):
        path = OUTPUT_DIR / f"M11_relative_offset_quartile_association_patterns.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def main() -> int:
    args = parse_args()
    if args.resamples != 1000:
        raise RuntimeError("This candidate contract requires exactly 1,000 resamples")
    if not np.isfinite(args.penalizer) or args.penalizer < 0.0:
        raise RuntimeError("penalizer must be finite and non-negative")
    prepare_directories()
    participant, values, reconstruction_audit = shared.load_components()
    analysis = direct.load_context_and_analysis(participant)

    delta_cells, delta_summary = compute_delta_c_matrix(penalizer=float(args.penalizer))
    mapping, mapping_summary = compute_mapping_permutations(delta_summary)
    first_bootstrap, first_permutations, first_summary = compute_first_event_ranking(
        participant,
        values,
        resamples=int(args.resamples),
        seed=int(args.seed),
    )
    contrasts, contrast_replicates = compute_best_offtarget_contrasts(
        delta_cells,
        delta_summary,
        resamples=int(args.resamples),
        seed=int(args.seed),
    )
    quartiles = fit_quartile_dose_response(
        participant,
        values,
        analysis,
        penalizer=float(args.penalizer),
    )
    specificity = pd.read_csv(SPECIFICITY_INPUT)
    outputs = [
        *render_atlas(delta_summary, first_summary, mapping, mapping_summary, contrasts),
        *render_dose_response(quartiles, specificity),
    ]

    best_skill = {
        "best_skill": "epi predictive validation + frozen-output specificity evidence atlas",
        "train_signal": "need direct organ-specific evidence beyond shared organ-gap correlation",
        "selection_split": "7x7 calibration-to-outer-test delta-C matrix and first-event ranking",
        "heldout_gate": (
            "same frozen outer model calibration-to-test evaluation; all seven organs retained; "
            "query permutation not mislabelled as neural masking"
        ),
        "accepted_patterns": [
            "same-model calibration-to-outer-test Cox scoring",
            "complete 7x7 reporting",
            "1,000 fixed-seed participant bootstrap and label permutation",
            "mixed organ results retained",
        ],
        "rejected_patterns": [
            "training-sample delta-C",
            "selecting only diagonal-positive organs",
            "calling output-label permutation internal query masking",
        ],
        "patch_scope": "new candidate artifacts only; current figure pointer unchanged",
        "reject_if": (
            "missing 7x7 cells, calibration/test overlap, fewer than five test folds, "
            "non-editable SVG, or input-hash failure"
        ),
    }
    contract = {
        "schema_version": "final_figure_0725.m11.specificity_atlas.contract.v1",
        "generated_at_utc": utc_now(),
        "participants": int(len(participant)),
        "neural_network_refit": False,
        "true_internal_query_masking_performed": False,
        "negative_control": "post-hoc relative-offset label mapping permutation",
        "delta_c_matrix_cells": int(len(delta_cells)),
        "delta_c_evaluation": "same-model calibration-to-outer-test",
        "first_event_bootstrap_replicates": int(args.resamples),
        "first_event_label_permutations": int(args.resamples),
        "exact_mapping_permutations": int(len(mapping)),
        "best_offtarget_contrast": "exploratory post-selection five-fold comparison",
        "quartile_cox_cells": int(len(quartiles)),
        "cox_penalizer": float(args.penalizer),
        "seed": int(args.seed),
        "component_reconstruction": reconstruction_audit,
        "skillopt_card": best_skill,
    }
    (SOURCE_DIR / "analysis_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    atlas_svg = OUTPUT_DIR / "M11_specificity_evidence_atlas.svg"
    dose_svg = OUTPUT_DIR / "M11_relative_offset_quartile_association_patterns.svg"
    atlas_text = atlas_svg.read_text(encoding="utf-8")
    dose_text = dose_svg.read_text(encoding="utf-8")
    matrix = delta_summary.pivot(
        index="source_organ",
        columns="target_outcome",
        values="mean_incremental_delta_c",
    ).loc[list(ORGANS), list(ORGANS)]
    diagonal_largest = sum(
        int(matrix.loc[target, target] >= matrix[target].max() - 1e-15)
        for target in ORGANS
    )
    q4 = quartiles.loc[quartiles["quartile"].eq(4)]
    matched_q4 = q4.loc[q4["is_matched_target"]]
    monotonic_count = 0
    matched_q4_greater_than_selected_offtarget = 0
    selected_offtarget = specificity.set_index("target_outcome")[
        "strongest_off_target_source"
    ].to_dict()
    for target in ORGANS:
        matched_sequence = np.concatenate(
            [
                [1.0],
                quartiles.loc[
                    quartiles["target_outcome"].eq(target)
                    & quartiles["source_organ"].eq(target)
                ]
                .sort_values("quartile")["hr_vs_q1"]
                .to_numpy(float),
            ]
        )
        monotonic_count += int(np.all(np.diff(matched_sequence) >= 0.0))
        off_source = str(selected_offtarget[target])
        matched_value = float(
            q4.loc[
                q4["target_outcome"].eq(target)
                & q4["source_organ"].eq(target),
                "hr_vs_q1",
            ].iloc[0]
        )
        off_value = float(
            q4.loc[
                q4["target_outcome"].eq(target)
                & q4["source_organ"].eq(off_source),
                "hr_vs_q1",
            ].iloc[0]
        )
        matched_q4_greater_than_selected_offtarget += int(matched_value > off_value)

    overlap_counts: dict[str, int] = {}
    for fold in range(5):
        calibration_path, test_path = direct.outer_prediction_paths(fold)
        calibration_ids = pd.Index(
            pd.read_parquet(calibration_path, columns=["participant_id"])[
                "participant_id"
            ]
            .astype(str)
            .unique()
        )
        test_ids = pd.Index(
            pd.read_parquet(test_path, columns=["participant_id"])["participant_id"]
            .astype(str)
            .unique()
        )
        overlap_counts[str(fold)] = int(len(calibration_ids.intersection(test_ids)))
    if max(overlap_counts.values()) != 0:
        raise RuntimeError(f"Calibration/test overlap detected: {overlap_counts}")

    reference_diagonal = pd.read_csv(DIAGONAL_DELTA_INPUT)[
        ["outer_fold", "target_organ", "incremental_delta_c"]
    ].rename(columns={"target_organ": "target_outcome", "incremental_delta_c": "reference_delta_c"})
    observed_diagonal = delta_cells.loc[delta_cells["is_matched_target"], [
        "outer_fold", "target_outcome", "incremental_delta_c"
    ]]
    diagonal_parity = observed_diagonal.merge(
        reference_diagonal,
        on=["outer_fold", "target_outcome"],
        how="inner",
        validate="one_to_one",
    )
    if len(diagonal_parity) != 35:
        raise RuntimeError("Diagonal delta-C parity table is incomplete")
    diagonal_max_abs_difference = float(
        np.max(
            np.abs(
                diagonal_parity["incremental_delta_c"].to_numpy(float)
                - diagonal_parity["reference_delta_c"].to_numpy(float)
            )
        )
    )
    if diagonal_max_abs_difference > 1e-12:
        raise RuntimeError(
            f"Diagonal delta-C parity failed: {diagonal_max_abs_difference}"
        )
    qa = {
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "candidate_only_pointer_unchanged": True,
        "participants": int(len(participant)),
        "delta_c_matrix_cells": int(len(delta_cells)),
        "delta_c_summary_cells": int(len(delta_summary)),
        "delta_c_outer_folds_min": int(
            delta_cells.groupby(["source_organ", "target_outcome"])["outer_fold"].nunique().min()
        ),
        "calibration_test_overlap_by_fold": overlap_counts,
        "diagonal_delta_c_max_abs_difference_vs_v2": diagonal_max_abs_difference,
        "diagonal_largest_delta_c_count": int(diagonal_largest),
        "identity_mapping_rank": int(mapping_summary["identity_rank_descending"]),
        "first_event_participants": int(first_summary["participants_with_first_event"]),
        "first_event_top1_accuracy": float(first_summary["observed_top1_accuracy"]),
        "first_event_top2_accuracy": float(first_summary["observed_top2_accuracy"]),
        "unique_first_event_top1_accuracy": float(
            first_summary["unique_first_observed_top1_accuracy"]
        ),
        "unique_first_event_top2_accuracy": float(
            first_summary["unique_first_observed_top2_accuracy"]
        ),
        "first_event_top1_permutation_mean": float(first_summary["permutation_top1_mean"]),
        "first_event_top2_permutation_mean": float(first_summary["permutation_top2_mean"]),
        "matched_minus_best_offtarget_positive_count": int(
            contrasts["matched_minus_best_offtarget_delta_c"].gt(0.0).sum()
        ),
        "matched_q4_hr_range": [
            float(matched_q4["hr_vs_q1"].min()),
            float(matched_q4["hr_vs_q1"].max()),
        ],
        "matched_q4_hr_gt_1_count": int(matched_q4["hr_vs_q1"].gt(1.0).sum()),
        "matched_q4_hr_gt_selected_offtarget_count": int(
            matched_q4_greater_than_selected_offtarget
        ),
        "matched_monotonic_q1_to_q4_count": int(monotonic_count),
        "atlas_svg_editable_text": "<text" in atlas_text,
        "atlas_svg_embedded_raster": "<image" in atlas_text or "data:image" in atlas_text,
        "dose_svg_editable_text": "<text" in dose_text,
        "dose_svg_embedded_raster": "<image" in dose_text or "data:image" in dose_text,
    }
    if (
        not qa["atlas_svg_editable_text"]
        or qa["atlas_svg_embedded_raster"]
        or not qa["dose_svg_editable_text"]
        or qa["dose_svg_embedded_raster"]
    ):
        raise RuntimeError("SVG editability QA failed")
    (QA_DIR / "QA_REPORT.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# M11器官特异性证据图集候选

本目录不替换当前Fig.11指针，不重新训练神经网络。

- `M11_specificity_evidence_atlas`：外层测试ΔC矩阵、首次器官事件排序、器官标签映射置换负对照、匹配与最佳非匹配ΔC对比；
- `M11_relative_offset_quartile_association_patterns`：七器官匹配offset与最强非匹配offset的四分位HR关联模式。结果并非普遍单调，因此不称为单调剂量–反应。

真实query遮蔽需要重新运行冻结模型推理，本目录没有进行该干预；这里的负对照是输出relative-offset标签映射置换。最佳非匹配offset由同一五折均值选择，其对比属于探索性结果。

QA摘要：ΔC对角线为各列最大者{diagonal_largest}/7；真实映射在5,040种映射中排名{mapping_summary['identity_rank_descending']}；首次事件Top-1={first_summary['observed_top1_accuracy']:.4f}、Top-2={first_summary['observed_top2_accuracy']:.4f}。
"""
    (OUTPUT_ROOT / "README_CN.md").write_text(readme, encoding="utf-8")

    inputs = [
        shared.OOF,
        shared.RUN_CONFIG,
        shared.FINAL_SELECTION,
        shared.CONTEXT,
        DIAGONAL_DELTA_INPUT,
        CROSS_HR_INPUT,
        SPECIFICITY_INPUT,
        *[
            path
            for fold in range(5)
            for path in direct.outer_prediction_paths(fold)
        ],
    ]
    artifacts = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "schema_version": "final_figure_0725.m11.specificity_atlas.candidate.v1",
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "candidate_only": True,
        "pointer_updated": False,
        "seed": int(args.seed),
        "inputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ],
        "code": {
            "path": SCRIPT.relative_to(ROOT).as_posix(),
            "bytes": SCRIPT.stat().st_size,
            "sha256": sha256(SCRIPT),
            "dependencies": [
                {
                    "path": dependency.relative_to(ROOT).as_posix(),
                    "bytes": dependency.stat().st_size,
                    "sha256": sha256(dependency),
                }
                for dependency in (shared.SCRIPT, direct.SCRIPT)
            ],
        },
        "artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }
    (OUTPUT_ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
