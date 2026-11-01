#!/usr/bin/env python3
"""Build a direct-evidence candidate figure for organ specificity.

The candidate contains three quantitative panels:
  a. model-derived shared versus within-participant organ-specific variance;
  b. a 7 source-offset x 7 future-outcome full-risk-adjusted Cox HR matrix;
  c. target-organ HRs and held-out incremental Harrell C-index values.

No neural-network model is trained or refit. For each outer fold, Cox
discrimination models are fit on the same frozen model's calibration
participants and evaluated on that model's held-out outer-test participants.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
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


ROOT = shared.ROOT
SCRIPT = Path(__file__).resolve()
FIGURE_ROOT = SCRIPT.parent.parent
OUTPUT_ROOT = FIGURE_ROOT / "CANDIDATE_v2_DIRECT_EVIDENCE"
SOURCE_DIR = OUTPUT_ROOT / "source_data"
OUTPUT_DIR = OUTPUT_ROOT / "outputs"
QA_DIR = OUTPUT_ROOT / "qa"

ORGANS = shared.ORGANS
ORGAN_LABELS = shared.ORGAN_LABELS
ORGAN_COLORS = shared.ORGAN_COLORS
GROUP_ORDER = shared.GROUP_ORDER
FULL_RISK_COVARIATES = shared.FULL_RISK_COVARIATES
COX_PENALIZER = shared.COX_PENALIZER
SHARED_COLOR = shared.SHARED_COLOR
SPECIFIC_COLOR = shared.SPECIFIC_COLOR

DECOMPOSITION_INPUT = (
    FIGURE_ROOT
    / "CANDIDATE_v1/source_data/shared_specific_variance_full_and_16groups.csv"
)
DECOMPOSITION_BOOTSTRAP_INPUT = (
    FIGURE_ROOT
    / "CANDIDATE_v1/source_data/shared_specific_variance_bootstrap_1000.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-statistics",
        action="store_true",
        help="Reuse complete source CSVs and redraw only.",
    )
    parser.add_argument("--penalizer", type=float, default=COX_PENALIZER)
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


def numeric_imputed(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    train_out = pd.DataFrame(index=train.index)
    score_out = pd.DataFrame(index=score.index)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in columns:
        train_values = pd.to_numeric(train[column], errors="coerce")
        score_values = pd.to_numeric(score[column], errors="coerce")
        median = float(train_values.median())
        train_values = train_values.fillna(median)
        score_values = score_values.fillna(median)
        mean = float(train_values.mean())
        scale = float(train_values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        train_out[column] = (train_values - mean) / scale
        score_out[column] = (score_values - mean) / scale
        means[column] = mean
        scales[column] = scale
    return train_out, score_out, means, scales


def load_context_table() -> pd.DataFrame:
    context = pd.read_parquet(
        shared.CONTEXT,
        columns=["participant_id", *FULL_RISK_COVARIATES],
    )
    context["participant_id"] = context["participant_id"].astype(str)
    return context.set_index("participant_id")


def attach_analysis(participant: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    analysis = context.reindex(participant.index).copy()
    analysis["base_gap"] = participant["base_gap"]
    analysis["mean_offset"] = participant["mean_offset"]
    if "outer_fold" in participant:
        analysis["outer_fold"] = participant["outer_fold"].astype(int)
    for organ in ORGANS:
        analysis[f"relative_offset_{organ}"] = participant[f"relative_offset_{organ}"]
    return analysis


def load_context_and_analysis(participant: pd.DataFrame) -> pd.DataFrame:
    return attach_analysis(participant, load_context_table())


def outer_prediction_paths(fold: int) -> tuple[Path, Path]:
    if fold == 0:
        root = (
            ROOT
            / "Revision-0710/03_ehr_age/06_codebook_v2/01_runs/"
            "shared_private_independent_view_huber1_fold0"
        )
        return (
            root / "fold1_calibration_predictions.parquet",
            root / "fold0_predictions.parquet",
        )
    root = (
        ROOT
        / "Revision-0710/03_ehr_age/06_codebook_v2/"
        f"02_fivefold_frozen_huber1/01_outer_folds/outer_fold_{fold}"
    )
    return (
        root / "calibration_predictions.parquet",
        root / "outer_predictions.parquet",
    )


def load_prediction_components(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    config = json.loads(shared.RUN_CONFIG.read_text(encoding="utf-8"))
    cap = float(config["model"]["risk_offset_cap_years"])
    temperature = float(config["model"]["risk_temperature"])
    columns = [
        "participant_id",
        "organ_system",
        "raw_age_gap",
        "model_risk_score",
        "duration",
        "event",
        "at_risk",
    ]
    long = pd.read_parquet(path, columns=columns)
    long["participant_id"] = long["participant_id"].astype(str)
    if long.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError(f"Duplicate participant-organ rows: {path}")
    if set(long["organ_system"].astype(str)) != set(ORGANS):
        raise RuntimeError(f"Organ contract changed: {path}")
    long["organ_offset"] = cap * np.tanh(
        pd.to_numeric(long["model_risk_score"], errors="raise").to_numpy(float)
        / temperature
    )
    long["base_gap_reconstructed"] = (
        pd.to_numeric(long["raw_age_gap"], errors="raise").to_numpy(float)
        - long["organ_offset"].to_numpy(float)
    )
    values: dict[str, pd.DataFrame] = {}
    for column in (
        "organ_offset",
        "base_gap_reconstructed",
        "duration",
        "event",
        "at_risk",
    ):
        values[column] = long.pivot(
            index="participant_id",
            columns="organ_system",
            values=column,
        ).loc[:, list(ORGANS)]
    participant = pd.DataFrame(index=values["organ_offset"].index)
    base = values["base_gap_reconstructed"].to_numpy(float)
    offset = values["organ_offset"].to_numpy(float)
    participant["base_gap"] = base.mean(axis=1)
    participant["mean_offset"] = offset.mean(axis=1)
    for index, organ in enumerate(ORGANS):
        participant[f"relative_offset_{organ}"] = (
            offset[:, index] - (offset.sum(axis=1) - offset[:, index]) / 6.0
        )
    return participant, values


def outcome_frame(
    target: str,
    participant: pd.DataFrame,
    values: dict[str, pd.DataFrame],
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    at_risk = values["at_risk"].loc[participant.index, target].to_numpy(int).astype(bool)
    frame = analysis.loc[at_risk].copy()
    frame["participant_id"] = frame.index.astype(str)
    frame["duration"] = values["duration"].loc[participant.index, target].to_numpy(float)[at_risk]
    frame["event"] = values["event"].loc[participant.index, target].to_numpy(int)[at_risk]
    if frame["participant_id"].duplicated().any():
        raise RuntimeError(f"Duplicate participant IDs for outcome {target}")
    if frame["event"].sum() < 1:
        raise RuntimeError(f"No events for outcome {target}")
    return frame


def fit_cross_organ_hr_matrix(
    participant: pd.DataFrame,
    values: dict[str, pd.DataFrame],
    analysis: pd.DataFrame,
    *,
    penalizer: float,
) -> pd.DataFrame:
    from lifelines import CoxPHFitter

    checkpoint = SOURCE_DIR / "_checkpoint_cross_organ_hr.csv"
    if checkpoint.exists():
        checkpoint_frame = pd.read_csv(checkpoint)
        rows: list[dict[str, object]] = checkpoint_frame.to_dict(orient="records")
    else:
        rows = []
    completed = {
        (str(row["source_organ"]), str(row["target_outcome"]))
        for row in rows
    }
    base_features = [*FULL_RISK_COVARIATES, "base_gap", "mean_offset"]
    relative_features = [f"relative_offset_{organ}" for organ in ORGANS]
    all_features = [*base_features, *relative_features]
    for target in ORGANS:
        target_frame = outcome_frame(target, participant, values, analysis)
        standardized, _, _, scales = numeric_imputed(
            target_frame,
            target_frame,
            all_features,
        )
        retained_base = [column for column in base_features if column in standardized]
        for source in ORGANS:
            if (source, target) in completed:
                print(
                    f"Cross-organ Cox target={target} source={source} [checkpoint]",
                    flush=True,
                )
                continue
            relative = f"relative_offset_{source}"
            if relative not in standardized:
                raise RuntimeError(f"No variation in {relative} for {target}")
            print(
                f"Cross-organ Cox target={target} source={source}",
                flush=True,
            )
            model_frame = standardized[[*retained_base, relative]].copy()
            model_frame["duration"] = target_frame["duration"].to_numpy(float)
            model_frame["event"] = target_frame["event"].to_numpy(int)
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
            item = model.summary.loc[relative]
            rows.append(
                {
                    "source_organ": source,
                    "source_organ_label": ORGAN_LABELS[source],
                    "target_outcome": target,
                    "target_outcome_label": ORGAN_LABELS[target],
                    "is_matched_target": bool(source == target),
                    "n_at_risk": int(len(model_frame)),
                    "events": int(model_frame["event"].sum()),
                    "relative_offset_sd_years": float(scales[relative]),
                    "coef_per_sd": float(item["coef"]),
                    "hr_per_sd": float(item["exp(coef)"]),
                    "hr_ci95_low": float(item["exp(coef) lower 95%"]),
                    "hr_ci95_high": float(item["exp(coef) upper 95%"]),
                    "se": float(item["se(coef)"]),
                    "z": float(item["z"]),
                    "p_value": float(item["p"]),
                    "penalizer": float(penalizer),
                    "model": (
                        "cause-specific Cox: full clinical/record-density baseline + "
                        "base gap + mean offset + one source-relative offset"
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
    result = pd.DataFrame(rows)
    result["q_value_bh_49"] = shared.bh_adjust(result["p_value"].to_numpy(float))
    result["q_value_bh_within_target_7"] = np.nan
    for target, indices in result.groupby("target_outcome", sort=False).groups.items():
        result.loc[indices, "q_value_bh_within_target_7"] = shared.bh_adjust(
            result.loc[indices, "p_value"].to_numpy(float)
        )
    return result


def fit_and_score_cox(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    penalizer: float,
) -> np.ndarray:
    from lifelines import CoxPHFitter

    train_z, test_z, _, _ = numeric_imputed(train, test, features)
    retained = [column for column in features if column in train_z]
    model_frame = train_z[retained].copy()
    model_frame["duration"] = train["duration"].to_numpy(float)
    model_frame["event"] = train["event"].to_numpy(int)
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
    return model.predict_log_partial_hazard(test_z[retained]).to_numpy(np.float64)


def harrell_c(frame: pd.DataFrame, risk: np.ndarray) -> float:
    from lifelines.utils import concordance_index

    return float(
        concordance_index(
            frame["duration"].to_numpy(float),
            -np.asarray(risk, dtype=float),
            frame["event"].to_numpy(int).astype(bool),
        )
    )


def crossfit_incremental_c(
    *,
    penalizer: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint = SOURCE_DIR / "_checkpoint_calibration_test_delta_c.csv"
    if checkpoint.exists():
        checkpoint_frame = pd.read_csv(checkpoint)
        rows: list[dict[str, object]] = checkpoint_frame.to_dict(orient="records")
    else:
        rows = []
    completed = {
        (str(row["target_organ"]), int(row["outer_fold"]))
        for row in rows
    }
    base_features = [*FULL_RISK_COVARIATES, "base_gap", "mean_offset"]
    context = load_context_table()
    for fold in range(5):
        calibration_path, test_path = outer_prediction_paths(fold)
        calibration_participant, calibration_values = load_prediction_components(
            calibration_path
        )
        test_participant, test_values = load_prediction_components(test_path)
        overlap = calibration_participant.index.intersection(test_participant.index)
        if len(overlap):
            raise RuntimeError(
                f"Calibration/test participant overlap for outer_fold={fold}: {len(overlap)}"
            )
        calibration_analysis = attach_analysis(calibration_participant, context)
        test_analysis = attach_analysis(test_participant, context)
        for target in ORGANS:
            if (target, fold) in completed:
                print(
                    f"Held-out C-index target={target} outer_fold={fold} [checkpoint]",
                    flush=True,
                )
                continue
            relative = f"relative_offset_{target}"
            train = outcome_frame(
                target,
                calibration_participant,
                calibration_values,
                calibration_analysis,
            )
            test = outcome_frame(
                target,
                test_participant,
                test_values,
                test_analysis,
            )
            print(
                f"Calibration-to-test C-index target={target} outer_fold={fold} "
                f"train={len(train)} test={len(test)}",
                flush=True,
            )
            base_score = fit_and_score_cox(
                train,
                test,
                base_features,
                penalizer=penalizer,
            )
            augmented_score = fit_and_score_cox(
                train,
                test,
                [*base_features, relative],
                penalizer=penalizer,
            )
            c_base = harrell_c(test, base_score)
            c_augmented = harrell_c(test, augmented_score)
            rows.append(
                {
                    "target_organ": target,
                    "target_organ_label": ORGAN_LABELS[target],
                    "outer_fold": fold,
                    "n_train": int(len(train)),
                    "events_train": int(train["event"].sum()),
                    "n_test": int(len(test)),
                    "events_test": int(test["event"].sum()),
                    "c_shared_baseline": c_base,
                    "c_plus_target_relative_offset": c_augmented,
                    "incremental_delta_c": float(c_augmented - c_base),
                    "penalizer": float(penalizer),
                    "fit_partition": "same-outer-model calibration participants",
                    "evaluation_partition": "same-outer-model outer-test participants",
                    "calibration_prediction_path": calibration_path.relative_to(ROOT).as_posix(),
                    "test_prediction_path": test_path.relative_to(ROOT).as_posix(),
                }
            )
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
    cells = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for target in ORGANS:
        values_target = cells.loc[
            cells["target_organ"].eq(target), "incremental_delta_c"
        ].to_numpy(float)
        mean = float(values_target.mean())
        se = float(values_target.std(ddof=1) / np.sqrt(len(values_target)))
        critical = float(stats.t.ppf(0.975, df=len(values_target) - 1))
        summaries.append(
            {
                "target_organ": target,
                "target_organ_label": ORGAN_LABELS[target],
                "outer_folds": int(len(values_target)),
                "mean_incremental_delta_c": mean,
                "min_fold_incremental_delta_c": float(values_target.min()),
                "max_fold_incremental_delta_c": float(values_target.max()),
                "paired_fold_t_ci95_low": mean - critical * se,
                "paired_fold_t_ci95_high": mean + critical * se,
                "folds_positive": int(np.sum(values_target > 0.0)),
            }
        )
    return cells, pd.DataFrame(summaries)


def specificity_summary(cross_hr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in ORGANS:
        subset = cross_hr.loc[cross_hr["target_outcome"].eq(target)].copy()
        diagonal = subset.loc[subset["source_organ"].eq(target)].iloc[0]
        off = subset.loc[subset["source_organ"].ne(target)]
        descending = subset.sort_values("coef_per_sd", ascending=False).reset_index(drop=True)
        rank = int(descending.index[descending["source_organ"].eq(target)][0] + 1)
        rows.append(
            {
                "target_outcome": target,
                "target_outcome_label": ORGAN_LABELS[target],
                "matched_hr": float(diagonal["hr_per_sd"]),
                "matched_log_hr": float(diagonal["coef_per_sd"]),
                "mean_off_target_hr": float(off["hr_per_sd"].mean()),
                "mean_off_target_log_hr": float(off["coef_per_sd"].mean()),
                "matched_minus_mean_off_target_log_hr": float(
                    diagonal["coef_per_sd"] - off["coef_per_sd"].mean()
                ),
                "matched_rank_among_7_sources": rank,
                "matched_is_largest": bool(rank == 1),
                "strongest_off_target_source": str(
                    off.sort_values("coef_per_sd", ascending=False).iloc[0]["source_organ"]
                ),
                "strongest_off_target_hr": float(
                    off.sort_values("coef_per_sd", ascending=False).iloc[0]["hr_per_sd"]
                ),
            }
        )
    return pd.DataFrame(rows)


def panel_label(axis: plt.Axes, label: str, x: float = -0.12) -> None:
    axis.text(
        x,
        1.075,
        label,
        transform=axis.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_panel_a(
    bar_axis: plt.Axes,
    dot_axis: plt.Axes,
    decomposition: pd.DataFrame,
) -> None:
    panel_label(bar_axis, "a", x=-0.18)
    bar_axis.set_title(
        "Model-derived variance decomposition",
        fontsize=13.5,
        fontweight="bold",
        pad=12,
    )
    full = decomposition.loc[decomposition["analysis_group"].eq("full_cohort")].iloc[0]
    shared_percent = 100.0 * float(full["shared_variance_fraction"])
    specific_percent = 100.0 * float(full["specific_variance_fraction"])
    bar_axis.barh([0], [shared_percent], color=SHARED_COLOR, height=0.54)
    bar_axis.barh(
        [0],
        [specific_percent],
        left=[shared_percent],
        color=SPECIFIC_COLOR,
        height=0.54,
    )
    bar_axis.text(
        shared_percent / 2.0,
        0,
        f"Shared {shared_percent:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=10.5,
        fontweight="bold",
    )
    bar_axis.text(
        shared_percent + specific_percent / 2.0,
        0,
        f"{specific_percent:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        fontweight="bold",
        rotation=90,
    )
    bar_axis.set_xlim(0.0, 100.0)
    bar_axis.set_yticks([])
    bar_axis.set_xlabel("Variance fraction (%)", fontsize=9)
    bar_axis.spines[["top", "right", "left"]].set_visible(False)

    groups = decomposition.loc[
        decomposition["analysis_group"].ne("full_cohort")
    ].copy()
    groups["order"] = groups["analysis_group"].map(
        {group: index for index, group in enumerate(GROUP_ORDER)}
    )
    groups = groups.sort_values("order", ascending=False)
    y = np.arange(len(groups))
    values = 100.0 * groups["specific_variance_fraction"].to_numpy(float)
    dot_axis.hlines(y, 0.0, values, color="#D4D4D4", linewidth=1.0)
    dot_axis.scatter(
        values,
        y,
        color=SPECIFIC_COLOR,
        s=28,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    dot_axis.axvline(
        specific_percent,
        color=SHARED_COLOR,
        linestyle="--",
        linewidth=1.1,
    )
    dot_axis.text(
        0.985,
        0.985,
        f"Dashed: full cohort ({specific_percent:.2f}%)",
        transform=dot_axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=SHARED_COLOR,
    )
    dot_axis.set_yticks(y, groups["analysis_group_label"], fontsize=7.5)
    dot_axis.set_xlabel("Organ-specific variance (%)", fontsize=9)
    dot_axis.set_xlim(0.0, max(5.2, float(values.max()) * 1.12))
    dot_axis.grid(axis="x", color="#E5E5E5", linewidth=0.65)
    dot_axis.spines[["top", "right", "left"]].set_visible(False)
    dot_axis.tick_params(axis="y", length=0)


def draw_panel_b(axis: plt.Axes, cross_hr: pd.DataFrame) -> None:
    panel_label(axis, "b", x=-0.14)
    axis.set_title(
        "Cross-organ prospective associations",
        fontsize=13.5,
        fontweight="bold",
        pad=12,
    )
    coefficients = cross_hr.pivot(
        index="source_organ",
        columns="target_outcome",
        values="coef_per_sd",
    ).loc[list(ORGANS), list(ORGANS)]
    hrs = cross_hr.pivot(
        index="source_organ",
        columns="target_outcome",
        values="hr_per_sd",
    ).loc[list(ORGANS), list(ORGANS)]
    q_values = cross_hr.pivot(
        index="source_organ",
        columns="target_outcome",
        values="q_value_bh_49",
    ).loc[list(ORGANS), list(ORGANS)]
    annotation = np.empty((len(ORGANS), len(ORGANS)), dtype=object)
    for row in range(len(ORGANS)):
        for column in range(len(ORGANS)):
            marker = "*" if float(q_values.iloc[row, column]) < 0.05 else ""
            annotation[row, column] = f"{hrs.iloc[row, column]:.2f}{marker}"
    max_abs = max(0.08, float(np.abs(coefficients.to_numpy(float)).max()))
    sns.heatmap(
        coefficients,
        ax=axis,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        square=True,
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": 7.2},
        linewidths=0.9,
        linecolor="white",
        cbar=False,
    )
    labels = [ORGAN_LABELS[organ] for organ in ORGANS]
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(labels, rotation=0, fontsize=8)
    axis.set_xlabel("Future organ event", fontsize=9.5)
    axis.set_ylabel("Source relative organ offset", fontsize=9.5)
    for index in range(len(ORGANS)):
        axis.add_patch(
            Rectangle(
                (index, index),
                1,
                1,
                fill=False,
                edgecolor="#202020",
                linewidth=2.0,
            )
        )
    axis.text(
        0.0,
        -0.24,
        "Cells: adjusted HR per 1-SD source offset; black boxes: matched organ; * BH q<0.05.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )


def draw_panel_c(
    hr_axis: plt.Axes,
    delta_axis: plt.Axes,
    cross_hr: pd.DataFrame,
    delta_cells: pd.DataFrame,
    delta_summary: pd.DataFrame,
) -> None:
    panel_label(hr_axis, "c", x=-0.18)
    hr_axis.text(
        0.0,
        1.095,
        "Target-organ association and added discrimination",
        transform=hr_axis.transAxes,
        fontsize=13.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    diagonal = (
        cross_hr.loc[cross_hr["is_matched_target"]]
        .set_index("target_outcome")
        .loc[list(reversed(ORGANS))]
        .reset_index()
    )
    y = np.arange(len(diagonal))
    for index, row in diagonal.iterrows():
        organ = str(row["target_outcome"])
        hr = float(row["hr_per_sd"])
        low = float(row["hr_ci95_low"])
        high = float(row["hr_ci95_high"])
        hr_axis.errorbar(
            hr,
            index,
            xerr=np.asarray([[hr - low], [high - hr]]),
            fmt="o",
            color=ORGAN_COLORS[organ],
            markersize=6,
            capsize=2.5,
            linewidth=1.4,
            zorder=3,
        )
        hr_axis.text(high + 0.004, index, f"{hr:.2f}", va="center", fontsize=7.5)
    hr_axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    hr_axis.set_yticks(y, diagonal["target_outcome_label"], fontsize=8.5)
    hr_axis.set_xlabel("Adjusted HR (95% CI)", fontsize=9)
    hr_axis.set_title("Matched outcome", fontsize=10.5, fontweight="bold", pad=7)
    hr_axis.set_xlim(
        min(0.98, float(diagonal["hr_ci95_low"].min()) - 0.01),
        float(diagonal["hr_ci95_high"].max()) + 0.055,
    )
    hr_axis.grid(axis="x", color="#E5E5E5", linewidth=0.65)
    hr_axis.spines[["top", "right", "left"]].set_visible(False)
    hr_axis.tick_params(axis="y", length=0)

    order = list(reversed(ORGANS))
    summary = delta_summary.set_index("target_organ").loc[order]
    fold_offsets = np.linspace(-0.15, 0.15, 5)
    all_delta = delta_cells["incremental_delta_c"].to_numpy(float)
    x_padding = max(0.00015, float(np.ptp(all_delta)) * 0.12)
    for index, organ in enumerate(order):
        folds = (
            delta_cells.loc[delta_cells["target_organ"].eq(organ)]
            .sort_values("outer_fold")
        )
        fold_values = folds["incremental_delta_c"].to_numpy(float)
        delta_axis.hlines(
            index,
            float(fold_values.min()),
            float(fold_values.max()),
            color="#B8B8B8",
            linewidth=1.1,
            zorder=1,
        )
        delta_axis.scatter(
            fold_values,
            index + fold_offsets,
            s=15,
            facecolor="white",
            edgecolor=ORGAN_COLORS[organ],
            linewidth=0.8,
            zorder=2,
        )
        mean = float(summary.loc[organ, "mean_incremental_delta_c"])
        delta_axis.scatter(
            mean,
            index,
            s=44,
            marker="D",
            color=ORGAN_COLORS[organ],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        delta_axis.text(
            max(mean, float(fold_values.max())) + x_padding * 0.25,
            index,
            f"{mean:+.4f}",
            va="center",
            fontsize=7.3,
        )
    delta_axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    delta_axis.tick_params(axis="y", left=False, labelleft=False)
    delta_axis.set_xlabel(r"Held-out incremental $\Delta$C", fontsize=9)
    delta_axis.set_title("Added discrimination", fontsize=10.5, fontweight="bold", pad=7)
    delta_axis.grid(axis="x", color="#E5E5E5", linewidth=0.65)
    delta_axis.spines[["top", "right", "left"]].set_visible(False)
    delta_axis.tick_params(axis="y", length=0)
    lower = min(-x_padding, float(all_delta.min()) - x_padding)
    upper = float(all_delta.max()) + x_padding * 2.1
    delta_axis.set_xlim(lower, upper)
    delta_axis.text(
        0.0,
        -0.17,
        "Open circles: five held-out outer folds; diamonds: fold means.",
        transform=delta_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )


def render_figure(
    decomposition: pd.DataFrame,
    cross_hr: pd.DataFrame,
    delta_cells: pd.DataFrame,
    delta_summary: pd.DataFrame,
) -> list[Path]:
    shared.setup_style()
    figure = plt.figure(figsize=(17.0, 6.25))
    outer = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.05, 1.02, 1.42),
        left=0.05,
        right=0.985,
        top=0.92,
        bottom=0.15,
        wspace=0.34,
    )
    a_grid = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=(0.9, 3.5),
        hspace=0.35,
    )
    bar_axis = figure.add_subplot(a_grid[0, 0])
    dot_axis = figure.add_subplot(a_grid[1, 0])
    heat_axis = figure.add_subplot(outer[0, 1])
    c_grid = outer[0, 2].subgridspec(
        1,
        2,
        width_ratios=(0.92, 1.12),
        wspace=0.16,
    )
    hr_axis = figure.add_subplot(c_grid[0, 0])
    delta_axis = figure.add_subplot(c_grid[0, 1], sharey=hr_axis)
    draw_panel_a(bar_axis, dot_axis, decomposition)
    draw_panel_b(heat_axis, cross_hr)
    draw_panel_c(hr_axis, delta_axis, cross_hr, delta_cells, delta_summary)
    outputs: list[Path] = []
    for extension in ("svg", "pdf", "png"):
        path = OUTPUT_DIR / f"M11_candidate_direct_organ_specificity.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def write_readme(
    specificity: pd.DataFrame,
    delta_summary: pd.DataFrame,
) -> None:
    diagonal_largest = int(specificity["matched_is_largest"].sum())
    positive_delta = int(delta_summary["mean_incremental_delta_c"].gt(0.0).sum())
    text = f"""# M11器官特异性直接证据候选正文图

本目录为作者审核候选图，不替换当前Fig.11指针，也没有重新训练神经网络。

- A：模型输出的共享参与者成分与参与者内器官特异成分方差分解；
- B：7个来源器官相对offset对7类未来器官事件的49个完整调整cause-specific Cox模型；
- C左：B面板对角线对应的靶器官HR及Wald 95% CI；
- C右：完整临床和记录密度基线、共享基础gap及平均offset模型，与进一步加入对应器官相对offset模型的五折外层测试增量Harrell C-index。每个外层折均使用同一外层模型的校准参与者拟合Cox，再在该模型未见的外层测试参与者中评估；小圆点为5个外层测试折，菱形为折均值。

当前49项比较中，对应器官offset为该结局七个来源offset中最强者的结局数为{diagonal_largest}/7；七个器官中平均增量C-index为正的数量为{positive_delta}/7。具体结果必须依据CSV逐项解释，不把横断器官对比解释为因果传播。
"""
    (OUTPUT_ROOT / "README_CN.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.penalizer) or args.penalizer < 0.0:
        raise RuntimeError("penalizer must be finite and non-negative")
    prepare_directories()
    participant, values, reconstruction_audit = shared.load_components()
    analysis = load_context_and_analysis(participant)

    decomposition = pd.read_csv(DECOMPOSITION_INPUT)
    decomposition_bootstrap = pd.read_csv(DECOMPOSITION_BOOTSTRAP_INPUT)
    full = decomposition.loc[decomposition["analysis_group"].eq("full_cohort")].iloc[0]
    if int(full["n"]) != len(participant):
        raise RuntimeError("Decomposition participant count does not match OOF cohort")
    if len(decomposition_bootstrap) != 1000:
        raise RuntimeError("Expected 1000 decomposition bootstrap replicates")
    decomposition.to_csv(
        SOURCE_DIR / "shared_specific_variance_full_and_16groups.csv",
        index=False,
    )
    decomposition_bootstrap.to_csv(
        SOURCE_DIR / "shared_specific_variance_bootstrap_1000.csv",
        index=False,
    )

    cross_path = SOURCE_DIR / "cross_organ_relative_offset_fullrisk_cox_49.csv"
    delta_cells_path = SOURCE_DIR / "target_relative_offset_heldout_delta_c_35cells.csv"
    delta_summary_path = SOURCE_DIR / "target_relative_offset_heldout_delta_c_7organs.csv"
    specificity_path = SOURCE_DIR / "matched_vs_offtarget_specificity_summary.csv"
    cross_valid = False
    if cross_path.exists():
        cross_hr = pd.read_csv(cross_path)
        cross_valid = (
            len(cross_hr) == 49
            and np.allclose(cross_hr["penalizer"].to_numpy(float), float(args.penalizer))
        )
    if not cross_valid:
        cross_hr = fit_cross_organ_hr_matrix(
            participant,
            values,
            analysis,
            penalizer=float(args.penalizer),
        )
        cross_hr.to_csv(cross_path, index=False)

    delta_valid = False
    if delta_cells_path.exists() and delta_summary_path.exists():
        delta_cells = pd.read_csv(delta_cells_path)
        delta_summary = pd.read_csv(delta_summary_path)
        delta_valid = (
            len(delta_cells) == 35
            and set(delta_cells["fit_partition"].astype(str))
            == {"same-outer-model calibration participants"}
            and np.allclose(
                delta_cells["penalizer"].to_numpy(float),
                float(args.penalizer),
            )
        )
    if not delta_valid:
        delta_cells, delta_summary = crossfit_incremental_c(
            penalizer=float(args.penalizer),
        )
        delta_cells.to_csv(delta_cells_path, index=False)
        delta_summary.to_csv(delta_summary_path, index=False)
    specificity = specificity_summary(cross_hr)
    specificity.to_csv(specificity_path, index=False)
    for checkpoint in (
        SOURCE_DIR / "_checkpoint_cross_organ_hr.csv",
        SOURCE_DIR / "_checkpoint_heldout_delta_c.csv",
        SOURCE_DIR / "_checkpoint_calibration_test_delta_c.csv",
    ):
        checkpoint.unlink(missing_ok=True)

    if len(cross_hr) != 49 or cross_hr[["source_organ", "target_outcome"]].duplicated().any():
        raise RuntimeError("Cross-organ Cox matrix is not a complete unique 7x7 grid")
    if len(delta_cells) != 35 or delta_cells.groupby("target_organ")["outer_fold"].nunique().ne(5).any():
        raise RuntimeError("Held-out delta-C table is not a complete 7x5 grid")
    if len(delta_summary) != 7:
        raise RuntimeError("Held-out delta-C summary does not contain seven organs")

    contract = {
        "schema_version": "final_figure_0725.m11.direct_specificity.contract.v1",
        "generated_at_utc": utc_now(),
        "participants": int(len(participant)),
        "neural_network_refit": False,
        "cross_organ_hr_models": 49,
        "cross_organ_hr_adjustment": [
            *FULL_RISK_COVARIATES,
            "base_gap",
            "mean_offset",
            "one source relative offset",
        ],
        "relative_offset_definition": "source offset minus mean of the other six offsets",
        "cox_penalizer": float(args.penalizer),
        "incremental_c_baseline": (
            "full clinical/record-density covariates + base gap + mean offset"
        ),
        "incremental_c_augmented": (
            "baseline + matched target relative offset"
        ),
        "incremental_c_fit": "same-outer-model calibration participants",
        "incremental_c_evaluation": "same-outer-model outer-test participants",
        "incremental_c_uncertainty_display": (
            "five held-out fold points and fold mean; no participant bootstrap CI"
        ),
        "decomposition_bootstrap_replicates": 1000,
        "component_reconstruction": reconstruction_audit,
    }
    (SOURCE_DIR / "analysis_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    outputs = render_figure(decomposition, cross_hr, delta_cells, delta_summary)
    write_readme(specificity, delta_summary)

    svg = next(path for path in outputs if path.suffix == ".svg")
    svg_text = svg.read_text(encoding="utf-8")
    matched = cross_hr.loc[cross_hr["is_matched_target"]]
    qa = {
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "candidate_only_pointer_unchanged": True,
        "participants": int(len(participant)),
        "cross_organ_cox_cells": int(len(cross_hr)),
        "heldout_delta_c_cells": int(len(delta_cells)),
        "heldout_outer_folds_per_organ": int(
            delta_cells.groupby("target_organ")["outer_fold"].nunique().min()
        ),
        "matched_hr_range": [
            float(matched["hr_per_sd"].min()),
            float(matched["hr_per_sd"].max()),
        ],
        "matched_is_largest_count": int(specificity["matched_is_largest"].sum()),
        "matched_rank_range": [
            int(specificity["matched_rank_among_7_sources"].min()),
            int(specificity["matched_rank_among_7_sources"].max()),
        ],
        "mean_incremental_delta_c_range": [
            float(delta_summary["mean_incremental_delta_c"].min()),
            float(delta_summary["mean_incremental_delta_c"].max()),
        ],
        "organs_with_positive_mean_incremental_delta_c": int(
            delta_summary["mean_incremental_delta_c"].gt(0.0).sum()
        ),
        "incremental_c_evaluation_contract": sorted(
            set(delta_cells["fit_partition"].astype(str))
        ),
        "svg_editable_text": "<text" in svg_text,
        "svg_embedded_raster": "<image" in svg_text or "data:image" in svg_text,
    }
    if not qa["svg_editable_text"] or qa["svg_embedded_raster"]:
        raise RuntimeError("SVG editability QA failed")
    (QA_DIR / "QA_REPORT.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    inputs = [
        shared.OOF,
        shared.RUN_CONFIG,
        shared.FINAL_SELECTION,
        shared.CONTEXT,
        shared.MEMBERSHIP,
        DECOMPOSITION_INPUT,
        DECOMPOSITION_BOOTSTRAP_INPUT,
        *[
            path
            for fold in range(5)
            for path in outer_prediction_paths(fold)
        ],
    ]
    artifacts = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "schema_version": "final_figure_0725.m11.direct_specificity.candidate.v1",
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "candidate_only": True,
        "pointer_updated": False,
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
            "dependency": {
                "path": shared.SCRIPT.relative_to(ROOT).as_posix(),
                "bytes": shared.SCRIPT.stat().st_size,
                "sha256": sha256(shared.SCRIPT),
            },
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
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
