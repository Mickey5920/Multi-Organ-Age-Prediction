from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[2]
REPO_ROOT = REVISION_ROOT.parent
OOF_PATH = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "02_fivefold_frozen_huber1"
    / "02_summary"
    / "oof_predictions.parquet"
)
EXPECTED_OOF_SHA256 = "bfc9e0de88121bb4b520bd6c878ac4d655d00b7d1dfa1f64b5f7a4de0b633239"
CACHE_ROOT = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
INDEX_ROOT = REVISION_ROOT / "01_data_rebuild" / "07_corrected_primary_v1" / "01_index_and_events"
SOURCE_DIR = HERE / "Source_Data"
FIGURE_DIR = HERE / "Figures"

ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
DENSITY_SPECS = (
    ("l32_total_active_count", "log1p"),
    ("l32_icd_active_count", "log1p"),
    ("l32_gp_active_count", "log1p"),
    ("l32_zero_event", "identity"),
    ("l32_saturated_32", "identity"),
    ("preindex_icd_count", "log1p"),
    ("preindex_gp_count", "log1p"),
    ("has_preindex_icd", "identity"),
    ("has_preindex_gp", "identity"),
    ("phenotype_missing_fraction", "identity"),
)
DISPLAY_NAMES = {
    "l32_total_active_count": "L32 total events",
    "l32_icd_active_count": "L32 ICD events",
    "l32_gp_active_count": "L32 GP events",
    "l32_zero_event": "No L32 event",
    "l32_saturated_32": "L32 saturated",
    "preindex_icd_count": "Pre-index ICD count",
    "preindex_gp_count": "Pre-index GP count",
    "has_preindex_icd": "Any pre-index ICD",
    "has_preindex_gp": "Any pre-index GP",
    "phenotype_missing_fraction": "Phenotype missingness",
}
ORGAN_DISPLAY = {
    "cardiovascular": "Cardiovascular",
    "hepatic": "Hepatic",
    "immune": "Immune",
    "metabolic": "Metabolic",
    "musculoskeletal": "Musculoskeletal",
    "pulmonary": "Pulmonary",
    "renal": "Renal",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be aligned, finite, and nondegenerate")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    coefficient = float((x @ y) / denominator)
    coefficient = float(np.clip(coefficient, -1.0, 1.0))
    if abs(coefficient) >= 1.0:
        p_value = 0.0
    else:
        statistic = coefficient * math.sqrt((len(x) - 2) / max(1e-15, 1.0 - coefficient**2))
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=len(x) - 2))
    clipped = float(np.clip(coefficient, -0.999999999, 0.999999999))
    se = 1.0 / math.sqrt(len(x) - 3)
    z = math.atanh(clipped)
    return coefficient, math.tanh(z - 1.959963984540054 * se), math.tanh(
        z + 1.959963984540054 * se
    ), p_value


def _load_oof() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "participant_id",
        "outer_fold",
        "organ_system",
        "actual_age",
        "predicted_age",
        "corrected_age_gap",
        "strict_healthy_reference_60m",
        "sex",
    ]
    oof = pd.read_parquet(OOF_PATH, columns=columns)
    oof["participant_id"] = oof["participant_id"].astype(str)
    if len(oof) != 456_773 * 7 or oof.duplicated(["participant_id", "organ_system"]).any():
        raise ValueError("canonical OOF table has unexpected coverage")
    if set(oof["organ_system"].unique()) != set(ORGANS):
        raise ValueError("canonical OOF organ set changed")

    invariant = oof.groupby("participant_id", sort=False).agg(
        outer_fold=("outer_fold", "nunique"),
        actual_age=("actual_age", "nunique"),
        strict=("strict_healthy_reference_60m", "nunique"),
        sex=("sex", "nunique"),
        organs=("organ_system", "size"),
    )
    if not (
        invariant["outer_fold"].eq(1).all()
        and invariant["actual_age"].eq(1).all()
        and invariant["strict"].eq(1).all()
        and invariant["sex"].eq(1).all()
        and invariant["organs"].eq(7).all()
    ):
        raise ValueError("participant-level OOF invariants failed")

    participant = oof.drop_duplicates("participant_id")[[
        "participant_id",
        "outer_fold",
        "actual_age",
        "strict_healthy_reference_60m",
        "sex",
    ]].copy()
    gaps = oof.pivot(index="participant_id", columns="organ_system", values="corrected_age_gap")
    predictions = oof.pivot(index="participant_id", columns="organ_system", values="predicted_age")
    gaps = gaps.loc[:, list(ORGANS)].rename(columns=lambda value: f"gap__{value}").reset_index()
    predictions = predictions.loc[:, list(ORGANS)].rename(
        columns=lambda value: f"prediction__{value}"
    ).reset_index()
    participant = participant.merge(gaps, on="participant_id", validate="one_to_one")
    participant = participant.merge(predictions, on="participant_id", validate="one_to_one")
    return oof, participant, predictions


def _load_cache_density() -> pd.DataFrame:
    manifest = json.loads((CACHE_ROOT / "batch_manifest.json").read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for record in manifest:
        path = (CACHE_ROOT / record["path"]).resolve()
        with np.load(path, allow_pickle=False) as payload:
            participant_id = payload["participant_id"].astype(str)
            valid = np.asarray(payload["event_valid_mask"], dtype=bool)
            event_type = np.asarray(payload["event_type_id"], dtype=np.int64)
            missing = np.asarray(payload["phenotype_missing_mask"], dtype=bool)
        total = valid.sum(axis=1).astype(np.int16)
        frames.append(
            pd.DataFrame(
                {
                    "participant_id": participant_id,
                    "l32_total_active_count": total,
                    "l32_icd_active_count": (valid & event_type.eq(1) if hasattr(event_type, "eq") else valid & (event_type == 1)).sum(axis=1).astype(np.int16),
                    "l32_gp_active_count": (valid & (event_type == 2)).sum(axis=1).astype(np.int16),
                    "l32_zero_event": total == 0,
                    "l32_saturated_32": total == 32,
                    "phenotype_missing_fraction": missing.mean(axis=1).astype(np.float32),
                }
            )
        )
    density = pd.concat(frames, ignore_index=True)
    if len(density) != 456_773 or density["participant_id"].duplicated().any():
        raise ValueError("compact-cache density coverage changed")

    icd = pd.read_parquet(
        INDEX_ROOT / "icd_participant_summary.parquet",
        columns=["participant_id", "model_input_icd_count", "has_model_input_icd"],
    ).rename(
        columns={
            "model_input_icd_count": "preindex_icd_count",
            "has_model_input_icd": "has_preindex_icd",
        }
    )
    gp = pd.read_parquet(
        INDEX_ROOT / "gp_participant_summary.parquet",
        columns=["participant_id", "model_input_gp_count", "has_model_input_gp"],
    ).rename(
        columns={
            "model_input_gp_count": "preindex_gp_count",
            "has_model_input_gp": "has_preindex_gp",
        }
    )
    for frame in (icd, gp):
        frame["participant_id"] = frame["participant_id"].astype(str)
    density = density.merge(icd, on="participant_id", validate="one_to_one")
    density = density.merge(gp, on="participant_id", validate="one_to_one")
    for name, _ in DENSITY_SPECS:
        density[name] = pd.to_numeric(density[name], errors="raise").astype(float)
        if not np.isfinite(density[name]).all():
            raise ValueError(f"nonfinite density variable: {name}")
    return density


def _analysis_values(frame: pd.DataFrame, name: str, transform: str) -> np.ndarray:
    values = frame[name].to_numpy(dtype=np.float64)
    if transform == "log1p":
        values = np.log1p(values)
    elif transform != "identity":
        raise ValueError(transform)
    return values


def _correlation_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    age_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    scopes = {
        "full_oof": np.ones(len(frame), dtype=bool),
        "strict_healthy": frame["strict_healthy_reference_60m"].eq(1).to_numpy(),
    }
    for scope, mask in scopes.items():
        scoped = frame.loc[mask]
        responses: dict[str, np.ndarray] = {"age": scoped["actual_age"].to_numpy(float)}
        responses.update({organ: scoped[f"gap__{organ}"].to_numpy(float) for organ in ORGANS})
        errors = {organ: np.abs(scoped[f"gap__{organ}"].to_numpy(float)) for organ in ORGANS}
        density_values = {
            name: _analysis_values(scoped, name, transform)
            for name, transform in DENSITY_SPECS
        }
        for method in ("spearman", "pearson"):
            if method == "spearman":
                x_values = {name: stats.rankdata(value) for name, value in density_values.items()}
                response_values = {name: stats.rankdata(value) for name, value in responses.items()}
                error_values = {name: stats.rankdata(value) for name, value in errors.items()}
            else:
                x_values = density_values
                response_values = responses
                error_values = errors
            for name, transform in DENSITY_SPECS:
                coefficient, low, high, p_value = _correlation(
                    x_values[name], response_values["age"]
                )
                age_rows.append(
                    {
                        "scope": scope,
                        "method": method,
                        "density_variable": name,
                        "density_transform": transform,
                        "n": len(scoped),
                        "coefficient": coefficient,
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": p_value,
                        "family_id": f"density_age::{scope}::{method}",
                    }
                )
                for organ in ORGANS:
                    coefficient, low, high, p_value = _correlation(
                        x_values[name], response_values[organ]
                    )
                    gap_rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "organ_system": organ,
                            "density_variable": name,
                            "density_transform": transform,
                            "n": len(scoped),
                            "coefficient": coefficient,
                            "ci95_low": low,
                            "ci95_high": high,
                            "p_value": p_value,
                            "family_id": f"density_gap::{scope}::{method}::{organ}",
                        }
                    )
                    coefficient, low, high, p_value = _correlation(
                        x_values[name], error_values[organ]
                    )
                    error_rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "organ_system": organ,
                            "density_variable": name,
                            "density_transform": transform,
                            "n": len(scoped),
                            "coefficient": coefficient,
                            "ci95_low": low,
                            "ci95_high": high,
                            "p_value": p_value,
                            "family_id": f"density_absolute_error::{scope}::{method}::{organ}",
                        }
                    )
    outputs = []
    for rows in (age_rows, gap_rows, error_rows):
        output = pd.DataFrame(rows)
        output["q_value"] = np.nan
        for _, index in output.groupby("family_id", sort=False).groups.items():
            family_index = np.asarray(list(index), dtype=int)
            finite = np.isfinite(output.loc[family_index, "p_value"].to_numpy(float))
            if finite.any():
                output.loc[family_index[finite], "q_value"] = benjamini_hochberg(
                    output.loc[family_index[finite], "p_value"].to_numpy(float)
                )
        output["status"] = np.where(np.isfinite(output["coefficient"]), "OK", "NOT_ESTIMABLE_CONSTANT")
        outputs.append(output)
    return tuple(outputs)  # type: ignore[return-value]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(predicted - actual)))
    pearson = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 2 else float("nan")
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum((actual - predicted) ** 2) / denominator) if denominator > 0 else float("nan")
    return mae, pearson, r2


def _density_stratified_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = np.full(len(frame), "", dtype=object)
    threshold_rows: list[dict[str, object]] = []
    counts = frame["l32_total_active_count"].to_numpy(float)
    folds = frame["outer_fold"].to_numpy(int)
    for fold in range(5):
        reference = counts[(folds != fold) & (counts > 0) & (counts < 32)]
        low, high = np.quantile(reference, [1.0 / 3.0, 2.0 / 3.0], method="linear")
        target = folds == fold
        assigned[target & (counts == 0)] = "zero"
        assigned[target & (counts == 32)] = "saturated"
        assigned[target & (counts > 0) & (counts < 32) & (counts <= low)] = "low"
        assigned[target & (counts > low) & (counts < 32) & (counts <= high)] = "middle"
        assigned[target & (counts > high) & (counts < 32)] = "high"
        threshold_rows.append(
            {
                "outer_fold": fold,
                "reference_folds": ",".join(str(value) for value in range(5) if value != fold),
                "reference_n": len(reference),
                "low_threshold": low,
                "high_threshold": high,
            }
        )
    if np.any(assigned == ""):
        raise ValueError("density stratum assignment is incomplete")
    frame = frame.copy()
    frame["density_stratum"] = assigned
    strict = frame.loc[frame["strict_healthy_reference_60m"].eq(1)].copy()
    strict["prediction__pooled_mean"] = strict[
        [f"prediction__{organ}" for organ in ORGANS]
    ].mean(axis=1)
    rows: list[dict[str, object]] = []
    for estimator in ("pooled_mean", *ORGANS):
        for stratum in ("overall", "zero", "low", "middle", "high", "saturated"):
            part = strict if stratum == "overall" else strict.loc[strict["density_stratum"].eq(stratum)]
            mae, pearson, r2 = _metrics(
                part["actual_age"].to_numpy(float),
                part[f"prediction__{estimator}"].to_numpy(float),
            )
            rows.append(
                {
                    "estimator_id": estimator,
                    "density_stratum": stratum,
                    "n": len(part),
                    "actual_age_min": float(part["actual_age"].min()),
                    "actual_age_max": float(part["actual_age"].max()),
                    "mae": mae,
                    "pearson": pearson,
                    "r2": r2,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows)


def _plot(age: pd.DataFrame, gap: pd.DataFrame, error: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(12.0, 8.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.55], hspace=0.34, wspace=0.30)
    ax_age = fig.add_subplot(grid[:, 0])
    ax_gap = fig.add_subplot(grid[0, 1])
    ax_error = fig.add_subplot(grid[1, 1])

    subset = age.loc[age["method"].eq("spearman")].copy()
    order = [name for name, _ in DENSITY_SPECS]
    y = np.arange(len(order))
    for scope, offset, color, marker, label in (
        ("full_oof", -0.12, "#0072B2", "o", "Full OOF"),
        ("strict_healthy", 0.12, "#D55E00", "s", "Strict Health"),
    ):
        part = subset.loc[subset["scope"].eq(scope)].set_index("density_variable").loc[order]
        ax_age.errorbar(
            part["coefficient"],
            y + offset,
            xerr=np.vstack(
                [part["coefficient"] - part["ci95_low"], part["ci95_high"] - part["coefficient"]]
            ),
            fmt=marker,
            ms=4.5,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            label=label,
        )
    ax_age.axvline(0, color="#555555", lw=0.8, ls="--")
    ax_age.set_yticks(y, [DISPLAY_NAMES[name] for name in order])
    ax_age.invert_yaxis()
    ax_age.set_xlabel("Spearman correlation with chronological age")
    ax_age.set_title("A  Data density and chronological age", loc="left", fontweight="bold")
    ax_age.legend(frameon=False, loc="lower right")
    ax_age.spines[["top", "right"]].set_visible(False)

    def heatmap(ax: plt.Axes, source: pd.DataFrame, title: str) -> None:
        part = source.loc[
            source["scope"].eq("full_oof") & source["method"].eq("spearman")
        ]
        matrix = part.pivot(index="organ_system", columns="density_variable", values="coefficient")
        matrix = matrix.loc[list(ORGANS), order]
        limit = max(0.05, float(np.nanmax(np.abs(matrix.to_numpy()))))
        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_yticks(np.arange(len(ORGANS)), [ORGAN_DISPLAY[organ] for organ in ORGANS])
        ax.set_xticks(
            np.arange(len(order)),
            [DISPLAY_NAMES[name].replace(" ", "\n", 1) for name in order],
            rotation=35,
            ha="right",
        )
        ax.set_title(title, loc="left", fontweight="bold")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        colorbar.set_label("Spearman correlation")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    heatmap(ax_gap, gap, "B  Data density and corrected organ-age gaps")
    heatmap(ax_error, error, "C  Data density and absolute corrected gap magnitude")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURE_DIR / f"Supplementary_Density_Age_Correlation.{extension}",
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    oof_sha256 = sha256_file(OOF_PATH)
    if oof_sha256 != EXPECTED_OOF_SHA256:
        raise RuntimeError(
            "final five-fold OOF input hash changed: "
            f"expected {EXPECTED_OOF_SHA256}, observed {oof_sha256}"
        )
    _, participant, _ = _load_oof()
    density = _load_cache_density()
    analysis = participant.merge(density, on="participant_id", validate="one_to_one")
    if len(analysis) != 456_773:
        raise ValueError("participant/density join is incomplete")

    age, gap, error = _correlation_tables(analysis)
    stratified, thresholds = _density_stratified_metrics(analysis)
    outputs = {
        "age_density_correlations.csv": age,
        "gap_density_correlations.csv": gap,
        "absolute_error_density_correlations.csv": error,
        "density_stratified_strict_age_metrics.csv": stratified,
        "density_stratum_crossfit_thresholds.csv": thresholds,
    }
    for name, frame in outputs.items():
        frame.to_csv(SOURCE_DIR / name, index=False)
    _plot(age, gap, error)

    age_spearman = age.loc[age["scope"].eq("full_oof") & age["method"].eq("spearman")]
    max_gap_row = gap.loc[
        gap.loc[gap["scope"].eq("full_oof") & gap["method"].eq("spearman"), "coefficient"]
        .abs()
        .idxmax()
    ]
    max_magnitude_row = error.loc[
        error.loc[
            error["scope"].eq("full_oof") & error["method"].eq("spearman"),
            "coefficient",
        ]
        .abs()
        .idxmax()
    ]
    pooled_strata = stratified.loc[stratified["estimator_id"].eq("pooled_mean")].set_index(
        "density_stratum"
    )

    qa = {
        "schema_version": "revision0710.density_age_correlation.v2",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_contract": "final frozen five-fold participant-level outer-test predictions",
        "participant_n": len(analysis),
        "strict_healthy_n": int(analysis["strict_healthy_reference_60m"].sum()),
        "age_correlation_rows": len(age),
        "gap_correlation_rows": len(gap),
        "absolute_error_correlation_rows": len(error),
        "density_stratified_metric_rows": len(stratified),
        "expected_rows": {"age": 40, "gap": 280, "absolute_error": 280, "stratified": 48},
        "all_expected_rows_present": bool(
            len(age) == 40 and len(gap) == 280 and len(error) == 280 and len(stratified) == 48
        ),
        "frozen_key_results": {
            "full_oof_age_spearman": {
                name: float(
                    age_spearman.loc[
                        age_spearman["density_variable"].eq(name), "coefficient"
                    ].iloc[0]
                )
                for name in (
                    "l32_total_active_count",
                    "l32_icd_active_count",
                    "l32_saturated_32",
                )
            },
            "maximum_absolute_full_oof_gap_density_spearman": {
                "organ_system": str(max_gap_row["organ_system"]),
                "density_variable": str(max_gap_row["density_variable"]),
                "coefficient": float(max_gap_row["coefficient"]),
                "ci95_low": float(max_gap_row["ci95_low"]),
                "ci95_high": float(max_gap_row["ci95_high"]),
            },
            "maximum_absolute_full_oof_gap_magnitude_density_spearman": {
                "organ_system": str(max_magnitude_row["organ_system"]),
                "density_variable": str(max_magnitude_row["density_variable"]),
                "coefficient": float(max_magnitude_row["coefficient"]),
                "ci95_low": float(max_magnitude_row["ci95_low"]),
                "ci95_high": float(max_magnitude_row["ci95_high"]),
            },
            "strict_healthy_pooled_mean": {
                stratum: {
                    "n": int(pooled_strata.loc[stratum, "n"]),
                    "mae": float(pooled_strata.loc[stratum, "mae"]),
                    "pearson": float(pooled_strata.loc[stratum, "pearson"]),
                    "r2": float(pooled_strata.loc[stratum, "r2"]),
                }
                for stratum in ("overall", "zero", "low", "middle", "high", "saturated")
            },
        },
        "inputs": {
            str(OOF_PATH.relative_to(REPO_ROOT)): oof_sha256,
            str((CACHE_ROOT / "batch_manifest.json").relative_to(REPO_ROOT)): sha256_file(
                CACHE_ROOT / "batch_manifest.json"
            ),
            str((INDEX_ROOT / "icd_participant_summary.parquet").relative_to(REPO_ROOT)): sha256_file(
                INDEX_ROOT / "icd_participant_summary.parquet"
            ),
            str((INDEX_ROOT / "gp_participant_summary.parquet").relative_to(REPO_ROOT)): sha256_file(
                INDEX_ROOT / "gp_participant_summary.parquet"
            ),
        },
    }
    if not qa["all_expected_rows_present"]:
        raise RuntimeError("density output row contract failed")
    qa_path = HERE / "QA_REPORT.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    output_paths = [
        *(SOURCE_DIR / name for name in outputs),
        *(FIGURE_DIR / f"Supplementary_Density_Age_Correlation.{extension}" for extension in ("png", "pdf", "svg")),
        qa_path,
    ]
    manifest = {
        "schema_version": "revision0710.density_age_correlation.manifest.v2",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_description": (
            "Descriptive density correlations and cross-fitted density-stratified age metrics "
            "using final frozen five-fold outer-test predictions; no model retraining."
        ),
        "input_sha256": qa["inputs"],
        "script": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "outputs": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }
    (HERE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
