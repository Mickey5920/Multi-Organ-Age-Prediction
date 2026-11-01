from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
REVISION_ROOT = REPO_ROOT / "Revision-0710"
OUTPUT_ROOT = (
    REVISION_ROOT
    / "Final_Fig_0716"
    / "Reviewer_Statistics"
    / "21_r3_12_bias_correction_sensitivity"
)
SOURCE_DATA = OUTPUT_ROOT / "Source_Data"

FINAL_ROOT = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "02_fivefold_frozen_huber1"
)
OOF_PATH = FINAL_ROOT / "02_summary" / "oof_predictions.parquet"
FOLD0_ROOT = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "01_runs"
    / "shared_private_independent_view_huber1_fold0"
)

ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
EXPECTED_PARTICIPANTS = 456_773
EXPECTED_ROWS = EXPECTED_PARTICIPANTS * len(ORGANS)
FORMS = {"continuous_linear": 1, "continuous_quadratic": 2}
FIVE_YEAR_EDGES = np.array([35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "role": role,
        "path": path.as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def fold_sources(outer_fold: int) -> tuple[Path, Path]:
    if outer_fold == 0:
        return (
            FOLD0_ROOT / "fold1_calibration_predictions.parquet",
            FOLD0_ROOT / "age_correction.json",
        )
    fold_root = FINAL_ROOT / "01_outer_folds" / f"outer_fold_{outer_fold}"
    return fold_root / "calibration_predictions.parquet", fold_root / "age_correction.json"


def design_matrix(age: np.ndarray, *, center: float, scale: float, degree: int) -> np.ndarray:
    z = (age - center) / scale
    return np.column_stack([np.power(z, power) for power in range(degree + 1)])


def fit_expected_gap(
    age: np.ndarray,
    raw_gap: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, float, float]:
    center = float(age.mean())
    scale = float(age.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("calibration reference age has no variation")
    design = design_matrix(age, center=center, scale=scale, degree=degree)
    coefficients, *_ = np.linalg.lstsq(design, raw_gap, rcond=None)
    return coefficients, center, scale


def expected_gap(
    age: np.ndarray,
    coefficients: np.ndarray,
    *,
    center: float,
    scale: float,
) -> np.ndarray:
    return design_matrix(
        age,
        center=center,
        scale=scale,
        degree=len(coefficients) - 1,
    ) @ coefficients


def correlation(x: pd.Series, y: pd.Series, *, method: str = "pearson") -> float:
    return float(x.corr(y, method=method))


def slope(age: pd.Series, gap: pd.Series) -> float:
    return float(np.polyfit(age.to_numpy(dtype=np.float64), gap.to_numpy(dtype=np.float64), 1)[0])


def age_group_labels(age: pd.Series, scheme: str) -> pd.Series:
    if scheme == "submitted_three_group_display":
        return pd.cut(
            age,
            bins=[-np.inf, 60.0, 70.0, np.inf],
            labels=["<60", "60-<70", ">=70"],
            right=False,
        )
    if scheme == "shifted_three_group_display":
        return pd.cut(
            age,
            bins=[-np.inf, 55.0, 65.0, np.inf],
            labels=["<55", "55-<65", ">=65"],
            right=False,
        )
    if scheme == "five_year_display":
        return pd.cut(age, bins=FIVE_YEAR_EDGES, right=False, include_lowest=True)
    raise ValueError(f"unknown display scheme: {scheme}")


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    required_columns = [
        "participant_id",
        "organ_system",
        "actual_age",
        "predicted_age",
        "strict_healthy_reference_60m",
        "raw_age_gap",
        "expected_age_gap",
        "corrected_age_gap",
        "outer_fold",
    ]
    oof = pd.read_parquet(OOF_PATH, columns=required_columns)
    oof["participant_id"] = oof["participant_id"].astype(str)
    if len(oof) != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected OOF row count: {len(oof)}")
    if oof["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise RuntimeError("unexpected OOF participant count")
    if set(oof["organ_system"].astype(str)) != set(ORGANS):
        raise RuntimeError("OOF organ set mismatch")
    if set(oof["outer_fold"].astype(int)) != set(range(5)):
        raise RuntimeError("OOF fold set mismatch")
    if oof.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("duplicate OOF participant-organ rows")
    raw_parity = np.max(
        np.abs(
            oof["raw_age_gap"].to_numpy(dtype=np.float64)
            - (
                oof["predicted_age"].to_numpy(dtype=np.float64)
                - oof["actual_age"].to_numpy(dtype=np.float64)
            )
        )
    )

    evaluated_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    input_records = [file_record(OOF_PATH, "final_fivefold_oof_predictions", rows=len(oof))]
    max_expected_parity = 0.0
    max_corrected_parity = 0.0
    disjoint_all = True

    for outer_fold in range(5):
        calibration_path, correction_path = fold_sources(outer_fold)
        calibration = pd.read_parquet(
            calibration_path,
            columns=[
                "participant_id",
                "organ_system",
                "actual_age",
                "predicted_age",
                "strict_healthy_reference_60m",
            ],
        )
        calibration["participant_id"] = calibration["participant_id"].astype(str)
        calibration["raw_age_gap"] = (
            calibration["predicted_age"].to_numpy(dtype=np.float64)
            - calibration["actual_age"].to_numpy(dtype=np.float64)
        )
        test = oof.loc[oof["outer_fold"].astype(int).eq(outer_fold)].copy()
        overlap = set(calibration["participant_id"]).intersection(test["participant_id"])
        disjoint_all = disjoint_all and not overlap
        if overlap:
            raise RuntimeError(f"calibration/test overlap in outer fold {outer_fold}: {len(overlap)}")

        correction = json.loads(correction_path.read_text(encoding="utf-8"))
        sealed = {row["organ_system"]: row for row in correction["parameters"]}
        if set(sealed) != set(ORGANS):
            raise RuntimeError(f"sealed correction organ mismatch in fold {outer_fold}")
        input_records.extend(
            [
                file_record(
                    calibration_path,
                    f"outer_fold_{outer_fold}_calibration_predictions",
                    rows=len(calibration),
                ),
                file_record(correction_path, f"outer_fold_{outer_fold}_sealed_linear_correction"),
            ]
        )

        for organ in ORGANS:
            cal_organ = calibration.loc[calibration["organ_system"].astype(str).eq(organ)].copy()
            cal_reference = cal_organ.loc[
                cal_organ["strict_healthy_reference_60m"].astype(int).eq(1)
            ].copy()
            test_organ = test.loc[test["organ_system"].astype(str).eq(organ)].copy()
            age_ref = cal_reference["actual_age"].to_numpy(dtype=np.float64)
            gap_ref = cal_reference["raw_age_gap"].to_numpy(dtype=np.float64)
            age_test = test_organ["actual_age"].to_numpy(dtype=np.float64)
            raw_test = test_organ["raw_age_gap"].to_numpy(dtype=np.float64)

            result = test_organ[
                [
                    "participant_id",
                    "organ_system",
                    "outer_fold",
                    "actual_age",
                    "strict_healthy_reference_60m",
                    "raw_age_gap",
                ]
            ].copy()
            result["raw"] = raw_test
            for form, degree in FORMS.items():
                coefficients, center, scale = fit_expected_gap(age_ref, gap_ref, degree=degree)
                fitted_expected = expected_gap(
                    age_test,
                    coefficients,
                    center=center,
                    scale=scale,
                )
                result[form] = raw_test - fitted_expected
                parameter_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "organ_system": organ,
                        "correction_form": form,
                        "degree": degree,
                        "calibration_participants": int(calibration["participant_id"].nunique()),
                        "calibration_reference_rows": int(len(cal_reference)),
                        "age_center": center,
                        "age_scale": scale,
                        "coefficients_standardized_age_basis": json.dumps(
                            [float(value) for value in coefficients],
                            separators=(",", ":"),
                        ),
                        "fit_population": "calibration-role strict disease-free reference",
                        "model_retraining_performed": False,
                    }
                )

                if form == "continuous_linear":
                    sealed_expected = (
                        float(sealed[organ]["intercept"])
                        + float(sealed[organ]["age_slope"]) * age_test
                    )
                    max_expected_parity = max(
                        max_expected_parity,
                        float(np.max(np.abs(fitted_expected - sealed_expected))),
                        float(
                            np.max(
                                np.abs(
                                    fitted_expected
                                    - test_organ["expected_age_gap"].to_numpy(dtype=np.float64)
                                )
                            )
                        ),
                    )
                    max_corrected_parity = max(
                        max_corrected_parity,
                        float(
                            np.max(
                                np.abs(
                                    result[form].to_numpy(dtype=np.float64)
                                    - test_organ["corrected_age_gap"].to_numpy(dtype=np.float64)
                                )
                            )
                        ),
                    )
            evaluated_parts.append(result)

    evaluated = pd.concat(evaluated_parts, ignore_index=True)
    if len(evaluated) != EXPECTED_ROWS:
        raise RuntimeError("evaluated row count mismatch")
    if evaluated.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("evaluated participant-organ duplication")

    strict = evaluated.loc[evaluated["strict_healthy_reference_60m"].astype(int).eq(1)].copy()
    summary_rows: list[dict[str, object]] = []
    for organ in ORGANS:
        organ_all = evaluated.loc[evaluated["organ_system"].astype(str).eq(organ)]
        organ_strict = strict.loc[strict["organ_system"].astype(str).eq(organ)]
        for form in ("raw", *FORMS):
            values = organ_strict[form]
            age = organ_strict["actual_age"]
            fixed_bins = pd.cut(age, bins=FIVE_YEAR_EDGES, right=False, include_lowest=True)
            fixed_summary = (
                pd.DataFrame({"age_bin": fixed_bins, "gap": values})
                .groupby("age_bin", observed=True)["gap"]
                .agg(["size", "mean"])
            )
            eligible_bins = fixed_summary.loc[fixed_summary["size"].ge(100)]
            if form == "continuous_linear":
                cross_form_pearson = 1.0
                cross_form_spearman = 1.0
                mean_abs_difference = 0.0
                q95_abs_difference = 0.0
            elif form == "raw":
                cross_form_pearson = np.nan
                cross_form_spearman = np.nan
                mean_abs_difference = np.nan
                q95_abs_difference = np.nan
            else:
                difference = (
                    organ_all[form].to_numpy(dtype=np.float64)
                    - organ_all["continuous_linear"].to_numpy(dtype=np.float64)
                )
                cross_form_pearson = correlation(
                    organ_all[form], organ_all["continuous_linear"]
                )
                cross_form_spearman = correlation(
                    organ_all[form],
                    organ_all["continuous_linear"],
                    method="spearman",
                )
                mean_abs_difference = float(np.mean(np.abs(difference)))
                q95_abs_difference = float(np.quantile(np.abs(difference), 0.95))
            summary_rows.append(
                {
                    "organ_system": organ,
                    "correction_form": form,
                    "strict_reference_rows": int(len(organ_strict)),
                    "mean_gap_years": float(values.mean()),
                    "sd_gap_years": float(values.std(ddof=1)),
                    "gap_age_pearson": correlation(age, values),
                    "gap_age_spearman": correlation(age, values, method="spearman"),
                    "gap_age_linear_slope_years_per_year": slope(age, values),
                    "max_abs_five_year_bin_mean_n_ge_100": float(
                        eligible_bins["mean"].abs().max()
                    ),
                    "all_participant_pearson_vs_primary_linear": cross_form_pearson,
                    "all_participant_spearman_vs_primary_linear": cross_form_spearman,
                    "all_participant_mean_abs_difference_vs_primary_linear_years": mean_abs_difference,
                    "all_participant_q95_abs_difference_vs_primary_linear_years": q95_abs_difference,
                    "age_groups_used_for_fit": False,
                    "model_retraining_performed": False,
                }
            )
    summary = pd.DataFrame(summary_rows)
    parameters = pd.DataFrame(parameter_rows)

    display_rows: list[dict[str, object]] = []
    for scheme in (
        "submitted_three_group_display",
        "shifted_three_group_display",
        "five_year_display",
    ):
        strict[f"_{scheme}"] = age_group_labels(strict["actual_age"], scheme)
        grouped = strict.groupby(
            ["organ_system", f"_{scheme}"], observed=True
        )["continuous_linear"].agg(["size", "mean", "std"])
        for (organ, label), row in grouped.iterrows():
            display_rows.append(
                {
                    "display_scheme": scheme,
                    "organ_system": organ,
                    "age_group_label": str(label),
                    "n_participants": int(row["size"]),
                    "mean_primary_linear_gap_years": float(row["mean"]),
                    "sd_primary_linear_gap_years": float(row["std"]),
                    "age_groups_used_for_correction_fit": False,
                    "correction_values_change_when_display_scheme_changes": False,
                }
            )
    display = pd.DataFrame(display_rows)

    primary = summary.loc[summary["correction_form"].eq("continuous_linear")]
    quadratic = summary.loc[summary["correction_form"].eq("continuous_quadratic")]
    raw = summary.loc[summary["correction_form"].eq("raw")]
    key_results = {
        "participant_n": EXPECTED_PARTICIPANTS,
        "participant_organ_rows": EXPECTED_ROWS,
        "strict_reference_participant_n": int(
            strict["participant_id"].nunique()
        ),
        "primary_method": (
            "fold-local organ-specific continuous linear regression of raw age gap on "
            "chronological age, fitted only in the participant-disjoint calibration-role "
            "strict disease-free reference"
        ),
        "primary_formula": (
            "corrected_gap = (predicted_age - chronological_age) "
            "- (intercept + age_slope * chronological_age)"
        ),
        "age_groups_used_for_primary_fit": False,
        "raw_gap_age_pearson_range": [
            float(raw["gap_age_pearson"].min()),
            float(raw["gap_age_pearson"].max()),
        ],
        "primary_linear_gap_age_pearson_range": [
            float(primary["gap_age_pearson"].min()),
            float(primary["gap_age_pearson"].max()),
        ],
        "primary_linear_gap_age_spearman_range": [
            float(primary["gap_age_spearman"].min()),
            float(primary["gap_age_spearman"].max()),
        ],
        "primary_linear_slope_range_years_per_year": [
            float(primary["gap_age_linear_slope_years_per_year"].min()),
            float(primary["gap_age_linear_slope_years_per_year"].max()),
        ],
        "primary_linear_max_abs_five_year_bin_mean_range": [
            float(primary["max_abs_five_year_bin_mean_n_ge_100"].min()),
            float(primary["max_abs_five_year_bin_mean_n_ge_100"].max()),
        ],
        "quadratic_pearson_vs_primary_linear_range_all_participants": [
            float(quadratic["all_participant_pearson_vs_primary_linear"].min()),
            float(quadratic["all_participant_pearson_vs_primary_linear"].max()),
        ],
        "quadratic_spearman_vs_primary_linear_range_all_participants": [
            float(quadratic["all_participant_spearman_vs_primary_linear"].min()),
            float(quadratic["all_participant_spearman_vs_primary_linear"].max()),
        ],
        "quadratic_gap_age_spearman_range": [
            float(quadratic["gap_age_spearman"].min()),
            float(quadratic["gap_age_spearman"].max()),
        ],
        "quadratic_max_abs_five_year_bin_mean_range": [
            float(quadratic["max_abs_five_year_bin_mean_n_ge_100"].min()),
            float(quadratic["max_abs_five_year_bin_mean_n_ge_100"].max()),
        ],
        "display_scheme_role": (
            "post-correction descriptive QC only; changing display cutpoints does not "
            "refit or alter any participant's corrected gap"
        ),
        "model_retraining_performed": False,
    }

    summary_path = SOURCE_DATA / "correction_form_sensitivity_by_organ.csv"
    parameter_path = SOURCE_DATA / "correction_parameter_lineage.csv"
    display_path = SOURCE_DATA / "age_group_display_definition_qc.csv"
    key_path = SOURCE_DATA / "r3_12_bias_correction_summary.json"
    summary.to_csv(summary_path, index=False)
    parameters.to_csv(parameter_path, index=False)
    display.to_csv(display_path, index=False)
    key_path.write_text(json.dumps(key_results, indent=2), encoding="utf-8")

    qa = {
        "status": "PASS",
        "final_oof_participant_count": int(oof["participant_id"].nunique()) == EXPECTED_PARTICIPANTS,
        "final_oof_row_count": len(oof) == EXPECTED_ROWS,
        "unique_participant_organ_rows": not oof.duplicated(
            ["participant_id", "organ_system"]
        ).any(),
        "raw_gap_formula_parity_max_abs": float(raw_parity),
        "calibration_outer_test_participant_disjoint": bool(disjoint_all),
        "sealed_expected_gap_parity_max_abs": float(max_expected_parity),
        "sealed_corrected_gap_parity_max_abs": float(max_corrected_parity),
        "sealed_linear_parity_pass": bool(
            max_expected_parity < 1e-10 and max_corrected_parity < 1e-10
        ),
        "expected_parameter_rows": len(parameters) == 5 * len(ORGANS) * len(FORMS),
        "expected_summary_rows": len(summary) == len(ORGANS) * (1 + len(FORMS)),
        "all_primary_fits_are_continuous_not_grouped": bool(
            (~summary.loc[summary["correction_form"].eq("continuous_linear"), "age_groups_used_for_fit"]).all()
        ),
        "model_retraining_performed": False,
    }
    if not all(
        value
        for key, value in qa.items()
        if key
        in {
            "final_oof_participant_count",
            "final_oof_row_count",
            "unique_participant_organ_rows",
            "calibration_outer_test_participant_disjoint",
            "sealed_linear_parity_pass",
            "expected_parameter_rows",
            "expected_summary_rows",
            "all_primary_fits_are_continuous_not_grouped",
        }
    ):
        qa["status"] = "FAIL"
    qa_path = OUTPUT_ROOT / "QA_REPORT.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    outputs = [
        file_record(summary_path, "correction_form_sensitivity_by_organ", rows=len(summary)),
        file_record(parameter_path, "correction_parameter_lineage", rows=len(parameters)),
        file_record(display_path, "age_group_display_definition_qc", rows=len(display)),
        file_record(key_path, "r3_12_bias_correction_summary"),
        file_record(qa_path, "qa_report"),
    ]
    manifest = {
        "schema_version": "r3_12.bias_correction_sensitivity.v1",
        "status": "COMPLETE" if qa["status"] == "PASS" else "FAILED_QA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "reviewer_R3_12_bias_correction_definition_and_form_sensitivity",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "contract": {
            "primary_correction": "continuous_linear",
            "sensitivity_correction": "continuous_quadratic",
            "fit_population": "fold-local calibration-role strict disease-free reference",
            "score_population": "participant-disjoint outer test fold",
            "age_groups_used_for_primary_fit": False,
            "model_retraining_performed": False,
        },
        "key_results": key_results,
        "inputs": input_records,
        "script": file_record(SCRIPT_PATH, "r3_12_bias_correction_sensitivity_builder"),
        "outputs": outputs,
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if qa["status"] != "PASS":
        raise RuntimeError("R3.12 QA failed")
    print(json.dumps(key_results, indent=2))
    print(f"QA: {qa['status']}")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
