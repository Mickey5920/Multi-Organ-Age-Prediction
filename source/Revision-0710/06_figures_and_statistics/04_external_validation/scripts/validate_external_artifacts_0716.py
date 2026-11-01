from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
FINAL = ROOT / "Revision-0710/Final_Fig_0716"
FIG13 = FINAL / "Fig13"
FIG14 = FINAL / "Fig14"
WORK = ROOT / "Revision-0710/06_figures_and_statistics/04_external_validation/core_0716"
ORGANS = {
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def load_csv(path: Path, checks: list[dict[str, Any]], allow_empty: bool = False) -> pd.DataFrame:
    require(path.exists(), f"required CSV exists: {path.relative_to(ROOT)}", checks)
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if not allow_empty:
            raise
        frame = pd.DataFrame()
    if allow_empty:
        require(True, f"optional-empty CSV parsed: {path.name}", checks)
    else:
        require(len(frame) > 0, f"required CSV has rows: {path.name}", checks)
    return frame


def check_organs(frame: pd.DataFrame, name: str, checks: list[dict[str, Any]]) -> None:
    observed = set(frame["organ_system"].dropna().astype(str).unique())
    require(observed == ORGANS, f"seven-organ coverage: {name}", checks)


def check_main_png(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    require(path.exists(), f"main PNG exists: {path.relative_to(ROOT)}", checks)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        width, height = image.size
    gray = rgb.mean(axis=2)
    nonwhite_fraction = float(np.mean(gray < 250))
    require(width >= 4000 and height >= 2500, f"publication PNG dimensions: {path.name}", checks)
    require(float(gray.std()) > 8.0, f"nonblank PNG variance: {path.name}", checks)
    require(nonwhite_fraction > 0.02, f"nonblank PNG ink fraction: {path.name}", checks)
    return {
        "path": str(path.relative_to(ROOT)),
        "width": width,
        "height": height,
        "nonwhite_fraction": nonwhite_fraction,
        "gray_std": float(gray.std()),
    }


def validate_sources(checks: list[dict[str, Any]]) -> dict[str, Any]:
    source13 = FIG13 / "Source_Data"
    source14 = FIG14 / "Source_Data"

    age13 = load_csv(source13 / "Fig13B_raw_age_metrics_checkpoint_range.csv", checks)
    risk13 = load_csv(source13 / "Fig13D_paired_incremental_risk_metrics.csv", checks)
    hr13 = load_csv(source13 / "Fig13E_cause_specific_hr.csv", checks)
    calibration13 = load_csv(source13 / "Fig13F_risk_calibration_deciles.csv", checks)
    failures13 = load_csv(
        source13 / "MIMIC_risk_model_failures.csv", checks, allow_empty=True
    )
    check_organs(age13, "Fig13 age metrics", checks)
    check_organs(risk13, "Fig13 paired risk", checks)
    check_organs(hr13, "Fig13 cause-specific HR", checks)
    require(len(age13) == 42, "Fig13 age metrics: 7 organs x 6 metrics", checks)
    require(len(risk13) == 84, "Fig13 risk: 7 organs x 2 horizons x 2 comparisons x 3 metrics", checks)
    require(len(hr13) == 14, "Fig13 HR: 7 organs x 2 horizons", checks)
    require(len(failures13) == 0, "MIMIC OOF risk models have zero failures", checks)
    require(
        set(risk13["bootstrap_replicates"].unique()) == {200},
        "Fig13 paired risk uses 200 bootstrap replicates",
        checks,
    )
    require(
        set(calibration13["risk_decile"].unique()) == set(range(1, 11)),
        "Fig13 risk calibration contains all deciles",
        checks,
    )

    flow14 = load_csv(source14 / "Fig14A_release_cohort_flow.csv", checks)
    qc14 = load_csv(source14 / "Fig14B_release_slot_qc.csv", checks)
    age14 = load_csv(source14 / "Fig14C_raw_age_metrics_checkpoint_range.csv", checks)
    recal14 = load_csv(source14 / "Fig14E_local_crossfit_recalibration.csv", checks)
    or14 = load_csv(source14 / "Fig14F_questionnaire_or.csv", checks)
    check_organs(age14, "Fig14 age metrics", checks)
    check_organs(recal14, "Fig14 local recalibration", checks)
    check_organs(or14, "Fig14 questionnaire OR", checks)
    require(len(flow14) == 2, "Fig14 keeps P and L releases separate", checks)
    require(len(qc14) == 156, "Fig14 slot QC: 2 releases x 78 slots", checks)
    require(len(age14) == 84, "Fig14 age metrics: 2 releases x 7 organs x 6 metrics", checks)
    require(len(or14) == 14, "Fig14 OR: 2 releases x 7 organs", checks)
    require(
        int(or14["estimability_status"].eq("not_estimable_by_design").sum()) == 2,
        "Fig14 records two release-specific immune outcomes as not estimable by design",
        checks,
    )
    require(
        int(qc14["qc_status"].eq("pass").sum()) == 105
        and int(qc14["qc_status"].eq("not_measured").sum()) == 51,
        "Fig14 slot QC has 105 pass and 51 not-measured rows",
        checks,
    )
    return {
        "fig13": {
            "age_metric_rows": len(age13),
            "paired_risk_rows": len(risk13),
            "hr_rows": len(hr13),
            "risk_calibration_rows": len(calibration13),
            "risk_model_failures": len(failures13),
        },
        "fig14": {
            "release_rows": len(flow14),
            "slot_qc_rows": len(qc14),
            "age_metric_rows": len(age14),
            "local_recalibration_rows": len(recal14),
            "questionnaire_or_rows": len(or14),
        },
    }


def build_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for figure in (FIG13, FIG14):
        for path in sorted(
            item
            for item in figure.rglob("*")
            if item.is_file() and item.name != "MANIFEST.json"
        ):
            row: dict[str, Any] = {
                "figure": figure.name,
                "relative_path": str(path.relative_to(FINAL)),
                "extension": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "csv_rows": "",
                "image_width": "",
                "image_height": "",
            }
            if path.suffix.lower() == ".csv":
                try:
                    row["csv_rows"] = len(pd.read_csv(path))
                except pd.errors.EmptyDataError:
                    row["csv_rows"] = 0
            elif path.suffix.lower() == ".png":
                with Image.open(path) as image:
                    row["image_width"], row["image_height"] = image.size
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    checks: list[dict[str, Any]] = []
    source_summary = validate_sources(checks)
    images = [
        check_main_png(
            FIG13 / "Main_Figure/Fig13_MIMIC_external_validation.png", checks
        ),
        check_main_png(
            FIG14 / "Main_Figure/Fig14_NHANES_external_validation.png", checks
        ),
    ]
    for figure, stem in (
        (FIG13, "Fig13_MIMIC_external_validation"),
        (FIG14, "Fig14_NHANES_external_validation"),
    ):
        for extension in ("png", "pdf", "svg"):
            require(
                (figure / "Main_Figure" / f"{stem}.{extension}").exists(),
                f"main figure format exists: {stem}.{extension}",
                checks,
            )
        for panel in "ABCDEF":
            for extension in ("png", "pdf", "svg"):
                require(
                    (figure / "Panels" / f"{figure.name}{panel}.{extension}").exists(),
                    f"panel format exists: {figure.name}{panel}.{extension}",
                    checks,
                )

    inventory = build_inventory()
    inventory_path = FINAL / "EXTERNAL_VALIDATION_FILE_INVENTORY.csv"
    atomic_csv(inventory_path, inventory)

    receipts = {
        "prediction_seal": json.loads(
            (WORK / "shared/04_inference_work/PREDICTION_SEAL_RECEIPT.json").read_text(
                encoding="utf-8"
            )
        ),
        "outcome_read": json.loads(
            (WORK / "shared/05_outcomes_after_prediction_seal/OUTCOME_READ_RECEIPT.json").read_text(
                encoding="utf-8"
            )
        ),
        "basic_statistics": json.loads(
            (WORK / "shared/06_statistics/BASIC_STATISTICS_RECEIPT.json").read_text(
                encoding="utf-8"
            )
        ),
        "risk_statistics": json.loads(
            (WORK / "shared/06_statistics/RISK_STATISTICS_RECEIPT.json").read_text(
                encoding="utf-8"
            )
        ),
        "render": json.loads(
            (WORK / "shared/07_render/RENDER_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    manifest = {
        "schema_version": "revision0710.external_validation.final_artifacts.v1",
        "status": "PASS",
        "created_at": now_iso(),
        "figures": ["Fig13", "Fig14"],
        "source_summary": source_summary,
        "image_qc": images,
        "checks": checks,
        "inventory": str(inventory_path.relative_to(ROOT)),
        "inventory_rows": len(inventory),
        "receipt_status": {
            key: value.get("status", "") for key, value in receipts.items()
        },
        "receipt_sha256": {
            "prediction_seal": sha256_file(
                WORK / "shared/04_inference_work/PREDICTION_SEAL_RECEIPT.json"
            ),
            "outcome_read": sha256_file(
                WORK / "shared/05_outcomes_after_prediction_seal/OUTCOME_READ_RECEIPT.json"
            ),
            "basic_statistics": sha256_file(
                WORK / "shared/06_statistics/BASIC_STATISTICS_RECEIPT.json"
            ),
            "risk_statistics": sha256_file(
                WORK / "shared/06_statistics/RISK_STATISTICS_RECEIPT.json"
            ),
            "render": sha256_file(WORK / "shared/07_render/RENDER_MANIFEST.json"),
        },
    }
    manifest_path = FINAL / "EXTERNAL_VALIDATION_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    for figure in (FIG13, FIG14):
        figure_manifest = {
            **manifest,
            "figure": figure.name,
            "inventory_rows": int(inventory["figure"].eq(figure.name).sum()),
            "files": inventory.loc[inventory["figure"].eq(figure.name)].to_dict(
                orient="records"
            ),
        }
        atomic_json(figure / "MANIFEST.json", figure_manifest)
    print(json.dumps({"status": "PASS", "checks": len(checks), "inventory_rows": len(inventory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
