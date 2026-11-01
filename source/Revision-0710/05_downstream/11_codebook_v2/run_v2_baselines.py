from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
REVISION = DOWNSTREAM.parent
EVALUATOR = DOWNSTREAM / "scripts" / "evaluate_method_5fold.py"
SUMMARIZER = DOWNSTREAM / "scripts" / "summarize_baseline_envelope.py"
OUTCOMES = REVISION / "01_data_rebuild" / "09_v2_labels" / "organ_outcomes_60m.parquet"
MASKS = (
    REVISION
    / "01_data_rebuild"
    / "07_corrected_primary_v1"
    / "02_outcomes_60m"
    / "participant_followup_and_evaluation_masks.parquet"
)
FOLD_ROLES = REVISION / "01_data_rebuild" / "03_folds" / "outer_fold_roles.parquet"
REGISTRY_ROOT = DOWNSTREAM / "01_prediction_registries"
METHODS = (
    ("retain", "tier_a", REGISTRY_ROOT / "tier_a" / "retain.json"),
    ("behrt_style", "tier_a", REGISTRY_ROOT / "tier_a" / "behrt_style.json"),
    ("medbert_style", "tier_a", REGISTRY_ROOT / "tier_a" / "medbert_style.json"),
    ("ehrformer_style", "tier_a", REGISTRY_ROOT / "tier_a" / "ehrformer_style.json"),
    (
        "retain_organquery",
        "tier_b",
        REGISTRY_ROOT / "tier_b" / "retain_organquery.json",
    ),
    (
        "behrt_style_organquery",
        "tier_b",
        REGISTRY_ROOT / "tier_b" / "behrt_style_organquery.json",
    ),
    (
        "medbert_style_organquery",
        "tier_b",
        REGISTRY_ROOT / "tier_b" / "medbert_style_organquery.json",
    ),
    (
        "ehrformer_style_organquery",
        "tier_b",
        REGISTRY_ROOT / "tier_b" / "ehrformer_style_organquery.json",
    ),
)
EVALUATION_ROOT = HERE / "01_method_results"
SUMMARY_ROOT = HERE / "02_summary"


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [EVALUATOR, SUMMARIZER, OUTCOMES, MASKS, FOLD_ROLES]
    required.extend(path for _, _, path in METHODS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v2 baseline inputs: {missing}")

    entries = []
    for method, tier, registry in METHODS:
        output = EVALUATION_ROOT / method
        if not (output / "evaluation_summary.json").is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--prediction-registry",
                    str(registry),
                    "--method",
                    method,
                    "--participant-masks",
                    str(MASKS),
                    "--organ-outcomes",
                    str(OUTCOMES),
                    "--fold-roles",
                    str(FOLD_ROLES),
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
        summary_path = output / "evaluation_summary.json"
        entries.append(
            {
                "method": method,
                "tier": tier,
                "summary_path": str(
                    Path("../01_method_results") / method / "evaluation_summary.json"
                ),
                "sha256": sha256_file(summary_path),
            }
        )

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    registry_path = SUMMARY_ROOT / "baseline_evaluation_registry.json"
    write_json(
        registry_path,
        {
            "schema_version": "revision0710_baseline_evaluation_registry_v1",
            "methods": entries,
        },
    )
    summary_table = SUMMARY_ROOT / "baseline_method_table.json"
    rebuild_summary = True
    if summary_table.is_file():
        current = json.loads(summary_table.read_text(encoding="utf-8"))
        current_methods = current.get("methods", []) if isinstance(current, dict) else []
        rebuild_summary = {
            row.get("method") for row in current_methods if isinstance(row, dict)
        } != {method for method, _, _ in METHODS}
    if rebuild_summary:
        subprocess.run(
            [
                sys.executable,
                str(SUMMARIZER),
                "--registry",
                str(registry_path),
                "--output-dir",
                str(SUMMARY_ROOT),
                "--overwrite",
            ],
            check=True,
        )

    methods = json.loads(
        (SUMMARY_ROOT / "baseline_method_table.json").read_text(encoding="utf-8")
    )["methods"]
    best = {
        "strict_mae": min(methods, key=lambda row: float(row["strict_mae"])),
        "strict_pearson": max(methods, key=lambda row: float(row["strict_pearson"])),
        "strict_r2": max(methods, key=lambda row: float(row["strict_r2"])),
        "full_risk_macro_delta_c": max(
            methods, key=lambda row: float(row["full_risk_macro_delta_c"])
        ),
    }
    targets = {
        "status": "PASS",
        "definition": "metric-wise best of eight fixed five-fold baseline predictions on codebook-v2 outcomes",
        "best": {
            metric: {
                "method": row["method"],
                "value": row[metric],
            }
            for metric, row in best.items()
        },
    }
    write_json(HERE / "baseline_targets.json", targets)
    print(json.dumps(targets, indent=2), flush=True)


if __name__ == "__main__":
    main()
