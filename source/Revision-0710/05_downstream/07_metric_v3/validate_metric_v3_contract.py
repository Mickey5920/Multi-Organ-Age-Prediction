from __future__ import annotations

import hashlib
import json
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = REVISION_ROOT / "00_contract" / "metric_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_entry(entry: dict) -> None:
    path = REVISION_ROOT / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"byte-size mismatch: {path}")
    observed = sha256(path)
    if observed != entry["sha256"]:
        raise RuntimeError(f"sha256 mismatch: {path}")


def validate() -> dict:
    manifest_path = CONTRACT_ROOT / "ARTIFACT_MANIFEST.v3.json"
    manifest = load_json(manifest_path)
    if manifest["schema_version"] != "revision0710.metric.v3.artifact_manifest":
        raise RuntimeError("wrong metric-v3 manifest schema")

    sidecar = (CONTRACT_ROOT / "ARTIFACT_MANIFEST.v3.sha256").read_text(
        encoding="ascii"
    ).strip()
    expected_manifest_hash, expected_name = sidecar.split()
    if expected_name != "ARTIFACT_MANIFEST.v3.json":
        raise RuntimeError("wrong metric-v3 sidecar filename")
    if sha256(manifest_path) != expected_manifest_hash:
        raise RuntimeError("metric-v3 manifest sidecar mismatch")

    for section in ("artifacts", "immutable_upstream", "verification_files"):
        for entry in manifest[section].values():
            _verify_entry(entry)

    criteria = load_json(CONTRACT_ROOT / "SUCCESS_CRITERIA.v3.json")
    if criteria["schema_version"] != "revision0710.metric.v3.success_criteria":
        raise RuntimeError("wrong metric-v3 criteria schema")
    if not criteria["non_retroactivity"]["metric_v2_is_immutable"]:
        raise RuntimeError("metric-v2 immutability is not enforced")

    gates = criteria["stage_a1_development_selection"]["gates"]
    expected_gates = {
        "strict_mae_exclusive_max": 3.6952458145871845,
        "strict_pearson_exclusive_min": 0.794184395549544,
        "strict_r2_exclusive_min": 0.6259793843492366,
        "age_sex_harrell_macro_delta_c_exclusive_min": 0.026913434679319033,
        "positive_delta_c_organs_required": 7,
        "organs_beating_metricwise_baseline_min": 5,
    }
    if gates != expected_gates:
        raise RuntimeError("metric-v3 changed the frozen metric-v2 Stage A1 gates")

    stage_a2 = criteria["stage_a2_single_candidate_confirmation"]
    age = stage_a2["age_point_superiority_gates"]
    if age["strict_mae"]["minimum_inclusive"] != 0.01:
        raise RuntimeError("wrong Stage A2 MAE superiority margin")
    if age["strict_pearson"]["minimum_inclusive"] != 0.001:
        raise RuntimeError("wrong Stage A2 Pearson superiority margin")
    if age["strict_r2"]["minimum_inclusive"] != 0.002:
        raise RuntimeError("wrong Stage A2 R2 superiority margin")
    risk = stage_a2["primary_risk_gate"]
    if risk["point_contrast_minimum_inclusive"] != 0.01:
        raise RuntimeError("wrong full-clinical Uno-C superiority margin")
    if risk["participant_paired_95_ci_lower_bound_exclusive_min"] != 0.0:
        raise RuntimeError("wrong full-clinical Uno-C confidence bound")
    if risk["absolute_delta_c_gate"] is not None:
        raise RuntimeError("an absolute delta-C gate was reintroduced")
    if stage_a2["candidate_input"] != "exactly one hash-frozen Stage A1 winner":
        raise RuntimeError("Stage A2 is not restricted to one frozen candidate")

    d03 = load_json(
        REVISION_ROOT
        / "03_ehr_age"
        / "03_runs"
        / "fold0"
        / "development"
        / "06b1fd87ec50"
        / "development_summary.json"
    )
    if d03["status"] != "NON_PROMOTABLE":
        raise RuntimeError("D03 historical status changed")
    if d03["outer_test_loaded"] or d03["outer_test_predictions_written"]:
        raise RuntimeError("D03 outer-test boundary changed")

    return {
        "schema_version": manifest["schema_version"],
        "status": "VALID",
        "stage_a1_inherited": True,
        "stage_a2_all_four_point_superiority": True,
        "outer_test_used": False,
        "verified_artifacts": sum(len(manifest[x]) for x in (
            "artifacts", "immutable_upstream", "verification_files"
        )),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
