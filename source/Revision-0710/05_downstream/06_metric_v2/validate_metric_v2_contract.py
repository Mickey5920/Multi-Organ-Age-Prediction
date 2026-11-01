#!/usr/bin/env python
"""Validate the isolated Revision-0710 metric-v2 contract and SHA lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


SUCCESS_SCHEMA = "revision0710.metric.v2.success_criteria"
ENVELOPE_SCHEMA = "revision0710.metric.v2.fold0_development_baseline_envelope"
MANIFEST_SCHEMA = "revision0710.metric.v2.artifact_manifest"
EXPECTED_METHODS = (
    "retain",
    "behrt_style",
    "medbert_style",
    "retain_organquery",
    "behrt_style_organquery",
    "medbert_style_organquery",
)
EXPECTED_ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
EXPECTED_GATES = {
    "strict_mae": {
        "direction": "lower",
        "operator": "<",
        "baseline": Decimal("3.7052458145871845"),
        "margin": Decimal("0.01"),
        "threshold": Decimal("3.6952458145871845"),
    },
    "strict_pearson": {
        "direction": "higher",
        "operator": ">",
        "baseline": Decimal("0.793184395549544"),
        "margin": Decimal("0.001"),
        "threshold": Decimal("0.794184395549544"),
    },
    "strict_r2": {
        "direction": "higher",
        "operator": ">",
        "baseline": Decimal("0.6239793843492366"),
        "margin": Decimal("0.002"),
        "threshold": Decimal("0.6259793843492366"),
    },
    "age_sex_harrell_full_risk_macro_delta_c_index": {
        "direction": "higher",
        "operator": ">",
        "baseline": Decimal("0.016913434679319033"),
        "margin": Decimal("0.01"),
        "threshold": Decimal("0.026913434679319033"),
    },
}


class MetricV2ContractError(ValueError):
    """Raised when a v2 artifact violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise MetricV2ContractError(f"non-finite JSON constant is forbidden: {value}")


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=Decimal,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricV2ContractError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise MetricV2ContractError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetricV2ContractError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MetricV2ContractError(f"{label} must be a list")
    return value


def _require_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise MetricV2ContractError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise MetricV2ContractError(f"{label} must be a finite decimal value")
    if isinstance(value, float) and not math.isfinite(value):
        raise MetricV2ContractError(f"{label} must be finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise MetricV2ContractError(f"{label} must be a decimal value") from exc
    if not result.is_finite():
        raise MetricV2ContractError(f"{label} must be finite")
    return result


def _validate_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise MetricV2ContractError(f"{label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise MetricV2ContractError(f"{label} must be a SHA-256 string")
    return normalized


def _resolve_revision_path(revision_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MetricV2ContractError(f"{label} must be a non-empty relative path")
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise MetricV2ContractError(f"{label} must stay inside the Revision-0710 root")
    root = revision_root.resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MetricV2ContractError(f"{label} escapes the Revision-0710 root") from exc
    if not resolved.is_file():
        raise MetricV2ContractError(f"{label} does not exist: {resolved}")
    return resolved


def _validate_gate(metric: str, value: object, *, label: str) -> None:
    gate = _mapping(value, label=label)
    expected = EXPECTED_GATES[metric]
    _require_equal(gate.get("direction"), expected["direction"], label=f"{label}.direction")
    _require_equal(
        gate.get("comparison_operator"), expected["operator"], label=f"{label}.comparison_operator"
    )
    for key, expected_key in (
        ("baseline_value", "baseline"),
        ("required_margin", "margin"),
        ("threshold_exclusive", "threshold"),
        ("threshold_exclusive_decimal", "threshold"),
    ):
        actual = _decimal(gate.get(key), label=f"{label}.{key}")
        _require_equal(actual, expected[expected_key], label=f"{label}.{key}")
    arithmetic = (
        expected["baseline"] - expected["margin"]
        if expected["direction"] == "lower"
        else expected["baseline"] + expected["margin"]
    )
    _require_equal(arithmetic, expected["threshold"], label=f"{label}.threshold_arithmetic")
    if metric == "strict_mae":
        _require_equal(
            _decimal(gate.get("absolute_safety_ceiling_exclusive"), label=f"{label}.absolute_ceiling"),
            Decimal("4.35"),
            label=f"{label}.absolute_ceiling",
        )
    if metric == "age_sex_harrell_full_risk_macro_delta_c_index":
        if gate.get("absolute_gate", "missing") is not None:
            raise MetricV2ContractError(f"{label}.absolute_gate must be null")
        _require_equal(gate.get("absolute_gate_policy"), "none", label=f"{label}.absolute_gate_policy")


def validate_envelope_payload(payload: Mapping[str, Any]) -> None:
    _require_equal(payload.get("schema_version"), ENVELOPE_SCHEMA, label="envelope.schema_version")
    _require_equal(payload.get("status"), "frozen", label="envelope.status")
    scope = _mapping(payload.get("scope"), label="envelope.scope")
    _require_equal(scope.get("outer_fold"), 0, label="envelope.scope.outer_fold")
    _require_equal(scope.get("source_split"), "validation", label="envelope.scope.source_split")
    _require_equal(scope.get("fit_role"), "calibration", label="envelope.scope.fit_role")
    _require_equal(scope.get("score_role"), "screen", label="envelope.scope.score_role")
    _require_equal(scope.get("outer_test_used"), False, label="envelope.scope.outer_test_used")
    _require_equal(scope.get("risk_model_base"), "chronological_age_and_sex", label="risk_model_base")
    _require_equal(scope.get("risk_discrimination_estimator"), "harrell_c_index", label="estimator")
    methods = tuple(_sequence(payload.get("methods"), label="envelope.methods"))
    _require_equal(methods, EXPECTED_METHODS, label="envelope.methods")

    gates = _mapping(payload.get("development_metric_envelope"), label="development_metric_envelope")
    _require_equal(set(gates), set(EXPECTED_GATES), label="development_metric_envelope.keys")
    for metric in EXPECTED_GATES:
        _validate_gate(metric, gates[metric], label=f"development_metric_envelope.{metric}")

    organs = _sequence(payload.get("organ_delta_c_envelope"), label="organ_delta_c_envelope")
    organ_names = tuple(_mapping(row, label="organ row").get("organ_system") for row in organs)
    _require_equal(organ_names, EXPECTED_ORGANS, label="organ_delta_c_envelope.organ_system")
    requirements = _mapping(
        payload.get("ehr_age_development_requirements"), label="ehr_age_development_requirements"
    )
    _require_equal(requirements.get("all_four_metric_gates_required"), True, label="all_four_gates")
    _require_equal(requirements.get("positive_organs_required"), 7, label="positive_organs_required")
    _require_equal(
        requirements.get("organs_beating_metricwise_baseline_required"),
        5,
        label="organs_beating_metricwise_baseline_required",
    )
    _require_equal(
        requirements.get("bootstrap_allowed_for_epoch_ranking"),
        False,
        label="bootstrap_allowed_for_epoch_ranking",
    )
    boundary = _mapping(payload.get("confirmation_boundary"), label="confirmation_boundary")
    _require_equal(
        boundary.get("confirmation_metrics_computed_in_this_artifact"),
        False,
        label="confirmation_metrics_computed_in_this_artifact",
    )
    expected_future = {
        "full_clinical_uno_c",
        "time_dependent_auc",
        "ipcw_brier_and_ibs",
        "calibration",
        "decision_curve_analysis",
    }
    _require_equal(
        set(_sequence(boundary.get("required_future_metrics"), label="required_future_metrics")),
        expected_future,
        label="required_future_metrics",
    )


def validate_success_payload(payload: Mapping[str, Any]) -> None:
    _require_equal(payload.get("schema_version"), SUCCESS_SCHEMA, label="success.schema_version")
    _require_equal(payload.get("status"), "frozen_development_contract", label="success.status")
    history = _mapping(payload.get("historical_v1_boundary"), label="historical_v1_boundary")
    _require_equal(
        _mapping(history.get("success_criteria"), label="historical success").get("immutable"),
        True,
        label="historical success immutable",
    )
    d01 = _mapping(history.get("d01_development_summary"), label="historical D01")
    _require_equal(d01.get("immutable"), True, label="historical D01 immutable")
    _require_equal(d01.get("historical_status"), "NON_PROMOTABLE", label="historical D01 status")

    integrity = _mapping(payload.get("integrity_requirements"), label="integrity_requirements")
    _require_equal(integrity.get("outer_test_used"), False, label="integrity.outer_test_used")
    _require_equal(integrity.get("development_source_split"), "validation", label="development split")
    _require_equal(integrity.get("participant_disjoint_fit_and_score"), True, label="disjoint roles")
    _require_equal(integrity.get("fine_tuning_allowed"), False, label="fine_tuning_allowed")
    _require_equal(integrity.get("distillation_allowed"), False, label="distillation_allowed")

    screen = _mapping(payload.get("stage_a1_deterministic_epoch_screen"), label="stage_a1")
    _require_equal(screen.get("all_gates_required"), True, label="stage_a1.all_gates_required")
    gates = _mapping(screen.get("gates"), label="stage_a1.gates")
    _require_equal(set(gates), set(EXPECTED_GATES), label="stage_a1.gates.keys")
    for metric in EXPECTED_GATES:
        _validate_gate(metric, gates[metric], label=f"stage_a1.gates.{metric}")

    organs = _mapping(screen.get("organ_requirements"), label="stage_a1.organ_requirements")
    positive = _mapping(organs.get("positive_delta_c_organs"), label="positive_delta_c_organs")
    beating = _mapping(
        organs.get("organs_beating_metricwise_baseline"), label="organs_beating_metricwise_baseline"
    )
    _require_equal((positive.get("comparison_operator"), positive.get("required"), positive.get("denominator")),
                   ("==", 7, 7), label="positive organ rule")
    _require_equal((beating.get("comparison_operator"), beating.get("required"), beating.get("denominator")),
                   (">=", 5, 7), label="organs beating rule")
    ranking = _mapping(screen.get("candidate_ranking"), label="candidate_ranking")
    _require_equal(ranking.get("candidate_count_to_freeze"), 1, label="candidate_count_to_freeze")
    _require_equal(ranking.get("bootstrap_used_for_epoch_ranking"), False, label="bootstrap ranking")
    _require_equal(
        ranking.get("confirmation_metrics_used_for_epoch_ranking"), False, label="confirmation ranking"
    )

    confirmation = _mapping(payload.get("stage_a2_single_candidate_confirmation"), label="stage_a2")
    bootstrap = _mapping(confirmation.get("paired_bootstrap"), label="stage_a2.paired_bootstrap")
    _require_equal(bootstrap.get("unit"), "participant", label="bootstrap.unit")
    _require_equal(bootstrap.get("replicates"), 2000, label="bootstrap.replicates")
    _require_equal(bootstrap.get("used_for_epoch_ranking"), False, label="bootstrap.used_for_epoch_ranking")
    _require_equal(bootstrap.get("run_for_every_epoch"), False, label="bootstrap.run_for_every_epoch")
    _require_equal(bootstrap.get("run_after_candidate_hash_is_frozen"), True, label="bootstrap.freeze_first")
    _require_equal(
        bootstrap.get("failure_allows_fallback_to_next_epoch"), False, label="bootstrap.no_fallback"
    )
    primary = _mapping(confirmation.get("primary_confirmation_metric"), label="primary_confirmation_metric")
    _require_equal(
        primary.get("name"),
        "60_month_full_clinical_uno_c_incremental_gain_contrast",
        label="primary confirmation name",
    )
    _require_equal(
        _decimal(primary.get("point_contrast_min_inclusive"), label="Uno point margin"),
        Decimal("0.01"),
        label="Uno point margin",
    )
    _require_equal(
        primary.get("paired_95_ci_lower_bound_comparison"), ">0", label="Uno paired CI rule"
    )
    secondary = _mapping(confirmation.get("key_secondary_metrics"), label="key_secondary_metrics")
    _require_equal(
        set(secondary),
        {"time_dependent_auc", "ipcw_brier_and_ibs", "calibration", "decision_curve_analysis"},
        label="key_secondary_metrics.keys",
    )
    for name, metric in secondary.items():
        _require_equal(
            _mapping(metric, label=f"key_secondary_metrics.{name}").get("used_for_epoch_ranking"),
            False,
            label=f"key_secondary_metrics.{name}.used_for_epoch_ranking",
        )
    failure = _mapping(confirmation.get("failure_policy"), label="failure_policy")
    _require_equal(failure.get("automatic_next_epoch_fallback"), False, label="failure no fallback")

    outer = _mapping(payload.get("outer_evaluation_release"), label="outer_evaluation_release")
    _require_equal(outer.get("allowed_before_stage_a1_candidate_freeze"), False, label="outer before freeze")
    _require_equal(
        outer.get("allowed_before_stage_a2_decision_record"), False, label="outer before confirmation record"
    )


def _validate_file_reference(
    revision_root: Path, reference: object, *, label: str, validate_schema: bool = True
) -> tuple[Path, dict[str, Any] | None]:
    ref = _mapping(reference, label=label)
    path = _resolve_revision_path(revision_root, ref.get("path"), label=f"{label}.path")
    expected_hash = _validate_sha(ref.get("sha256"), label=f"{label}.sha256")
    _require_equal(sha256_file(path), expected_hash, label=f"{label}.sha256")
    if "bytes" in ref:
        _require_equal(path.stat().st_size, ref.get("bytes"), label=f"{label}.bytes")
    payload: dict[str, Any] | None = None
    if validate_schema and "schema_version" in ref:
        payload = read_json(path, label=label)
        _require_equal(payload.get("schema_version"), ref.get("schema_version"), label=f"{label}.schema")
    return path, payload


def _reference_identity(reference: object, *, label: str) -> tuple[str, str]:
    ref = _mapping(reference, label=label)
    path = ref.get("path")
    if not isinstance(path, str):
        raise MetricV2ContractError(f"{label}.path must be a string")
    return path, _validate_sha(ref.get("sha256"), label=f"{label}.sha256")


def validate_contract(revision_root: Path, manifest_path: Path) -> dict[str, Any]:
    revision_root = revision_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path, label="manifest")
    _require_equal(manifest.get("schema_version"), MANIFEST_SCHEMA, label="manifest.schema_version")
    _require_equal(manifest.get("status"), "frozen", label="manifest.status")

    artifacts = _mapping(manifest.get("artifacts"), label="manifest.artifacts")
    _require_equal(
        set(artifacts),
        {"success_criteria_v2", "fold0_development_baseline_envelope_v2"},
        label="manifest.artifacts.keys",
    )
    _, success = _validate_file_reference(
        revision_root, artifacts["success_criteria_v2"], label="success_criteria_v2"
    )
    _, envelope = _validate_file_reference(
        revision_root,
        artifacts["fold0_development_baseline_envelope_v2"],
        label="fold0_development_baseline_envelope_v2",
    )
    assert success is not None and envelope is not None
    validate_success_payload(success)
    validate_envelope_payload(envelope)

    upstream = _mapping(manifest.get("immutable_upstream"), label="manifest.immutable_upstream")
    required_upstream = {
        "historical_success_criteria_v1",
        "historical_d01_development_summary",
        "historical_fold0_development_envelope_v1",
        "historical_fold0_development_registry_v1",
        "revised_metric_plan_after_d01",
    }
    _require_equal(set(upstream), required_upstream, label="manifest.immutable_upstream.keys")
    for name, reference in upstream.items():
        ref = _mapping(reference, label=f"immutable_upstream.{name}")
        _require_equal(ref.get("immutable"), True, label=f"immutable_upstream.{name}.immutable")
        _validate_file_reference(
            revision_root, ref, label=f"immutable_upstream.{name}", validate_schema=False
        )

    verification = _mapping(manifest.get("verification_files"), label="manifest.verification_files")
    _require_equal(set(verification), {"validator", "tests"}, label="manifest.verification_files.keys")
    for name, reference in verification.items():
        _validate_file_reference(
            revision_root, reference, label=f"verification_files.{name}", validate_schema=False
        )

    success_envelope_ref = _mapping(
        success.get("fold0_development_baseline_envelope"), label="success envelope reference"
    )
    _require_equal(
        success_envelope_ref.get("sha256"),
        _mapping(artifacts["fold0_development_baseline_envelope_v2"], label="manifest envelope").get(
            "sha256"
        ),
        label="success-to-envelope hash lineage",
    )
    success_history = _mapping(success.get("historical_v1_boundary"), label="success history")
    envelope_lineage = _mapping(envelope.get("source_lineage"), label="envelope source_lineage")
    internal_to_manifest = (
        (
            success_history.get("success_criteria"),
            upstream["historical_success_criteria_v1"],
            "success historical criteria lineage",
        ),
        (
            success_history.get("d01_development_summary"),
            upstream["historical_d01_development_summary"],
            "success D01 lineage",
        ),
        (
            success.get("rationale_lineage"),
            upstream["revised_metric_plan_after_d01"],
            "success rationale lineage",
        ),
        (
            envelope_lineage.get("historical_v1_envelope"),
            upstream["historical_fold0_development_envelope_v1"],
            "envelope v1 lineage",
        ),
        (
            envelope_lineage.get("historical_v1_registry"),
            upstream["historical_fold0_development_registry_v1"],
            "envelope registry lineage",
        ),
    )
    for internal_ref, manifest_ref, label in internal_to_manifest:
        _require_equal(
            _reference_identity(internal_ref, label=f"{label}.internal"),
            _reference_identity(manifest_ref, label=f"{label}.manifest"),
            label=label,
        )

    edges = _sequence(manifest.get("lineage_edges"), label="manifest.lineage_edges")
    observed_edges = {
        (
            _mapping(edge, label="lineage edge").get("from"),
            _mapping(edge, label="lineage edge").get("to"),
            _mapping(edge, label="lineage edge").get("relation"),
        )
        for edge in edges
    }
    expected_edges = {
        (
            "historical_fold0_development_envelope_v1",
            "fold0_development_baseline_envelope_v2",
            "derive_relative_thresholds_without_overwrite",
        ),
        (
            "historical_fold0_development_registry_v1",
            "fold0_development_baseline_envelope_v2",
            "freeze_six_method_source_registry",
        ),
        (
            "revised_metric_plan_after_d01",
            "success_criteria_v2",
            "adopt_metric_v2_policy",
        ),
        (
            "historical_success_criteria_v1",
            "success_criteria_v2",
            "supersede_without_mutation",
        ),
        (
            "historical_d01_development_summary",
            "success_criteria_v2",
            "preserve_non_retroactive_boundary",
        ),
        (
            "fold0_development_baseline_envelope_v2",
            "success_criteria_v2",
            "supply_development_point_gates",
        ),
    }
    _require_equal(observed_edges, expected_edges, label="manifest.lineage_edges")
    invariants = _mapping(manifest.get("invariants"), label="manifest.invariants")
    for key in (
        "v1_artifacts_are_read_only",
        "outer_test_used",
        "bootstrap_used_for_epoch_ranking",
        "legacy_absolute_downstream_gate_enabled",
    ):
        expected = key == "v1_artifacts_are_read_only"
        _require_equal(invariants.get(key), expected, label=f"manifest.invariants.{key}")

    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise MetricV2ContractError(f"manifest SHA sidecar is missing: {sidecar}")
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    if len(tokens) != 2 or tokens[1] != manifest_path.name:
        raise MetricV2ContractError("manifest SHA sidecar must use '<sha256>  <filename>' format")
    _require_equal(_validate_sha(tokens[0], label="manifest sidecar SHA"), sha256_file(manifest_path),
                   label="manifest sidecar SHA")

    return {
        "status": "VALID",
        "schema_version": MANIFEST_SCHEMA,
        "verified_v2_artifacts": len(artifacts),
        "verified_immutable_upstream": len(upstream),
        "verified_verification_files": len(verification),
        "outer_test_used": False,
        "development_gate_count": len(EXPECTED_GATES),
    }


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "00_contract" / "metric_v2" / "ARTIFACT_MANIFEST.v2.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_contract(args.revision_root, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
