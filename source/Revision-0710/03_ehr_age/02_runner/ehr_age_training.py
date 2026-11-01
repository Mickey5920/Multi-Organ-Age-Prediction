from __future__ import annotations

import json
import hashlib
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from ehr_age_data import (
    BASELINE_COMMON,
    MODEL_DIR,
    REVISION_ROOT,
    Fold0DevelopmentData,
    JointBatch,
    JointVectorizedBatcher,
    canonical_hash,
    compact_event_dropout,
    load_fold0_development_data,
    sha256_file,
    verify_pass_gate,
)

for _path in (MODEL_DIR, BASELINE_COMMON):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ehr_age_losses import AgePreservingLossConfig, ehr_age_joint_loss  # noqa: E402
from ehr_age_model import EHRAGEConfig, EHRAGEModel, ORGAN_SYSTEMS  # noqa: E402

DOWNSTREAM_COMMON = REVISION_ROOT / "05_downstream" / "common"
if str(DOWNSTREAM_COMMON) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_COMMON))
from crossfit_metrics import (  # noqa: E402
    apply_linear_age_correction,
    apply_linear_age_correction_to_validation,
    evaluate_cox_cell,
    fit_linear_age_correction,
)

METRIC_V2_DIR = REVISION_ROOT / "05_downstream" / "06_metric_v2"
if str(METRIC_V2_DIR) not in sys.path:
    sys.path.insert(0, str(METRIC_V2_DIR))
from validate_metric_v2_contract import validate_contract as validate_metric_v2_contract  # noqa: E402


BASELINE_ENVELOPE_SCHEMA_V1 = "revision0710_fold0_development_baseline_envelope_v1"
BASELINE_ENVELOPE_SCHEMA_V2 = "revision0710.metric.v2.fold0_development_baseline_envelope"
# Backward-compatible import name used by the original runner tests and tooling.
BASELINE_ENVELOPE_SCHEMA = BASELINE_ENVELOPE_SCHEMA_V1
METRIC_CONTRACT_V1 = "v1"
METRIC_CONTRACT_V2 = "metric_v2"
METRIC_V2_MANIFEST = REVISION_ROOT / "00_contract" / "metric_v2" / "ARTIFACT_MANIFEST.v2.json"
FREEZE_SCHEMA = "revision0710_ehr_age_fold0_freeze_v1"
FINALIZE_CONFIRMATION = "READ_OUTER_FOLD0_ONCE"
FREEZE_CONFIRMATION = "FREEZE_SELECTED_CONFIGURATION"


@dataclass(frozen=True)
class SelectionGates:
    strict_mae_max: float
    strict_pearson_min: float
    strict_r2_min: float
    delta_c_min: float = -math.inf
    positive_organs_min: int = 0
    organs_beating_baseline_min: int = 0
    baseline_organ_delta_c: Mapping[str, float] | None = None
    threshold_decimals: Mapping[str, str] | None = None
    source: str = "unspecified"
    baseline_artifact_hashes: Mapping[str, Any] | None = None
    metric_contract_version: str = METRIC_CONTRACT_V1
    envelope_schema_version: str = BASELINE_ENVELOPE_SCHEMA_V1
    metric_contract_hashes: Mapping[str, str] | None = None

    @staticmethod
    def normalize_contract_version(value: object) -> str:
        normalized = str(value).strip().lower()
        aliases = {
            "v1": METRIC_CONTRACT_V1,
            "1": METRIC_CONTRACT_V1,
            "1.0": METRIC_CONTRACT_V1,
            "metric_v1": METRIC_CONTRACT_V1,
            "v2": METRIC_CONTRACT_V2,
            "2": METRIC_CONTRACT_V2,
            "2.0": METRIC_CONTRACT_V2,
            "2.0.0": METRIC_CONTRACT_V2,
            "metric_v2": METRIC_CONTRACT_V2,
        }
        if normalized not in aliases:
            raise ValueError(f"unknown metric contract version: {value!r}")
        return aliases[normalized]

    @classmethod
    def from_baseline_envelope(
        cls,
        path: Path,
        *,
        expected_contract_version: str | None = None,
    ) -> "SelectionGates":
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema_version")
        if schema == BASELINE_ENVELOPE_SCHEMA_V1:
            observed = METRIC_CONTRACT_V1
            gates = cls._from_v1(path, payload)
        elif schema == BASELINE_ENVELOPE_SCHEMA_V2:
            observed = METRIC_CONTRACT_V2
            gates = cls._from_v2(path, payload)
        else:
            raise ValueError(f"unsupported baseline-envelope schema {schema!r}: {path}")
        if expected_contract_version is not None:
            expected = cls.normalize_contract_version(expected_contract_version)
            if observed != expected:
                raise ValueError(
                    f"metric contract/envelope mismatch: config={expected}, envelope={observed}"
                )
        return gates

    @classmethod
    def _from_v1(cls, path: Path, payload: Mapping[str, Any]) -> "SelectionGates":
        scope = payload.get("scope", {})
        expected_scope = {
            "outer_fold": 0,
            "source_split": "validation",
            "source_fold_id": 1,
            "fit_role": "calibration",
            "score_role": "screen",
            "outer_test_used": False,
        }
        mismatches = {
            key: (scope.get(key), expected)
            for key, expected in expected_scope.items()
            if scope.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"baseline envelope has invalid development scope: {mismatches}")
        metrics = payload.get("metric_envelope", {})
        required_metrics = {
            "screen_strict_mae",
            "screen_strict_pearson",
            "screen_strict_r2",
            "full_risk_macro_delta_c",
        }
        if set(metrics) != required_metrics:
            raise ValueError("baseline envelope must contain exactly the four frozen gate metrics")
        for metric in required_metrics:
            if (
                "baseline_value" not in metrics[metric]
                or "ehr_age_threshold_exclusive_decimal" not in metrics[metric]
            ):
                raise ValueError(
                    f"baseline envelope metric {metric} lacks baseline value/exclusive decimal"
                )
        organ_payload = payload.get("organ_delta_c_envelope", {})
        if not isinstance(organ_payload, list):
            raise ValueError("baseline organ delta C envelope must be a row list")
        organ_values = {
            str(row["organ_system"]): float(row["baseline_delta_c"])
            for row in organ_payload
        }
        if set(organ_values) != set(ORGAN_SYSTEMS):
            raise ValueError("baseline envelope must contain all seven organ delta C values")
        requirements = payload.get("ehr_age_requirements", {})
        artifact_hashes = payload.get("source_artifact_sha256", {})
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("baseline envelope must record non-empty input artifact SHA256 values")
        registry_path = path.parent / "fold0_development_evaluation_registry.json"
        if sha256_file(registry_path) != payload.get("source_registry_sha256"):
            raise ValueError("baseline envelope source registry hash mismatch")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        expected_methods = set(payload.get("methods", []))
        if len(expected_methods) != 6 or {row["method"] for row in registry.get("methods", [])} != expected_methods:
            raise ValueError("baseline envelope must be backed by exactly six registered methods")
        for row in registry["methods"]:
            for kind in ("summary", "manifest"):
                artifact_path = (registry_path.parent / row[f"{kind}_path"]).resolve()
                if sha256_file(artifact_path) != row[f"{kind}_sha256"]:
                    raise ValueError(f"registered baseline {kind} hash mismatch for {row['method']}")
        shared_paths = {
            "fold0_validation_calibration_screen": REVISION_ROOT
            / "01_data_rebuild"
            / "03_folds"
            / "fold0_validation_calibration_screen.parquet",
            "participant_masks": REVISION_ROOT
            / "01_data_rebuild"
            / "02_outcomes_60m"
            / "participant_followup_and_evaluation_masks.parquet",
            "organ_outcomes": REVISION_ROOT
            / "01_data_rebuild"
            / "02_outcomes_60m"
            / "organ_outcomes_60m.parquet",
            "evaluator_code": REVISION_ROOT
            / "05_downstream"
            / "scripts"
            / "evaluate_fold0_development.py",
        }
        shared_hashes = artifact_hashes.get("shared", {})
        for name, artifact_path in shared_paths.items():
            if sha256_file(artifact_path) != shared_hashes.get(name):
                raise ValueError(f"baseline envelope shared artifact hash mismatch: {name}")
        threshold_decimals = {
            "strict_mae": str(metrics["screen_strict_mae"]["ehr_age_threshold_exclusive_decimal"]),
            "strict_pearson": str(
                metrics["screen_strict_pearson"]["ehr_age_threshold_exclusive_decimal"]
            ),
            "strict_r2": str(metrics["screen_strict_r2"]["ehr_age_threshold_exclusive_decimal"]),
            "risk_delta_c": str(
                metrics["full_risk_macro_delta_c"]["ehr_age_threshold_exclusive_decimal"]
            ),
        }
        return cls(
            strict_mae_max=float(threshold_decimals["strict_mae"]),
            strict_pearson_min=float(threshold_decimals["strict_pearson"]),
            strict_r2_min=float(threshold_decimals["strict_r2"]),
            delta_c_min=float(threshold_decimals["risk_delta_c"]),
            positive_organs_min=int(requirements.get("positive_organs_required", 7)),
            organs_beating_baseline_min=int(
                requirements.get("organs_beating_metricwise_baseline_required", 5)
            ),
            baseline_organ_delta_c=organ_values,
            threshold_decimals=threshold_decimals,
            source=str(path.resolve()),
            baseline_artifact_hashes=artifact_hashes,
            metric_contract_version=METRIC_CONTRACT_V1,
            envelope_schema_version=BASELINE_ENVELOPE_SCHEMA_V1,
            metric_contract_hashes={
                "baseline_envelope": sha256_file(path),
                "baseline_registry": sha256_file(registry_path),
            },
        )

    @classmethod
    def _from_v2(cls, path: Path, payload: Mapping[str, Any]) -> "SelectionGates":
        manifest_path = METRIC_V2_MANIFEST.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        envelope_ref = manifest.get("artifacts", {}).get(
            "fold0_development_baseline_envelope_v2", {}
        )
        canonical_envelope = (REVISION_ROOT / str(envelope_ref.get("path", ""))).resolve()
        if path.resolve() != canonical_envelope:
            raise ValueError(
                "metric_v2 envelope must be the canonical artifact referenced by ARTIFACT_MANIFEST.v2"
            )
        if sha256_file(path) != envelope_ref.get("sha256"):
            raise ValueError("metric_v2 envelope hash disagrees with ARTIFACT_MANIFEST.v2")
        validation = validate_metric_v2_contract(REVISION_ROOT, manifest_path)
        if validation.get("status") != "VALID" or validation.get("outer_test_used") is not False:
            raise ValueError("metric_v2 contract validator did not return the frozen VALID state")

        metrics = payload.get("development_metric_envelope", {})
        required_metrics = {
            "strict_mae",
            "strict_pearson",
            "strict_r2",
            "age_sex_harrell_full_risk_macro_delta_c_index",
        }
        if set(metrics) != required_metrics:
            raise ValueError("metric_v2 envelope must contain exactly four Stage A1 gates")
        threshold_decimals = {
            "strict_mae": str(metrics["strict_mae"]["threshold_exclusive_decimal"]),
            "strict_pearson": str(metrics["strict_pearson"]["threshold_exclusive_decimal"]),
            "strict_r2": str(metrics["strict_r2"]["threshold_exclusive_decimal"]),
            "risk_delta_c": str(
                metrics["age_sex_harrell_full_risk_macro_delta_c_index"][
                    "threshold_exclusive_decimal"
                ]
            ),
        }
        if threshold_decimals["risk_delta_c"] != "0.026913434679319033":
            raise ValueError("metric_v2 Harrell relative delta-C threshold is not frozen at 0.026913434679319033")
        organ_payload = payload.get("organ_delta_c_envelope", [])
        organ_values = {
            str(row["organ_system"]): float(row["baseline_delta_c"])
            for row in organ_payload
        }
        if set(organ_values) != set(ORGAN_SYSTEMS):
            raise ValueError("metric_v2 envelope must contain all seven organ delta-C baselines")
        requirements = payload.get("ehr_age_development_requirements", {})
        if requirements.get("all_four_metric_gates_required") is not True:
            raise ValueError("metric_v2 requires all four Stage A1 metric gates")
        success_path = REVISION_ROOT / "00_contract" / "metric_v2" / "SUCCESS_CRITERIA.v2.json"
        sidecar_path = manifest_path.with_suffix(".sha256")
        return cls(
            strict_mae_max=float(threshold_decimals["strict_mae"]),
            strict_pearson_min=float(threshold_decimals["strict_pearson"]),
            strict_r2_min=float(threshold_decimals["strict_r2"]),
            delta_c_min=float(threshold_decimals["risk_delta_c"]),
            positive_organs_min=int(requirements.get("positive_organs_required", 7)),
            organs_beating_baseline_min=int(
                requirements.get("organs_beating_metricwise_baseline_required", 5)
            ),
            baseline_organ_delta_c=organ_values,
            threshold_decimals=threshold_decimals,
            source=str(path.resolve()),
            baseline_artifact_hashes=payload.get("source_lineage", {}),
            metric_contract_version=METRIC_CONTRACT_V2,
            envelope_schema_version=BASELINE_ENVELOPE_SCHEMA_V2,
            metric_contract_hashes={
                "baseline_envelope": sha256_file(path),
                "success_criteria": sha256_file(success_path),
                "artifact_manifest": sha256_file(manifest_path),
                "artifact_manifest_sidecar": sha256_file(sidecar_path),
                "validator": sha256_file(METRIC_V2_DIR / "validate_metric_v2_contract.py"),
            },
        )

    @classmethod
    def smoke(cls, reference: "SelectionGates | None" = None) -> "SelectionGates":
        reference = reference or cls(
            strict_mae_max=0.0,
            strict_pearson_min=0.0,
            strict_r2_min=0.0,
        )
        return cls(
            strict_mae_max=1.0e9,
            strict_pearson_min=-1.0e9,
            strict_r2_min=-1.0e9,
            delta_c_min=-1.0e9,
            positive_organs_min=0,
            organs_beating_baseline_min=0,
            baseline_organ_delta_c=reference.baseline_organ_delta_c,
            threshold_decimals=None,
            source=f"smoke_only_under_{reference.metric_contract_version}",
            baseline_artifact_hashes=reference.baseline_artifact_hashes,
            metric_contract_version=reference.metric_contract_version,
            envelope_schema_version=reference.envelope_schema_version,
            metric_contract_hashes=reference.metric_contract_hashes,
        )

    def evaluate(
        self,
        age_metrics: Mapping[str, float | int | None],
        cox_metrics: Mapping[str, Any],
    ) -> tuple[bool, list[str]]:
        failures: list[str] = []
        mae = age_metrics.get("mae")
        pearson = age_metrics.get("pearson")
        r2 = age_metrics.get("r2")
        delta = cox_metrics.get("macro_delta_c")
        positive = int(cox_metrics.get("positive_organs", 0))
        beating = int(cox_metrics.get("organs_beating_baseline", 0))
        decimals = self.threshold_decimals or {}

        def below(value: float, metric: str, fallback: float) -> bool:
            threshold = Decimal(decimals.get(metric, str(fallback)))
            return Decimal(str(value)) < threshold

        def above(value: float, metric: str, fallback: float) -> bool:
            threshold = Decimal(decimals.get(metric, str(fallback)))
            return Decimal(str(value)) > threshold

        if mae is None or not below(float(mae), "strict_mae", self.strict_mae_max):
            failures.append("strict_mae")
        if pearson is None or not above(
            float(pearson), "strict_pearson", self.strict_pearson_min
        ):
            failures.append("strict_pearson")
        if r2 is None or not above(float(r2), "strict_r2", self.strict_r2_min):
            failures.append("strict_r2")
        if delta is None or not above(
            float(delta), "risk_delta_c", self.delta_c_min
        ):
            failures.append("delta_c")
        if positive < self.positive_organs_min:
            failures.append("positive_organs")
        if beating < self.organs_beating_baseline_min:
            failures.append("organs_beating_baseline")
        return not failures, failures


def metric_contract_record(gates: SelectionGates, envelope_path: Path) -> dict[str, Any]:
    return {
        "version": gates.metric_contract_version,
        "envelope_schema_version": gates.envelope_schema_version,
        "envelope_path": str(envelope_path.resolve()),
        "envelope_sha256": sha256_file(envelope_path),
        "lineage_hashes": dict(gates.metric_contract_hashes or {}),
        "risk_gate_name": (
            "age_sex_harrell_full_risk_macro_delta_c_index"
            if gates.metric_contract_version == METRIC_CONTRACT_V2
            else "legacy_full_risk_macro_delta_c"
        ),
        "risk_gate_threshold_exclusive_decimal": (
            None
            if gates.threshold_decimals is None
            else gates.threshold_decimals.get("risk_delta_c")
        ),
    }


def verify_metric_contract_record(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    expected = SelectionGates.normalize_contract_version(
        protocol.get("metric_contract_version", METRIC_CONTRACT_V1)
    )
    envelope_value = config.get("baseline_envelope_path")
    if not envelope_value:
        raise RuntimeError("run config does not record a baseline envelope path")
    envelope_path = Path(str(envelope_value)).resolve()
    gates = SelectionGates.from_baseline_envelope(
        envelope_path, expected_contract_version=expected
    )
    observed_hash = sha256_file(envelope_path)
    if observed_hash != config.get("baseline_envelope_sha256"):
        raise RuntimeError("baseline envelope changed after run materialization")
    expected_record = metric_contract_record(gates, envelope_path)
    stored = config.get("metric_contract")
    if stored is None:
        if expected != METRIC_CONTRACT_V1:
            raise RuntimeError("metric_v2 run config is missing its explicit metric_contract record")
        # Historical v1 run configs predate the explicit record; infer only from frozen v1 lineage.
        return expected_record
    if not isinstance(stored, Mapping):
        raise RuntimeError("run config metric_contract must be a mapping")
    for key in (
        "version",
        "envelope_schema_version",
        "envelope_path",
        "envelope_sha256",
        "lineage_hashes",
        "risk_gate_name",
        "risk_gate_threshold_exclusive_decimal",
    ):
        if stored.get(key) != expected_record.get(key):
            raise RuntimeError(f"run config metric contract mismatch: {key}")
    return dict(stored)


@dataclass(frozen=True)
class PredictionBundle:
    participant_indices: np.ndarray
    chronological_age: np.ndarray
    strict_healthy: np.ndarray
    duration_months: np.ndarray
    event: np.ndarray
    at_risk: np.ndarray
    sex: np.ndarray
    predicted_age: np.ndarray
    risk_score: np.ndarray


def configure_reproducibility(seed: int, *, deterministic: bool, tf32: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def trainable_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def pooled_age_metrics(
    chronological_age: np.ndarray,
    predicted_age: np.ndarray,
    *,
    participant_mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    age = np.asarray(chronological_age, dtype=np.float64)
    prediction = np.asarray(predicted_age, dtype=np.float64)
    if prediction.shape != (len(age), len(ORGAN_SYSTEMS)):
        raise ValueError("predicted age must have shape [participant,organ]")
    selected = np.ones(len(age), dtype=bool) if participant_mask is None else np.asarray(participant_mask, dtype=bool)
    if selected.shape != (len(age),):
        raise ValueError("participant mask must align with chronological age")
    if not selected.any():
        return {"participant_n": 0, "row_n": 0, "mae": None, "pearson": None, "r2": None}
    target = np.repeat(age[selected], len(ORGAN_SYSTEMS))
    score = prediction[selected].reshape(-1)
    error = score - target
    denominator = np.square(target - target.mean()).sum()
    pearson = None
    if len(target) >= 2 and target.std() > 0.0 and score.std() > 0.0:
        pearson = _finite_or_none(float(np.corrcoef(target, score)[0, 1]))
    r2 = None if denominator <= 0.0 else _finite_or_none(1.0 - np.square(error).sum() / denominator)
    return {
        "participant_n": int(selected.sum()),
        "row_n": int(selected.sum() * len(ORGAN_SYSTEMS)),
        "mae": float(np.abs(error).mean()),
        "pearson": pearson,
        "r2": r2,
    }


def _evaluation_frame(
    prediction: PredictionBundle,
    participant_ids: list[str],
) -> pd.DataFrame:
    n = len(participant_ids)
    if n != len(prediction.chronological_age):
        raise ValueError("participant IDs and prediction bundle are not aligned")
    return pd.DataFrame(
        {
            "participant_id": np.repeat(np.asarray(participant_ids, dtype=object), len(ORGAN_SYSTEMS)),
            "organ_system": np.tile(np.asarray(ORGAN_SYSTEMS, dtype=object), n),
            "actual_age": np.repeat(prediction.chronological_age.astype(np.float64), len(ORGAN_SYSTEMS)),
            "predicted_age": prediction.predicted_age.astype(np.float64).reshape(-1),
            "strict_healthy_reference_60m": np.repeat(
                prediction.strict_healthy.astype(np.int8), len(ORGAN_SYSTEMS)
            ),
            "sex": np.repeat(prediction.sex.astype(np.float64), len(ORGAN_SYSTEMS)),
            "duration": prediction.duration_months.astype(np.float64).reshape(-1),
            "event": prediction.event.astype(np.int8).reshape(-1),
            "at_risk": prediction.at_risk.astype(np.int8).reshape(-1),
        }
    )


def calibration_to_screen_cox(
    calibration: PredictionBundle,
    screen: PredictionBundle,
    *,
    calibration_participant_ids: list[str],
    screen_participant_ids: list[str],
    baseline_organ_delta_c: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit correction/Cox on Fold 1 calibration and score only Fold 1 screen."""

    calibration_frame = _evaluation_frame(calibration, calibration_participant_ids)
    screen_frame = _evaluation_frame(screen, screen_participant_ids)
    correction = fit_linear_age_correction(calibration_frame)
    calibration_corrected = apply_linear_age_correction_to_validation(
        calibration_frame, correction
    )
    screen_corrected = apply_linear_age_correction(screen_frame, correction)
    organ_metrics: dict[str, dict[str, float | int]] = {}
    deltas: list[float] = []
    beating = 0
    for organ in ORGAN_SYSTEMS:
        calibration_organ = calibration_corrected.loc[
            calibration_corrected["organ_system"].astype(str).eq(organ)
            & calibration_corrected["at_risk"].eq(1)
        ].copy()
        screen_organ = screen_corrected.loc[
            screen_corrected["organ_system"].astype(str).eq(organ)
            & screen_corrected["at_risk"].eq(1)
        ].copy()
        cell = evaluate_cox_cell(
            calibration_organ,
            screen_organ,
            fold=0,
            organ_system=organ,
        )
        row = asdict(cell)
        organ_metrics[organ] = row
        deltas.append(cell.delta_c)
        if baseline_organ_delta_c is not None:
            if organ not in baseline_organ_delta_c:
                raise ValueError(f"baseline envelope is missing organ delta C for {organ}")
            beating += int(cell.delta_c > float(baseline_organ_delta_c[organ]))
    return {
        "definition": "Fold1 calibration-fit linear correction and Cox; Fold1 screen C-index",
        "macro_delta_c": float(np.mean(deltas)),
        "positive_organs": int(sum(value > 0.0 for value in deltas)),
        "organs_beating_baseline": int(beating),
        "evaluable_organs": len(deltas),
        "organs": organ_metrics,
    }


def _move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _move_labels(batch: JointBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "chronological_age": batch.chronological_age.to(device, non_blocking=True),
        "strict_healthy_reference": batch.strict_healthy.to(device, non_blocking=True),
        "duration_months": batch.duration_months.to(device, non_blocking=True),
        "event": batch.event.to(device, non_blocking=True),
        "at_risk": batch.at_risk.to(device, non_blocking=True),
    }


def _train_epoch(
    *,
    model: EHRAGEModel,
    batcher: JointVectorizedBatcher,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    interval_edges: torch.Tensor,
    loss_config: AgePreservingLossConfig,
    survival_multiplier: float,
    event_dropout_probability: float,
    gradient_clip_norm: float,
    amp: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    observed_n = 0
    optimizer_steps = 0
    amp_skipped_steps = 0
    for batch_index, batch in enumerate(batcher):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        inputs = _move_inputs(batch.inputs, device)
        labels = _move_labels(batch, device)
        thinned = compact_event_dropout(inputs, probability=event_dropout_probability)
        with _autocast(device, amp and device.type == "cuda"):
            output = model(**inputs)
            augmented = model(**thinned) if event_dropout_probability > 0.0 else None
            losses = ehr_age_joint_loss(
                output,
                **labels,
                interval_edges_months=interval_edges,
                config=loss_config,
                survival_multiplier=survival_multiplier,
                augmented_output=augmented,
            )
        scale_before = float(scaler.get_scale())
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        step_skipped = scaler.is_enabled() and float(scaler.get_scale()) < scale_before
        amp_skipped_steps += int(step_skipped)
        optimizer_steps += int(not step_skipped)
        batch_n = len(batch.chronological_age)
        observed_n += batch_n
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_n
    if observed_n == 0:
        raise RuntimeError("training batcher produced no batches")
    result = {key: value / observed_n for key, value in totals.items()}
    result["optimizer_steps"] = float(optimizer_steps)
    result["amp_skipped_steps"] = float(amp_skipped_steps)
    return result


def _predict(
    *,
    model: EHRAGEModel,
    batcher: JointVectorizedBatcher,
    device: torch.device,
    amp: bool,
    max_batches: int | None,
) -> PredictionBundle:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "participant_indices",
            "chronological_age",
            "strict_healthy",
            "duration_months",
            "event",
            "at_risk",
            "sex",
            "predicted_age",
            "risk_score",
        )
    }
    with torch.inference_mode():
        for batch_index, batch in enumerate(batcher):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = _move_inputs(batch.inputs, device)
            with _autocast(device, amp and device.type == "cuda"):
                output = model(**inputs)
            chunks["participant_indices"].append(batch.participant_indices.numpy())
            chunks["chronological_age"].append(batch.chronological_age.numpy())
            chunks["strict_healthy"].append(batch.strict_healthy.numpy())
            chunks["duration_months"].append(batch.duration_months.numpy())
            chunks["event"].append(batch.event.numpy())
            chunks["at_risk"].append(batch.at_risk.numpy())
            chunks["sex"].append(batch.inputs["sex"].numpy())
            chunks["predicted_age"].append(output.predicted_age.float().cpu().numpy())
            chunks["risk_score"].append(output.risk_score.float().cpu().numpy())
    if not chunks["participant_indices"]:
        raise RuntimeError("prediction batcher produced no batches")
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    return PredictionBundle(**arrays)


def _model_config(payload: Mapping[str, Any]) -> EHRAGEConfig:
    edges = payload["survival_interval_edges_months"]
    return EHRAGEConfig(
        hidden_dim=int(payload["hidden_dim"]),
        num_heads=int(payload["num_heads"]),
        num_transformer_layers=int(payload["num_transformer_layers"]),
        num_gru_layers=int(payload["num_gru_layers"]),
        dropout=float(payload["dropout"]),
        freeze_event_embeddings=bool(payload["freeze_event_embeddings"]),
        risk_offset_cap_years=float(payload["risk_offset_cap_years"]),
        risk_temperature=float(payload["risk_temperature"]),
        survival_intervals=len(edges) - 1,
        baseline_log_rate_init=float(payload.get("baseline_log_rate_init", -7.5)),
        base_age_mode=str(payload.get("base_age_mode", "shared_fusion")),
        base_age_head_mode=str(payload.get("base_age_head_mode", "linear")),
        base_age_pooling_mode=str(payload.get("base_age_pooling_mode", "global_fused")),
    )


def _loss_config(payload: Mapping[str, Any]) -> AgePreservingLossConfig:
    allowed = set(AgePreservingLossConfig.__dataclass_fields__)
    return AgePreservingLossConfig(**{key: payload[key] for key in allowed if key in payload})


def _protocol_guard(payload: Mapping[str, Any]) -> None:
    required_false = (
        "warm_start",
        "fine_tune",
        "teacher",
        "distillation",
        "baseline_predictions_as_targets",
        "outer_test_during_development",
    )
    violations = [key for key in required_false if payload.get(key) is not False]
    if violations:
        raise RuntimeError(f"from-scratch protocol violation: {violations}")


def _code_hashes() -> dict[str, str]:
    paths = {
        "ehr_age_model": MODEL_DIR / "ehr_age_model.py",
        "ehr_age_losses": MODEL_DIR / "ehr_age_losses.py",
        "ehr_age_data": Path(__file__).resolve().parent / "ehr_age_data.py",
        "ehr_age_training": Path(__file__).resolve(),
        "compact_cache_dataset": BASELINE_COMMON / "compact_cache_dataset.py",
        "crossfit_metrics": DOWNSTREAM_COMMON / "crossfit_metrics.py",
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def _long_prediction_frame(
    participant_ids: list[str],
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    split_role: str = "validation",
) -> pd.DataFrame:
    if split_role not in {"validation", "test"}:
        raise ValueError("split_role must be validation or test")
    frame = pd.DataFrame(
        {
            "participant_id": np.repeat(np.asarray(participant_ids, dtype=object), len(ORGAN_SYSTEMS)),
            "organ_system": np.tile(np.asarray(ORGAN_SYSTEMS, dtype=object), len(participant_ids)),
            "actual_age": np.repeat(np.asarray(target, dtype=np.float64), len(ORGAN_SYSTEMS)),
            "predicted_age": np.asarray(prediction, dtype=np.float32).reshape(-1),
            "outer_fold": np.zeros(len(participant_ids) * len(ORGAN_SYSTEMS), dtype=np.int8),
            "split_role": split_role,
        }
    )
    return frame


def _risk_frame(
    participant_ids: list[str],
    risk_score: np.ndarray,
    inner_role: list[str],
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        "participant_id": participant_ids,
        "outer_fold": np.zeros(len(participant_ids), dtype=np.int8),
        "split_role": "validation",
        "inner_role": inner_role,
    }
    for index, organ in enumerate(ORGAN_SYSTEMS):
        payload[f"risk_score_{organ}"] = risk_score[:, index].astype(np.float32)
    return pd.DataFrame(payload)


def _survival_multiplier(epoch: int, loss_payload: Mapping[str, Any]) -> float:
    start = float(loss_payload.get("survival_ramp_start_fraction", 0.1))
    ramp_epochs = int(loss_payload.get("survival_ramp_epochs", 15))
    if not 0.0 <= start <= 1.0 or ramp_epochs <= 0:
        raise ValueError("invalid survival ramp configuration")
    progress = min(max(epoch - 1, 0) / max(ramp_epochs - 1, 1), 1.0)
    return start + (1.0 - start) * progress


def run_fold0_development(
    *,
    config_path: Path,
    cache_dir: Path,
    label_cache_path: Path,
    age_target_cache_path: Path,
    label_manifest_path: Path,
    output_root: Path,
    baseline_envelope_path: Path | None,
    batch_size_override: int | None = None,
    max_epochs_override: int | None = None,
    max_batches: int | None = None,
) -> Path:
    started = time.perf_counter()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if int(raw["training"]["outer_fold"]) != 0:
        raise ValueError("this runner is deliberately restricted to outer Fold 0")
    _protocol_guard(raw["protocol"])
    training = dict(raw["training"])
    if batch_size_override is not None:
        training["batch_size"] = int(batch_size_override)
    if max_epochs_override is not None:
        training["max_epochs"] = int(max_epochs_override)
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    smoke_only = max_batches is not None
    declared_metric_contract = SelectionGates.normalize_contract_version(
        raw["protocol"].get("metric_contract_version", METRIC_CONTRACT_V1)
    )
    baseline_reference_gates: SelectionGates | None = None
    baseline_envelope_hash: str | None = None
    if baseline_envelope_path is not None:
        baseline_reference_gates = SelectionGates.from_baseline_envelope(
            baseline_envelope_path,
            expected_contract_version=declared_metric_contract,
        )
        baseline_envelope_hash = sha256_file(baseline_envelope_path)
    if smoke_only:
        gates = SelectionGates.smoke(baseline_reference_gates)
    else:
        if baseline_reference_gates is None:
            raise ValueError(
                "formal development requires the explicit Fold1 calibration-to-screen six-baseline envelope"
            )
        gates = baseline_reference_gates
    if declared_metric_contract == METRIC_CONTRACT_V2 and baseline_reference_gates is None:
        raise ValueError("metric_v2 runs require the frozen canonical metric_v2 envelope")
    contract_record = (
        None
        if baseline_reference_gates is None or baseline_envelope_path is None
        else metric_contract_record(baseline_reference_gates, baseline_envelope_path)
    )

    seed = int(training["seed"])
    configure_reproducibility(
        seed,
        deterministic=bool(training["deterministic"]),
        tf32=bool(training["tf32"]),
    )
    cache_dir = cache_dir.resolve()
    source_hashes = {
        "cache_hard_gate": verify_pass_gate(cache_dir / "cache_hard_gate.json"),
        "leakage_gate": verify_pass_gate(cache_dir.parent / "05_leakage_audit" / "leakage_gate.json"),
        "batch_manifest": sha256_file(cache_dir / "batch_manifest.json"),
        "embedding_tables": sha256_file(cache_dir / "embedding_tables.pt"),
        "development_label_manifest": sha256_file(label_manifest_path),
        "development_labels": sha256_file(label_cache_path),
        "development_age_targets": sha256_file(age_target_cache_path),
    }
    code_hashes = _code_hashes()
    materialized: dict[str, Any] = {
        "schema_version": raw["schema_version"],
        "model": raw["model"],
        "loss": raw["loss"],
        "training": training,
        "protocol": raw["protocol"],
        "cache_dir": str(cache_dir),
        "label_cache_path": str(label_cache_path.resolve()),
        "age_target_cache_path": str(age_target_cache_path.resolve()),
        "label_manifest_path": str(label_manifest_path.resolve()),
        "selection_gates": asdict(gates),
        "formal_baseline_selection_gates": (
            None if baseline_reference_gates is None else asdict(baseline_reference_gates)
        ),
        "baseline_envelope_path": (
            None if baseline_envelope_path is None else str(baseline_envelope_path.resolve())
        ),
        "baseline_envelope_sha256": baseline_envelope_hash,
        "metric_contract": contract_record,
        "max_batches": max_batches,
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "hard_split_boundary": {
            "train_folds": [2, 3, 4],
            "validation_fold": 1,
            "validation_inner_roles": ["calibration", "screen"],
            "outer_test_fold_loaded": False,
        },
    }
    config_hash = canonical_hash(materialized)
    run_dir = output_root.resolve() / ("smoke" if smoke_only else "development") / config_hash[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        if canonical_hash(previous) != canonical_hash(materialized):
            raise RuntimeError("existing run directory contains a different materialized configuration")
    run_config_path.write_text(json.dumps(materialized, indent=2), encoding="utf-8")
    log_path = run_dir / "training.log"
    log_path.write_text("", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    data = load_fold0_development_data(
        cache_dir=cache_dir,
        label_cache_path=label_cache_path,
        age_target_cache_path=age_target_cache_path,
        label_manifest_path=label_manifest_path,
    )
    age_anchor = float(data.cache.dataset.chronological_age.index_select(0, data.train_indices).mean())
    batch_size = int(training["batch_size"])
    batchers = {
        "train": JointVectorizedBatcher(
            data, data.train_indices, batch_size=batch_size, shuffle=True, seed=seed
        ),
        "calibration": JointVectorizedBatcher(
            data, data.calibration_indices, batch_size=batch_size, shuffle=False, seed=seed
        ),
        "screen": JointVectorizedBatcher(
            data, data.screen_indices, batch_size=batch_size, shuffle=False, seed=seed
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = _model_config(raw["model"])
    model = EHRAGEModel(
        data.cache.code_embedding,
        data.cache.text_embedding,
        age_anchor=age_anchor,
        config=model_config,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    warmup_epochs = int(training["warmup_epochs"])

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / max(warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]) and device.type == "cuda")
    interval_edges = torch.tensor(
        raw["model"]["survival_interval_edges_months"], dtype=torch.float32, device=device
    )
    loss_config = _loss_config(raw["loss"])
    selected_checkpoint = run_dir / "selected_checkpoint.pt"
    last_state = run_dir / "last_training_state.pt"
    best_score = -math.inf
    best_strict_mae = math.inf
    selected_epoch: int | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    initial_trainable_hash = trainable_parameter_hash(model)
    emit(
        f"START EHR-AGE fold=0 device={device} train={len(data.train_indices):,} "
        f"calibration={len(data.calibration_indices):,} screen={len(data.screen_indices):,} "
        f"params={parameter_count:,} trainable={trainable_count:,} smoke={smoke_only} "
        f"metric_contract={declared_metric_contract} "
        f"base_age_mode={model_config.base_age_mode} "
        f"base_age_head_mode={model_config.base_age_head_mode} "
        f"base_age_pooling_mode={model_config.base_age_pooling_mode}"
    )
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        survival_multiplier = _survival_multiplier(epoch, raw["loss"])
        losses = _train_epoch(
            model=model,
            batcher=batchers["train"],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            interval_edges=interval_edges,
            loss_config=loss_config,
            survival_multiplier=survival_multiplier,
            event_dropout_probability=float(training["event_dropout_probability"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            amp=bool(training["amp"]),
            max_batches=max_batches,
        )
        if losses["optimizer_steps"] < 1:
            raise RuntimeError(
                "epoch completed without an optimizer update; AMP skipped every batch"
            )
        calibration = _predict(
            model=model,
            batcher=batchers["calibration"],
            device=device,
            amp=bool(training["amp"]),
            max_batches=max_batches,
        )
        screen = _predict(
            model=model,
            batcher=batchers["screen"],
            device=device,
            amp=bool(training["amp"]),
            max_batches=max_batches,
        )
        strict_metrics = pooled_age_metrics(
            screen.chronological_age,
            screen.predicted_age,
            participant_mask=screen.strict_healthy,
        )
        observed_calibration_ids = [
            data.cache.dataset.participant_ids[int(index)]
            for index in calibration.participant_indices.tolist()
        ]
        observed_screen_ids = [
            data.cache.dataset.participant_ids[int(index)]
            for index in screen.participant_indices.tolist()
        ]
        try:
            cox_metrics = calibration_to_screen_cox(
                calibration,
                screen,
                calibration_participant_ids=observed_calibration_ids,
                screen_participant_ids=observed_screen_ids,
                baseline_organ_delta_c=gates.baseline_organ_delta_c,
            )
        except Exception as exc:
            if not smoke_only:
                raise
            cox_metrics = {
                "definition": "not evaluable in bounded smoke subset",
                "macro_delta_c": 0.0,
                "positive_organs": 0,
                "organs_beating_baseline": 0,
                "evaluable_organs": 0,
                "organs": {},
                "smoke_exception": f"{type(exc).__name__}: {exc}",
            }
        eligible, failures = gates.evaluate(strict_metrics, cox_metrics)
        raw_score = cox_metrics["macro_delta_c"]
        score = -math.inf if raw_score is None else float(raw_score)
        strict_mae = (
            math.inf if strict_metrics["mae"] is None else float(strict_metrics["mae"])
        )
        improved = eligible and (
            selected_epoch is None
            or score > best_score + 0.001
            or (abs(score - best_score) <= 0.001 and strict_mae < best_strict_mae)
        )
        if improved:
            best_score = score
            best_strict_mae = strict_mae
            selected_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "age_anchor_train_only": age_anchor,
                    "config_hash": config_hash,
                    "strict_screen_metrics": strict_metrics,
                    "calibration_to_screen_cox": cox_metrics,
                    "from_scratch": True,
                    "metric_contract": contract_record,
                },
                selected_checkpoint,
            )
        elif selected_epoch is not None:
            stale_epochs += 1
        if losses["optimizer_steps"] > 0:
            scheduler.step()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "config_hash": config_hash,
                "from_scratch_same_run_only": True,
                "metric_contract": contract_record,
            },
            last_state,
        )
        epoch_row = {
            "epoch": epoch,
            "survival_multiplier": survival_multiplier,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "losses": losses,
            "strict_screen_metrics": strict_metrics,
            "calibration_to_screen_cox": cox_metrics,
            "eligible": eligible,
            "gate_failures": failures,
            "selected": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_row)
        emit(
            f"EPOCH {epoch} loss={losses['total']:.5f} strict_mae={strict_metrics['mae']} "
            f"strict_r={strict_metrics['pearson']} strict_r2={strict_metrics['r2']} "
            f"cal_to_screen_delta_c={raw_score} eligible={eligible} selected={improved} "
            f"optimizer_steps={int(losses['optimizer_steps'])} "
            f"amp_skipped={int(losses['amp_skipped_steps'])} seconds={epoch_row['seconds']:.1f}"
        )
        if (
            selected_epoch is not None
            and epoch >= int(training["minimum_epochs"])
            and stale_epochs >= int(training["patience"])
        ):
            break

    history_path = run_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    final_trainable_hash = trainable_parameter_hash(model)
    if final_trainable_hash == initial_trainable_hash:
        raise RuntimeError("training completed without changing any trainable model parameter")
    validation_artifacts: dict[str, str] = {}
    if selected_epoch is not None:
        checkpoint = torch.load(selected_checkpoint, map_location=device, weights_only=True)
        if checkpoint["config_hash"] != config_hash or checkpoint.get("from_scratch") is not True:
            raise RuntimeError("selected checkpoint provenance mismatch")
        if checkpoint.get("metric_contract") != contract_record:
            raise RuntimeError("selected checkpoint metric-contract provenance mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        calibration = _predict(
            model=model,
            batcher=batchers["calibration"],
            device=device,
            amp=bool(training["amp"]),
            max_batches=max_batches,
        )
        screen = _predict(
            model=model,
            batcher=batchers["screen"],
            device=device,
            amp=bool(training["amp"]),
            max_batches=max_batches,
        )
        combined_indices = np.concatenate((calibration.participant_indices, screen.participant_indices))
        combined_target = np.concatenate((calibration.chronological_age, screen.chronological_age))
        combined_prediction = np.concatenate((calibration.predicted_age, screen.predicted_age))
        combined_risk = np.concatenate((calibration.risk_score, screen.risk_score))
        participant_ids = [data.cache.dataset.participant_ids[int(index)] for index in combined_indices]
        inner_roles = ["calibration"] * len(calibration.participant_indices) + ["screen"] * len(screen.participant_indices)
        long_path = run_dir / "validation_predictions_long.parquet"
        risk_path = run_dir / "validation_risk_scores.parquet"
        _long_prediction_frame(participant_ids, combined_target, combined_prediction).to_parquet(
            long_path, index=False
        )
        _risk_frame(participant_ids, combined_risk, inner_roles).to_parquet(risk_path, index=False)
        validation_artifacts = {
            "validation_predictions_long": str(long_path),
            "validation_risk_scores": str(risk_path),
        }

    status = "SMOKE_ONLY" if smoke_only else ("PROMOTABLE" if selected_epoch is not None else "NON_PROMOTABLE")
    summary = {
        "schema_version": "revision0710_ehr_age_fold0_development_run_v1",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "selected_epoch": selected_epoch,
        "selected_calibration_to_screen_delta_c": (
            None if selected_epoch is None else best_score
        ),
        "outer_test_loaded": False,
        "outer_test_predictions_written": False,
        "from_scratch": True,
        "metric_contract": contract_record,
        "metric_contract_version": declared_metric_contract,
        "base_age_mode": model_config.base_age_mode,
        "base_age_head_mode": model_config.base_age_head_mode,
        "base_age_pooling_mode": model_config.base_age_pooling_mode,
        "age_anchor_train_only": age_anchor,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "initial_trainable_parameter_sha256": initial_trainable_hash,
        "final_trainable_parameter_sha256": final_trainable_hash,
        "trainable_parameters_changed": True,
        "optimizer_steps": int(
            sum(row["losses"]["optimizer_steps"] for row in history)
        ),
        "amp_skipped_steps": int(
            sum(row["losses"]["amp_skipped_steps"] for row in history)
        ),
        "split_counts": {
            "train": len(data.train_indices),
            "calibration": len(data.calibration_indices),
            "screen": len(data.screen_indices),
            "test_loaded": 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
        "artifacts": validation_artifacts,
    }
    summary_path = run_dir / "development_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    artifacts: dict[str, Path] = {
        "run_config": run_config_path,
        "training_history": history_path,
        "training_log": log_path,
        "last_training_state": last_state,
        "development_summary": summary_path,
    }
    if selected_checkpoint.exists():
        artifacts["selected_checkpoint"] = selected_checkpoint
    artifacts.update({key: Path(value) for key, value in validation_artifacts.items()})
    emit(f"COMPLETE status={status} selected_epoch={selected_epoch} run_dir={run_dir}")
    hashes = {key: sha256_file(path) for key, path in artifacts.items()}
    hashes["canonical_config_hash"] = config_hash
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return run_dir


def verify_freeze_manifest(run_dir: Path, freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != FREEZE_SCHEMA or freeze.get("outer_test_authorized") is not True:
        raise RuntimeError("invalid or unauthorized Fold 0 freeze manifest")
    if Path(freeze.get("run_dir", "")).resolve() != run_dir.resolve():
        raise RuntimeError("freeze manifest belongs to a different run directory")
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    if hashes.get("selected_checkpoint") != sha256_file(run_dir / "selected_checkpoint.pt"):
        raise RuntimeError("selected checkpoint changed after development selection")
    if freeze.get("selected_checkpoint_sha256") != hashes["selected_checkpoint"]:
        raise RuntimeError("freeze manifest checkpoint hash mismatch")
    if freeze.get("run_config_sha256") != sha256_file(run_dir / "run_config.json"):
        raise RuntimeError("freeze manifest run-config hash mismatch")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    verified_contract = verify_metric_contract_record(config)
    frozen_contract = freeze.get("metric_contract")
    if frozen_contract is None and verified_contract.get("version") == METRIC_CONTRACT_V1:
        # Historical v1 freeze manifests may predate explicit contract records.
        frozen_contract = verified_contract
    if frozen_contract != verified_contract:
        raise RuntimeError("freeze manifest metric-contract mismatch")
    checkpoint = torch.load(
        run_dir / "selected_checkpoint.pt", map_location="cpu", weights_only=True
    )
    checkpoint_contract = checkpoint.get("metric_contract")
    if checkpoint_contract is None and verified_contract.get("version") == METRIC_CONTRACT_V1:
        checkpoint_contract = verified_contract
    if checkpoint_contract != verified_contract:
        raise RuntimeError("selected checkpoint metric-contract mismatch")
    return freeze
