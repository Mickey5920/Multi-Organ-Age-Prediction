from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import v8_contract
import v8_inputs
from v8_training import TrainingBatch, synthetic_batch


REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parent
CANONICAL_RUNNER_DIR = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "02_runner"
if str(CANONICAL_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(CANONICAL_RUNNER_DIR))


EXPECTED_CACHE_SHARD_ARRAY_KEYS = {
    "participant_id",
    "participant_row",
    "fold_id",
    "phenotype_raw_zero",
    "phenotype_missing_mask",
    "sex",
    "event_code_id",
    "event_text_id",
    "event_relative_days",
    "event_type_id",
    "event_valid_mask",
    "event_count",
    "storage_event_code_id",
    "storage_event_text_id",
    "storage_event_relative_days",
    "storage_event_type_id",
    "storage_event_valid_mask",
    "storage_event_count",
}
EXPECTED_TRANSFORM_ARRAY_KEYS = {
    "median", "mean", "std", "feature_hash", "train_id_hash",
    "outer_fold", "validation_fold",
}


@dataclass(frozen=True)
class RoleBundle:
    role: str
    outer_fold: int
    participant_ids: tuple[str, ...]
    inputs: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]

    def as_training_batch(self) -> TrainingBatch:
        return TrainingBatch(self.participant_ids, self.inputs, self.labels)


class RoleBatcher:
    def __init__(self, bundle: RoleBundle, *, batch_size: int, seed: int, shuffle: bool) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.bundle = bundle
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.iteration = 0

    def __iter__(self):
        n = len(self.bundle.participant_ids)
        positions = torch.arange(n)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.iteration)
            positions = positions[torch.randperm(n, generator=generator)]
        self.iteration += 1
        for start in range(0, n, self.batch_size):
            selected = positions[start : start + self.batch_size]
            yield TrainingBatch(
                participant_ids=tuple(self.bundle.participant_ids[index] for index in selected.tolist()),
                inputs={key: value.index_select(0, selected) for key, value in self.bundle.inputs.items()},
                labels={key: value.index_select(0, selected) for key, value in self.bundle.labels.items()},
            )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_role_bundle_create_only(path: Path, bundle: RoleBundle) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"role bundle exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "revision0710.fig04_fig05.synthetic_role_bundle.v8",
        "role": bundle.role,
        "outer_fold": bundle.outer_fold,
        "participant_ids": bundle.participant_ids,
        "inputs": bundle.inputs,
        "labels": bundle.labels,
    }
    torch.save(payload, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_role_bundle(path: Path, *, expected_role: str) -> RoleBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"schema_version", "role", "outer_fold", "participant_ids", "inputs", "labels"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("role bundle keys changed")
    if payload["schema_version"] != "revision0710.fig04_fig05.synthetic_role_bundle.v8":
        raise ValueError("role bundle schema changed")
    if payload["role"] != expected_role:
        raise ValueError("role bundle role mismatch")
    v8_inputs.validate_inputs(payload["inputs"])
    label_keys = {
        "chronological_age",
        "strict_healthy_reference",
        "duration_months",
        "event",
        "at_risk",
    }
    if set(payload["labels"]) != label_keys:
        raise ValueError("role label schema changed")
    n = len(payload["participant_ids"])
    if any(len(value) != n for value in payload["inputs"].values()):
        raise ValueError("role input length changed")
    if any(len(value) != n for value in payload["labels"].values()):
        raise ValueError("role label length changed")
    return RoleBundle(
        role=str(payload["role"]),
        outer_fold=int(payload["outer_fold"]),
        participant_ids=tuple(str(value) for value in payload["participant_ids"]),
        inputs=payload["inputs"],
        labels=payload["labels"],
    )


def create_physically_separate_synthetic_roles(
    root: Path,
    *,
    outer_fold: int = 0,
    train_n: int = 16,
    calibration_n: int = 16,
    outer_n: int = 16,
) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    ids: dict[str, set[str]] = {}
    for ordinal, (role, n, seed) in enumerate(
        (("train", train_n, 8101), ("calibration", calibration_n, 8201), ("outer_test", outer_n, 8301)),
        start=1,
    ):
        batch = synthetic_batch(n, seed=seed)
        bundle = RoleBundle(
            role=role,
            outer_fold=outer_fold,
            participant_ids=tuple(f"{role}_{index:04d}" for index in range(n)),
            inputs=batch.inputs,
            labels=batch.labels,
        )
        path = root / f"{ordinal:02d}_{role}" / "role_bundle.pt"
        roles[role] = write_role_bundle_create_only(path, bundle)
        ids[role] = set(bundle.participant_ids)
    if any(ids[a].intersection(ids[b]) for a, b in (("train", "calibration"), ("train", "outer_test"), ("calibration", "outer_test"))):
        raise RuntimeError("synthetic role participant IDs overlap")
    return roles


def load_revision0710_role(
    *,
    role_manifest_path: Path,
    expected_manifest_sha256: str,
    cache_dir: Path,
    outer_fold: int,
    expected_role: str,
    cache_artifact_bindings: dict[str, Any],
    expected_cache_artifact_bindings_sha256: str,
) -> tuple[RoleBundle, torch.Tensor, torch.Tensor]:
    """Production loader for one physically isolated canonical role manifest."""

    if (
        v8_contract.cache_artifact_bindings_sha256(cache_artifact_bindings)
        != expected_cache_artifact_bindings_sha256
    ):
        raise ValueError("cache artifact binding SHA mismatch")
    v8_contract.validate_cache_artifact_bindings(
        cache_artifact_bindings, PACKAGE_ROOT, verify_hashes=False
    )
    if cache_dir.resolve() != (REPO_ROOT / v8_contract.CACHE_RELATIVE_ROOT).resolve():
        raise ValueError("cache directory changed")
    if sha256_file(role_manifest_path) != expected_manifest_sha256:
        raise ValueError("role manifest SHA mismatch")
    manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("role") != expected_role or int(manifest.get("outer_fold", -1)) != outer_fold:
        raise ValueError("role manifest identity changed")
    artifact = manifest["artifact"]
    selected_fold_ids = tuple(int(value) for value in artifact["fold_ids"])
    expected_participant_order = _validate_cache_files_for_load(
        cache_dir=cache_dir,
        bindings=cache_artifact_bindings,
        outer_fold=outer_fold,
        fold_ids=selected_fold_ids,
    )
    age_path = Path(artifact["age_targets"]["path"])
    labels_path = Path(artifact["joint_labels"]["path"])
    if sha256_file(age_path) != artifact["age_targets"]["sha256"]:
        raise ValueError("role age-target SHA mismatch")
    if sha256_file(labels_path) != artifact["joint_labels"]["sha256"]:
        raise ValueError("role joint-label SHA mismatch")
    from ehr_age_data import load_restricted_compact_cache

    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=outer_fold,
        fold_ids=selected_fold_ids,
        age_target_cache_path=age_path,
    )
    participant_ids = tuple(cache.dataset.participant_ids)
    if participant_ids != expected_participant_order:
        raise ValueError("cache participant order/content binding changed")
    if len(participant_ids) != int(artifact["participant_count"]):
        raise ValueError("role participant count changed")
    participant_ids_sha256 = v8_contract.canonical_json_sha256(
        {"participant_ids": sorted(str(value) for value in participant_ids)}
    )
    if participant_ids_sha256 != artifact["participant_ids_canonical_sha256"]:
        raise ValueError("role participant ID coverage changed")
    if int(artifact["participant_organ_rows"]) != len(participant_ids) * len(v8_contract.ORGAN_SYSTEMS):
        raise ValueError("role participant-organ row count changed")
    labels = pd.read_parquet(labels_path)
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels["organ_system"] = labels["organ_system"].astype(str)
    expected_index = pd.MultiIndex.from_product(
        [participant_ids, v8_contract.ORGAN_SYSTEMS], names=["participant_id", "organ_system"]
    )
    indexed = labels.set_index(["participant_id", "organ_system"]).reindex(expected_index)
    if indexed[["duration_months", "event", "at_risk"]].isna().any().any():
        raise ValueError("role joint labels are incomplete")
    n = len(participant_ids)
    duration = indexed["duration_months"].to_numpy(np.float32).reshape(n, 7)
    event = indexed["event"].to_numpy(np.int64).reshape(n, 7)
    at_risk = indexed["at_risk"].to_numpy(np.int64).reshape(n, 7)
    strict_matrix = indexed["strict_healthy_reference_60m"].to_numpy(np.int64).reshape(n, 7)
    if not np.all(strict_matrix == strict_matrix[:, :1]):
        raise ValueError("strict-reference flag differs across organs")
    bundle = RoleBundle(
        role=expected_role,
        outer_fold=outer_fold,
        participant_ids=participant_ids,
        inputs=cache.dataset.inputs,
        labels={
            "chronological_age": cache.dataset.chronological_age,
            "strict_healthy_reference": torch.from_numpy(strict_matrix[:, 0]),
            "duration_months": torch.from_numpy(duration),
            "event": torch.from_numpy(event.astype(bool)),
            "at_risk": torch.from_numpy(at_risk.astype(bool)),
        },
    )
    return bundle, cache.code_embedding, cache.text_embedding


def _normalize_manifest_path(value: str) -> str:
    return str(value).replace("\\", "/")


def _validate_cache_files_for_load(
    *,
    cache_dir: Path,
    bindings: dict[str, Any],
    outer_fold: int,
    fold_ids: tuple[int, ...],
) -> tuple[str, ...]:
    """Rehash and validate every cache file the role loader will open."""

    batch_record = bindings["batch_manifest"]
    batch_path = cache_dir / str(batch_record["path"])
    if batch_path.stat().st_size != int(batch_record["bytes"]) or sha256_file(batch_path) != batch_record["sha256"]:
        raise ValueError("batch manifest bytes/SHA changed")
    observed_manifest = json.loads(batch_path.read_text(encoding="utf-8"))
    if not isinstance(observed_manifest, list) or len(observed_manifest) != 60:
        raise ValueError("batch manifest schema/cardinality changed")
    bound_shards = bindings["shards"]
    for ordinal, (observed, bound) in enumerate(zip(observed_manifest, bound_shards, strict=True)):
        if set(observed) != {
            "path", "fold_id", "batch_in_fold", "n", "active_events",
            "storage_active_events", "sha256",
        }:
            raise ValueError(f"batch manifest row schema changed: {ordinal}")
        normalized = dict(observed)
        normalized["path"] = _normalize_manifest_path(normalized["path"])
        expected = {
            key: bound[key]
            for key in (
                "path", "fold_id", "batch_in_fold", "n", "active_events",
                "storage_active_events", "sha256",
            )
        }
        if normalized != expected or int(bound["ordinal"]) != ordinal:
            raise ValueError(f"batch manifest order/content binding changed: {ordinal}")

    selected = [row for row in bound_shards if int(row["fold_id"]) in set(fold_ids)]
    if not selected:
        raise ValueError("cache binding selected no shards")
    participant_order: list[str] = []
    for row in selected:
        shard_path = cache_dir / str(row["path"])
        if shard_path.stat().st_size != int(row["bytes"]) or sha256_file(shard_path) != row["sha256"]:
            raise ValueError(f"cache shard bytes/SHA changed: {row['path']}")
        with np.load(shard_path, allow_pickle=False) as payload:
            if set(payload.files) != EXPECTED_CACHE_SHARD_ARRAY_KEYS:
                raise ValueError(f"cache shard array schema changed: {row['path']}")
            n = int(row["n"])
            if any(np.asarray(payload[key]).shape[0] != n for key in payload.files):
                raise ValueError(f"cache shard array length changed: {row['path']}")
            ids = np.asarray(payload["participant_id"])
            if ids.dtype.kind not in {"U", "S"}:
                raise ValueError(f"cache shard participant_id dtype changed: {row['path']}")
            observed_folds = set(np.asarray(payload["fold_id"], dtype=np.int64).tolist())
            if observed_folds != {int(row["fold_id"])}:
                raise ValueError(f"cache shard fold content changed: {row['path']}")
            participant_order.extend(str(value) for value in ids.tolist())
    if len(participant_order) != len(set(participant_order)):
        raise ValueError("selected cache shards contain duplicate participant IDs")

    transform_manifest_record = bindings["fold_transform_manifest"]
    transform_manifest_path = cache_dir / str(transform_manifest_record["path"])
    if (
        transform_manifest_path.stat().st_size != int(transform_manifest_record["bytes"])
        or sha256_file(transform_manifest_path) != transform_manifest_record["sha256"]
    ):
        raise ValueError("fold transform manifest bytes/SHA changed")
    transform_manifest = json.loads(transform_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(transform_manifest, list) or len(transform_manifest) != 5:
        raise ValueError("fold transform manifest schema/cardinality changed")
    bound_transforms = bindings["fold_transforms"]
    for ordinal, (observed, bound) in enumerate(zip(transform_manifest, bound_transforms, strict=True)):
        if set(observed) != {
            "outer_fold", "validation_fold", "train_folds", "train_n",
            "feature_hash", "train_id_hash", "path", "sha256",
        }:
            raise ValueError(f"fold transform manifest row schema changed: {ordinal}")
        normalized = dict(observed)
        normalized["path"] = _normalize_manifest_path(normalized["path"])
        expected = {
            key: bound[key]
            for key in (
                "outer_fold", "validation_fold", "train_folds", "train_n",
                "feature_hash", "train_id_hash", "path", "sha256",
            )
        }
        if normalized != expected or int(bound["ordinal"]) != ordinal:
            raise ValueError(f"fold transform manifest order/content binding changed: {ordinal}")
    transform = bound_transforms[int(outer_fold)]
    transform_path = cache_dir / str(transform["path"])
    if transform_path.stat().st_size != int(transform["bytes"]) or sha256_file(transform_path) != transform["sha256"]:
        raise ValueError("selected fold transform bytes/SHA changed")
    with np.load(transform_path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_TRANSFORM_ARRAY_KEYS:
            raise ValueError("selected fold transform array schema changed")
        if int(payload["outer_fold"]) != int(transform["outer_fold"]):
            raise ValueError("selected fold transform outer-fold content changed")
        if int(payload["validation_fold"]) != int(transform["validation_fold"]):
            raise ValueError("selected fold transform validation-fold content changed")
        if str(payload["feature_hash"]) != transform["feature_hash"]:
            raise ValueError("selected fold transform feature hash content changed")
        if str(payload["train_id_hash"]) != transform["train_id_hash"]:
            raise ValueError("selected fold transform train-id hash content changed")
    return tuple(participant_order)


def prediction_frame(
    model: torch.nn.Module,
    bundle: RoleBundle,
    *,
    variant_id: str,
    run_id: str,
    evaluation_epoch: int,
    device: torch.device,
    inference_batch_size: int,
) -> pd.DataFrame:
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    if not bundle.participant_ids:
        raise ValueError("prediction role is empty")
    model.to(device)
    model.eval()
    predicted_chunks: list[np.ndarray] = []
    risk_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(bundle.participant_ids), inference_batch_size):
            stop = min(start + inference_batch_size, len(bundle.participant_ids))
            inputs = {
                key: value[start:stop].to(device)
                for key, value in bundle.inputs.items()
            }
            output = model(**inputs)
            predicted_chunks.append(output.predicted_age.float().cpu().numpy())
            risk_chunks.append(output.risk_score.float().cpu().numpy())
    predicted = np.concatenate(predicted_chunks, axis=0)
    risk = np.concatenate(risk_chunks, axis=0)
    age = bundle.labels["chronological_age"].numpy()
    strict = bundle.labels["strict_healthy_reference"].numpy().astype(int)
    sex = bundle.inputs["sex"].numpy().astype(int)
    duration = bundle.labels["duration_months"].numpy()
    event = bundle.labels["event"].numpy().astype(int)
    at_risk = bundle.labels["at_risk"].numpy().astype(int)
    rows: list[dict[str, Any]] = []
    for participant_index, participant_id in enumerate(bundle.participant_ids):
        for organ_index, organ in enumerate(v8_contract.ORGAN_SYSTEMS):
            rows.append(
                {
                    "participant_id": participant_id,
                    "run_id": run_id,
                    "variant_id": variant_id,
                    "outer_fold": bundle.outer_fold,
                    "split_role": bundle.role,
                    "evaluation_epoch": evaluation_epoch,
                    "organ_system": organ,
                    "actual_age": float(age[participant_index]),
                    "predicted_age": float(predicted[participant_index, organ_index]),
                    "raw_age_gap": float(predicted[participant_index, organ_index] - age[participant_index]),
                    "corrected_age_gap": np.nan,
                    "sex": int(sex[participant_index]),
                    "strict_healthy_reference_60m": int(strict[participant_index]),
                    "duration_months": float(duration[participant_index, organ_index]),
                    "event": int(event[participant_index, organ_index]),
                    "at_risk": int(at_risk[participant_index, organ_index]),
                    "risk_score": float(risk[participant_index, organ_index]),
                }
            )
    return pd.DataFrame(rows)


def fit_age_correction(calibration: pd.DataFrame) -> dict[str, Any]:
    parameters = []
    strict = calibration.loc[calibration["strict_healthy_reference_60m"].eq(1)]
    for organ in v8_contract.ORGAN_SYSTEMS:
        cell = strict.loc[strict["organ_system"].eq(organ)]
        if len(cell) < 3 or cell["actual_age"].nunique() < 2:
            raise ValueError(f"calibration strict reference is insufficient for {organ}")
        design = np.column_stack([np.ones(len(cell)), cell["actual_age"].to_numpy(float)])
        coef, *_ = np.linalg.lstsq(design, cell["raw_age_gap"].to_numpy(float), rcond=None)
        parameters.append(
            {
                "organ_system": organ,
                "intercept": float(coef[0]),
                "age_slope": float(coef[1]),
                "fit_participants": int(cell["participant_id"].nunique()),
                "fit_rows": int(len(cell)),
                "fit_participant_ids_sha256": v8_contract.canonical_json_sha256(
                    {"participant_ids": sorted(cell["participant_id"].astype(str).unique().tolist())}
                ),
            }
        )
    return {
        "schema_version": "revision0710.fig04_fig05.age_correction.v8",
        "fit_role": "calibration",
        "model_selection_used": False,
        "parameters": parameters,
    }


def apply_age_correction(frame: pd.DataFrame, correction: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    by_organ = {row["organ_system"]: row for row in correction["parameters"]}
    expected = result.apply(
        lambda row: by_organ[row["organ_system"]]["intercept"]
        + by_organ[row["organ_system"]]["age_slope"] * row["actual_age"],
        axis=1,
    )
    result["corrected_age_gap"] = result["raw_age_gap"] - expected.to_numpy(float)
    return result


def fit_cox_nuisance(calibration: pd.DataFrame, *, penalizer: float) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning

    models = []
    for organ in v8_contract.ORGAN_SYSTEMS:
        cell = calibration.loc[
            calibration["organ_system"].eq(organ) & calibration["at_risk"].eq(1)
        ].copy()
        if cell["event"].sum() < 1:
            raise ValueError(f"calibration has no event for {organ}")
        for model_name, covariates in (
            ("base", ("actual_age", "sex")),
            ("full", ("actual_age", "sex", "corrected_age_gap")),
        ):
            train = cell[["duration_months", "event", *covariates]].copy()
            means = {name: float(train[name].mean()) for name in covariates}
            scales = {name: float(train[name].std(ddof=0)) for name in covariates}
            if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
                raise ValueError(f"calibration Cox covariate lacks variation for {organ}/{model_name}")
            z_names = []
            for name in covariates:
                z_name = f"z_{name}"
                train[z_name] = (train[name] - means[name]) / scales[name]
                z_names.append(z_name)
            model = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    model.fit(
                        train[["duration_months", "event", *z_names]],
                        duration_col="duration_months",
                        event_col="event",
                        show_progress=False,
                    )
                except ConvergenceWarning as exc:
                    raise ValueError(
                        f"calibration Cox convergence warning for {organ}/{model_name}"
                    ) from exc
            coefficients = {name: float(model.params_[f"z_{name}"]) for name in covariates}
            models.append(
                {
                    "organ_system": organ,
                    "model": model_name,
                    "covariates": list(covariates),
                    "means": means,
                    "scales": scales,
                    "coefficients": coefficients,
                    "fit_participants": int(cell["participant_id"].nunique()),
                    "fit_events": int(cell["event"].sum()),
                    "fit_participant_ids_sha256": v8_contract.canonical_json_sha256(
                        {"participant_ids": sorted(cell["participant_id"].astype(str).unique().tolist())}
                    ),
                    "convergence_warning": False,
                }
            )
    return {
        "schema_version": "revision0710.fig04_fig05.cox_nuisance.v8",
        "fit_role": "calibration",
        "model_selection_used": False,
        "penalizer": penalizer,
        "models": models,
    }


def score_outer_metrics(
    outer: pd.DataFrame,
    nuisance: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from lifelines.utils import concordance_index

    strict = outer.loc[outer["strict_healthy_reference_60m"].eq(1)]
    actual = strict["actual_age"].to_numpy(float)
    predicted = strict["predicted_age"].to_numpy(float)
    residual = predicted - actual
    denominator = np.square(actual - actual.mean()).sum()
    fold_metrics = {
        "strict_n_participants": int(strict["participant_id"].nunique()),
        "strict_n_participant_organ_rows": int(len(strict)),
        "raw_strict_mae": float(np.abs(residual).mean()),
        "raw_strict_pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "raw_strict_r2": float(1.0 - np.square(residual).sum() / denominator),
    }
    by_key = {(row["organ_system"], row["model"]): row for row in nuisance["models"]}
    organ_rows = []
    for organ in v8_contract.ORGAN_SYSTEMS:
        cell = outer.loc[outer["organ_system"].eq(organ) & outer["at_risk"].eq(1)].copy()
        scores = {}
        for model_name in ("base", "full"):
            model = by_key[(organ, model_name)]
            risk = np.zeros(len(cell), dtype=float)
            for name in model["covariates"]:
                risk += (
                    (cell[name].to_numpy(float) - model["means"][name])
                    / model["scales"][name]
                    * model["coefficients"][name]
                )
            scores[model_name] = float(
                concordance_index(
                    cell["duration_months"].to_numpy(float),
                    -risk,
                    cell["event"].to_numpy(bool),
                )
            )
        organ_rows.append(
            {
                "organ_system": organ,
                "at_risk_n": int(len(cell)),
                "event_n": int(cell["event"].sum()),
                "age_sex_harrell_base_c": scores["base"],
                "age_sex_gap_harrell_full_c": scores["full"],
                "age_sex_harrell_delta_c": scores["full"] - scores["base"],
            }
        )
    fold_metrics["age_sex_harrell_macro_delta_c"] = float(
        np.mean([row["age_sex_harrell_delta_c"] for row in organ_rows])
    )
    return fold_metrics, organ_rows
