from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


MODEL_INPUT_KEYS = (
    "phenotype",
    "phenotype_missing_mask",
    "sex",
    "event_code_id",
    "event_text_id",
    "event_relative_days",
    "event_type_id",
    "event_valid_mask",
)

TARGET_KEYS = (
    "chronological_age",
    "strict_healthy_reference",
    "duration_months",
    "event",
    "at_risk",
)

REQUIRED_ARRAYS = MODEL_INPUT_KEYS[2:] + TARGET_KEYS + ("phenotype", "fold")


@dataclass(frozen=True)
class Standardization:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "median": torch.from_numpy(self.median.copy()),
            "mean": torch.from_numpy(self.mean.copy()),
            "std": torch.from_numpy(self.std.copy()),
        }


@dataclass(frozen=True)
class DataBundle:
    dataset: "EHRDataset"
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    outer_indices: np.ndarray
    standardization: Standardization


class EHRDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        lengths = {len(value) for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("all tensors must contain the same number of participants")
        self.tensors = tensors

    def __len__(self) -> int:
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def load_embeddings(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            code = torch.from_numpy(np.asarray(archive["code_embedding"], dtype=np.float32))
            text = torch.from_numpy(np.asarray(archive["text_embedding"], dtype=np.float32))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        code = torch.as_tensor(payload["code_embedding"], dtype=torch.float32)
        text = torch.as_tensor(payload["text_embedding"], dtype=torch.float32)
    if code.ndim != 2 or text.ndim != 2 or code.shape[1] != text.shape[1]:
        raise ValueError("code and text embeddings must be aligned matrices")
    if code.shape[0] < 2 or text.shape[0] < 2:
        raise ValueError("embedding tables must include padding and active rows")
    if not torch.isfinite(code).all() or not torch.isfinite(text).all():
        raise ValueError("embedding tables must be finite")
    return code, text


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_ARRAYS if key not in archive.files]
        if missing:
            raise ValueError(f"missing arrays: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in REQUIRED_ARRAYS}
    validate_arrays(arrays)
    return arrays


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    n = len(arrays["phenotype"])
    if arrays["phenotype"].shape != (n, 78):
        raise ValueError("phenotype must have shape [N,78]")
    if arrays["sex"].shape not in ((n,), (n, 1)):
        raise ValueError("sex must have shape [N] or [N,1]")
    if arrays["chronological_age"].shape != (n,):
        raise ValueError("chronological_age must have shape [N]")
    if arrays["strict_healthy_reference"].shape != (n,):
        raise ValueError("strict_healthy_reference must have shape [N]")
    if arrays["fold"].shape != (n,):
        raise ValueError("fold must have shape [N]")
    sequence_shape = (n, 32)
    for key in (
        "event_code_id",
        "event_text_id",
        "event_relative_days",
        "event_type_id",
        "event_valid_mask",
    ):
        if arrays[key].shape != sequence_shape:
            raise ValueError(f"{key} must have shape [N,32]")
    target_shape = (n, 7)
    for key in ("duration_months", "event", "at_risk"):
        if arrays[key].shape != target_shape:
            raise ValueError(f"{key} must have shape [N,7]")
    if not np.isin(arrays["fold"], np.arange(5)).all():
        raise ValueError("fold must contain values 0 to 4")
    valid = arrays["event_valid_mask"].astype(bool)
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("active events must be left aligned")
    if np.any(valid & (arrays["event_relative_days"] >= 0)):
        raise ValueError("active event times must be strictly pre-index")
    if np.any(valid & (arrays["event_type_id"] == 2) & (arrays["event_code_id"] != 0)):
        raise ValueError("GP prescription slots must use code_id=0")
    if not np.isfinite(arrays["sex"]).all() or not np.isfinite(arrays["chronological_age"]).all():
        raise ValueError("sex and chronological age must be finite")
    if np.isinf(arrays["phenotype"]).any():
        raise ValueError("phenotype cannot contain infinite values")
    event = arrays["event"].astype(bool)
    at_risk = arrays["at_risk"].astype(bool)
    duration = arrays["duration_months"].astype(np.float32)
    if np.any(event & ~at_risk):
        raise ValueError("events must occur within the at-risk population")
    if np.any(duration[at_risk] <= 0) or np.any(duration[at_risk] > 60):
        raise ValueError("at-risk durations must lie in (0,60]")


def fit_standardization(phenotype: np.ndarray) -> Standardization:
    phenotype = np.asarray(phenotype, dtype=np.float32)
    median = np.zeros(phenotype.shape[1], dtype=np.float32)
    for column in range(phenotype.shape[1]):
        observed = phenotype[np.isfinite(phenotype[:, column]), column]
        median[column] = float(np.median(observed)) if len(observed) else 0.0
    imputed = np.where(np.isfinite(phenotype), phenotype, median)
    mean = imputed.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = imputed.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Standardization(median=median, mean=mean, std=std)


def apply_standardization(
    phenotype: np.ndarray, standardization: Standardization
) -> tuple[np.ndarray, np.ndarray]:
    phenotype = np.asarray(phenotype, dtype=np.float32)
    missing = ~np.isfinite(phenotype)
    imputed = np.where(missing, standardization.median, phenotype)
    standardized = (imputed - standardization.mean) / standardization.std
    return standardized.astype(np.float32), missing.astype(bool)


def prepare_arrays(arrays: dict[str, np.ndarray], outer_fold: int) -> DataBundle:
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be between 0 and 4")
    folds = arrays["fold"].astype(np.int64)
    calibration_fold = (outer_fold + 1) % 5
    train_indices = np.flatnonzero((folds != outer_fold) & (folds != calibration_fold))
    calibration_indices = np.flatnonzero(folds == calibration_fold)
    outer_indices = np.flatnonzero(folds == outer_fold)
    if min(len(train_indices), len(calibration_indices), len(outer_indices)) == 0:
        raise ValueError("train, calibration, and outer roles must all be nonempty")
    standardization = fit_standardization(arrays["phenotype"][train_indices])
    phenotype, phenotype_missing = apply_standardization(
        arrays["phenotype"], standardization
    )
    tensors = {
        "phenotype": torch.from_numpy(phenotype),
        "phenotype_missing_mask": torch.from_numpy(phenotype_missing),
        "sex": torch.from_numpy(arrays["sex"].reshape(-1).astype(np.float32)),
        "event_code_id": torch.from_numpy(arrays["event_code_id"].astype(np.int64)),
        "event_text_id": torch.from_numpy(arrays["event_text_id"].astype(np.int64)),
        "event_relative_days": torch.from_numpy(
            arrays["event_relative_days"].astype(np.float32)
        ),
        "event_type_id": torch.from_numpy(arrays["event_type_id"].astype(np.int64)),
        "event_valid_mask": torch.from_numpy(arrays["event_valid_mask"].astype(bool)),
        "chronological_age": torch.from_numpy(
            arrays["chronological_age"].astype(np.float32)
        ),
        "strict_healthy_reference": torch.from_numpy(
            arrays["strict_healthy_reference"].astype(bool)
        ),
        "duration_months": torch.from_numpy(arrays["duration_months"].astype(np.float32)),
        "event": torch.from_numpy(arrays["event"].astype(bool)),
        "at_risk": torch.from_numpy(arrays["at_risk"].astype(bool)),
    }
    return DataBundle(
        dataset=EHRDataset(tensors),
        train_indices=train_indices,
        calibration_indices=calibration_indices,
        outer_indices=outer_indices,
        standardization=standardization,
    )


def load_data_bundle(path: str | Path, outer_fold: int) -> DataBundle:
    return prepare_arrays(load_arrays(path), outer_fold)


def make_loaders(
    bundle: DataBundle,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train": DataLoader(
            Subset(bundle.dataset, bundle.train_indices.tolist()),
            shuffle=True,
            generator=generator,
            **common,
        ),
        "calibration": DataLoader(
            Subset(bundle.dataset, bundle.calibration_indices.tolist()),
            shuffle=False,
            **common,
        ),
        "outer": DataLoader(
            Subset(bundle.dataset, bundle.outer_indices.tolist()),
            shuffle=False,
            **common,
        ),
    }


def make_synthetic_bundle(
    n: int = 80,
    seed: int = 7210,
) -> tuple[DataBundle, torch.Tensor, torch.Tensor]:
    if n < 20:
        raise ValueError("synthetic smoke data require at least 20 participants")
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 73.0, size=n).astype(np.float32)
    phenotype = rng.normal(size=(n, 78)).astype(np.float32)
    phenotype += ((age - age.mean()) / age.std())[:, None] * rng.normal(
        0.0, 0.15, size=(1, 78)
    ).astype(np.float32)
    phenotype[rng.random((n, 78)) < 0.08] = np.nan
    event_code_id = np.zeros((n, 32), dtype=np.int64)
    event_text_id = np.zeros((n, 32), dtype=np.int64)
    event_relative_days = np.zeros((n, 32), dtype=np.float32)
    event_type_id = np.zeros((n, 32), dtype=np.int64)
    event_valid_mask = np.zeros((n, 32), dtype=bool)
    for row in range(n):
        length = int(rng.integers(0, 9))
        if length == 0:
            continue
        event_valid_mask[row, :length] = True
        event_relative_days[row, :length] = -np.sort(
            rng.integers(1, 5000, size=length)
        )[::-1]
        types = rng.choice(np.array([1, 2]), size=length, p=np.array([0.65, 0.35]))
        event_type_id[row, :length] = types
        event_text_id[row, :length] = rng.integers(1, 256, size=length)
        icd = types == 1
        event_code_id[row, :length][icd] = rng.integers(1, 128, size=int(icd.sum()))
    at_risk = rng.random((n, 7)) > 0.08
    duration = rng.uniform(1.0, 60.0, size=(n, 7)).astype(np.float32)
    duration[~at_risk] = 0.0
    event = (rng.random((n, 7)) < 0.18) & at_risk
    arrays = {
        "phenotype": phenotype,
        "sex": rng.integers(0, 2, size=n).astype(np.float32),
        "event_code_id": event_code_id,
        "event_text_id": event_text_id,
        "event_relative_days": event_relative_days,
        "event_type_id": event_type_id,
        "event_valid_mask": event_valid_mask,
        "chronological_age": age,
        "strict_healthy_reference": (rng.random(n) < 0.25),
        "duration_months": duration,
        "event": event,
        "at_risk": at_risk,
        "fold": np.arange(n, dtype=np.int64) % 5,
    }
    validate_arrays(arrays)
    bundle = prepare_arrays(arrays, outer_fold=0)
    code_embedding = torch.randn(128, 512, generator=torch.Generator().manual_seed(seed))
    text_embedding = torch.randn(256, 512, generator=torch.Generator().manual_seed(seed + 1))
    code_embedding[0].zero_()
    text_embedding[0].zero_()
    return bundle, code_embedding, text_embedding
