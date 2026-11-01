from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
REBUILD_DIR = ROOT / "Revision-0710" / "01_data_rebuild"
CORE_DIR = REBUILD_DIR / "scripts"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from compact_cache_core import (  # noqa: E402
    EVENT_TYPE_IDS,
    TEXT_ROLES,
    build_vocab,
    event_text_keys,
    fit_phenotype_transform,
    pack_compact_sequences,
    select_unified_timeline,
)


DEFAULT_SOURCE_CONFIG = REBUILD_DIR / "00_config" / "source_paths.json"
DEFAULT_CACHE_CONFIG = REBUILD_DIR / "04_cache_v1" / "cache_config.json"
DEFAULT_INDEX_DIR = REBUILD_DIR / "01_index_and_events"
DEFAULT_FOLD_DIR = REBUILD_DIR / "03_folds"
DEFAULT_OUTPUT_DIR = REBUILD_DIR / "04_cache_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_clip_pretrained(model_name: str, requested: str) -> Path | str:
    requested_path = Path(requested)
    if requested_path.exists():
        return requested_path.resolve()
    if model_name == "ViT-B-16" and requested == "openai":
        cache_root = (
            Path.home()
            / ".cache/huggingface/hub/models--timm--vit_base_patch16_clip_224.openai/snapshots"
        )
        candidates = sorted(
            cache_root.glob("*/open_clip_model.safetensors"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.stat().st_size > 100_000_000:
                return candidate.resolve()
    return requested


def encode_text_table(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    raw_text: list[str],
    prompts: list[str],
    device: torch.device,
    batch_size: int,
    embedding_dim: int,
) -> tuple[torch.Tensor, float]:
    output = torch.zeros((len(raw_text), embedding_dim), dtype=torch.float32)
    formatted = [prompt.format(text=text) if text else "" for text, prompt in zip(raw_text, prompts)]
    nonempty = [(index, text) for index, text in enumerate(formatted) if text]
    for start in range(0, len(nonempty), batch_size):
        batch = nonempty[start : start + batch_size]
        indices = [index for index, _ in batch]
        tokens = tokenizer([text for _, text in batch]).to(device)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            encoded = model.encode_text(tokens)
        encoded = torch.nan_to_num(encoded.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        output[torch.as_tensor(indices, dtype=torch.long)] = encoded
        if start == 0 or (start // batch_size + 1) % 10 == 0:
            print(f"Encoded {min(start + batch_size, len(nonempty)):,}/{len(nonempty):,} texts", flush=True)
    if not torch.isfinite(output).all():
        raise ValueError("CLIP embedding table contains non-finite values")
    half = output.half()
    sample = output[1 : min(len(output), 4097)]
    sample_half = half[1 : min(len(output), 4097)].float()
    if len(sample):
        cosine = torch.nn.functional.cosine_similarity(sample, sample_half, dim=1)
        min_cosine = float(cosine.min())
    else:
        min_cosine = 1.0
    half[0].zero_()
    return half, min_cosine


def build_embedding_tables(
    *,
    output_dir: Path,
    cache_config: dict[str, Any],
    encode_batch_size: int,
) -> dict[str, Any]:
    global torch
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise SystemExit("open_clip is required; run this stage in the Torch_25 environment") from exc

    code_vocab = pd.read_csv(output_dir / "vocab" / "event_code_vocab.csv.gz")
    text_vocab = pd.read_csv(output_dir / "vocab" / "event_text_vocab.csv.gz")
    model_name = str(cache_config["clip_model"])
    resolved = resolve_clip_pretrained(model_name, str(cache_config["clip_pretrained"]))
    print(f"Loading {model_name} text encoder from {resolved}", flush=True)
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=str(resolved))
    tokenizer = open_clip.get_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    embedding_dim = int(getattr(model, "text_projection").shape[-1])

    code_prompts = [str(cache_config["code_prompt"])] * len(code_vocab)
    role_to_prompt = {
        "icd_description": str(cache_config["description_prompt"]),
        "gp_drug": str(cache_config["drug_prompt"]),
        "padding": "{text}",
    }
    text_prompts = [role_to_prompt[role] for role in text_vocab["semantic_role"].tolist()]
    print(f"Encoding {len(code_vocab) - 1:,} ICD codes", flush=True)
    code_embedding, code_cosine = encode_text_table(
        model=model,
        tokenizer=tokenizer,
        raw_text=code_vocab["raw_text"].fillna("").astype(str).tolist(),
        prompts=code_prompts,
        device=device,
        batch_size=encode_batch_size,
        embedding_dim=embedding_dim,
    )
    print(f"Encoding {len(text_vocab) - 1:,} namespaced descriptions/drugs", flush=True)
    text_embedding, text_cosine = encode_text_table(
        model=model,
        tokenizer=tokenizer,
        raw_text=text_vocab["raw_text"].fillna("").astype(str).tolist(),
        prompts=text_prompts,
        device=device,
        batch_size=encode_batch_size,
        embedding_dim=embedding_dim,
    )
    table_path = output_dir / "embedding_tables.pt"
    torch.save(
        {
            "code_embedding": code_embedding.cpu(),
            "text_embedding": text_embedding.cpu(),
            "padding_id": 0,
            "embedding_dim": embedding_dim,
        },
        table_path,
    )
    numpy_table_path = output_dir / "embedding_tables.npz"
    np.savez(
        numpy_table_path,
        code_embedding=code_embedding.cpu().numpy(),
        text_embedding=text_embedding.cpu().numpy(),
    )
    weight_sha = sha256_file(resolved) if isinstance(resolved, Path) else None
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "pretrained_requested": cache_config["clip_pretrained"],
        "pretrained_resolved": str(resolved),
        "pretrained_sha256": weight_sha,
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": str(device),
        "embedding_dim": embedding_dim,
        "dtype": "float16",
        "normalized": bool(cache_config["embedding_normalized"]),
        "code_vocab_n": int(len(code_vocab)),
        "text_vocab_n": int(len(text_vocab)),
        "code_fp16_min_cosine_vs_fp32": code_cosine,
        "text_fp16_min_cosine_vs_fp32": text_cosine,
        "padding_rows_zero": bool(
            torch.count_nonzero(code_embedding[0]).item() == 0
            and torch.count_nonzero(text_embedding[0]).item() == 0
        ),
        "embedding_table_sha256": sha256_file(table_path),
        "embedding_numpy_sha256": sha256_file(numpy_table_path),
    }
    (output_dir / "embedding_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return metadata


def modality_starvation_audit(
    *,
    icd: pd.DataFrame,
    gp: pd.DataFrame,
    stored: pd.DataFrame,
    primary: pd.DataFrame,
) -> dict[str, Any]:
    icd_raw = icd.groupby("participant_id").size()
    gp_raw = gp.groupby("participant_id").size()
    raw = pd.concat([icd_raw.rename("icd_raw"), gp_raw.rename("gp_raw")], axis=1).fillna(0)
    storage_counts = pd.crosstab(stored["participant_id"], stored["source"])
    primary_counts = pd.crosstab(primary["participant_id"], primary["source"])
    audit = raw.join(storage_counts.add_prefix("storage_"), how="left").join(
        primary_counts.add_prefix("primary_"), how="left"
    ).fillna(0)
    dual = audit["icd_raw"].gt(0) & audit["gp_raw"].gt(0)
    storage_icd = audit.get("storage_hospital_icd", pd.Series(0, index=audit.index))
    storage_gp = audit.get("storage_gp_prescription", pd.Series(0, index=audit.index))
    primary_icd = audit.get("primary_hospital_icd", pd.Series(0, index=audit.index))
    primary_gp = audit.get("primary_gp_prescription", pd.Series(0, index=audit.index))

    stored_counts = stored.groupby("participant_id").size()
    primary_total = primary.groupby("participant_id").size()
    truncated_ids = stored_counts[stored_counts.gt(primary_total.reindex(stored_counts.index, fill_value=0))].index
    stored_by_day = stored.groupby(["participant_id", "event_date"]).size().rename("storage_n")
    primary_by_day = primary.groupby(["participant_id", "event_date"]).size().rename("primary_n")
    day_counts = stored_by_day.to_frame().join(primary_by_day, how="left").fillna(0)
    cutoff_split = day_counts["primary_n"].gt(0) & day_counts["primary_n"].lt(day_counts["storage_n"])

    dual_n = int(dual.sum())
    primary_icd_starved = int((dual & primary_icd.eq(0)).sum())
    primary_gp_starved = int((dual & primary_gp.eq(0)).sum())
    return {
        "storage_length": int(stored.groupby("participant_id").size().max()),
        "primary_length": int(primary.groupby("participant_id").size().max()),
        "participants_with_any_icd": int(icd_raw.size),
        "participants_with_any_gp": int(gp_raw.size),
        "participants_with_both_modalities_raw": dual_n,
        "storage_icd_starved_among_raw_dual_n": int((dual & storage_icd.eq(0)).sum()),
        "storage_gp_starved_among_raw_dual_n": int((dual & storage_gp.eq(0)).sum()),
        "primary_icd_starved_among_raw_dual_n": primary_icd_starved,
        "primary_gp_starved_among_raw_dual_n": primary_gp_starved,
        "primary_icd_starved_among_raw_dual_fraction": primary_icd_starved / dual_n if dual_n else 0.0,
        "primary_gp_starved_among_raw_dual_fraction": primary_gp_starved / dual_n if dual_n else 0.0,
        "participants_truncated_at_primary_length": int(len(truncated_ids)),
        "participants_with_any_same_day_group_split": int(
            cutoff_split.groupby(level="participant_id").any().sum()
        ),
        "primary_rule_review_recommended": bool(
            dual_n and (primary_icd_starved / dual_n > 0.01 or primary_gp_starved / dual_n > 0.01)
        ),
    }


def build_compact_cache(
    *,
    source_config: dict[str, Any],
    cache_config: dict[str, Any],
    index_dir: Path,
    fold_dir: Path,
    output_dir: Path,
    max_participants: int | None,
) -> dict[str, Any]:
    storage_length = int(cache_config["storage_sequence_length"])
    primary_length = int(cache_config["primary_train_sequence_length"])
    storage_minimum = {
        "hospital_icd": int(cache_config["storage_minimum_icd"]),
        "gp_prescription": int(cache_config["storage_minimum_gp"]),
    }
    primary_minimum = {
        "hospital_icd": int(cache_config["balanced_crop_sensitivity"]["minimum_icd"]),
        "gp_prescription": int(cache_config["balanced_crop_sensitivity"]["minimum_gp"]),
    }
    batch_size = int(cache_config["participant_batch_size"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    (output_dir / "vocab").mkdir(parents=True, exist_ok=True)
    (output_dir / "fold_transforms").mkdir(parents=True, exist_ok=True)
    (output_dir / "targets").mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(index_dir / "age_model_cohort.parquet")
    folds = pd.read_parquet(fold_dir / "fold_manifest_5fold.parquet")
    participant = cohort.merge(
        folds[["participant_id", "fold_id"]], on="participant_id", validate="one_to_one"
    ).sort_values(["fold_id", "participant_id"], kind="mergesort")
    if max_participants is not None:
        per_fold = max(1, int(np.ceil(max_participants / 5)))
        participant = (
            participant.groupby("fold_id", sort=True, group_keys=False)
            .head(per_fold)
            .head(max_participants)
            .copy()
        )
    participant_ids = participant["participant_id"].astype(str).tolist()
    allowed = set(participant_ids)

    gp_limit = int(source_config["gp_source_retention_limit"])
    icd = pd.read_parquet(index_dir / "canonical_preindex_icd.parquet")
    gp = pd.read_parquet(index_dir / f"canonical_preindex_gp_l{gp_limit}.parquet")
    if max_participants is not None:
        icd = icd[icd["participant_id"].astype(str).isin(allowed)].copy()
        gp = gp[gp["participant_id"].astype(str).isin(allowed)].copy()
    print(
        f"Selecting unified L{storage_length} timeline from {len(icd):,} ICD and {len(gp):,} GP rows",
        flush=True,
    )
    timeline = select_unified_timeline(
        icd,
        gp,
        participant_ids=participant_ids,
        max_len=storage_length,
        minimum_per_source=storage_minimum,
    )
    timeline.to_parquet(
        output_dir / f"storage_timeline_balanced_l{storage_length}.parquet",
        index=False,
        compression="zstd",
    )
    primary_timeline = select_unified_timeline(
        timeline[timeline["source"].eq("hospital_icd")],
        timeline[timeline["source"].eq("gp_prescription")],
        participant_ids=participant_ids,
        max_len=primary_length,
        minimum_per_source=primary_minimum,
    )
    primary_timeline.to_parquet(
        output_dir / f"primary_timeline_balanced_l{primary_length}.parquet",
        index=False,
        compression="zstd",
    )

    code_vocab, code_to_id = build_vocab(timeline["code"])
    text_keys = event_text_keys(timeline)
    text_vocab, text_to_id = build_vocab(text_keys)
    code_frame = pd.DataFrame(
        {"vocab_id": np.arange(len(code_vocab), dtype=np.int32), "raw_text": code_vocab}
    )
    code_frame["semantic_role"] = np.where(code_frame["vocab_id"].eq(0), "padding", "icd_code")
    text_rows = []
    for vocab_id, key in enumerate(text_vocab):
        if vocab_id == 0:
            text_rows.append({"vocab_id": 0, "semantic_role": "padding", "raw_text": ""})
        else:
            role, raw_text = key.split("\x1f", maxsplit=1)
            text_rows.append({"vocab_id": vocab_id, "semantic_role": role, "raw_text": raw_text})
    text_frame = pd.DataFrame(text_rows)
    code_frame.to_parquet(output_dir / "vocab" / "event_code_vocab.parquet", index=False)
    text_frame.to_parquet(output_dir / "vocab" / "event_text_vocab.parquet", index=False)
    code_frame.to_csv(output_dir / "vocab" / "event_code_vocab.csv.gz", index=False, compression="gzip")
    text_frame.to_csv(output_dir / "vocab" / "event_text_vocab.csv.gz", index=False, compression="gzip")

    print("Packing compact event arrays", flush=True)
    storage_packed = pack_compact_sequences(
        timeline,
        participant_order=participant_ids,
        max_len=storage_length,
        code_to_id=code_to_id,
        text_to_id=text_to_id,
    )
    packed = pack_compact_sequences(
        primary_timeline,
        participant_order=participant_ids,
        max_len=primary_length,
        code_to_id=code_to_id,
        text_to_id=text_to_id,
    )
    starvation = modality_starvation_audit(
        icd=icd,
        gp=gp,
        stored=timeline,
        primary=primary_timeline,
    )
    (output_dir / "modality_starvation_audit.json").write_text(
        json.dumps(starvation, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    phenotype = pd.read_parquet(index_dir / "raw_phenotypes.parquet").set_index("participant_id")
    phenotype_missing = pd.read_parquet(index_dir / "phenotype_missing_mask.parquet").set_index("participant_id")
    feature_columns = json.loads((index_dir / "phenotype_feature_columns.json").read_text(encoding="utf-8"))
    phenotype = phenotype.loc[participant_ids]
    phenotype_missing = phenotype_missing.loc[participant_ids]
    raw_values = phenotype[feature_columns].to_numpy(dtype=np.float32)
    missing_mask = phenotype_missing[feature_columns].to_numpy(dtype=bool)
    if not np.array_equal(np.isnan(raw_values), missing_mask):
        raise ValueError("raw phenotype NaN positions disagree with stored missing mask")
    raw_zero = np.where(missing_mask, 0.0, raw_values).astype(np.float32)
    sex_text = phenotype["sex"].astype("string").fillna("").str.strip().str.lower()
    sex = sex_text.map({"female": 0, "male": 1})
    if sex.isna().any():
        raise ValueError(f"unmapped sex rows: {int(sex.isna().sum())}")
    sex_array = sex.to_numpy(dtype=np.int8)
    fold_array = participant["fold_id"].to_numpy(dtype=np.int16)

    targets = participant[["participant_id", "fold_id", "chronological_age"]].copy()
    targets.to_parquet(output_dir / "targets" / "chronological_age_targets.parquet", index=False)
    batch_manifest = []
    for fold_id in range(5):
        fold_positions = np.flatnonzero(fold_array == fold_id)
        for batch_in_fold, start in enumerate(range(0, len(fold_positions), batch_size)):
            positions = fold_positions[start : start + batch_size]
            payload = {
                "participant_id": np.asarray([participant_ids[index] for index in positions]),
                "participant_row": positions.astype(np.int32),
                "fold_id": fold_array[positions],
                "phenotype_raw_zero": raw_zero[positions],
                "phenotype_missing_mask": missing_mask[positions],
                "sex": sex_array[positions],
                "event_code_id": packed["event_code_id"][positions],
                "event_text_id": packed["event_text_id"][positions],
                "event_relative_days": packed["event_relative_days"][positions],
                "event_type_id": packed["event_type_id"][positions],
                "event_valid_mask": packed["event_valid_mask"][positions],
                "event_count": packed["event_count"][positions],
                "storage_event_code_id": storage_packed["event_code_id"][positions],
                "storage_event_text_id": storage_packed["event_text_id"][positions],
                "storage_event_relative_days": storage_packed["event_relative_days"][positions],
                "storage_event_type_id": storage_packed["event_type_id"][positions],
                "storage_event_valid_mask": storage_packed["event_valid_mask"][positions],
                "storage_event_count": storage_packed["event_count"][positions],
            }
            batch_path = output_dir / "shards" / f"fold{fold_id}_batch{batch_in_fold:03d}.npz"
            np.savez_compressed(batch_path, **payload)
            batch_manifest.append(
                {
                    "path": str(batch_path.relative_to(output_dir)),
                    "fold_id": fold_id,
                    "batch_in_fold": batch_in_fold,
                    "n": int(len(positions)),
                    "active_events": int(packed["event_valid_mask"][positions].sum()),
                    "storage_active_events": int(
                        storage_packed["event_valid_mask"][positions].sum()
                    ),
                    "sha256": sha256_file(batch_path),
                }
            )

    transform_manifest = []
    feature_hash = sha256_json(feature_columns)
    for outer_fold in range(5):
        validation_fold = (outer_fold + 1) % 5
        train_mask = (fold_array != outer_fold) & (fold_array != validation_fold)
        transform = fit_phenotype_transform(raw_zero, missing_mask, train_mask=train_mask)
        train_ids = [participant_ids[index] for index in np.flatnonzero(train_mask)]
        train_id_hash = hashlib.sha256("\n".join(train_ids).encode("utf-8")).hexdigest()
        path = output_dir / "fold_transforms" / f"outer_fold_{outer_fold}.npz"
        np.savez_compressed(
            path,
            median=transform["median"],
            mean=transform["mean"],
            std=transform["std"],
            feature_hash=np.asarray(feature_hash),
            train_id_hash=np.asarray(train_id_hash),
            outer_fold=np.asarray(outer_fold, dtype=np.int8),
            validation_fold=np.asarray(validation_fold, dtype=np.int8),
        )
        transform_manifest.append(
            {
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "train_folds": sorted(set(range(5)) - {outer_fold, validation_fold}),
                "train_n": int(train_mask.sum()),
                "feature_hash": feature_hash,
                "train_id_hash": train_id_hash,
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
        )

    forbidden_cache_keys = {
        "event",
        "duration",
        "death",
        "outcome",
        "strict_healthy",
        "future_disease",
        "system_mask",
        "survival_time",
        "reported_age_at_recruitment",
        "chronological_age",
        "label",
    }
    observed_keys = set()
    for path in sorted((output_dir / "shards").glob("*.npz"))[:1]:
        with np.load(path, allow_pickle=False) as payload:
            observed_keys.update(payload.files)
    forbidden_found = sorted(observed_keys & forbidden_cache_keys)
    active = packed["event_valid_mask"]
    inactive = ~active
    storage_active = storage_packed["event_valid_mask"]
    storage_inactive = ~storage_active
    cache_gate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not forbidden_found else "FAIL",
        "participants": int(len(participant_ids)),
        "storage_sequence_length": storage_length,
        "primary_train_sequence_length": primary_length,
        "storage_crop_rule": cache_config["storage_crop_rule"],
        "primary_crop_rule": cache_config["primary_crop_rule"],
        "storage_timeline_rows": int(len(timeline)),
        "primary_timeline_rows": int(len(primary_timeline)),
        "participants_with_events": int((packed["event_count"] > 0).sum()),
        "participants_without_events": int((packed["event_count"] == 0).sum()),
        "active_event_slots": int(active.sum()),
        "nonnegative_active_time_slots": int((packed["event_relative_days"][active] >= 0).sum()),
        "inactive_nonzero_code_slots": int((packed["event_code_id"][inactive] != 0).sum()),
        "inactive_nonzero_text_slots": int((packed["event_text_id"][inactive] != 0).sum()),
        "inactive_nonzero_time_slots": int((packed["event_relative_days"][inactive] != 0).sum()),
        "inactive_nonzero_type_slots": int((packed["event_type_id"][inactive] != 0).sum()),
        "storage_nonnegative_active_time_slots": int(
            (storage_packed["event_relative_days"][storage_active] >= 0).sum()
        ),
        "storage_inactive_nonzero_slots": int(
            (storage_packed["event_code_id"][storage_inactive] != 0).sum()
            + (storage_packed["event_text_id"][storage_inactive] != 0).sum()
            + (storage_packed["event_relative_days"][storage_inactive] != 0).sum()
            + (storage_packed["event_type_id"][storage_inactive] != 0).sum()
        ),
        "feature_count": int(len(feature_columns)),
        "phenotype_missing_cells": int(missing_mask.sum()),
        "phenotype_mask_mismatches": int(np.not_equal(np.isnan(raw_values), missing_mask).sum()),
        "batch_count": int(len(batch_manifest)),
        "cache_keys": sorted(observed_keys),
        "forbidden_cache_keys_found": forbidden_found,
        "outcomes_in_cache": False,
        "age_target_in_input_shards": False,
        "reported_age_in_cache": False,
        "modality_starvation": starvation,
    }
    if any(
        cache_gate[key] != 0
        for key in [
            "nonnegative_active_time_slots",
            "inactive_nonzero_code_slots",
            "inactive_nonzero_text_slots",
            "inactive_nonzero_time_slots",
            "inactive_nonzero_type_slots",
            "storage_nonnegative_active_time_slots",
            "storage_inactive_nonzero_slots",
            "phenotype_mask_mismatches",
        ]
    ):
        cache_gate["status"] = "FAIL"
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "fold_transform_manifest.json").write_text(
        json.dumps(transform_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "cache_input_gate.json").write_text(
        json.dumps(cache_gate, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "shard_format": "numpy_npz",
        "source_config_sha256": sha256_file(DEFAULT_SOURCE_CONFIG),
        "cache_config_sha256": sha256_file(DEFAULT_CACHE_CONFIG),
        "feature_hash": feature_hash,
        "cache_gate": cache_gate,
        "batch_manifest_sha256": sha256_file(output_dir / "batch_manifest.json"),
        "fold_transform_manifest_sha256": sha256_file(output_dir / "fold_transform_manifest.json"),
    }
    (output_dir / "cache_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    if cache_gate["status"] != "PASS":
        raise RuntimeError("Compact cache input gate failed")
    return cache_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact, masked Revision-0710 event cache.")
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--cache-config", type=Path, default=DEFAULT_CACHE_CONFIG)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-participants", type=int)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--only-embeddings", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=2048)
    args = parser.parse_args()

    source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
    cache_config = json.loads(args.cache_config.read_text(encoding="utf-8"))
    if args.max_participants is not None and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("--max-participants requires an explicit non-production --output-dir")
    if not args.only_embeddings:
        gate = build_compact_cache(
            source_config=source_config,
            cache_config=cache_config,
            index_dir=args.index_dir,
            fold_dir=args.fold_dir,
            output_dir=args.output_dir,
            max_participants=args.max_participants,
        )
        print(json.dumps(gate, indent=2), flush=True)
    if not args.skip_embeddings:
        embedding = build_embedding_tables(
            output_dir=args.output_dir,
            cache_config=cache_config,
            encode_batch_size=args.encode_batch_size,
        )
        print(json.dumps(embedding, indent=2), flush=True)


if __name__ == "__main__":
    main()
