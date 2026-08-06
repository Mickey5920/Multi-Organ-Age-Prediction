from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from Dataloader import (
    MODEL_INPUT_KEYS,
    load_data_bundle,
    load_embeddings,
    make_loaders,
    make_synthetic_bundle,
)
from model import EHRAGEConfig, EHRAGEModel, ORGAN_SYSTEMS
from train import (
    LossConfig,
    apply_osag_correction,
    configure_reproducibility,
    ehr_age_loss,
    fit_osag_correction,
    predict,
    save_checkpoint,
    split_batch,
    strict_age_metrics,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the current EHR-AGE architecture")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--outer-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--scheduler-horizon", type=int, default=60)
    parser.add_argument("--event-dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7210)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    horizon_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_epochs < 0 or horizon_epochs <= warmup_epochs:
        raise ValueError("invalid scheduler settings")

    def factor(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(horizon_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def save_predictions(
    output_dir: Path,
    prediction: dict[str, torch.Tensor],
    osag: torch.Tensor,
) -> None:
    np.savez_compressed(
        output_dir / "outer_predictions.npz",
        chronological_age=prediction["chronological_age"].numpy(),
        predicted_age=prediction["predicted_age"].numpy(),
        base_age=prediction["base_age"].numpy(),
        contextual_age=prediction["base_view_ages"][:, 0].numpy(),
        reverse_age=prediction["base_view_ages"][:, 1].numpy(),
        risk_score=prediction["risk_score"].numpy(),
        risk_age_offset=prediction["risk_age_offset"].numpy(),
        osag=osag.numpy(),
        organ_systems=np.asarray(ORGAN_SYSTEMS, dtype="U32"),
    )


def validate_forward(
    model: EHRAGEModel,
    loader,
    device: torch.device,
    interval_edges: torch.Tensor,
    loss_config: LossConfig,
    amp: bool,
) -> dict[str, object]:
    batch = next(iter(loader))
    inputs, targets = split_batch(batch, device)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        output = model(**inputs)
        losses = ehr_age_loss(
            output,
            **targets,
            interval_edges_months=interval_edges,
            config=loss_config,
        )
    return {
        "status": "PASS",
        "batch_size": len(targets["chronological_age"]),
        "predicted_age_shape": list(output.predicted_age.shape),
        "base_view_ages_shape": list(output.base_view_ages.shape),
        "risk_score_shape": list(output.risk_score.shape),
        "interval_log_rate_shape": list(output.interval_log_rate.shape),
        "loss": float(losses["total"]),
    }


def main() -> None:
    args = parse_args()
    configure_reproducibility(args.seed)
    if args.smoke_test:
        bundle, code_embedding, text_embedding = make_synthetic_bundle(seed=args.seed)
    else:
        if args.data is None or args.embeddings is None:
            raise SystemExit("--data and --embeddings are required unless --smoke-test is used")
        bundle = load_data_bundle(args.data, args.outer_fold)
        code_embedding, text_embedding = load_embeddings(args.embeddings)
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke_test else 47)
    batch_size = args.batch_size if args.batch_size is not None else (8 if args.smoke_test else 2048)
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    loaders = make_loaders(
        bundle,
        batch_size=batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    device = resolve_device(args.device)
    train_age = bundle.dataset.tensors["chronological_age"][bundle.train_indices]
    age_anchor = float(train_age.mean())
    model_config = EHRAGEConfig()
    loss_config = LossConfig()
    model = EHRAGEModel(
        code_embedding,
        text_embedding,
        age_anchor=age_anchor,
        config=model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        horizon_epochs=args.scheduler_horizon,
    )
    amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    interval_edges = torch.tensor(
        [0.0, 12.0, 24.0, 36.0, 48.0, 60.0],
        dtype=torch.float32,
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if args.validate_only:
        result = validate_forward(
            model,
            loaders["train"],
            device,
            interval_edges,
            loss_config,
            amp,
        )
        result["parameter_count"] = parameter_count
        result["trainable_parameter_count"] = trainable_count
        print(json.dumps(result, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        losses = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            epoch,
            loss_config,
            interval_edges,
            event_dropout_probability=args.event_dropout,
            gradient_clip_norm=args.gradient_clip,
            amp=amp,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **losses,
        }
        history.append(row)
        print(json.dumps(row))
    calibration_prediction = predict(model, loaders["calibration"], device, amp=amp)
    correction = fit_osag_correction(calibration_prediction)
    outer_prediction = predict(model, loaders["outer"], device, amp=amp)
    osag = apply_osag_correction(
        outer_prediction["predicted_age"],
        outer_prediction["chronological_age"],
        correction,
    )
    save_checkpoint(
        args.output_dir / "ehr_age_checkpoint.pt",
        model,
        optimizer,
        epochs,
        model_config,
        loss_config,
        bundle.standardization.as_dict(),
    )
    torch.save(correction, args.output_dir / "osag_age_correction.pt")
    save_predictions(args.output_dir, outer_prediction, osag)
    summary = {
        "status": "COMPLETE",
        "outer_fold": args.outer_fold,
        "calibration_fold": (args.outer_fold + 1) % 5,
        "train_folds": [
            fold
            for fold in range(5)
            if fold not in {args.outer_fold, (args.outer_fold + 1) % 5}
        ],
        "epochs": epochs,
        "scheduler_horizon": args.scheduler_horizon,
        "seed": args.seed,
        "age_anchor": age_anchor,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "strict_outer_age_metrics": strict_age_metrics(outer_prediction),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
