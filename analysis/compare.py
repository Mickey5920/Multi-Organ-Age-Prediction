"""Train an objective-matched comparator from the public EHR-AGE array format."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from Dataloader import MODEL_INPUT_KEYS, load_arrays, load_embeddings, prepare_arrays
from .cache import CompactAgeDataset, CompactCacheBundle, PhenotypeTransform
from .comparison_models import CONTEMPORARY_MODELS, ProspectiveBaselineAdapter, build_contemporary_backbone
from .data import Fold0DevelopmentData, JointLabels, JointVectorizedBatcher
from .schedule import _scheduler
from .training import _loss_config, _predict, _survival_multiplier, _train_epoch, configure_reproducibility


def prepare_data(arrays, embeddings, outer_fold, transform_path=None):
    bundle = prepare_arrays(arrays, outer_fold)
    tensors = bundle.dataset.tensors
    transform = PhenotypeTransform(
        bundle.standardization.median, bundle.standardization.mean,
        bundle.standardization.std, outer_fold, (outer_fold + 1) % 5,
    )
    if transform_path is not None:
        transform = PhenotypeTransform.load(Path(transform_path))
        if (transform.outer_fold, transform.validation_fold) != (outer_fold, (outer_fold + 1) % 5):
            raise ValueError("transform fold does not match the requested split")
        missing = ~np.isfinite(arrays['phenotype'])
        tensors['phenotype'] = torch.from_numpy(transform.apply(np.nan_to_num(arrays['phenotype']), missing))
    dataset = CompactAgeDataset(
        participant_ids=range(len(tensors['chronological_age'])),
        inputs={key:tensors[key] for key in MODEL_INPUT_KEYS},
        chronological_age=tensors['chronological_age'],
        fold_id=torch.from_numpy(arrays['fold'].astype(np.int64)),
    )
    roles = np.full(len(dataset), 'train', dtype='<U11')
    roles[bundle.calibration_indices] = 'calibration'
    roles[bundle.outer_indices] = 'test'
    labels = JointLabels(tensors['strict_healthy_reference'], tensors['duration_months'],
                         tensors['event'], tensors['at_risk'], tuple(roles))
    labels.validate(len(dataset))
    return Fold0DevelopmentData(
        CompactCacheBundle(dataset, *embeddings, transform), labels,
        torch.from_numpy(bundle.train_indices), torch.from_numpy(bundle.calibration_indices),
        torch.from_numpy(bundle.outer_indices), {},
    )


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    spec, shared = config['objective_matched'], config['shared']
    seed = int(shared['seed_base']) + args.outer_fold
    configure_reproducibility(seed, deterministic=shared['deterministic'], tf32=shared['tf32'])
    arrays = load_arrays(args.data)
    data = prepare_data(arrays, load_embeddings(args.embeddings), args.outer_fold, args.transform)
    device = torch.device(args.device)
    backbone = build_contemporary_backbone(
        args.model, data.cache.code_embedding, data.cache.text_embedding,
        hidden_dim=spec['hidden_dim'], num_heads=spec['num_heads'],
        num_layers=spec['num_layers'], dropout=spec['dropout'], freeze_embeddings=True,
    )
    edges = spec['survival_interval_edges_months']
    model = ProspectiveBaselineAdapter(
        backbone, hidden_dim=spec['hidden_dim'],
        age_anchor=float(data.cache.dataset.chronological_age[data.train_indices].mean()),
        dropout=spec['dropout'], risk_offset_cap_years=spec['risk_offset_cap_years'],
        risk_temperature=spec['risk_temperature'], survival_intervals=len(edges)-1,
        baseline_log_rate_init=spec['baseline_log_rate_init'],
    ).to(device)
    if args.validate_only:
        model.eval()
        with torch.inference_mode():
            inputs = {k:v[data.train_indices[:2]].to(device) for k,v in data.cache.dataset.inputs.items()}
            output = model(**inputs)
            assert output.predicted_age.shape == (len(inputs['sex']), 7)
            assert torch.isfinite(output.predicted_age).all()
        print('Input and forward checks passed.')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = _scheduler(optimizer, warmup_epochs=spec['warmup_epochs'], horizon_epochs=spec['scheduler_horizon_epochs'])
    amp = bool(shared['amp']) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    batcher = JointVectorizedBatcher(data, data.train_indices, batch_size=spec['batch_size'], shuffle=True, seed=seed)
    interval_edges = torch.tensor(edges, dtype=torch.float32, device=device)
    loss_config = _loss_config(spec['loss'])
    for epoch in range(1, spec['fixed_epochs'] + 1):
        losses = _train_epoch(model=model, batcher=batcher, optimizer=optimizer, scaler=scaler,
            device=device, interval_edges=interval_edges, loss_config=loss_config,
            survival_multiplier=_survival_multiplier(epoch, spec['loss']),
            event_dropout_probability=spec['event_dropout_probability'],
            gradient_clip_norm=spec['gradient_clip_norm'], amp=amp, max_batches=None)
        scheduler.step()
        print(json.dumps({'epoch':epoch, **losses}))
    torch.save({'model_state':model.state_dict(), 'model':args.model, 'config':config,
                'outer_fold':args.outer_fold}, args.output/'model.pt')
    np.savez_compressed(args.output/'phenotype_transform.npz', **asdict(data.cache.phenotype_transform))
    for role, indices in [('calibration', data.calibration_indices), ('test', data.screen_indices)]:
        predicted = _predict(model=model,
            batcher=JointVectorizedBatcher(data, indices, batch_size=spec['batch_size'], shuffle=False, seed=seed),
            device=device, amp=amp, max_batches=None)
        np.savez_compressed(args.output/f'{role}_predictions.npz', **asdict(predicted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--model', choices=CONTEMPORARY_MODELS, required=True)
    parser.add_argument('--outer-fold', type=int, choices=range(5), required=True)
    parser.add_argument('--transform', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('comparison_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--validate-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
