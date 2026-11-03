from __future__ import annotations

import argparse
from pathlib import Path

from dataclasses import dataclass

import numpy as np

import pandas as pd

from numba import njit

@njit(cache=True)
def prefix(tree, pos):
    result = 0.0
    while pos > 0:
        result += tree[pos]
        pos -= pos & -pos
    return result

@njit(cache=True)
def harrell(counts, order, duration, event, rank, unique):
    """Exact historical ties: exclude all equal-duration pairs."""
    tree = np.zeros(unique + 1)
    later = 0.0
    numerator = 0.0
    denominator = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        t = duration[order[start]]
        while stop < len(order) and duration[order[stop]] == t:
            stop += 1
        for j in range(start, stop):
            i = order[j]
            if event[i] == 1 and counts[i] > 0:
                lo = prefix(tree, rank[i]-1)
                hi = prefix(tree, rank[i])
                numerator += counts[i] * (lo + 0.5*(hi-lo))
                denominator += counts[i] * later
        for j in range(start, stop):
            i = order[j]
            pos = rank[i]
            while pos < len(tree):
                tree[pos] += counts[i]
                pos += pos & -pos
            later += counts[i]
        start = stop
    return numerator / denominator

@njit(cache=True)
def auc(counts, order, probability, outcome, weights):
    numerator = 0.0
    total_case = 0.0
    control_before = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        p = probability[order[start]]
        while stop < len(order) and probability[order[stop]] == p:
            stop += 1
        cases = 0.0
        controls = 0.0
        for j in range(start, stop):
            i = order[j]
            if weights[i] > 0:
                w = weights[i] * counts[i]
                if outcome[i] == 1:
                    cases += w
                elif outcome[i] == 0:
                    controls += w
        numerator += cases * (control_before + 0.5*controls)
        control_before += controls
        total_case += cases
        start = stop
    return numerator / (total_case * control_before)

@dataclass
class Prepared:
    stage: str
    comparison: str
    organ: str
    frame: pd.DataFrame
    positions: np.ndarray
    duration_order: np.ndarray
    probability_orders: list
    ranks: list
    uniques: list
    loss: list
    duration: np.ndarray
    event: np.ndarray
    outcome: np.ndarray
    weight: np.ndarray
    scores: list
    probabilities: list

    def evaluate(self, counts):
        n = float(counts.sum())
        result = []
        for k in range(2):
            result.extend([
                harrell(counts, self.duration_order, self.duration, self.event,
                        self.ranks[k], self.uniques[k]),
                auc(counts, self.probability_orders[k], self.probabilities[k],
                    self.outcome, self.weight),
                float(np.dot(counts, self.loss[k])/n),
            ])
        b = np.asarray(result[:3])
        a = np.asarray(result[3:])
        delta = a-b
        delta[2] *= -1
        return b, a, delta


def prepare_frames(direct, adapted, horizon_months=12):
    """Align the two prediction tables without reading any local study files."""
    frames = {}
    keys = ['subject_id','organ_system','horizon_months','comparison','outer_fold']
    aligned = keys + ['observed_time_days','event_indicator','competing_death_indicator',
                      'ipcw_outcome','ipcw_weight','ipcw_observed']
    prediction_columns = ['base_risk','augmented_risk','base_probability','augmented_probability']
    for stage, raw in [('direct',direct), ('adapted',adapted)]:
        missing = set(aligned+prediction_columns)-set(raw.columns)
        if missing:
            raise ValueError(f'Missing prediction columns: {sorted(missing)}')
        f = raw.loc[raw.horizon_months.eq(horizon_months)].copy()
        f['subject_id'] = f.subject_id.astype(str)
        f = f.sort_values(keys, kind='stable').reset_index(drop=True)
        if f.empty or f.duplicated(keys).any():
            raise ValueError('Empty horizon or duplicated participant/comparison rows')
        numeric = f[['observed_time_days','event_indicator','ipcw_weight']+prediction_columns].to_numpy(float)
        if not np.isfinite(numeric).all() or (f.ipcw_weight < 0).any():
            raise ValueError('Invalid time, weight, or prediction values')
        observed = f.ipcw_weight.gt(0)
        if not f.loc[observed, 'ipcw_outcome'].isin([0,1]).all():
            raise ValueError('IPCW outcome must be binary where weight is positive')
        frames[stage] = f
    pd.testing.assert_frame_equal(frames['direct'][aligned], frames['adapted'][aligned])
    people = frames['direct'][['subject_id','outer_fold']].drop_duplicates().sort_values('subject_id')
    if people.subject_id.duplicated().any() or not people.outer_fold.isin(range(5)).all():
        raise ValueError('Each participant must have one fold in 0..4')
    universe = people.subject_id.to_numpy()
    folds = [np.flatnonzero(people.outer_fold.to_numpy()==k) for k in range(5)]
    cells = []
    for stage, frame in frames.items():
        for (comparison, organ), f in frame.groupby(['comparison','organ_system'], sort=True):
            if f.subject_id.duplicated().any():
                raise ValueError('Repeated participant within a comparison/organ')
            positions = np.searchsorted(universe, f.subject_id.to_numpy())
            duration = f.observed_time_days.to_numpy(float)
            event = f.event_indicator.to_numpy(np.int8)
            outcome = f.ipcw_outcome.to_numpy(float)
            weight = f.ipcw_weight.to_numpy(float)
            scores = [f[k].to_numpy(float) for k in ['base_risk','augmented_risk']]
            probabilities = [f[k].to_numpy(float) for k in ['base_probability','augmented_probability']]
            unique = [np.unique(s) for s in scores]
            cells.append(Prepared(stage, comparison, organ, f, positions,
                np.argsort(-duration, kind='stable'),
                [np.argsort(p, kind='stable') for p in probabilities],
                [np.searchsorted(u,s).astype(np.int64)+1 for u,s in zip(unique,scores)],
                [len(u) for u in unique],
                [np.where(weight>0, weight*(outcome-p)**2, 0.) for p in probabilities],
                duration, event, outcome, weight, scores, probabilities))
    return cells, universe, folds


def bootstrap_tables(direct, adapted, n_resamples=1000, seed=20261031):
    """Joint participant resampling within folds; seven-organ macro contrasts."""
    if n_resamples < 2:
        raise ValueError('At least two replicates are required')
    cells, universe, folds = prepare_frames(direct, adapted)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_resamples, len(cells), 3))
    for replicate in range(n_resamples):
        counts = np.zeros(len(universe), dtype=np.int64)
        for ids in folds:
            if len(ids):
                counts[ids] = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)))
        for index, cell in enumerate(cells):
            if counts[cell.positions].sum() == 0:
                raise ValueError('Empty bootstrap cell; sample is too small')
            samples[replicate,index] = cell.evaluate(counts[cell.positions])[2]
    if not np.isfinite(samples).all():
        raise ValueError('Non-finite bootstrap metrics; inspect event counts and sample size')
    metrics = ['harrell_c','ipcw_auc','ipcw_brier_improvement']
    points = []
    for index, cell in enumerate(cells):
        base, augmented, delta = cell.evaluate(np.ones(len(cell.frame), dtype=np.int64))
        for j, metric in enumerate(metrics):
            low, high = np.quantile(samples[:,index,j],[.025,.975])
            points.append({'stage':cell.stage,'comparison':cell.comparison,'organ_system':cell.organ,
                'metric':metric,'base':base[j],'augmented':augmented[j],'improvement':delta[j],
                'ci_lower':low,'ci_upper':high})
    macro, contrasts, macro_samples = [], [], {}
    organs = {'cardiovascular','hepatic','immune','metabolic','musculoskeletal','pulmonary','renal'}
    points = pd.DataFrame(points)
    for stage in ['direct','adapted']:
        for comparison in sorted(set(c.comparison for c in cells)):
            indices = [i for i,c in enumerate(cells) if c.stage==stage and c.comparison==comparison]
            if {cells[i].organ for i in indices} != organs:
                raise ValueError('A macro contrast requires all seven organs')
            draws = samples[:,indices].mean(axis=1)
            macro_samples[(stage,comparison)] = draws
            for j, metric in enumerate(metrics):
                selected = points.loc[points.stage.eq(stage)&points.comparison.eq(comparison)&points.metric.eq(metric)]
                low, high = np.quantile(draws[:,j],[.025,.975])
                macro.append({'stage':stage,'comparison':comparison,'metric':metric,
                    'improvement':selected.improvement.mean(),'ci_lower':low,'ci_upper':high})
    macro = pd.DataFrame(macro)
    for comparison in sorted(set(c.comparison for c in cells)):
        differences = macro_samples[('adapted',comparison)]-macro_samples[('direct',comparison)]
        for j, metric in enumerate(metrics):
            subset = macro.loc[macro.comparison.eq(comparison)&macro.metric.eq(metric)].set_index('stage')
            low, high = np.quantile(differences[:,j],[.025,.975])
            contrasts.append({'comparison':comparison,'metric':metric,
                'adapted_minus_direct':subset.loc['adapted','improvement']-subset.loc['direct','improvement'],
                'ci_lower':low,'ci_upper':high})
    return points, macro, pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description='Paired external comparison from prepared predictions')
    parser.add_argument('--direct', required=True, type=Path)
    parser.add_argument('--adapted', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    tables = bootstrap_tables(pd.read_parquet(args.direct), pd.read_parquet(args.adapted), args.replicates, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in zip(['organ_comparisons','macro_comparisons','paired_transfer_contrasts'], tables):
        table.to_csv(args.output/f'{name}.csv', index=False)


if __name__ == '__main__':
    main()
