from __future__ import annotations

import numpy as np

import pandas as pd

import warnings
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from .statistics.cox_scoring import fit_lifelines_cox_scores
from .statistics.survival_metrics import build_metric_time_grid, evaluate_censoring_aware_metrics
from .statistics.clinical_utility_metrics import harrell_c_index

ORGANS = ('cardiovascular','hepatic','immune','metabolic','musculoskeletal','pulmonary','renal')
CONTEXT = ('chronological_age','sex_male','smoking_current','smoking_missing',
    'education_higher','education_missing','townsend','townsend_missing',
    'employment_employed','employment_retired','employment_unemployed','employment_other',
    'employment_missing','sleep_duration','sleep_missing','phenotype_missing_fraction',
    'log1p_preindex_icd_count','log1p_preindex_gp_count','has_preindex_icd','has_preindex_gp',
    'prevalent_organ_count')
PHENOTYPES = tuple(f'phenotype_{i:02d}' for i in range(78))
MISSING = tuple(f'phenotype_missing_{i:02d}' for i in range(78))
RISK = tuple(f'medbert_risk__{organ}' for organ in ORGANS)
GAPS = tuple(f'gap__{organ}' for organ in ORGANS)
REFERENCE_COLUMNS = CONTEXT + PHENOTYPES + MISSING + RISK


class ConvergentCoxPHFitter(CoxPHFitter):
    """The deterministic step-size retry used for the wider clinical design."""
    def fit(self, *args, **kwargs):
        for fit_options in (None, {'step_size':0.25, 'max_steps':1000},
                            {'step_size':0.05, 'max_steps':2000}):
            call_kwargs = dict(kwargs)
            if fit_options is not None:
                call_kwargs['fit_options'] = fit_options
            error = None
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
                try:
                    result = super().fit(*args, **call_kwargs)
                except (ConvergenceError, np.linalg.LinAlgError) as exc:
                    error = str(exc)
            numerical = [str(w.message) for w in captured
                         if issubclass(w.category, (ConvergenceWarning, RuntimeWarning))]
            if error is not None:
                numerical.append(error)
            if not numerical and np.isfinite(self.log_likelihood_):
                return result
        raise RuntimeError('Cox did not converge after step-size retries: ' + '; '.join(numerical))


def select_penalty(fit_raw, covariates, imputation_columns, seed):
    """Three-fold tuning within the fitting participants only."""
    splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    for inner_fold, (train_index, valid_index) in enumerate(splits.split(fit_raw, fit_raw['event'])):
        train, valid = fit_raw.iloc[train_index].copy(), fit_raw.iloc[valid_index].copy()
        train['split_role'], valid['split_role'] = 'validation', 'test'
        train, valid, _ = impute_clinical(train, valid, imputation_columns)
        grid = build_metric_time_grid(train, valid, report_times=(60.,), ibs_times=(60.,))
        for penalty in (0.0001, 0.001, 0.01, 0.1):
            bundle = fit_lifelines_cox_scores(train, valid, covariates=covariates,
                time_grid=grid, phase='formal', penalizer=penalty, fitter_class=ConvergentCoxPHFitter)
            value = float(harrell_c_index(valid, bundle.risk_score))
            if not np.isfinite(value):
                raise ValueError('Non-finite inner-validation C-index')
            rows.append({'inner_fold':inner_fold, 'penalizer':penalty, 'harrell_c':value})
    table = pd.DataFrame(rows)
    means = table.groupby('penalizer')['harrell_c'].mean()
    chosen = float(max(p for p,v in means.items() if means.max()-v <= 1e-12))
    return chosen, table


def fit_reference(fit, score, covariates, imputation_columns, seed=710):
    """Fit one organ/outer-fold reference or reference-plus-OSAG model."""
    chosen, cv = select_penalty(fit, covariates, imputation_columns, seed)
    fit, score, medians = impute_clinical(fit, score, imputation_columns)
    grid = build_metric_time_grid(fit, score)
    bundle = fit_lifelines_cox_scores(fit, score, covariates=covariates,
        time_grid=grid, phase='formal', penalizer=chosen, fitter_class=ConvergentCoxPHFitter)
    metrics, calibration = evaluate_censoring_aware_metrics(fit, score,
        risk_score=bundle.risk_score, survival_probability=bundle.survival_probability,
        time_grid=grid, calibration_bins=10)
    metrics['harrell_c'] = float(harrell_c_index(score, bundle.risk_score))
    return {'scores':bundle, 'time_grid':grid, 'cv':cv, 'medians':medians,
            'metrics':metrics, 'calibration':calibration}


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Clinical/EHR Cox reference with and without seven OSAGs')
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    fit, score = pd.read_parquet(args.fit), pd.read_parquet(args.test)
    for frame in (fit,score):
        if frame['organ_system'].nunique() != 1 or frame['outer_fold'].nunique() != 1:
            raise ValueError('Supply exactly one organ and one outer-fold evaluation per call')
    if set(fit.organ_system) != set(score.organ_system):
        raise ValueError('Fitting and test organs differ')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, columns in [('reference',REFERENCE_COLUMNS), ('reference_plus_osags',REFERENCE_COLUMNS+GAPS)]:
        result = fit_reference(fit,score,columns,CONTEXT,seed=args.seed)
        bundle = result['scores']
        prediction = score[['participant_id','organ_system','outer_fold','duration','event']].copy()
        prediction['risk_score'] = bundle.risk_score
        prediction.to_parquet(args.output/f'{name}_predictions.parquet',index=False)
        np.savez_compressed(args.output/f'{name}_survival.npz',times=bundle.times,
            survival_probability=bundle.survival_probability)
        result['cv'].to_csv(args.output/f'{name}_cv.csv',index=False)
        (args.output/f'{name}_metrics.json').write_text(json.dumps(result['metrics'],indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()

def impute_clinical(fit, score, columns):
    fit, score = fit.copy(), score.copy()
    medians = {}
    for column in columns:
        values = pd.to_numeric(fit[column], errors="coerce")
        median = float(values.median())
        if not np.isfinite(median):
            raise ValueError(f"No finite fit median for {column}")
        medians[column] = median
        fit[column] = values.fillna(median)
        score[column] = pd.to_numeric(score[column], errors="coerce").fillna(median)
    return fit, score, medians
