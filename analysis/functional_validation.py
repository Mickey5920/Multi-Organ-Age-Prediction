"""Follow-up functional and biochemical associations for the seven EHR-AGE systems.

Reproduce the 21-measure, 42-association analysis from participant-level
five-fold evaluation predictions and authorized local UK Biobank measurements.
All pooled associations enter BH and Holm correction. Continuous outcomes use
HC3 linear regression; slow walking uses HC0 logistic regression. This program
does not train neural networks. It exports aggregate statistics only.

See analysis/README.md for input schemas, measurement definitions and commands.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key]="4"
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits


ORGANS = ("cardiovascular", "hepatic", "immune", "metabolic", "musculoskeletal", "pulmonary", "renal")

DENSITY = ["phenotype_missing_fraction", "log1p_preindex_icd_count", "log1p_preindex_gp_count", "has_preindex_icd", "has_preindex_gp"]

BURDEN = ["prevalent_organ_count"] + [f"log1p_prevalent_{o}_count" for o in ORGANS]

def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def unique_id(frame, label):
    assert frame["participant_id"].notna().all(), label + ": missing ID"
    frame["participant_id"] = frame["participant_id"].astype(str)
    assert not frame["participant_id"].duplicated().any(), label + ": duplicate ID"
    return frame.set_index("participant_id")

CORE = ["chronological_age", "sex_male", "elapsed_years", *DENSITY, *BURDEN]

SPECS = [
    ("cardiovascular", "pulse_pressure", "Pulse pressure", "mmHg", 2, "identity", 1),
    ("cardiovascular", "systolic_bp", "Systolic blood pressure", "mmHg", 2, "identity", 1),
    ("cardiovascular", "pulse_rate", "Resting pulse rate", "beats/min", 2, "identity", 1),
    ("hepatic", "albumin", "Serum albumin", "g/L", 1, "identity", -1),
    ("hepatic", "alt", "Alanine aminotransferase", "U/L", 1, "log", 1),
    ("hepatic", "ggt", "Gamma glutamyltransferase", "U/L", 1, "log", 1),
    ("immune", "crp", "C-reactive protein", "mg/L", 1, "log", 1),
    ("immune", "wbc", "White blood cell count", "10^9 cells/L", 1, "log", 1),
    ("immune", "nlr", "Neutrophil-to-lymphocyte ratio", "ratio", 1, "log", 1),
    ("metabolic", "hba1c", "Glycated haemoglobin", "mmol/mol", 1, "identity", 1),
    ("metabolic", "triglycerides", "Triglycerides", "mmol/L", 1, "log", 1),
    ("metabolic", "hdl", "HDL cholesterol", "mmol/L", 1, "identity", -1),
    ("musculoskeletal", "grip", "Bilateral mean grip strength", "kg", 2, "identity", -1),
    ("musculoskeletal", "heel_bmd", "Bilateral ultrasound heel BMD", "g/cm^2", 2, "identity", -1),
    ("musculoskeletal", "slow_walking", "Self-reported slow walking pace", "odds ratio", 2, "logit", 1),
    ("pulmonary", "fev1", "Forced expiratory volume in 1 second", "mL", 2, "identity", -1),
    ("pulmonary", "fvc", "Forced vital capacity", "mL", 2, "identity", -1),
    ("pulmonary", "fev1_fvc", "FEV1/FVC", "percentage points", 2, "identity", -1),
    ("renal", "cystatin_c", "Cystatin C", "mg/L", 1, "log", 1),
    ("renal", "creatinine", "Serum creatinine", "umol/L", 1, "log", 1),
    ("renal", "urea", "Urea", "mmol/L", 1, "log", 1),
]

SPECS = [dict(zip(("organ", "endpoint", "label", "unit", "instance", "transform", "expected_sign"), x)) for x in SPECS]

LAB_NAMES = {
    "albumin": "Albumin", "alt": "Alanine aminotransferase", "ggt": "Gamma glutamyltransferase",
    "crp": "C-reactive protein", "wbc": "White blood cell (leukocyte) count",
    "neutrophils": "Neutrophill count", "lymphocytes": "Lymphocyte count",
    "hba1c": "Glycated haemoglobin (HbA1c)", "triglycerides": "Triglycerides",
    "hdl": "HDL cholesterol", "cystatin_c": "Cystatin C", "creatinine": "Creatinine", "urea": "Urea",
}

FIELD_IDS = {
    "albumin": [30600], "alt": [30620], "ggt": [30730], "crp": [30710], "wbc": [30000],
    "nlr": [30140, 30120], "hba1c": [30750], "triglycerides": [30870], "hdl": [30760],
    "cystatin_c": [30720], "creatinine": [30700], "urea": [30670], "heel_bmd": [4105, 4124],
    "slow_walking": [924], "pulse_pressure": [4080, 4079], "systolic_bp": [4080], "pulse_rate": [102],
    "grip": [46, 47], "fev1": [3063, 3061, 20152], "fvc": [3062, 3061, 20152], "fev1_fvc": [3063, 3062, 3061, 20152],
}

RULES = {
    "blood_pressure": "Both same-array positive finite SBP/DBP pairs (arrays 0 and 1), SBP>DBP; mean SBP and mean paired SBP-DBP at each visit. No disease-range trimming.",
    "pulse_rate": "Both positive finite automated readings, arrays 0 and 1; arithmetic mean at each visit.",
    "albumin": "Baseline field 30600 serum albumin only, not p23479; same assay at Instance 1.",
    "laboratory": "Finite positive observed values at both visits; no winsorization or disease-based exclusions. Natural logs as fixed in SPECS at baseline and follow-up.",
    "heel_bmd": "Both positive finite left and right ultrasound-estimated BMD at both visits; bilateral mean. No substitution of baseline-only field 3148; not DXA BMD.",
    "walking": "Slow, steady average, or brisk only; other/declined/missing excluded. Follow-up slow vs average/brisk; baseline slow and brisk indicators (average reference). Self-report, not measured gait speed.",
    "pulmonary": "Numerical-repeatability rule at Instances 0 and 2; max selected FEV1 and FVC, ratio of those maxima. Missing field 3061 means acceptability cannot be verified.",
    "nlr": "Positive finite neutrophil and lymphocyte counts at the same visit; ratio, then natural logarithm.",
    "dates": "Elapsed years from raw assessment-centre dates, not assay-processing dates; follow-up strictly after original index date.",
}

ANALYTIC_QC = {"heel_bmd": "Bilateral means at both visits strictly within (0.18, 1.06) g/cm2 for men and (0.12, 1.025) for women", "source": "https://doi.org/10.1371/journal.pone.0200785"}

def qualify_blow_values(fev1, fvc, flags=None, numeric_only=False):
    fev1, fvc = (np.asarray(x, dtype=float) for x in (fev1, fvc))
    assert fev1.shape == fvc.shape and fev1.ndim == 2
    if numeric_only:
        assert flags is None, "Do not silently ignore available quality flags"
        selected = np.isfinite(fev1) & np.isfinite(fvc) & (fev1 > 0) & (fvc > 0) & (fev1 <= fvc)
    else:
        assert flags is not None, "Quality flags required unless numeric-only mode is explicit"
        flags = np.asarray(flags, float)
        assert flags.shape == fev1.shape
        selected = np.isin(flags, [0., 32.])
    good_fev1 = selected & np.isfinite(fev1) & (fev1 > 0)
    good_fvc = selected & np.isfinite(fvc) & (fvc > 0)
    a = np.sort(np.where(good_fev1, fev1, -np.inf), axis=1)
    b = np.sort(np.where(good_fvc, fvc, -np.inf), axis=1)
    best_fev1, best_fvc = a[:, -1].copy(), b[:, -1].copy()
    have_two = (good_fev1.sum(axis=1) >= 2) & (good_fvc.sum(axis=1) >= 2)
    threshold = np.where(best_fvc <= 1., .10, .15)
    difference_fev1, difference_fvc = np.full(len(a), np.nan), np.full(len(a), np.nan)
    difference_fev1[have_two] = a[have_two, -1] - a[have_two, -2]
    difference_fvc[have_two] = b[have_two, -1] - b[have_two, -2]
    repeatable = have_two & (difference_fev1 <= threshold + 1e-10) & (difference_fvc <= threshold + 1e-10)
    best_fev1[~np.isfinite(best_fev1)] = np.nan
    best_fvc[~np.isfinite(best_fvc)] = np.nan
    return pd.DataFrame({"selected_fev1_l": best_fev1, "selected_fvc_l": best_fvc,
                         "n_selected_fev1": good_fev1.sum(axis=1), "n_selected_fvc": good_fvc.sum(axis=1),
                         "repeatable": repeatable, "fev1_top_difference_l": difference_fev1,
                         "fvc_top_difference_l": difference_fvc, "threshold_l": threshold})

def positive(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)

def mean_two(values):
    values = positive(values)
    assert values.ndim == 2 and values.shape[1] == 2
    return values.mean(axis=1)

def paired_bp(sbp, dbp):
    sbp, dbp = positive(sbp), positive(dbp)
    assert sbp.shape == dbp.shape and sbp.ndim == 2 and sbp.shape[1] == 2
    good = np.isfinite(sbp) & np.isfinite(dbp) & (sbp > dbp)
    return np.where(good, sbp, np.nan).mean(axis=1), np.where(good, sbp-dbp, np.nan).mean(axis=1)

def pace_code(series):
    allowed = {"Slow pace": 1., "Steady average pace": 2., "Brisk pace": 3.,
               "None of the above": np.nan, "Prefer not to answer": np.nan}
    assert set(series.dropna().unique()) <= set(allowed), "Unknown walking-pace category"
    return series.map(allowed)

def bmd_in_range(values, sex_male):
    values, sex_male = np.asarray(values, float), np.asarray(sex_male, float)
    assert np.isin(sex_male, [0, 1]).all()
    low, high = np.where(sex_male == 1, .18, .12), np.where(sex_male == 1, 1.06, 1.025)
    return np.isfinite(values) & (values > low) & (values < high)

def metadata(spec):
    return {**spec, "field_urls": [f"https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id={x}" for x in FIELD_IDS[spec["endpoint"]]]}

def load_measurements(paths):
    names = {"Standing height | Instance 0": "height_0"}
    for key, stem in LAB_NAMES.items():
        for visit in (0, 1):
            col = f"{stem} | Instance {visit}"
            if key == "albumin" and visit == 0:
                col += "(participant - p30600_i0)"
            names[col] = f"{key}_{visit}"
    for visit in (0, 2):
        for side in ("left", "right"):
            names[f"Heel bone mineral density (BMD) ({side}) | Instance {visit}"] = f"bmd_{side}_{visit}"
            names[f"Hand grip strength ({side}) | Instance {visit}"] = f"grip_{side}_{visit}"
        for array in (0, 1):
            for key, stem in [("sbp", "Systolic blood pressure"), ("dbp", "Diastolic blood pressure"), ("pulse", "Pulse rate")]:
                names[f"{stem}, automated reading | Instance {visit} | Array {array}"] = f"{key}_{visit}_{array}"
        for array in (0, 1, 2):
            names[f"Forced expiratory volume in 1-second (FEV1) | Instance {visit} | Array {array}"] = f"fev_{visit}_{array}"
            names[f"Forced vital capacity (FVC) | Instance {visit} | Array {array}"] = f"fvc_{visit}_{array}"
    raw = pd.read_csv(paths["grip"], usecols=["Participant ID", *names], dtype={"Participant ID": str})
    raw = unique_id(raw.rename(columns={"Participant ID": "participant_id", **names}), "raw measurements")
    for c in raw:
        parsed = pd.to_numeric(raw[c], errors="coerce")
        assert not (raw[c].notna() & parsed.isna()).any(), f"Unparsed value: {c}"
        raw[c] = parsed
    frame = pd.DataFrame(index=raw.index)
    frame["baseline_height_cm"] = positive(raw.height_0)
    for key in LAB_NAMES:
        for v in (0, 1): frame[f"{key}_{v}"] = positive(raw[f"{key}_{v}"])
    for v in (0, 1): frame[f"nlr_{v}"] = frame[f"neutrophils_{v}"] / frame[f"lymphocytes_{v}"]
    for v in (0, 2):
        sbp, pp = paired_bp(raw[[f"sbp_{v}_{a}" for a in (0, 1)]], raw[[f"dbp_{v}_{a}" for a in (0, 1)]])
        frame[f"systolic_bp_{v}"], frame[f"pulse_pressure_{v}"] = sbp, pp
        frame[f"pulse_rate_{v}"] = mean_two(raw[[f"pulse_{v}_{a}" for a in (0, 1)]])
        frame[f"heel_bmd_{v}"] = mean_two(raw[[f"bmd_left_{v}", f"bmd_right_{v}"]])
        g = raw[[f"grip_left_{v}", f"grip_right_{v}"]].to_numpy(float)
        frame[f"grip_{v}"] = np.where(np.isfinite(g) & (g >= 0), g, np.nan).mean(axis=1)
        q = qualify_blow_values(raw[[f"fev_{v}_{a}" for a in range(3)]], raw[[f"fvc_{v}_{a}" for a in range(3)]], numeric_only=True)
        q.index = raw.index
        frame[f"fev1_{v}"] = 1000 * q.selected_fev1_l.where(q.repeatable)
        frame[f"fvc_{v}"] = 1000 * q.selected_fvc_l.where(q.repeatable)
        frame[f"fev1_fvc_{v}"] = 100 * frame[f"fev1_{v}"] / frame[f"fvc_{v}"]
    walk_names = {f"Usual walking pace | Instance {v}": f"pace_{v}" for v in (0, 2)}
    walk = pd.read_csv(paths["walking"], usecols=["Participant ID", *walk_names], dtype={"Participant ID": str})
    walk = unique_id(walk.rename(columns={"Participant ID": "participant_id", **walk_names}), "walking pace")
    for v in (0, 2):
        frame[f"pace_{v}"] = pace_code(walk[f"pace_{v}"]).reindex(frame.index)
        frame[f"slow_walking_{v}"] = frame[f"pace_{v}"].eq(1).astype(float).where(frame[f"pace_{v}"].notna())
    frame["baseline_pace_slow"] = frame.pace_0.eq(1).astype(float).where(frame.pace_0.notna())
    frame["baseline_pace_brisk"] = frame.pace_0.eq(3).astype(float).where(frame.pace_0.notna())
    date_names = {f"Date of attending assessment centre | Instance {v}": f"visit_date_{v}" for v in (0, 1, 2)}
    dates = pd.read_csv(paths["dates"], usecols=["Participant ID", *date_names], dtype={"Participant ID": str})
    dates = unique_id(dates.rename(columns={"Participant ID": "participant_id", **date_names}), "assessment dates")
    for col in dates:
        parsed = pd.to_datetime(dates[col], errors="coerce")
        assert not (dates[col].notna() & parsed.isna()).any()
        dates[col] = parsed
    frame = frame.join(dates, validate="one_to_one")
    stats = []
    for spec in SPECS:
        key, visit = spec["endpoint"], spec["instance"]
        b, f = frame[f"{key}_0"], frame[f"{key}_{visit}"]
        row = {**spec, "raw_rows": len(frame), "baseline_nonmissing": int(b.notna().sum()),
               "followup_nonmissing": int(f.notna().sum()), "paired_measurements": int((b.notna() & f.notna()).sum())}
        for label, values in [("baseline", b), ("followup", f)]:
            for q, suffix in [(0, "min"), (.01, "p01"), (.5, "median"), (.99, "p99"), (1, "max")]:
                row[f"{label}_{suffix}"] = float(values.quantile(q))
        stats.append(row)
    return frame, pd.DataFrame(stats)

def assemble_context(paths, measurements):
    context = unique_id(pd.read_parquet(paths["context"], columns=["participant_id", "fold_id", "chronological_age", "sex_male", "smoking_current", "smoking_missing", *DENSITY]), "context")
    cohort = unique_id(pd.read_parquet(paths["cohort"], columns=["participant_id", "index_date"]), "cohort")
    folds = unique_id(pd.read_parquet(paths["folds"], columns=["participant_id", "fold_id"]), "folds")
    cols = ["participant_id", "index_date"] + [f"prevalent_{o}_count" for o in ORGANS] + [f"has_prevalent_{o}" for o in ORGANS]
    labels = unique_id(pd.read_parquet(paths["labels"], columns=cols), "labels")
    for x in (context, cohort, folds, labels):
        assert len(x) == len(cohort) and set(x.index) == set(cohort.index)
    assert context.fold_id.eq(folds.fold_id.reindex(context.index)).all()
    assert cohort.index_date.eq(labels.index_date.reindex(cohort.index)).all()
    frame = context.join(cohort, validate="one_to_one").join(measurements, validate="one_to_one")
    assert frame.index_date.eq(frame.visit_date_0).all()
    assert (frame.smoking_current.notna() | frame.smoking_missing.eq(1)).all()
    frame["smoking_current"] = frame.smoking_current.fillna(0)
    frame["prevalent_organ_count"] = labels[[f"has_prevalent_{o}" for o in ORGANS]].astype(int).sum(axis=1).reindex(frame.index)
    for o in ORGANS:
        n = labels[f"prevalent_{o}_count"].reindex(frame.index)
        assert (n >= 0).all() and n.gt(0).eq(labels[f"has_prevalent_{o}"].reindex(frame.index)).all()
        frame[f"log1p_prevalent_{o}_count"] = np.log1p(n)
    return frame, folds

def endpoint_frame(frame, spec):
    key, visit, transform = spec["endpoint"], spec["instance"], spec["transform"]
    out = frame.copy()
    out["elapsed_years"] = (out[f"visit_date_{visit}"] - out.index_date).dt.days / 365.25
    b, f = out[f"{key}_0"], out[f"{key}_{visit}"]
    out["baseline_measurement"], out["outcome"] = b, f
    if transform == "log":
        out["baseline_measurement"], out["outcome"] = np.log(b), np.log(f)
    controls = ["baseline_measurement", *CORE]
    if transform == "logit": controls = ["baseline_pace_slow", "baseline_pace_brisk", *CORE]
    if spec["organ"] == "pulmonary": controls += ["baseline_height_cm", "smoking_current", "smoking_missing"]
    mask, flow = pd.Series(True, index=out.index), []
    criteria = [
        ("Current EHR-AGE cohort", mask.copy()),
        ("Valid follow-up measurement", np.isfinite(f)),
        ("Valid baseline measurement", np.isfinite(b)),
        ("Follow-up strictly after index", out.elapsed_years.gt(0)),
        ("Complete specified covariates", np.isfinite(out[controls]).all(axis=1)),
    ]
    if key == "heel_bmd":
        criteria.insert(3, ("Published sex-specific eBMD bounds at both visits", bmd_in_range(b, out.sex_male) & bmd_in_range(f, out.sex_male)))
    for label, ok in criteria:
        old = int(mask.sum()); mask &= ok
        flow.append({"endpoint": key, "step": label, "retained": int(mask.sum()), "excluded_at_step": old-int(mask.sum())})
    columns = list(dict.fromkeys(["fold_id", "outcome", "baseline_measurement", "elapsed_years", *controls]))
    out = out.loc[mask, columns].copy()
    assert len(out) > 200 and set(out.fold_id) == set(range(5)), f"Insufficient usable data for {key}"
    return out, controls, flow

def decomposition(raw, risk, cap, temperature):
    offsets = cap * np.tanh(risk / temperature)
    bases = raw - offsets
    relative = offsets - (offsets.sum(axis=1, keepdims=True) - offsets) / 6
    return bases.mean(axis=1), offsets.mean(axis=1), relative, np.ptp(bases, axis=1)

def add_exposures(frame, components, organ):
    out = frame.join(components[["base_gap", "mean_offset", f"{organ}_osag", f"{organ}_relative"]], validate="one_to_one")
    scales = []
    for col in ["base_gap", "mean_offset", f"{organ}_osag", f"{organ}_relative"]:
        mean, sd = float(out[col].mean()), float(out[col].std(ddof=0))
        assert np.isfinite(out[col]).all() and sd > 1e-12
        out[col+"_z"] = (out[col]-mean)/sd
        scales.append({"variable": col, "n": len(out), "mean_years": mean, "sd_years_ddof0": sd})
    return out, scales

def fit_association(frame, spec, controls, kind, pooled=True):
    organ, trans = (spec['organ'], spec['transform'])
    exposure = f"{organ}_{('osag' if kind == 'full_OSAG' else 'relative')}_z"
    extra = [] if kind == 'full_OSAG' else ['base_gap_z', 'mean_offset_z']
    design = frame[[*controls, *extra, exposure]].astype(float).copy()
    if pooled:
        for f in range(1, 5):
            design[f'original_fold_{f}'] = frame.fold_id.eq(f).astype(float)
    constant = [c for c in design if design[c].nunique() <= 1]
    design = design.drop(columns=constant)
    assert exposure in design
    for c in design:
        if c != exposure:
            design[c] = (design[c] - design[c].mean()) / design[c].std(ddof=0)
    design.insert(0, 'intercept', 1.0)
    x, y = (design.to_numpy(float), frame.outcome.to_numpy(float))
    assert np.isfinite(x).all() and np.isfinite(y).all() and (np.linalg.matrix_rank(x) == x.shape[1])
    nuisance = design.drop(columns=exposure)
    if trans == 'logit':
        assert set(y) == {0.0, 1.0}
        fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(cov_type='HC0', maxiter=100)
        assert fit.converged
        partial, leverage = (np.nan, np.nan)
        covariance = 'HC0 logistic sandwich'
    else:
        fit = sm.OLS(y, design, hasconst=True).fit(cov_type='HC3', use_t=False)
        reduced = sm.OLS(y, nuisance, hasconst=True).fit()
        partial = float((reduced.ssr - fit.ssr) / reduced.ssr)
        residual = design[exposure].to_numpy() - nuisance.to_numpy() @ np.linalg.lstsq(nuisance, design[exposure], rcond=None)[0]
        beta_fwl = float(residual @ y / (residual @ residual))
        assert np.isclose(beta_fwl, fit.params[exposure], atol=1e-08, rtol=1e-08)
        leverage = float(np.sum(np.linalg.qr(x, mode='reduced')[0] ** 2, axis=1).max())
        covariance = 'HC3 linear'
    ci = fit.conf_int().loc[exposure]
    beta, low, high = (float(fit.params[exposure]), float(ci.iloc[0]), float(ci.iloc[1]))
    if trans == 'log':
        effect, lo, hi = [100 * np.expm1(v) for v in (beta, low, high)]
        unit = 'percent difference in conditional geometric mean'
    elif trans == 'logit':
        effect, lo, hi = [np.exp(v) for v in (beta, low, high)]
        unit = 'odds ratio'
    else:
        effect, lo, hi, unit = (beta, low, high, spec['unit'])
    row = {**spec, 'model': kind, 'n': len(frame), 'beta_analysis_scale': beta, 'se_analysis_scale': float(fit.bse[exposure]), 'ci95_low_analysis_scale': low, 'ci95_high_analysis_scale': high, 'effect_per_exposure_SD': float(effect), 'ci95_low': float(lo), 'ci95_high': float(hi), 'effect_unit': unit, 'p_two_sided': float(fit.pvalues[exposure]), 'partial_r2': partial, 'covariance': covariance, 'max_leverage': leverage, 'condition_number': float(np.linalg.cond(x)), 'constant_columns_omitted': ';'.join(constant), 'slow_walking_events': int(y.sum()) if trans == 'logit' else np.nan}
    coeff = pd.DataFrame({'endpoint': spec['endpoint'], 'model': kind, 'term': fit.params.index, 'beta_analysis_scale': fit.params.values, 'se': fit.bse.values, 'ci95_low': fit.conf_int()[0].values, 'ci95_high': fit.conf_int()[1].values, 'p_two_sided': fit.pvalues.values})
    return (row, coeff)

def self_test():
    sbp, pp = paired_bp([[120, 140], [120, 140], [80, 140]], [[80, 90], [np.nan, 90], [90, 90]])
    assert sbp[0] == 130 and pp[0] == 45 and np.isnan(pp[1:]).all()
    assert np.isnan(mean_two([[1, np.nan]])[0])
    assert np.array_equal(pace_code(pd.Series(["Slow pace", "Steady average pace", "Brisk pace"])), [1, 2, 3])
    assert pace_code(pd.Series([None, "Prefer not to answer"])).isna().all()
    assert bmd_in_range([.18, .181, 1.06, .12, .121, 1.025], [1, 1, 1, 0, 0, 0]).tolist() == [False, True, False, False, True, False]
    risk = np.arange(21.).reshape(3, 7)/10
    offsets = 8*np.tanh(risk/2); raw = offsets + np.array([1., 2., 3.])[:, None]
    base, mean, relative, spread = decomposition(raw, risk, 8, 2)
    assert np.allclose(base, [1, 2, 3]) and np.allclose(relative.sum(axis=1), 0)
    for j in range(7): assert np.allclose(relative[:, j], offsets[:, j]-np.delete(offsets, j, axis=1).mean(axis=1))
    assert np.allclose(spread, 0)
    # A known-effect fixture checks generic organ selection and continuous units.
    rng = np.random.default_rng(43); n = 600
    fixture = pd.DataFrame({"cardiovascular_osag_z": rng.normal(size=n), "baseline": rng.normal(size=n), "fold_id": np.arange(n)%5})
    fixture["outcome"] = 5+2*fixture.cardiovascular_osag_z+3*fixture.baseline+rng.normal(0, .1, n)
    row, _ = fit_association(fixture, SPECS[0], ["baseline"], "full_OSAG")
    assert abs(row["effect_per_exposure_SD"]-2) < .02 and row["ci95_low"] < 2 < row["ci95_high"]
    walking = next(s for s in SPECS if s["endpoint"] == "slow_walking")
    fixture = fixture.rename(columns={"cardiovascular_osag_z": "musculoskeletal_osag_z"})
    probs = 1/(1+np.exp(-(-1+.6*fixture.musculoskeletal_osag_z+.2*fixture.baseline)))
    fixture["outcome"] = rng.binomial(1, probs)
    row, _ = fit_association(fixture, walking, ["baseline"], "full_OSAG")
    assert row["effect_per_exposure_SD"] > 1 and np.isclose(np.log(row["effect_per_exposure_SD"]), row["beta_analysis_scale"])
    assert len(SPECS) == 21 and all(sum(s["organ"] == o for s in SPECS) == 3 for o in ORGANS)
    log("self_tests_passed", checks=["same-array BP pairing", "missing-value propagation", "pace categories", "all-seven component mapping", "linear units and independent coefficient", "logistic odds-ratio conversion"])


def prediction_components(paths, ids, frame, folds, cap=8., temperature=2.):
    if cap<=0 or temperature<=0: raise ValueError("Offset cap and temperature must be positive")
    columns=["participant_id", "organ_system", "actual_age", "sex", "outer_fold", "raw_age_gap", "corrected_age_gap", "model_risk_score"]
    pred=pd.read_parquet(paths["predictions"],columns=columns)
    pred["participant_id"]=pred.participant_id.astype(str)
    if pred.duplicated(["participant_id","organ_system"]).any(): raise ValueError("Duplicate participant-organ prediction")
    assert len(pred)==len(frame)*7 and set(pred.organ_system)==set(ORGANS)
    assert set(pred.participant_id)==set(frame.index)
    assert pred.outer_fold.eq(pred.participant_id.map(folds.fold_id)).all(), "Evaluation-fold mismatch"
    assert pred.sex.eq(pred.participant_id.map(frame.sex_male)).all(), "Sex coding mismatch"
    assert np.allclose(pred.actual_age,pred.participant_id.map(frame.chronological_age),atol=1e-5,rtol=0)
    pred=pred.loc[pred.participant_id.isin(ids)]
    assert len(pred)==len(ids)*7
    wide=lambda name:pred.pivot(index="participant_id",columns="organ_system",values=name).loc[ids,list(ORGANS)]
    raw,risk,gaps=wide("raw_age_gap"),wide("model_risk_score"),wide("corrected_age_gap")
    base,mean,relative,spread=decomposition(raw.to_numpy(float),risk.to_numpy(float),cap,temperature)
    result=pd.DataFrame({"base_gap":base,"mean_offset":mean},index=ids)
    for j,organ in enumerate(ORGANS):
        result[f"{organ}_osag"]=gaps[organ]
        result[f"{organ}_relative"]=relative[:,j]
    assert np.isfinite(result).all().all()
    return result,{"prediction_rows":len(frame)*7,"evaluation_folds":5,"maximum_reconstructed_base_span_years":float(spread.max())}



def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs={"phenotypes":"grip","assessment-dates":"dates","walking-pace":"walking","predictions":"predictions",
            "covariates":"context","baseline-labels":"labels","cohort":"cohort","folds":"folds"}
    for flag in inputs:p.add_argument("--"+flag,type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--offset-cap-years",type=float,default=8.)
    p.add_argument("--risk-temperature",type=float,default=2.)
    p.add_argument("--numeric-repeatability-only",action="store_true",
                   help="Explicitly use the reported numerical spirometry rule; does not certify manoeuvre acceptability")
    p.add_argument("--validate-only",action="store_true",help="Read and validate inputs without fitting or writing outputs")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test();return
    missing=[flag for flag in inputs if getattr(args,flag.replace("-","_")) is None]
    if missing:p.error("Missing input flags: "+", ".join(missing))
    if not args.numeric_repeatability_only:p.error("Exact reproduction requires explicit --numeric-repeatability-only")
    if not args.validate_only and args.output is None:p.error("--output is required for analysis")
    paths={key:getattr(args,flag.replace("-","_")).resolve(strict=True) for flag,key in inputs.items()}
    if args.output is not None and args.output.exists() and any(args.output.iterdir()):
        p.error("Output directory is not empty; choose a new directory")
    measurements,availability=load_measurements(paths)
    frame,folds=assemble_context(paths,measurements)
    frames,controls,flow,characteristics={},{},[],[]
    for spec in SPECS:
        key=spec["endpoint"];data,ctl,steps=endpoint_frame(frame,spec)
        frames[key],controls[key]=data,ctl;flow+=steps
        characteristics.append({**spec,"n":len(data),"followup_median_years":float(data.elapsed_years.median()),
          "followup_p25_years":float(data.elapsed_years.quantile(.25)),"followup_p75_years":float(data.elapsed_years.quantile(.75)),
          **{f"fold_{f}_n":int(data.fold_id.eq(f).sum()) for f in range(5)}})
    ids=pd.Index(sorted(set().union(*(set(f.index) for f in frames.values()))),name="participant_id")
    components,audit=prediction_components(paths,ids,frame,folds,args.offset_cap_years,args.risk_temperature)
    if args.validate_only:
        log("inputs_validated",participants=len(frame),participants_with_any_followup=len(ids),endpoints=len(SPECS),fits=0)
        return
    rows,fold_rows,coefficients,scales=[],[],[],[]
    with threadpool_limits(limits=4):
        for spec in SPECS:
            key=spec["endpoint"];data,scale=add_exposures(frames[key],components,spec["organ"])
            scales += [{"endpoint":key,**s} for s in scale]
            for kind in ("full_OSAG","relative_component"):
                row,coef=fit_association(data,spec,controls[key],kind);rows.append(row);coefficients.append(coef)
                for fold in range(5):
                    fr,_=fit_association(data.loc[data.fold_id.eq(fold)],spec,controls[key],kind,pooled=False)
                    fr["fold"]=fold;fold_rows.append(fr)
    result=pd.DataFrame(rows);fold_result=pd.DataFrame(fold_rows)
    assert len(result)==42 and len(fold_result)==210
    result["q_BH_42_tests"]=multipletests(result.p_two_sided,method="fdr_bh")[1]
    result["p_Holm_42_tests"]=multipletests(result.p_two_sided,method="holm")[1]
    result["direction_matches_expected"]=result.beta_analysis_scale*result.expected_sign>0
    result["folds_matching_expected_direction"]=[int(((fold_result.loc[fold_result.endpoint.eq(r.endpoint)&fold_result.model.eq(r.model),"beta_analysis_scale"]*r.expected_sign)>0).sum()) for r in result.itertuples()]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"all_associations.csv",index=False)
    fold_result.to_csv(out/"fold_associations.csv",index=False)
    pd.concat(coefficients,ignore_index=True).to_csv(out/"model_coefficients.csv",index=False)
    pd.DataFrame(characteristics).to_csv(out/"participant_characteristics.csv",index=False)
    pd.DataFrame(scales).to_csv(out/"exposure_scales.csv",index=False)
    pd.DataFrame(flow).to_csv(out/"participant_flow.csv",index=False)
    availability.to_csv(out/"measurement_availability.csv",index=False)
    write_json(out/"analysis_definitions.json",{"measures":[metadata(s) for s in SPECS],"rules":RULES,
        "quality_screen":ANALYTIC_QC,"controls":controls,"offset_cap_years":args.offset_cap_years,
        "risk_temperature":args.risk_temperature,"prediction_checks":audit})
    log("analysis_complete",endpoints=21,pooled_associations=42)


if __name__=="__main__":main()

