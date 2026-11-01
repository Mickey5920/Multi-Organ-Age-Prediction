"""Information-matched clinical/EHR reference for round-two Reviewer 1, comment 1.

All historical inputs are read-only. New files live beneath --run-root.
prepare/bootstrap use the existing Anaconda base; fit uses the original
age_prediction_py310 survival-analysis environment. No neural model is refitted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[3]
REV = ROOT / "Revision-0710"
OLD = REV / "06_figures_and_statistics/03_reviewer_statistics"
CLINICAL_SCRIPT = OLD / "03_clinical_utility/01_formal_run/scripts/run_0710_clinical_utility.py"
FINAL_SCRIPT = OLD / "07_final_model_clinical_utility/run_final_model_clinical_utility.py"
FAST_SCRIPT = OLD / "07_final_model_clinical_utility/run_fast_final_model_bootstrap.py"
BOOT_HELPER = OLD / "05_fig11_fig12_marginal_and_review_closure/01_scripts/run_clinical_utility_paired_bootstrap.py"
REGISTRY = REV / "02_baselines_5fold/04_objective_and_generic_20260808/03_evaluation/registries/medbert_style_prospective.json"
RAW_CACHE = REV / "01_data_rebuild/04_cache_v1"
OUTCOME_SHA = "3a9021f00080c25fd523debcdf075a5be8281c9658db261af99a8bdc11f4e735"
ORGANS = ("cardiovascular", "pulmonary", "musculoskeletal", "immune", "renal", "hepatic", "metabolic")
PHENO = tuple(f"phenotype_{i:02d}" for i in range(78))
MISSING = tuple(f"phenotype_missing_{i:02d}" for i in range(78))
MED = tuple(f"medbert_risk__{o}" for o in ORGANS)
GAPS = tuple(f"gap__{o}" for o in ORGANS)
PENALTIES = (0.0001, 0.001, 0.01, 0.1)
SEED = 20261031
TIER = "information_matched"


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def check_paths(run_root):
    root = Path(run_root).resolve()
    if root.drive.upper() != "E:" or ROOT.resolve() not in root.parents:
        raise RuntimeError(f"Run directory resolves outside canonical project: {root}")
    for name in ("outputs", "cache", "tmp"):
        p = (root / name).resolve()
        if root not in p.parents or p.drive.upper() == "C:":
            raise RuntimeError(f"Unsafe output: {p}")
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(root / "cache/numba"))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(root / "tmp"))
    return root


def prepare(run_root):
    """Convert only the small risk-score tables for the original no-Arrow environment."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    records = []
    for item in registry["run_directories"]:
        fold = int(item["outer_fold"])
        source = (REGISTRY.parent / item["path"]).resolve()
        config_path = source / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["status"] == "COMPLETE" and config["max_batches"] is None
        assert config["source_hashes"]["outcomes"] == OUTCOME_SHA
        assert digest(config_path) == item["run_config_sha256"]
        assert config["fold_roles"] == {
            "train": [f for f in range(5) if f not in (fold, (fold + 1) % 5)],
            "calibration": [(fold + 1) % 5], "outer_test": [fold],
        }
        for role in ("validation", "test"):
            path = source / f"{role}_risk_scores.parquet"
            df = pd.read_parquet(path)
            ids = df["participant_id"].astype(str).to_numpy(dtype="U7")
            assert len(set(ids)) == len(ids)
            assert set(df["outer_fold"].astype(int)) == {fold}
            assert set(df["split_role"]) == {role}
            scores = df[[f"risk_score_{o}" for o in ORGANS]].to_numpy(np.float32)
            assert np.isfinite(scores).all()
            assert len(df) == config["participant_counts"]["calibration" if role == "validation" else "outer_test"]
            output = run_root / "cache/medbert" / f"outer_{fold}__{role}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, participant_id=ids, scores=scores,
                                organs=np.asarray(ORGANS), outer_fold=fold, role=role)
            records.append({"outer_fold": fold, "role": role, "rows": len(df),
                            "source": str(path), "source_sha256": digest(path),
                            "cache": str(output), "cache_sha256": digest(output)})
    write_json(run_root / "outputs/analysis_manifest.json", {
        "status": "INPUTS_PREPARED", "project": str(ROOT.resolve()),
        "run_root": str(run_root), "source_code_sha256": digest(__file__),
        "inputs": records, "outcomes_sha256_recorded_by_training": OUTCOME_SHA,
        "raw_cache": str(RAW_CACHE.resolve()),
        "raw_cache_manifest_sha256": digest(RAW_CACHE / "cache_run_manifest.json"),
        "raw_cache_hash_evidence": "Existing per-shard manifests reused; bulk inputs not rehashed",
        "clinical_runner_sha256": digest(CLINICAL_SCRIPT),
        "final_prediction_adapter_sha256": digest(FINAL_SCRIPT),
        "protocol": {
            "reference": "C21 + all 78 phenotypes + 78 missing masks + seven frozen Med-BERT direct risk scores",
            "augmentation": "seven frozen EHR-AGE OSAGs", "organs": ORGANS,
            "neural_training": "existing three-fold models; no neural retraining",
            "fit_fold": "(outer_fold + 1) % 5", "test_fold": "outer_fold",
            "phenotype_preprocessing": "reuse original three-training-fold median/mean/std",
            "cox_scaling_imputation": "fit partition only; repeated inside calibration-only CV",
            "cv": "three event-stratified participant folds within calibration fold",
            "penalties": PENALTIES, "cv_metric": "mean held-out Harrell C",
            "penalty_ties": "larger penalty if scores tied to 1e-12",
            "seed": SEED, "bootstrap_replicates": 1000,
            "bootstrap": "paired participants within original outer folds, joint across organs; fixed fits",
            "aggregation": "equal mean across five folds, then equal mean across seven organs",
            "death": "cause-specific censoring; original right-open 60-month boundary",
            "selection_scope": "Med-BERT selected using existing matched comparison; secondary post-hoc analysis",
        },
    })
    log("prepared", files=len(records), bytes=sum(Path(r["cache"]).stat().st_size for r in records))


def modules():
    # The unchanged historical scorer suppresses warnings inside fit. Expose
    # convergence failures for this wider design without editing that scorer.
    import lifelines
    from lifelines.exceptions import ConvergenceWarning, ConvergenceError
    if not getattr(lifelines.CoxPHFitter, "_r11_audited", False):
        original = lifelines.CoxPHFitter

        class AuditedCoxPHFitter(original):
            _r11_audited = True

            def fit(self, *args, **kwargs):
                attempts = []
                options = (None, {"step_size": 0.25, "max_steps": 1000},
                           {"step_size": 0.05, "max_steps": 2000})
                for attempt, fit_options in enumerate(options, 1):
                    call_kwargs = dict(kwargs)
                    if fit_options is not None:
                        call_kwargs["fit_options"] = fit_options
                    error = None
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
                        try:
                            result = super().fit(*args, **call_kwargs)
                        except (ConvergenceError, np.linalg.LinAlgError) as exc:
                            error = str(exc)
                    numerical = [str(w.message) for w in captured
                                 if issubclass(w.category, (ConvergenceWarning, RuntimeWarning))]
                    if error is not None:
                        numerical.append(error)
                    attempts.append({"options": fit_options, "warnings": numerical})
                    if not numerical and np.isfinite(self.log_likelihood_):
                        AuditedCoxPHFitter._r11_last_attempts = attempts
                        return result
                    log("solver_retry", failed_attempt=attempt, next_attempt=attempt + 1,
                        warning_first_lines=[w.splitlines()[0] for w in numerical])
                raise RuntimeError("Cox did not converge after deterministic step-size retries: " +
                                   "; ".join(attempts[-1]["warnings"]))

        lifelines.CoxPHFitter = AuditedCoxPHFitter
    clinical = load_module("r11_legacy_clinical", CLINICAL_SCRIPT)
    final = load_module("r11_final_adapter", FINAL_SCRIPT)
    final.configure_runner(clinical)
    return clinical, final


def raw_phenotypes(actual_fold, outer_fold):
    manifest = json.loads((RAW_CACHE / "batch_manifest.json").read_text(encoding="utf-8"))
    pieces = []
    with np.load(RAW_CACHE / "fold_transforms" / f"outer_fold_{outer_fold}.npz") as z:
        assert int(z["outer_fold"]) == outer_fold
        assert int(z["validation_fold"]) == (outer_fold + 1) % 5
        median, mean, std = (z[k].astype(np.float32) for k in ("median", "mean", "std"))
    for entry in manifest:
        if int(entry["fold_id"]) != actual_fold:
            continue
        with np.load(RAW_CACHE / entry["path"], allow_pickle=False) as z:
            ids = z["participant_id"].astype(str)
            assert np.all(z["fold_id"] == actual_fold)
            raw, missing = z["phenotype_raw_zero"], z["phenotype_missing_mask"].astype(bool)
            values = ((np.where(missing, median[None, :], raw) - mean[None, :]) / std[None, :]).astype(np.float32)
            assert values.shape == missing.shape == (len(ids), 78)
            assert np.isfinite(values).all()
            frame = pd.DataFrame(np.concatenate((values, missing.astype(np.float32)), axis=1),
                                 columns=PHENO + MISSING)
            frame.insert(0, "participant_id", ids)
            pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    assert not result["participant_id"].duplicated().any()
    return result


def risk_features(run_root, outer_fold, role):
    with np.load(run_root / "cache/medbert" / f"outer_{outer_fold}__{role}.npz") as z:
        assert tuple(z["organs"].astype(str)) == ORGANS
        assert int(z["outer_fold"]) == outer_fold and str(z["role"]) == role
        result = pd.DataFrame(z["scores"].copy(), columns=MED)
        result.insert(0, "participant_id", z["participant_id"].astype(str))
    return result


def load_fold(run_root, outer_fold):
    clinical, final = modules()
    context = final.load_context()
    frames = []
    for role, actual in (("validation", (outer_fold + 1) % 5), ("test", outer_fold)):
        raw = raw_phenotypes(actual, outer_fold)
        risk = risk_features(run_root, outer_fold, role)
        assert set(raw["participant_id"]) == set(risk["participant_id"])
        frame = raw.merge(risk, on="participant_id", validate="one_to_one")
        actual_ids = set(context.loc[context["fold_id"].eq(actual), "participant_id"])
        assert set(frame["participant_id"]) == actual_ids
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    enriched = context.merge(features, on="participant_id", how="inner", validate="one_to_one")
    assert len(enriched) == len(features)
    np.testing.assert_allclose(enriched[list(MISSING)].mean(axis=1),
                               enriched["phenotype_missing_fraction"], atol=1e-7)
    sources = final.prediction_paths(outer_fold)
    fit_pred = final.load_prediction_frame(sources[0], outer_fold, "calibration")
    test_pred = final.load_prediction_frame(sources[1], outer_fold, "outer_test")
    fit_gaps, test_gaps = clinical.gap_wide(fit_pred), clinical.gap_wide(test_pred)
    universe = np.sort(context.loc[context["fold_id"].eq(outer_fold), "participant_id"].astype(np.int64).to_numpy())
    return clinical, enriched, fit_pred, test_pred, fit_gaps, test_gaps, universe


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


def cell_frames(data, outer_fold, organ):
    clinical, context, fp, sp, fg, sg, universe = data
    fit, score, _ = clinical.build_cell_frames(fp, sp, context, fg, sg,
                                               outer_fold=outer_fold, organ=organ)
    # Restore original clinical missing values so inner-CV medians use inner-fit only.
    indexed = context.set_index("participant_id")
    for frame in (fit, score):
        frame.loc[:, list(clinical.C_CONTEXT_DENSITY_PRIMARY)] = indexed.reindex(
            frame["participant_id"])[list(clinical.C_CONTEXT_DENSITY_PRIMARY)].to_numpy()
        assert np.isfinite(frame[list(PHENO + MISSING + MED + GAPS)].to_numpy()).all()
    return fit, score, universe


def choose_penalty(fit_raw, columns, clinical, seed, *, pilot=False):
    from sklearn.model_selection import StratifiedKFold
    splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(fit_raw, fit_raw["event"]))
    rows = []
    penalties = (0.01,) if pilot else PENALTIES
    for cv_index, (train_index, valid_index) in enumerate(splits[:1] if pilot else splits):
        train, valid = fit_raw.iloc[train_index].copy(), fit_raw.iloc[valid_index].copy()
        assert not set(train["participant_id"]) & set(valid["participant_id"])
        train["split_role"], valid["split_role"] = "validation", "test"
        train, valid, _ = impute_clinical(train, valid, clinical.C_CONTEXT_DENSITY_PRIMARY)
        grid = clinical.build_metric_time_grid(train, valid, report_times=(60.0,), ibs_times=(60.0,))
        for penalty in penalties:
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
                bundle = clinical.fit_lifelines_cox_scores(train, valid, covariates=columns,
                    time_grid=grid, phase="formal", penalizer=penalty)
            value = float(clinical.harrell_c_index(valid, bundle.risk_score))
            assert np.isfinite(value)
            rows.append({"inner_fold": cv_index, "penalizer": penalty,
                         "harrell_c": value, "fit_n": len(train), "validation_n": len(valid),
                         "fit_events": int(train["event"].sum()),
                         "validation_events": int(valid["event"].sum()),
                         "seconds": time.monotonic() - started,
                         "solver_attempts": len(__import__("lifelines").CoxPHFitter._r11_last_attempts),
                         "warning_count": len(captured)})
            log("cv", penalty=penalty, inner_fold=cv_index, c=value,
                seconds=rows[-1]["seconds"], dimensions=len(columns))
    table = pd.DataFrame(rows)
    means = table.groupby("penalizer")["harrell_c"].mean()
    maximum = means.max()
    chosen = float(max(p for p, v in means.items() if maximum - v <= 1e-12))
    return chosen, table


def evaluate_bundle(clinical, fit, score, bundle, grid):
    report, _ = clinical.evaluate_censoring_aware_metrics(fit, score,
        risk_score=bundle.risk_score, survival_probability=bundle.survival_probability,
        time_grid=grid, calibration_bins=10)
    metrics = {"harrell_c": float(clinical.harrell_c_index(score, bundle.risk_score)),
               **clinical.flatten_primary_metrics(report)}
    risks, calibration, dcas = {}, [], []
    for horizon in clinical.REPORT_TIMES:
        risks[horizon] = clinical.probability_at(bundle, grid, horizon, fit=False)
        fit_risk = clinical.probability_at(bundle, grid, horizon, fit=True)
        summary, bins = clinical.fixed_bin_calibration(fit, score, fit_risk=fit_risk,
            score_risk=risks[horizon], horizon=grid.effective_for(horizon),
            report_time=horizon, n_bins=10)
        metrics.update({f"cal_{int(horizon)}m_{k}": v for k, v in summary.items() if k != "time_months"})
        calibration.append(bins)
    state = clinical.ipcw_binary_state(fit, score, horizon=grid.effective_for(60.0))
    dca = clinical.ipcw_decision_curve(risks[60.0], state, thresholds=clinical.THRESHOLDS)
    return metrics, risks, pd.concat(calibration, ignore_index=True), dca, state


def save_bootstrap_inputs(path, fit, score, universe, base, aug, base_risk, aug_risk, horizon):
    from sksurv.nonparametric import CensoringDistributionEstimator
    helper = load_module("r11_old_bootstrap_helper", BOOT_HELPER)
    ids = score["participant_id"].astype(np.int64).to_numpy()
    positions = np.searchsorted(universe, ids)
    assert np.array_equal(universe[positions], ids)
    duration, event = score["duration"].to_numpy(float), score["event"].to_numpy(np.int8)
    censor = CensoringDistributionEstimator().fit(helper.survival(
        fit["duration"].to_numpy(float), fit["event"].to_numpy(np.int8)))
    uno_weight = np.square(censor.predict_ipcw(helper.survival(duration, event)))
    uno_weight[duration >= horizon] = 0.0
    case, control = helper.ipcw_components(duration, event, censor, horizon)
    np.savez_compressed(path, participant_id=ids, universe_size=len(universe),
        universe_positions=positions, duration=duration, event=event,
        uno_event_weight=uno_weight, case_weight=case, control_weight=control,
        base_score=base.risk_score, base_risk=base_risk,
        all_gaps_score=aug.risk_score, all_gaps_risk=aug_risk)


def fit_fold(run_root, outer_fold, *, pilot=False, only_organ=None):
    started = time.monotonic()
    folder = run_root / "outputs" / ("pilot" if pilot else "folds") / f"outer_fold_{outer_fold}"
    folder.mkdir(parents=True, exist_ok=True)
    data = load_fold(run_root, outer_fold)
    clinical = data[0]
    base_columns = tuple(clinical.C_CONTEXT_DENSITY_PRIMARY) + PHENO + MISSING + MED
    assert len(base_columns) == 184 and len(set(base_columns)) == 184
    metrics_rows, incremental_rows, cv_rows, coefficient_rows = [], [], [], []
    bin_rows, dca_rows, sample_rows = [], [], []
    for organ_index, organ in enumerate(ORGANS):
        if only_organ and organ != only_organ:
            continue
        checkpoint = folder / f"{organ}__complete.json"
        if checkpoint.exists() and not pilot:
            log("skip_complete", outer_fold=outer_fold, organ=organ)
            continue
        cell_started = time.monotonic()
        prior_pilot = None
        if pilot and (folder / f"{organ}__bootstrap.npz").exists():
            with np.load(folder / f"{organ}__bootstrap.npz") as old:
                prior_pilot = {k: old[k].copy() for k in ("base_score", "base_risk", "all_gaps_score", "all_gaps_risk")}
        raw_fit, raw_score, universe = cell_frames(data, outer_fold, organ)
        if pilot:
            # One representative full-size cell; one lambda and one inner split only.
            assert only_organ is not None
        fit, score, medians = impute_clinical(raw_fit, raw_score, clinical.C_CONTEXT_DENSITY_PRIMARY)
        grid = clinical.build_metric_time_grid(fit, score, report_times=clinical.REPORT_TIMES)
        predictions = {}
        local_cv, local_coeff, local_metrics, local_bins, local_dca = [], [], [], [], []
        for augmentation, columns in (("base", base_columns), ("all_gaps", base_columns + GAPS)):
            penalty, cv = choose_penalty(raw_fit, columns, clinical,
                SEED + 100 * outer_fold + organ_index, pilot=pilot)
            cv.insert(0, "augmentation", augmentation)
            local_cv.append(cv)
            fit_started = time.monotonic()
            bundle = clinical.fit_lifelines_cox_scores(fit, score, covariates=columns,
                time_grid=grid, phase="formal", penalizer=penalty)
            values, risks, bins, dca, state = evaluate_bundle(clinical, fit, score, bundle, grid)
            name = f"{TIER}__{augmentation}"
            local_metrics.append({"outer_fold": outer_fold, "organ_system": organ,
                "tier": TIER, "augmentation": augmentation, "model": name,
                "penalizer": penalty, "fit_n": len(fit), "fit_events": int(fit["event"].sum()),
                "test_n": len(score), "test_events": int(score["event"].sum()),
                "n_covariates": len(columns), "n_retained": len(bundle.coefficients),
                "solver_attempts": len(__import__("lifelines").CoxPHFitter._r11_last_attempts),
                "fit_seconds": time.monotonic() - fit_started, **values})
            for k, v in bundle.coefficients.items():
                local_coeff.append({"augmentation": augmentation, "covariate": k,
                                    "coefficient_per_fit_sd": v})
            for df in (bins, dca):
                df.insert(0, "model", name)
                df.insert(0, "organ_system", organ)
                df.insert(0, "outer_fold", outer_fold)
            local_bins.append(bins); local_dca.append(dca)
            predictions[augmentation] = (bundle, values, risks, dca)
            log("fit_evaluated", outer_fold=outer_fold, organ=organ,
                augmentation=augmentation, penalty=penalty, c=values["harrell_c"],
                seconds=local_metrics[-1]["fit_seconds"])
        base, bv, br, bd = predictions["base"]
        aug, av, ar, ad = predictions["all_gaps"]
        reclass = clinical.ipcw_reclassification(br[60.0], ar[60.0], state)
        delta_dca = bd[["threshold", "net_benefit"]].merge(
            ad[["threshold", "net_benefit"]], on="threshold", suffixes=("_base", "_aug"))
        delta_nb = delta_dca["net_benefit_aug"] - delta_dca["net_benefit_base"]
        increment = {"outer_fold": outer_fold, "organ_system": organ, "tier": TIER,
            "comparison": "information_matched__all_gaps_vs_base",
            "baseline_model": f"{TIER}__base", "augmented_model": f"{TIER}__all_gaps",
            "test_n": len(score), "test_events": int(score["event"].sum()),
            "delta_harrell_c": av["harrell_c"] - bv["harrell_c"],
            "delta_uno_c_60m": av["uno_c_60m"] - bv["uno_c_60m"],
            **{f"delta_auc_{int(t)}m": av[f"auc_{int(t)}m"] - bv[f"auc_{int(t)}m"] for t in clinical.REPORT_TIMES},
            **{f"brier_improvement_{int(t)}m": bv[f"brier_{int(t)}m"] - av[f"brier_{int(t)}m"] for t in clinical.REPORT_TIMES},
            "ibs_improvement": bv["ibs"] - av["ibs"], **reclass,
            "mean_delta_net_benefit": float(delta_nb.mean())}
        pd.DataFrame(local_metrics).to_csv(folder / f"{organ}__metrics.csv", index=False)
        pd.DataFrame([increment]).to_csv(folder / f"{organ}__increment.csv", index=False)
        pd.concat(local_cv).to_csv(folder / f"{organ}__cv.csv", index=False)
        pd.DataFrame(local_coeff).to_csv(folder / f"{organ}__coefficients.csv", index=False)
        pd.concat(local_bins).to_csv(folder / f"{organ}__calibration.csv", index=False)
        pd.concat(local_dca).to_csv(folder / f"{organ}__dca.csv", index=False)
        save_bootstrap_inputs(folder / f"{organ}__bootstrap.npz", fit, score, universe,
                              base, aug, br[60.0], ar[60.0], grid.effective_for(60.0))
        if prior_pilot is not None:
            with np.load(folder / f"{organ}__bootstrap.npz") as current:
                for key, values in prior_pilot.items():
                    np.testing.assert_allclose(current[key], values, rtol=1e-10, atol=1e-10)
        write_json(checkpoint, {"status": "PILOT_ONLY" if pilot else "COMPLETE",
            "outer_fold": outer_fold, "organ": organ, "fit_n": len(fit), "test_n": len(score),
            "test_events": int(score["event"].sum()), "imputation_medians": medians,
            "seconds": time.monotonic() - cell_started, "python": sys.executable,
            "numpy": np.__version__, "pandas": pd.__version__,
            "audited_convergence": True, "pilot_prediction_parity": prior_pilot is not None,
            "source_sha256": digest(__file__), "bootstrap_file": f"{organ}__bootstrap.npz"})
        log("cell_complete", outer_fold=outer_fold, organ=organ,
            delta_c=increment["delta_harrell_c"], seconds=time.monotonic() - cell_started)
        del predictions, base, aug, fit, score, raw_fit, raw_score, bundle
        gc.collect()
    log("fold_complete", outer_fold=outer_fold, pilot=pilot, seconds=time.monotonic() - started)


def load_bootstrap_cell(run_root, fold, organ, fast):
    path = run_root / "outputs/folds" / f"outer_fold_{fold}" / f"{organ}__bootstrap.npz"
    with np.load(path) as z:
        arrays = {key: z[key].copy() for key in z.files}
    duration, event = arrays["duration"], arrays["event"]
    case, control = arrays["case_weight"], arrays["control_weight"]
    base = fast.prepare_model(duration, event, case, control, arrays["base_score"])
    augmentation = fast.prepare_augmentation(duration, event, case, control,
        arrays["base_risk"], arrays["all_gaps_score"], arrays["all_gaps_risk"])
    return fast.CellPreparation(fold=fold, organ=organ,
        universe_size=int(arrays["universe_size"]),
        universe_positions=arrays["universe_positions"].astype(np.int64),
        duration=duration, event=event, uno_event_weight=arrays["uno_event_weight"],
        case_weight=case, control_weight=control,
        order=np.argsort(duration, kind="mergesort").astype(np.int64), base=base,
        base_brier_contribution=case * (1.0 - arrays["base_risk"]) ** 2 + control * arrays["base_risk"] ** 2,
        augmentations={"all_gaps": augmentation})


def bootstrap_fold(run_root, fold, replicates):
    fast = load_module("r11_fast_bootstrap", FAST_SCRIPT)
    cells = {organ: load_bootstrap_cell(run_root, fold, organ, fast) for organ in ORGANS}
    universe_n = next(iter(cells.values())).universe_size
    assert {c.universe_size for c in cells.values()} == {universe_n}
    folder = run_root / "outputs/folds" / f"outer_fold_{fold}"
    parity = []
    for organ, cell in cells.items():
        result = fast.incremental_metrics(cell, np.ones(len(cell.event), dtype=np.int64), "all_gaps")
        expected = pd.read_csv(folder / f"{organ}__increment.csv").iloc[0]
        for metric in fast.METRICS:
            error = abs(float(result[metric]) - float(expected[metric]))
            assert error < 2e-8, (fold, organ, metric, result[metric], expected[metric])
            parity.append({"organ": organ, "metric": metric, "absolute_error": error})
    write_json(folder / "bootstrap_parity.json", {"status": "PASS", "checks": parity})
    rng = np.random.default_rng(SEED + 10007 * fold)
    probabilities = np.full(universe_n, 1.0 / universe_n)
    rows = []
    started = time.monotonic()
    for replicate in range(replicates):
        counts = rng.multinomial(universe_n, probabilities)
        for organ, cell in cells.items():
            result = fast.incremental_metrics(cell, counts[cell.universe_positions].astype(np.int64), "all_gaps")
            assert all(np.isfinite(result[m]) for m in fast.METRICS)
            rows.append({"bootstrap": replicate, "outer_fold": fold,
                         "organ_system": organ, "augmentation": "all_gaps", **result})
        if (replicate + 1) % 100 == 0:
            log("bootstrap_progress", outer_fold=fold, complete=replicate + 1,
                total=replicates, seconds=time.monotonic() - started)
    pd.DataFrame(rows).to_csv(folder / "bootstrap_replicates.csv.gz", index=False)
    write_json(folder / "bootstrap_complete.json", {"status": "COMPLETE", "replicates": replicates,
        "seconds": time.monotonic() - started, "python": sys.executable,
        "numpy": np.__version__, "source_sha256": digest(__file__)})
    log("bootstrap_complete", outer_fold=fold, replicates=replicates, seconds=time.monotonic() - started)


def summarize(run_root):
    output = run_root / "outputs"
    model, incremental, cv, calibration, dca = [], [], [], [], []
    for fold in range(5):
        folder = output / "folds" / f"outer_fold_{fold}"
        for organ in ORGANS:
            assert json.loads((folder / f"{organ}__complete.json").read_text())["status"] == "COMPLETE"
            for target, suffix in ((model, "metrics"), (incremental, "increment"),
                                   (calibration, "calibration"), (dca, "dca")):
                target.append(pd.read_csv(folder / f"{organ}__{suffix}.csv"))
            frame = pd.read_csv(folder / f"{organ}__cv.csv")
            frame.insert(0, "organ_system", organ); frame.insert(0, "outer_fold", fold)
            cv.append(frame)
    combined = {"cell_model_metrics.csv": pd.concat(model, ignore_index=True),
                "cell_incremental_metrics.csv": pd.concat(incremental, ignore_index=True)}
    for name, frame in combined.items(): frame.to_csv(output / name, index=False)
    pd.concat(cv, ignore_index=True).to_csv(output / "cv_results.csv", index=False)
    pd.concat(calibration, ignore_index=True).to_csv(output / "calibration_bins.csv", index=False)
    pd.concat(dca, ignore_index=True).to_csv(output / "decision_curve.csv", index=False)
    clinical = load_module("r11_legacy_clinical", CLINICAL_SCRIPT)
    clinical.aggregate_results(output, combined)
    boot = pd.concat([pd.read_csv(output / "folds" / f"outer_fold_{fold}" / "bootstrap_replicates.csv.gz")
                      for fold in range(5)], ignore_index=True)
    fast = load_module("r11_fast_bootstrap", FAST_SCRIPT)
    metrics = list(fast.METRICS)
    assert len(boot) == 5 * 7 * 1000
    organ = boot.groupby(["bootstrap", "organ_system"], as_index=False)[metrics].mean()
    macro = organ.groupby("bootstrap", as_index=False)[metrics].mean()
    for scope, table, point in (("organ", organ, pd.read_csv(output / "organ_incremental_metrics.csv")),
                                ("macro", macro, pd.read_csv(output / "macro_incremental_metrics.csv"))):
        result = []
        groups = table.groupby("organ_system") if scope == "organ" else [("macro", table)]
        for name, group in groups:
            p = point.loc[point["organ_system"].eq(name)].iloc[0] if scope == "organ" else point.iloc[0]
            for metric in metrics:
                values = group[metric].to_numpy(float)
                assert len(values) == 1000 and np.isfinite(values).all()
                result.append({"organ_system": name, "metric": metric,
                    "point_estimate": float(p[metric]), "ci95_low": float(np.quantile(values, .025)),
                    "ci95_high": float(np.quantile(values, .975)),
                    "probability_favorable": float(np.mean(values > 0)), "bootstrap_replicates": len(values)})
        pd.DataFrame(result).to_csv(output / f"{scope}_paired_bootstrap_ci.csv", index=False)
    macro.to_csv(output / "macro_bootstrap_replicates.csv", index=False)
    # Descriptive original post-freeze rotations; no new model-selection claim.
    for name, frame in combined.items():
        keys = [k for k in ("organ_system", "augmentation", "model", "comparison") if k in frame]
        numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c != "outer_fold"]
        frame.loc[frame["outer_fold"].isin([2, 3, 4])].groupby(keys, as_index=False)[numeric].mean().to_csv(
            output / ("original_postfreeze_" + name), index=False)
    log("summarized", model_cells=len(combined["cell_model_metrics.csv"]), incremental_cells=len(combined["cell_incremental_metrics.csv"]))


def verify(run_root):
    output = run_root / "outputs"
    model = pd.read_csv(output / "cell_model_metrics.csv")
    increment = pd.read_csv(output / "cell_incremental_metrics.csv")
    assert len(model) == 70 and len(increment) == 35
    assert not model.duplicated(["outer_fold", "organ_system", "augmentation"]).any()
    assert not increment.duplicated(["outer_fold", "organ_system"]).any()
    old = pd.read_csv(REV / "Final_Fig_0716/Reviewer_Statistics/13_final_model_clinical_utility/01_point/cell_model_metrics.csv")
    old = old.loc[old["augmentation"].eq("base")].set_index(["outer_fold", "organ_system"])
    for _, row in model.iterrows():
        ref = old.loc[(row["outer_fold"], row["organ_system"])]
        for name in ("fit_n", "fit_events", "test_n", "test_events"):
            assert row[name] == ref[name], (row["outer_fold"], row["organ_system"], name, row[name], ref[name])
        assert row["penalizer"] in PENALTIES
    expected_cv = 5 * 7 * 2 * 3 * len(PENALTIES)
    cv = pd.read_csv(output / "cv_results.csv")
    assert len(cv) == expected_cv and cv["solver_attempts"].between(1, 3).all()
    for name in ("harrell_c", "uno_c_60m", "auc_60m", "brier_60m", "cal_60m_ipcw_logistic_calibration_slope"):
        assert np.isfinite(model[name]).all(), name
    for fold in range(5):
        p = output / "folds" / f"outer_fold_{fold}"
        assert json.loads((p / "bootstrap_parity.json").read_text())["status"] == "PASS"
        assert json.loads((p / "bootstrap_complete.json").read_text())["replicates"] == 1000
    manifest_path = output / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_code_sha256"] == digest(__file__)
    manifest.update(status="COMPLETE", verification={"model_cells": 70, "incremental_cells": 35,
        "inner_cv_candidates": expected_cv, "inner_cv_solver_attempts": int(cv["solver_attempts"].sum()),
        "sample_and_event_counts_match_existing": True,
        "bootstrap_parity": "PASS all 35 cells", "historical_inputs_modified": False})
    files = list(output.glob("*.csv"))
    manifest["result_files"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]
    write_json(manifest_path, manifest)
    log("verification_passed", model_cells=70, cv_fits=expected_cv)


def launch_workers(run_root, phase, workers):
    """One guarded experiment owns all work; workers write disjoint fold directories."""
    import subprocess
    executable = (Path(r"C:\Users\Mickey\.conda\envs\age_prediction_py310\python.exe")
                  if phase == "fit" else Path(r"D:\Anaconda\python.exe"))
    pending = list(range(5))
    active = {}
    while pending or active:
        while pending and len(active) < workers:
            fold = pending.pop(0)
            folder = run_root / "outputs/folds" / f"outer_fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            temp = run_root / "tmp" / f"{phase}_fold_{fold}"
            temp.mkdir(parents=True, exist_ok=True)
            env.update(TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp), JOBLIB_TEMP_FOLDER=str(temp),
                       NUMBA_CACHE_DIR=str(run_root / "cache/numba" / f"fold_{fold}"))
            handle = (folder / f"{phase}.log").open("a", encoding="utf-8")
            command = [str(executable), "-B", "-u", str(Path(__file__).resolve()), phase,
                       "--run-root", str(run_root), "--fold", str(fold)]
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            active[fold] = (proc, handle)
            log("worker_started", phase=phase, outer_fold=fold, pid=proc.pid)
        for fold, (proc, handle) in list(active.items()):
            result = proc.poll()
            if result is None:
                continue
            handle.close()
            del active[fold]
            log("worker_exited", phase=phase, outer_fold=fold, exit_code=result)
            if result != 0:
                # Do not interrupt other owned folds; preserve their completed work.
                for remaining, (other, stream) in active.items():
                    other.wait(); stream.close()
                raise RuntimeError(f"{phase} fold {fold} failed; inspect its log before resuming")
        write_json(run_root / "outputs/launch_state.json", {"phase": phase,
            "active": {str(f): p.pid for f, (p, _) in active.items()}, "pending": pending})
        if active:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "fit", "bootstrap", "summarize", "verify", "run-all"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--organ", choices=ORGANS)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run_root = check_paths(args.run_root)
    if args.phase == "prepare": prepare(run_root)
    elif args.phase in ("pilot", "fit"):
        fit_fold(run_root, args.fold, pilot=args.phase == "pilot", only_organ=args.organ)
    elif args.phase == "bootstrap": bootstrap_fold(run_root, args.fold, args.replicates)
    elif args.phase == "summarize": summarize(run_root)
    elif args.phase == "verify": verify(run_root)
    elif args.phase == "run-all":
        manifest_path = run_root / "outputs/analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("preparation_source_code_sha256", manifest["source_code_sha256"])
        manifest["source_code_sha256"] = digest(__file__)
        manifest["workers"] = args.workers
        manifest["protocol"]["solver"] = {
            "objective_and_penalty_unchanged": True,
            "retry_only_on_explicit_numerical_or_convergence_failure": True,
            "steps": ["lifelines defaults", {"step_size": 0.25, "max_steps": 1000},
                      {"step_size": 0.05, "max_steps": 2000}],
            "converged_fit_required_for_every_candidate": True,
        }
        manifest["storage"] = {"expected_gib": 1.0, "pilot_one_cell_bytes": 2983146,
            "projected_35_cell_bytes": 104410110,
            "cache_rebuild": "prepare phase reads ten existing Med-BERT risk tables; no neural inference/training"}
        write_json(manifest_path, manifest)
        launch_workers(run_root, "fit", args.workers)
        launch_workers(run_root, "bootstrap", args.workers)
        summarize(run_root)
        verify(run_root)


if __name__ == "__main__":
    main()
