"""R1.4 fixed-reference clinical evaluation, age-scale checks and associations.

prepare/summary/association/bootstrap: existing Anaconda runtime.
fit: existing age_prediction_py310 survival runtime.
Unchanged R1.1 reference predictions, metric code and CV rule are reused.
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
from types import SimpleNamespace

sys.dont_write_bytecode=True
for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable,"2")
if "PROJECT_CACHE_DIR" in os.environ:
    os.environ["NUMBA_CACHE_DIR"]=os.environ["PROJECT_CACHE_DIR"]
    os.environ["MPLCONFIGDIR"]=str(Path(os.environ["PROJECT_CACHE_DIR"])/"matplotlib")
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REV=ROOT/"Revision-0710"
PRIMARY=REV/"03_ehr_age/06_codebook_v2"
R11=ROOT/"Revisions-1030/work/r1_1_information_matched_20261031"
R11_CODE=R11/"run_information_matched.py"
S19_ROOT=ROOT/"Revision-0730/work/p0_core_sensitivity_20260809"
S19_CODE=S19_ROOT/"run_baseline_only_reference_sensitivity.py"
S19_OUTPUT=S19_ROOT/"outputs"
CASES=("strict_train","baseline_train")
ORGANS=("cardiovascular","pulmonary","musculoskeletal","immune","renal","hepatic","metabolic")
AGE_ORGANS=("cardiovascular","hepatic","immune","metabolic","musculoskeletal","pulmonary","renal")

def module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024**2),b""):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)

def log(event,**values):
    print(json.dumps({"event":event,**values},allow_nan=False),flush=True)

def case_root(case):
    return HERE/"analysis"/case

def prepared_dir(case,fold):
    return case_root(case)/"outputs/prepared"/f"outer_fold_{fold}"

def fit_dir(case,fold):
    return case_root(case)/"outputs/folds"/f"outer_fold_{fold}"

def original_dir(fold):
    return (PRIMARY/"01_runs/shared_private_independent_view_huber1_fold0" if fold==0
            else PRIMARY/f"02_fivefold_frozen_huber1/01_outer_folds/outer_fold_{fold}")

def original_prediction(fold,role):
    name=("fold1_calibration_predictions.parquet" if role=="calibration"
          else "fold0_predictions.parquet") if fold==0 else (
          "calibration_predictions.parquet" if role=="calibration" else "outer_predictions.parquet")
    f=pd.read_parquet(original_dir(fold)/name)
    f["participant_id"]=f.participant_id.astype(str)
    f["outer_fold"]=fold
    f["split_role"]="calibration" if role=="calibration" else "outer_test"
    if "duration" in f:f=f.rename(columns={"duration":"duration_months"})
    return f

def reference_membership():
    context=pd.read_pickle(REV/"06_figures_and_statistics/03_reviewer_statistics/03_clinical_utility/01_formal_run/02_input_cache/participant_context_covariates.pkl")
    context["participant_id"]=context.participant_id.astype(str)
    reference=context.set_index("participant_id").has_preindex_icd.eq(0)
    labels=pd.read_parquet(REV/"01_data_rebuild/09_v2_labels/organ_outcomes_60m.parquet",
                           columns=["participant_id","has_prevalent_target"])
    labels["participant_id"]=labels.participant_id.astype(str)
    count=labels.groupby("participant_id").has_prevalent_target.sum()
    reference &= count.reindex(reference.index).eq(0)
    assert len(reference)==456773 and int(reference.sum())==175637
    return reference

def age_metrics(frame,case,fold,reference):
    from scipy.stats import pearsonr
    rows=[]
    baseline=frame.participant_id.map(reference).to_numpy(bool)
    strict=frame.strict_healthy_reference_60m.to_numpy(bool)
    for population,mask in [("baseline_only",baseline),("strict_health",strict)]:
        for organ in AGE_ORGANS:
            use=mask & frame.organ_system.eq(organ).to_numpy()
            a=frame.loc[use,"actual_age"].to_numpy(float)
            p=frame.loc[use,"predicted_age"].to_numpy(float)
            rows.append({"case":case,"outer_fold":fold,"population":population,
                "organ_system":organ,"n":len(a),"mae":float(np.mean(np.abs(p-a))),
                "pearson":float(pearsonr(a,p).statistic),
                "r2":float(1-np.sum((p-a)**2)/np.sum((a-a.mean())**2))})
    return pd.DataFrame(rows)

def prepare(args):
    folder=prepared_dir(args.case,args.fold)
    assert args.output==folder.resolve()
    marker=folder/"prepared.json"
    if marker.exists():
        assert json.loads(marker.read_text())["status"]=="COMPLETE"
        log("skip_prepared",case=args.case,fold=args.fold)
        return
    s19=module("r14_s19",S19_CODE)
    reference=reference_membership()
    old_params=pd.read_csv(S19_OUTPUT/"baseline_only_correction_parameters.csv")
    old_params=old_params.loc[old_params.outer_fold.eq(args.fold)].copy()
    assert len(old_params)==7
    old_test=original_prediction(args.fold,"outer_test")
    old_alt=pd.read_parquet(S19_OUTPUT/f"baseline_only_oof_by_fold/outer_fold_{args.fold}.parquet")
    old_alt["participant_id"]=old_alt.participant_id.astype(str)
    frames={}
    parameters=None
    for role in ("calibration","outer_test"):
        old=original_prediction(args.fold,role)
        if args.case=="strict_train":
            if role=="calibration":
                current=old.copy()
                current["baseline_only_reference"]=current.participant_id.map(reference).astype(np.int8)
                current["fold_id"]=(args.fold+1)%5
                check=pd.DataFrame(s19.fit_correction(current,args.fold))
                for col in ("intercept_years","age_slope_years_per_year"):
                    np.testing.assert_allclose(check.set_index("organ_system")[col].sort_index(),
                        old_params.set_index("organ_system")[col].sort_index(),rtol=0,atol=3e-12)
                corrected=s19.apply_correction(current,old_params,args.fold)
                current["corrected_age_gap"]=corrected.baseline_only_corrected_age_gap.to_numpy()
                parameters=old_params
            else:
                current=old.merge(old_alt[["participant_id","organ_system",
                    "baseline_only_corrected_age_gap"]],on=["participant_id","organ_system"],
                    validate="one_to_one")
                current["corrected_age_gap"]=current.baseline_only_corrected_age_gap
        else:
            train_folder=HERE/"outputs/folds"/f"outer_fold_{args.fold}"
            receipt=json.loads((train_folder/"training_state.json").read_text())
            assert receipt["status"]=="COMPLETE"
            path=train_folder/("calibration_predictions.npz" if role=="calibration" else "outer_predictions.npz")
            assert sha(path)==receipt["output_sha256"][path.name]
            with np.load(path) as z:
                organs=z["organ_systems"].astype(str).tolist()
                assert tuple(organs)==AGE_ORGANS
                ids=z["participant_id"].astype(str)
                assert set(ids)==set(old.participant_id)
                new=pd.DataFrame({"participant_id":np.repeat(ids,7),
                    "organ_system":np.tile(organs,len(ids)),
                    "new_age":z["predicted_age"].astype(float).ravel(),
                    "new_actual_age":np.repeat(z["chronological_age"].astype(float),7),
                    "new_event":z["event"].ravel().astype(np.int8),
                    "new_at_risk":z["at_risk"].ravel().astype(np.int8),
                    "new_duration":z["duration_months"].astype(float).ravel()})
            current=old.merge(new,on=["participant_id","organ_system"],validate="one_to_one")
            assert len(current)==len(old)
            np.testing.assert_array_equal(current.event,current.new_event)
            np.testing.assert_array_equal(current.at_risk,current.new_at_risk)
            np.testing.assert_allclose(current.actual_age,current.new_actual_age,atol=0,rtol=0)
            use=current.at_risk.eq(1)
            np.testing.assert_allclose(current.loc[use,"duration_months"],current.loc[use,"new_duration"],
                                       atol=0,rtol=0)
            current["predicted_age"]=current.new_age
            current["baseline_only_reference"]=current.participant_id.map(reference).astype(np.int8)
            current["fold_id"]=(args.fold+1)%5 if role=="calibration" else args.fold
            if role=="calibration":
                parameters=pd.DataFrame(s19.fit_correction(current,args.fold))
            corrected=s19.apply_correction(current,parameters,args.fold)
            current["corrected_age_gap"]=corrected.baseline_only_corrected_age_gap.to_numpy()
        frames[role]=current
        columns=["participant_id","organ_system","duration_months","event","at_risk",
                 "corrected_age_gap","outer_fold","split_role"]
        current[columns].to_pickle(folder/f"{role}.pkl")
    parameters.to_csv(folder/"baseline_correction_parameters.csv",index=False)
    age_metrics(frames["outer_test"],args.case,args.fold,reference).to_csv(folder/"age_metrics.csv",index=False)
    if args.case=="baseline_train":
        from scipy.stats import spearmanr
        compare=frames["outer_test"].merge(old_alt[["participant_id","organ_system",
            "baseline_only_corrected_age_gap"]],on=["participant_id","organ_system"],validate="one_to_one")
        rows=[]
        for organ,g in compare.groupby("organ_system",sort=False):
            x=g.baseline_only_corrected_age_gap.to_numpy(float)
            y=g.corrected_age_gap.to_numpy(float)
            rows.append({"outer_fold":args.fold,"organ_system":organ,"n":len(g),
                "pearson":float(np.corrcoef(x,y)[0,1]),"spearman":float(spearmanr(x,y).statistic),
                "old_mean":float(x.mean()),"new_mean":float(y.mean()),
                "old_sd":float(x.std(ddof=1)),"new_sd":float(y.std(ddof=1)),
                "mean_difference":float((y-x).mean()),
                "mean_absolute_difference":float(np.abs(y-x).mean()),
                "rmse_difference":float(np.sqrt(np.mean((y-x)**2)))})
        pd.DataFrame(rows).to_csv(folder/"osag_agreement.csv",index=False)
    # Minimal long-format OSAG cache is also the association input.
    frames["outer_test"][["participant_id","organ_system","outer_fold","actual_age",
        "predicted_age","corrected_age_gap"]].to_pickle(folder/"age_osag.pkl")
    write_json(marker,{"status":"COMPLETE","case":args.case,"outer_fold":args.fold,
        "calibration_n":frames["calibration"].participant_id.nunique(),
        "prediction_n":frames["outer_test"].participant_id.nunique(),
        "old_correction_reused":args.case=="strict_train",
        "source_sha256":sha(__file__),"source_s19_sha256":sha(S19_CODE)})
    log("prepared",case=args.case,fold=args.fold,calibration_n=frames["calibration"].participant_id.nunique())

def load_fit_data(case,fold):
    r11=module("r14_r11",R11_CODE)
    clinical,final=r11.modules()
    def load_prediction(path,outer_fold,role):
        key="calibration" if role=="calibration" else "outer_test"
        return pd.read_pickle(prepared_dir(case,outer_fold)/f"{key}.pkl")
    final.load_prediction_frame=load_prediction
    return r11,r11.load_fold(R11,fold)

def fit(args):
    folder=args.output
    r11,data=load_fit_data(args.case,args.fold)
    clinical=data[0]
    covariates=tuple(clinical.C_CONTEXT_DENSITY_PRIMARY)+r11.PHENO+r11.MISSING+r11.MED+r11.GAPS
    assert len(covariates)==191
    for oi,organ in enumerate(r11.ORGANS):
        if args.organ and organ!=args.organ:continue
        marker=folder/f"{organ}__complete.json"
        if marker.exists():
            log("skip_fit_complete",case=args.case,fold=args.fold,organ=organ)
            continue
        begin=time.monotonic()
        raw_fit,raw_score,universe=r11.cell_frames(data,args.fold,organ)
        train,score,medians=r11.impute_clinical(raw_fit,raw_score,clinical.C_CONTEXT_DENSITY_PRIMARY)
        grid=clinical.build_metric_time_grid(train,score,report_times=clinical.REPORT_TIMES)
        original=R11/"outputs/folds"/f"outer_fold_{args.fold}"
        old_metrics=pd.read_csv(original/f"{organ}__metrics.csv")
        base_values=old_metrics.loc[old_metrics.augmentation.eq("base")].iloc[0].to_dict()
        assert len(train)==int(base_values["fit_n"]) and len(score)==int(base_values["test_n"])
        assert int(train.event.sum())==int(base_values["fit_events"])
        assert int(score.event.sum())==int(base_values["test_events"])
        with np.load(original/f"{organ}__bootstrap.npz") as z:
            arrays={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(arrays["participant_id"],score.participant_id.astype(np.int64))
        np.testing.assert_array_equal(arrays["event"],score.event.to_numpy(np.int8))
        np.testing.assert_allclose(arrays["duration"],score.duration.to_numpy(float),rtol=0,atol=0)
        penalty,cv=r11.choose_penalty(raw_fit,covariates,clinical,
            r11.SEED+100*args.fold+oi,pilot=args.pilot)
        bundle=clinical.fit_lifelines_cox_scores(train,score,covariates=covariates,
            time_grid=grid,phase="formal",penalizer=penalty)
        values,risks,bins,dca,state=r11.evaluate_bundle(clinical,train,score,bundle,grid)
        row={"outer_fold":args.fold,"organ_system":organ,"case":args.case,
            "tier":"information_matched","augmentation":"all_gaps","model":args.case+"__all_gaps",
            "penalizer":penalty,"fit_n":len(train),"test_n":len(score),
            "fit_events":int(train.event.sum()),"test_events":int(score.event.sum()),
            "n_covariates":191,"n_retained":len(bundle.coefficients),**values}
        base_values.update(case=args.case)
        base_dca=pd.read_csv(original/f"{organ}__dca.csv")
        base_dca=base_dca.loc[base_dca.model.eq("information_matched__base")]
        d=base_dca[["threshold","net_benefit"]].merge(dca[["threshold","net_benefit"]],
                    on="threshold",suffixes=("_base","_aug"))
        reclass=clinical.ipcw_reclassification(arrays["base_risk"],risks[60.],state)
        increment={"outer_fold":args.fold,"organ_system":organ,"case":args.case,
            "tier":"information_matched","comparison":args.case+"__all_gaps_vs_M",
            "baseline_model":"information_matched__base","augmented_model":args.case+"__all_gaps",
            "test_n":len(score),"test_events":int(score.event.sum()),
            "delta_harrell_c":values["harrell_c"]-base_values["harrell_c"],
            "delta_uno_c_60m":values["uno_c_60m"]-base_values["uno_c_60m"],
            **{f"delta_auc_{int(t)}m":values[f"auc_{int(t)}m"]-base_values[f"auc_{int(t)}m"]
               for t in clinical.REPORT_TIMES},
            **{f"brier_improvement_{int(t)}m":base_values[f"brier_{int(t)}m"]-values[f"brier_{int(t)}m"]
               for t in clinical.REPORT_TIMES},
            "ibs_improvement":base_values["ibs"]-values["ibs"],**reclass,
            "mean_delta_net_benefit":float((d.net_benefit_aug-d.net_benefit_base).mean())}
        arrays["all_gaps_score"]=bundle.risk_score
        arrays["all_gaps_risk"]=risks[60.]
        np.savez_compressed(folder/f"{organ}__bootstrap.npz",**arrays)
        pd.DataFrame([base_values,row]).to_csv(folder/f"{organ}__metrics.csv",index=False)
        pd.DataFrame([increment]).to_csv(folder/f"{organ}__increment.csv",index=False)
        cv.to_csv(folder/f"{organ}__cv.csv",index=False)
        pd.Series(bundle.coefficients).rename_axis("covariate").reset_index(name="coefficient_per_fit_sd").to_csv(
            folder/f"{organ}__coefficients.csv",index=False)
        bins.to_csv(folder/f"{organ}__calibration.csv",index=False)
        dca.to_csv(folder/f"{organ}__dca.csv",index=False)
        write_json(marker,{"status":"PILOT_ONLY" if args.pilot else "COMPLETE",
            "case":args.case,"fold":args.fold,"organ":organ,"n":len(score),
            "events":int(score.event.sum()),"penalizer":penalty,"seconds":time.monotonic()-begin,
            "reference_reused":True,"reference_bootstrap_sha256":sha(original/f"{organ}__bootstrap.npz"),
            "source_sha256":sha(__file__),"r11_source_sha256":sha(R11_CODE)})
        log("clinical_cell_complete",case=args.case,fold=args.fold,organ=organ,
            delta_c=increment["delta_harrell_c"],seconds=time.monotonic()-begin)
        del train,score,raw_fit,raw_score,bundle,arrays
        gc.collect()

def bootstrap(args):
    r11=module("r14_r11",R11_CODE)
    assert args.output==fit_dir(args.case,args.fold).resolve()
    complete=args.output/"bootstrap_complete.json"
    if complete.exists() and json.loads(complete.read_text())["replicates"]==args.replicates:
        log("skip_bootstrap_complete",case=args.case,fold=args.fold)
        return
    for organ in ORGANS:
        assert json.loads((args.output/f"{organ}__complete.json").read_text())["status"]=="COMPLETE"
    r11.bootstrap_fold(case_root(args.case),args.fold,args.replicates)

def summary(args):
    assert args.output==(HERE/"results").resolve()
    r11=module("r14_r11",R11_CODE)
    all_increments=[]
    all_age=[]
    all_boot=[]
    all_models=[]
    for case in CASES:
        for fold in range(5):
            folder=fit_dir(case,fold)
            assert json.loads((folder/"bootstrap_complete.json").read_text())["replicates"]==1000
            boot=pd.read_csv(folder/"bootstrap_replicates.csv.gz")
            boot.insert(0,"case",case)
            all_boot.append(boot)
            for organ in ORGANS:
                assert json.loads((folder/f"{organ}__complete.json").read_text())["status"]=="COMPLETE"
                all_increments.append(pd.read_csv(folder/f"{organ}__increment.csv"))
                all_models.append(pd.read_csv(folder/f"{organ}__metrics.csv"))
            all_age.append(pd.read_csv(prepared_dir(case,fold)/"age_metrics.csv"))
    increments=pd.concat(all_increments,ignore_index=True)
    models=pd.concat(all_models,ignore_index=True)
    ages=pd.concat(all_age,ignore_index=True)
    boot=pd.concat(all_boot,ignore_index=True)
    increments.to_csv(args.output/"cell_incremental_metrics.csv",index=False)
    models.to_csv(args.output/"cell_model_metrics.csv",index=False)
    ages.to_csv(args.output/"fold_organ_age_metrics.csv",index=False)
    metric_cols=[c for c in increments if c.startswith(("delta_","brier_improvement")) or
                 c in ("ibs_improvement","mean_delta_net_benefit","ipcw_idi_60m","ipcw_continuous_nri_60m")]
    macro=increments.groupby("case")[metric_cols].mean()
    macro.to_csv(args.output/"macro_incremental_metrics.csv")
    increments.groupby(["case","organ_system"])[metric_cols].mean().to_csv(
        args.output/"organ_incremental_metrics.csv")
    model_cols=["harrell_c","uno_c_60m","auc_12m","auc_24m","auc_60m","brier_12m","brier_24m",
        "brier_60m","ibs","cal_60m_observed_expected_ratio","cal_60m_ipcw_logistic_calibration_slope"]
    models.groupby(["case","augmentation"])[model_cols].mean().to_csv(args.output/"macro_model_metrics.csv")
    ages.groupby(["case","population"])[["mae","pearson","r2"]].mean().to_csv(args.output/"macro_age_metrics.csv")
    ages.groupby(["case","population","organ_system"])[["mae","pearson","r2"]].mean().to_csv(
        args.output/"organ_age_metrics.csv")
    agreement=pd.concat([pd.read_csv(prepared_dir("baseline_train",f)/"osag_agreement.csv")
                         for f in range(5)],ignore_index=True)
    agreement.to_csv(args.output/"fold_osag_agreement.csv",index=False)
    agreement.groupby("organ_system").mean(numeric_only=True).to_csv(args.output/"organ_osag_agreement.csv")
    metrics=list(r11.load_module("r14_fast",r11.FAST_SCRIPT).METRICS)
    macro_boot=boot.groupby(["case","bootstrap"])[metrics].mean()
    macro_boot.to_csv(args.output/"macro_bootstrap_replicates.csv")
    ci=[]
    for case in CASES:
        values=macro_boot.loc[case]
        for metric in metrics:
            lo,hi=np.quantile(values[metric],[.025,.975])
            ci.append({"case":case,"organ_system":"macro","metric":metric,"point_estimate":macro.loc[case,metric],
                       "ci95_low":lo,"ci95_high":hi,"replicates":1000})
        for organ,part in boot.loc[boot.case.eq(case)].groupby("organ_system"):
            vals=part.groupby("bootstrap")[metrics].mean()
            point=increments.loc[increments.case.eq(case)&increments.organ_system.eq(organ)][metrics].mean()
            for metric in metrics:
                lo,hi=np.quantile(vals[metric],[.025,.975])
                ci.append({"case":case,"organ_system":organ,"metric":metric,"point_estimate":point[metric],
                           "ci95_low":lo,"ci95_high":hi,"replicates":1000})
    pd.DataFrame(ci).to_csv(args.output/"paired_bootstrap_ci.csv",index=False)
    # Same participant universes and RNG schedule must underlie paired contrasts.
    for fold in range(5):
        for organ in ORGANS:
            with np.load(fit_dir("strict_train",fold)/f"{organ}__bootstrap.npz") as a,\
                 np.load(fit_dir("baseline_train",fold)/f"{organ}__bootstrap.npz") as b,\
                 np.load(R11/"outputs/folds"/f"outer_fold_{fold}"/f"{organ}__bootstrap.npz") as c:
                for key in ["participant_id","universe_positions","event","duration","base_score","base_risk"]:
                    np.testing.assert_array_equal(a[key],b[key])
                    np.testing.assert_array_equal(a[key],c[key])
    original=pd.concat([pd.read_csv(R11/f"outputs/folds/outer_fold_{f}/bootstrap_replicates.csv.gz")
                         for f in range(5)],ignore_index=True)
    original_macro=original.groupby("bootstrap")[metrics].mean()
    original_point=pd.read_csv(R11/"outputs/macro_incremental_metrics.csv").iloc[0]
    contrasts=[]
    for label,values,point in [
        ("training_reference_effect_at_baseline_calibration",
         macro_boot.loc["baseline_train"]-macro_boot.loc["strict_train"],
         macro.loc["baseline_train"]-macro.loc["strict_train"]),
        ("both_reference_roles_replaced_vs_original",
         macro_boot.loc["baseline_train"]-original_macro,
         macro.loc["baseline_train"]-original_point[metric_cols])]:
        for metric in metrics:
            lo,hi=np.quantile(values[metric],[.025,.975])
            contrasts.append({"contrast":label,"metric":metric,"point_estimate":point[metric],
                              "ci95_low":lo,"ci95_high":hi,"replicates":1000})
    pd.DataFrame(contrasts).to_csv(args.output/"reference_effect_paired_ci.csv",index=False)
    write_json(args.output/"analysis_manifest.json",{"status":"CLINICAL_AND_AGE_COMPLETE",
        "cases":{"strict_train":"strict training, baseline-only age correction",
                 "baseline_train":"baseline-only training and age correction"},
        "new_neural_models":5,"fixed_epoch":47,"new_augmented_cox_models":70,
        "reference_M_refitted":False,"cv_candidates":840,"paired_bootstrap_replicates":1000,
        "ci_scope":"conditional on fixed neural/Cox/censor fits; organ intervals unadjusted",
        "source_sha256":sha(__file__),"r11_source_sha256":sha(R11_CODE)})
    log("clinical_summary_complete",models=70,cv_candidates=840)

def association(args):
    assert args.output==(HERE/"results/associations").resolve()
    s19=module("r14_s19",S19_CODE)
    e47=s19.load_e47_module()
    context,audit=e47.load_context(s19.DEFAULT_CONTEXT,s19.DEFAULT_LABELS)
    gaps=pd.concat([pd.read_pickle(prepared_dir("baseline_train",f)/"age_osag.pkl")
                    for f in range(5)],ignore_index=True)
    assert len(gaps)==456773*7
    all_results=[]
    for organ in AGE_ORGANS:
        marker=args.output/f"{organ}__complete.json"
        if marker.exists():
            all_results.append(pd.read_csv(args.output/f"{organ}__hr.csv").iloc[0].to_dict())
            continue
        old=pd.read_parquet(s19.DEFAULT_CANONICAL,columns=["participant_id","organ_system",
            "outer_fold","actual_age","sex","at_risk","duration","event"],
            filters=[("organ_system","==",organ)])
        old["participant_id"]=old.participant_id.astype(str)
        old=old.rename(columns={"duration":"duration_months"})
        new=gaps.loc[gaps.organ_system.eq(organ),["participant_id","outer_fold","corrected_age_gap"]]
        frame=old.merge(new,on=["participant_id","outer_fold"],validate="one_to_one")
        frame=frame.merge(context,on="participant_id",validate="one_to_one")
        assert len(frame)==456773
        result,gap_audit,cov_audit,cframe=s19.fit_detailed_cox(e47,frame,organ,
            "corrected_age_gap","baseline_only_training_and_calibration")
        pd.DataFrame([result]).to_csv(args.output/f"{organ}__hr.csv",index=False)
        gap_audit.to_csv(args.output/f"{organ}__exposure_scale.csv",index=False)
        cov_audit.to_csv(args.output/f"{organ}__covariates.csv",index=False)
        cframe.to_csv(args.output/f"{organ}__association_c.csv",index=False)
        write_json(marker,{"status":"COMPLETE","organ":organ,"n":result["n"],"events":result["events"],
                          "source_sha256":sha(__file__)})
        all_results.append(result)
        log("association_complete",organ=organ,hr=result["gap_hr_per_crossfit_sd"])
    pd.DataFrame(all_results).to_csv(args.output/"new_reference_detailed_burden_hr.csv",index=False)
    old=pd.read_csv(S19_OUTPUT/"baseline_only_reference_detailed_burden_cox.csv")
    pd.concat([old,pd.DataFrame(all_results)],ignore_index=True).to_csv(
        args.output/"reference_comparison_detailed_burden_hr.csv",index=False)
    write_json(args.output/"association_manifest.json",{"status":"COMPLETE","models":7,
        "existing_controls_reused":14,"context_audit":audit,"source_sha256":sha(__file__),
        "s19_source_sha256":sha(S19_CODE)})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["prepare","fit","bootstrap","summary","association"],required=True)
    parser.add_argument("--case",choices=CASES,default="strict_train")
    parser.add_argument("--fold",type=int,choices=range(5),default=0)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--organ",choices=ORGANS)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--replicates",type=int,default=1000)
    args=parser.parse_args()
    args.output=args.output.resolve()
    assert args.output.is_relative_to((ROOT/"Revisions-1030").resolve())
    assert os.environ.get("PROJECT_RUN_DIR")==str(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    globals()[args.phase](args)

if __name__=="__main__":
    main()
