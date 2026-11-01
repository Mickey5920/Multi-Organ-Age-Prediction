"""R1.4: replace only the training-reference membership in frozen EHR-AGE.

Uses the unchanged primary architecture/training/data helpers. Original labels,
inputs, models and results are read-only. Formal runs stop at fixed epoch 47.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("NUMEXPR_NUM_THREADS","4")
for variable, subdir in [
    ("CUDA_CACHE_PATH","cuda"),("TORCH_EXTENSIONS_DIR","torch_extensions"),
    ("TRITON_CACHE_DIR","triton")]:
    if "PROJECT_CACHE_DIR" in os.environ:
        os.environ[variable]=str(Path(os.environ["PROJECT_CACHE_DIR"])/subdir)

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
REV=ROOT/"Revision-0710"
PRIMARY=REV/"03_ehr_age/06_codebook_v2"
sys.path.insert(0,str(PRIMARY))
import run_frozen_outer_fold as primary
import run_fold0_current as factory
import ehr_age_training as training
from ehr_age_data import JointVectorizedBatcher

CONTEXT=REV/"06_figures_and_statistics/03_reviewer_statistics/03_clinical_utility/01_formal_run/02_input_cache/participant_context_covariates.pkl"
EXPECTED_OUTCOME="3a9021f00080c25fd523debcdf075a5be8281c9658db261af99a8bdc11f4e735"
EXPECTED_INITIAL="d14ab2dc3a080aa5803064d05fea7f6257d004cebff54796c3843e07945f8632"
ORIGINAL_FILES=[
    PRIMARY/"01_runs/shared_private_independent_view_huber1_fold0/checkpoint.pt",
    *[PRIMARY/f"02_fivefold_frozen_huber1/01_outer_folds/outer_fold_{k}/checkpoint.pt" for k in range(1,5)],
]
LOG=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    assert path.resolve().is_relative_to((ROOT/"Revisions-1030").resolve())
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(path)

def log(event,**values):
    value=json.dumps({"event":event,**values},allow_nan=False)
    print(value,flush=True)
    if LOG:
        with LOG.open("a",encoding="utf-8") as f:f.write(value+"\n")

def dependency_hashes():
    files=[Path(primary.__file__),Path(factory.__file__),Path(training.__file__),
           REV/"03_ehr_age/02_runner/ehr_age_data.py",
           REV/"03_ehr_age/01_model/ehr_age_model.py",
           REV/"03_ehr_age/01_model/ehr_age_losses.py",
           REV/"03_ehr_age/01_model/private_retain_age.py",
           REV/"02_baselines_5fold/common/compact_cache_dataset.py",
           primary.SOURCE_CONFIG,
           primary.DEFAULT_CACHE/"cache_run_manifest.json"]
    return {str(p):digest(p) for p in files}

def original_directory(fold):
    return (PRIMARY/"01_runs/shared_private_independent_view_huber1_fold0" if fold==0
            else PRIMARY/f"02_fivefold_frozen_huber1/01_outer_folds/outer_fold_{fold}")

def original_calibration_ids(fold):
    path=original_directory(fold)/(
        "fold1_calibration_predictions.parquet" if fold==0 else "calibration_predictions.parquet")
    frame=pd.read_parquet(path,columns=["participant_id"])
    return frame.participant_id.astype(str).drop_duplicates().tolist()

def evaluation_data(labels,fold,role):
    data=primary.load_role_data(labels,outer_fold=fold,role=role)
    if role=="calibration":
        ids=original_calibration_ids(fold)
        lookup={str(pid):i for i,pid in enumerate(data.cache.dataset.participant_ids)}
        selected=torch.tensor([lookup[pid] for pid in ids],dtype=torch.long)
        data=replace(data,calibration_indices=selected)
    return data

def load_references():
    labels=pd.read_parquet(primary.OUTCOMES_PATH,
        columns=list(primary.OUTCOME_COLUMNS)+["has_prevalent_target"])
    labels["participant_id"]=labels.participant_id.astype(str)
    assert not labels.duplicated(["participant_id","organ_system"]).any()
    context=pd.read_pickle(CONTEXT)[["participant_id","has_preindex_icd"]].copy()
    context["participant_id"]=context.participant_id.astype(str)
    assert not context.participant_id.duplicated().any()
    burden=labels.groupby("participant_id").has_prevalent_target.sum()
    context["baseline_organ_count"]=context.participant_id.map(burden)
    assert context.baseline_organ_count.notna().all()
    reference=pd.Series((context.has_preindex_icd.eq(0)&context.baseline_organ_count.eq(0)).to_numpy(),
                        index=context.participant_id)
    strict=labels.groupby("participant_id").strict_healthy_reference_60m.first().astype(bool)
    assert len(reference)==456773 and int(reference.sum())==175637
    assert int(strict.sum())==102129
    assert reference.reindex(strict.index)[strict].all()
    assert digest(primary.OUTCOMES_PATH)==EXPECTED_OUTCOME
    return labels[list(primary.OUTCOME_COLUMNS)],reference,strict

def role_membership(fold):
    ids={}
    for role,directory in primary.ROLE_DIRECTORIES.items():
        p=primary.ROLE_AGE_ROOT/f"outer_fold_{fold}"/directory/"age_targets.parquet"
        frame=pd.read_parquet(p,columns=["participant_id","fold_id"])
        assert set(frame.fold_id.astype(int))==set(primary.role_fold_ids(fold)[role])
        ids[role]=set(frame.participant_id.astype(str))
    assert not ids["train"]&ids["calibration"]
    assert not ids["train"]&ids["outer_test"]
    assert not ids["calibration"]&ids["outer_test"]
    original_cal=set(original_calibration_ids(fold))
    assert original_cal.issubset(ids["calibration"])
    reserved=ids["calibration"]-original_cal
    ids["calibration"]=original_cal
    ids["reserved_historical_screen"]=reserved
    assert (fold==0 and len(original_cal)==45599) or not reserved
    assert len(set.union(*ids.values()))==456773
    return ids

def train_data(labels,reference,fold):
    data=primary.load_role_data(labels,outer_fold=fold,role="train")
    ids=[str(v) for v in data.cache.dataset.participant_ids]
    baseline=reference.reindex(ids)
    assert baseline.notna().all()
    old=data.labels.strict_healthy.clone()
    new=torch.from_numpy(baseline.to_numpy(dtype=bool))
    modified=replace(data,labels=replace(data.labels,strict_healthy=new))
    for name in ["event","duration_months","at_risk"]:
        assert getattr(modified.labels,name) is getattr(data.labels,name)
    assert modified.cache is data.cache
    return modified,old

def make_model(data):
    raw=json.loads(primary.SOURCE_CONFIG.read_text(encoding="utf-8"))
    primary._apply_variant(raw)
    cfg=training._loss_config(raw["loss"])
    assert cfg.strict_offset_weight==.75 and cfg.strict_final_huber_weight==0
    assert cfg.strict_center_weight==cfg.strict_slope_weight==0
    assert cfg.base_view_huber_weight==2 and cfg.huber_beta_years==1
    model,device,optimizer,scheduler,scaler,edges,anchor=factory.make_model_and_optimizer(raw,data)
    assert training.trainable_parameter_hash(model)==EXPECTED_INITIAL
    assert torch.__version__.split("+")[0]=="2.5.1"
    return raw,model,device,optimizer,scheduler,scaler,edges,anchor

def batcher(data,role,shuffle=False):
    indices={"train":data.train_indices,"calibration":data.calibration_indices,
             "outer_test":data.screen_indices}[role]
    return JointVectorizedBatcher(data,indices,batch_size=2048,shuffle=shuffle,seed=7210)

def save_prediction(path,data,prediction,reference,fold):
    ids=np.array(factory.prediction_ids(data,prediction))
    arrays={name:getattr(prediction,name) for name in
            ["chronological_age","strict_healthy","duration_months","event","at_risk",
             "sex","predicted_age","risk_score"]}
    assert arrays["predicted_age"].shape==(len(ids),7)
    assert np.isfinite(arrays["predicted_age"]).all()
    assert np.isfinite(arrays["risk_score"]).all()
    np.savez_compressed(path,participant_id=ids,baseline_reference=reference.reindex(ids).to_numpy(bool),
                        outer_fold=fold,organ_systems=np.array(primary.ORGAN_SYSTEMS),**arrays)

def pilot(args,labels,reference,data,old_strict):
    raw,model,device,optimizer,scheduler,scaler,edges,anchor=make_model(data)
    # Fixed-output loss algebra isolates reference membership from other objectives.
    first=next(iter(batcher(data,"train",False)))
    inp=training._move_inputs(first.inputs,device)
    with torch.no_grad(),training._autocast(device,True):
        out=model(**inp)
        kwargs=training._move_labels(first,device)
        loss_new=training.ehr_age_joint_loss(out,**kwargs,interval_edges_months=edges,
            config=training._loss_config(raw["loss"]),survival_multiplier=.1)
        old=old_strict.index_select(0,first.participant_indices).to(device)
        kwargs["strict_healthy_reference"]=old
        loss_old=training.ehr_age_joint_loss(out,**kwargs,interval_edges_months=edges,
            config=training._loss_config(raw["loss"]),survival_multiplier=.1)
        for key in loss_new:
            if key!="total" and not key.startswith("strict"):
                torch.testing.assert_close(loss_new[key],loss_old[key],rtol=0,atol=0)
        difference=(loss_new["total"]-loss_old["total"]).float()
        expected=.75*(loss_new["strict_offset"]-loss_old["strict_offset"]).float()
        torch.testing.assert_close(difference,expected,rtol=1e-4,atol=2e-6)
    del out,inp,loss_new,loss_old,kwargs,model,optimizer,scheduler,scaler
    gc.collect();torch.cuda.empty_cache()
    # Frozen source-checkpoint inference reproduces saved calibration predictions.
    calibration=evaluation_data(labels,args.fold,"calibration")
    raw,model,device,optimizer,scheduler,scaler,edges,anchor=make_model(data)
    factory.restore_checkpoint(original_directory(args.fold)/"checkpoint.pt",model)
    prediction=training._predict(model=model,batcher=batcher(calibration,"calibration"),
        device=device,amp=True,max_batches=1)
    ids=factory.prediction_ids(calibration,prediction)
    old_path=original_directory(args.fold)/(
        "fold1_calibration_predictions.parquet" if args.fold==0 else "calibration_predictions.parquet")
    old=pd.read_parquet(old_path)
    old["participant_id"]=old.participant_id.astype(str)
    idx=pd.MultiIndex.from_product([ids,primary.ORGAN_SYSTEMS],
                                  names=["participant_id","organ_system"])
    aligned=old.set_index(["participant_id","organ_system"]).reindex(idx)
    assert aligned.predicted_age.notna().all()
    expected=aligned.predicted_age.to_numpy(float).reshape(-1,7)
    age_error=float(np.max(np.abs(prediction.predicted_age-expected)))
    np.testing.assert_allclose(prediction.predicted_age,expected,rtol=1e-6,atol=1e-4)
    risk_error=None
    if "model_risk_score" in aligned:
        expected_risk=aligned.model_risk_score.to_numpy(float).reshape(-1,7)
        risk_error=float(np.max(np.abs(prediction.risk_score-expected_risk)))
        np.testing.assert_allclose(prediction.risk_score,expected_risk,rtol=1e-6,atol=1e-4)
    del calibration,prediction,model,optimizer,scheduler,scaler
    gc.collect();torch.cuda.empty_cache()
    # Pilot weights are discarded; every formal fold starts again from scratch.
    raw,model,device,optimizer,scheduler,scaler,edges,anchor=make_model(data)
    torch.cuda.reset_peak_memory_stats(device)
    begin=time.monotonic()
    losses=training._train_epoch(model=model,batcher=batcher(data,"train",True),
        optimizer=optimizer,scaler=scaler,device=device,interval_edges=edges,
        loss_config=training._loss_config(raw["loss"]),
        survival_multiplier=training._survival_multiplier(1,raw["loss"]),
        event_dropout_probability=float(raw["training"]["event_dropout_probability"]),
        gradient_clip_norm=float(raw["training"]["gradient_clip_norm"]),amp=True,max_batches=12)
    assert all(np.isfinite(v) for v in losses.values())
    assert losses["optimizer_steps"]>0
    result={"status":"PILOT_COMPLETE","fold":args.fold,"training_reference_n":int(data.labels.strict_healthy.sum()),
            "original_strict_training_n":int(old_strict.sum()),"source_checkpoint_age_max_abs_error":age_error,
            "source_checkpoint_risk_max_abs_error":risk_error,"reference_loss_algebra":"PASS",
            "full_batches":12,"batch_size":2048,"seconds":time.monotonic()-begin,
            "peak_cuda_allocated_mb":torch.cuda.max_memory_allocated(device)/1024**2,
            "event_input_width":data.cache.dataset.inputs["event_valid_mask"].shape[1],
            "losses":losses,"source_sha256":digest(__file__)}
    write_json(args.output/"pilot_result.json",result)
    log("pilot_complete",**result)

def compact_model_state(model):
    return {"trainable":{k:v.detach().cpu().clone() for k,v in model.named_parameters()
                         if v.requires_grad},
            "buffers":{k:v.detach().cpu().clone() for k,v in model.named_buffers()}}

def restore_compact_state(model,state):
    parameters={k:v for k,v in model.named_parameters() if v.requires_grad}
    buffers=dict(model.named_buffers())
    assert set(parameters)==set(state["trainable"])
    assert set(buffers)==set(state["buffers"])
    with torch.no_grad():
        for k,v in parameters.items():v.copy_(state["trainable"][k].to(v))
        for k,v in buffers.items():v.copy_(state["buffers"][k].to(v))

def check_checkpoint_io(output):
    model=torch.nn.Sequential(torch.nn.Embedding(10,3),torch.nn.Linear(3,1))
    model[0].weight.requires_grad_(False)
    model.register_buffer("constant",torch.tensor([3.]))
    old_frozen=model[0].weight.detach().clone()
    state=compact_model_state(model)
    path=output/"checkpoint_io_test.pt"
    torch.save(state,path)
    with torch.no_grad():
        model[1].weight.add_(1)
        model.constant.add_(2)
    restore_compact_state(model,torch.load(path,weights_only=True))
    assert torch.equal(model[1].weight,state["trainable"]["1.weight"])
    assert torch.equal(model.constant,state["buffers"]["constant"])
    assert torch.equal(model[0].weight,old_frozen)
    log("checkpoint_io_passed",frozen_embeddings_not_duplicated=True)

def save_resume(path,model,optimizer,scheduler,scaler,epoch,train_batcher,source_hash):
    temp=path.with_suffix(".tmp")
    torch.save({"epoch":epoch,"model":compact_model_state(model),"optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
        "random_state":random.getstate(),"numpy_state":np.random.get_state(),
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
        "batcher_iteration":train_batcher.iteration,"source_sha256":source_hash},temp)
    temp.replace(path)

def run_training(args,labels,reference,data,old_strict,membership):
    started=time.monotonic()
    source_hash=digest(__file__)
    raw,model,device,optimizer,scheduler,scaler,edges,anchor=make_model(data)
    train_batcher=batcher(data,"train",True)
    state_path=args.output/"resume_state.pt"
    start_epoch=1
    if state_path.exists():
        state=torch.load(state_path,map_location="cpu",weights_only=False)
        assert state["source_sha256"]==source_hash
        restore_compact_state(model,state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["random_state"]);np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state_all(state["cuda_rng"])
        train_batcher.iteration=state["batcher_iteration"]
        start_epoch=int(state["epoch"])+1
        log("resumed",fold=args.fold,next_epoch=start_epoch)
    config={"status":"TRAINING","fold":args.fold,"fixed_epoch":47,"seed":7210,
            "scheduler_horizon_epochs":60,"from_scratch":True,"checkpoint_input":None,
            "training_reference_definition":"has_preindex_icd == 0 and v2 prevalent-organ count == 0; no future condition",
            "training_reference_n":int(data.labels.strict_healthy.sum()),
            "original_strict_training_n":int(old_strict.sum()),
            "role_counts":{k:len(v) for k,v in membership.items()},
            "train_folds":list(primary.role_fold_ids(args.fold)["train"]),
            "calibration_fold":(args.fold+1)%5,"prediction_fold":args.fold,
            "model":raw["model"],"training":raw["training"],"loss":raw["loss"],
            "initial_trainable_parameter_sha256":EXPECTED_INITIAL,"source_sha256":source_hash,
            "dependencies":dependency_hashes(),"outcomes_sha256":EXPECTED_OUTCOME,
            "context_sha256":digest(CONTEXT),"gpu":torch.cuda.get_device_name(0),
            "python":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,
            "numpy":np.__version__,"pandas":pd.__version__}
    write_json(args.output/"run_config.json",config)
    originals={str(p):digest(p) for p in ORIGINAL_FILES}
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch,48):
        begin=time.monotonic()
        losses=training._train_epoch(model=model,batcher=train_batcher,optimizer=optimizer,
            scaler=scaler,device=device,interval_edges=edges,
            loss_config=training._loss_config(raw["loss"]),
            survival_multiplier=training._survival_multiplier(epoch,raw["loss"]),
            event_dropout_probability=float(raw["training"]["event_dropout_probability"]),
            gradient_clip_norm=float(raw["training"]["gradient_clip_norm"]),amp=True,max_batches=None)
        assert all(np.isfinite(v) for v in losses.values()),(epoch,losses)
        assert losses["optimizer_steps"]>0
        scheduler.step()
        log("epoch",fold=args.fold,epoch=epoch,total_epochs=47,
            seconds=time.monotonic()-begin,learning_rate=float(optimizer.param_groups[0]["lr"]),
            losses={k:float(v) for k,v in losses.items()})
        write_json(args.output/"training_state.json",{"status":"TRAINING","fold":args.fold,
            "epoch":epoch,"fixed_epoch":47,"seconds":time.monotonic()-started})
        if epoch%10==0 or epoch==47:
            save_resume(state_path,model,optimizer,scheduler,scaler,epoch,train_batcher,source_hash)
    factory.save_checkpoint(args.output/"checkpoint.pt",model,47,anchor)
    train_reference_count=int(data.labels.strict_healthy.sum())
    del data,train_batcher,optimizer,scheduler,scaler
    gc.collect()
    for role,name in [("calibration","calibration_predictions.npz"),("outer_test","outer_predictions.npz")]:
        bundle=evaluation_data(labels,args.fold,role)
        pred=training._predict(model=model,batcher=batcher(bundle,role),
                              device=device,amp=True,max_batches=None)
        save_prediction(args.output/name,bundle,pred,reference,args.fold)
        log("prediction_saved",fold=args.fold,role=role,n=len(pred.chronological_age))
        del bundle,pred
        gc.collect()
    assert all(digest(Path(p))==h for p,h in originals.items())
    result={"status":"COMPLETE","fold":args.fold,"fixed_epoch":47,
            "train_reference_n":train_reference_count,"seconds":time.monotonic()-started,
            "peak_cuda_allocated_mb":torch.cuda.max_memory_allocated(device)/1024**2,
            "original_checkpoints_unchanged":True,"source_sha256":source_hash,
            "output_sha256":{p.name:digest(p) for p in args.output.glob("*")
                             if p.name in {"checkpoint.pt","calibration_predictions.npz","outer_predictions.npz"}}}
    write_json(args.output/"training_state.json",result)
    log("fold_complete",**result)

def main():
    global LOG
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["pilot","train","io-check"],required=True)
    parser.add_argument("--fold",type=int,choices=range(5),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    assert args.output.is_relative_to((ROOT/"Revisions-1030").resolve())
    assert os.environ.get("PROJECT_RUN_DIR")==str(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    LOG=args.output/"run.log"
    if args.mode=="io-check":
        check_checkpoint_io(args.output)
        return
    if (args.output/"training_state.json").exists():
        state=json.loads((args.output/"training_state.json").read_text(encoding="utf-8"))
        if state.get("status")=="COMPLETE":
            log("skip_complete",fold=args.fold)
            return
    membership=role_membership(args.fold)
    log("loading",fold=args.fold,role_counts={k:len(v) for k,v in membership.items()})
    labels,reference,strict=load_references()
    data,old=train_data(labels,reference,args.fold)
    assert set(map(str,data.cache.dataset.participant_ids))==membership["train"]
    log("reference_checked",fold=args.fold,baseline_reference_total=int(reference.sum()),
        strict_reference_total=int(strict.sum()),train_reference=int(data.labels.strict_healthy.sum()))
    if args.mode=="pilot":
        pilot(args,labels,reference,data,old)
    else:
        run_training(args,labels,reference,data,old,membership)

if __name__=="__main__":
    main()
