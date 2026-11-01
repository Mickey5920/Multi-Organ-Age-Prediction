"""R1.3: joint participant bootstrap of sealed MIMIC OOF predictions.

No neural inference or downstream refitting. Historical inputs are read-only.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
if "PROJECT_CACHE_DIR" in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = os.environ["PROJECT_CACHE_DIR"]
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "Revision-0710/06_figures_and_statistics/04_external_validation"
SOURCE = ROOT / "Revision-0710/06_figures_and_statistics/00_plan/external_validation_prefreeze_v8/execute_external_validation_v8.py"
INPUTS = {
    "direct": OLD / "core_0717_codebook_v2_huber1/shared/06_statistics/mimic_oof_risk_predictions.parquet",
    "adapted": OLD / "core_0717_adapter_mimic_common53/shared/06_statistics/mimic_oof_risk_predictions.parquet",
}
PUBLISHED = ROOT / "Revision-0730/4-Data/MIMIC_12m_delta_c.csv"
DETAIL_ROOT = ROOT / "Revision-0710/Final_Fig_0716/Codebook_v2_Huber1"
PUBLISHED_DETAIL = {
    "Direct transfer": DETAIL_ROOT / "Fig13/Source_Data/Fig13D_paired_incremental_risk_metrics.csv",
    "Support aligned": DETAIL_ROOT / "External_Adaptation/mimic_common53/Fig13/Source_Data/Fig13D_paired_incremental_risk_metrics.csv",
}
ORGANS = ("cardiovascular", "hepatic", "immune", "metabolic", "musculoskeletal", "pulmonary", "renal")
COMPARISONS = ("age_sex_plus_gap", "age_sex_density_plus_gap")
METRICS = ("delta_harrell_c", "delta_auc_12m", "brier_improvement_12m")
SEED = 20261031

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

def log(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}, allow_nan=False), flush=True)

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

def historical_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"harrell_c", "weighted_auc", "weighted_brier"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace

def basic_tests(reference):
    t = np.array([1.,1.,2.,3.,4.,5.])
    e = np.array([1,0,1,0,1,0], dtype=np.int8)
    r = np.array([3.,3.,2.,2.,1.,0.])
    y = np.array([1.,0.,1.,0.,1.,0.])
    p = np.array([.8,.8,.6,.6,.3,.1])
    w = np.array([1.,.4,2.,1.,0.,.7])
    counts = np.array([2,3,0,1,2,1], dtype=np.int64)
    ix = np.repeat(np.arange(len(t)), counts)
    score_rank = np.searchsorted(np.unique(r), r).astype(np.int64)+1
    got = harrell(counts, np.argsort(-t, kind="stable"), t, e, score_rank, len(np.unique(r)))
    assert abs(got-reference["harrell_c"](t[ix], e[ix], r[ix])) < 1e-14
    got = auc(counts, np.argsort(p, kind="stable"), p, y, w)
    assert abs(got-reference["weighted_auc"](y[ix], p[ix], w[ix])) < 1e-14
    loss = w*(y-p)**2
    got = np.dot(counts, loss)/counts.sum()
    expected = reference["weighted_brier"](y[ix], p[ix], w[ix], normalization_n=len(ix))
    assert abs(got-expected) < 1e-14

def prepare(reference):
    frames = {}
    keys = ["subject_id","organ_system","horizon_months","comparison","outer_fold"]
    for stage, path in INPUTS.items():
        frame = pd.read_parquet(path)
        frame = frame.loc[frame.horizon_months.eq(12)].copy()
        frame["subject_id"] = frame["subject_id"].astype(str)
        frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        assert not frame.duplicated(keys).any()
        frames[stage] = frame
    aligned = keys+["observed_time_days","event_indicator","competing_death_indicator",
                    "ipcw_outcome","ipcw_weight","ipcw_observed"]
    pd.testing.assert_frame_equal(frames["direct"][aligned], frames["adapted"][aligned])
    people = frames["direct"][["subject_id","outer_fold"]].drop_duplicates().sort_values("subject_id")
    assert not people.subject_id.duplicated().any()
    universe = people.subject_id.to_numpy()
    fold_indices = [np.flatnonzero(people.outer_fold.to_numpy()==k) for k in range(5)]
    published = pd.concat([pd.read_csv(path).assign(stage=stage)
                           for stage,path in PUBLISHED_DETAIL.items()],ignore_index=True)
    published = published.loc[published.horizon_months.eq(12)]
    current_c = pd.read_csv(PUBLISHED)
    keys_c = ["stage","organ_system","comparison","metric"]
    joined = current_c.merge(published,on=keys_c,suffixes=("_current","_detail"),validate="one_to_one")
    assert len(joined)==len(current_c)
    np.testing.assert_allclose(joined.delta_current,joined.delta_detail,atol=1e-14,rtol=0)
    cells, parity = [], []
    first_counts = None
    rng = np.random.default_rng(SEED)
    master = np.zeros(len(universe), dtype=np.int64)
    for ids in fold_indices:
        master[ids] = rng.multinomial(len(ids), np.full(len(ids),1/len(ids)))
    for stage in ("direct","adapted"):
        for comparison in COMPARISONS:
            for organ in ORGANS:
                f = frames[stage].loc[frames[stage].comparison.eq(comparison)&
                                       frames[stage].organ_system.eq(organ)].copy()
                assert not f.empty and not f.subject_id.duplicated().any()
                positions = np.searchsorted(universe, f.subject_id.to_numpy())
                assert np.array_equal(universe[positions], f.subject_id.to_numpy())
                duration = f.observed_time_days.to_numpy(float)
                event = f.event_indicator.to_numpy(np.int8)
                outcome = f.ipcw_outcome.to_numpy(float)
                weight = f.ipcw_weight.to_numpy(float)
                scores = [f[k].to_numpy(float) for k in ("base_risk","augmented_risk")]
                probs = [f[k].to_numpy(float) for k in ("base_probability","augmented_probability")]
                uniques = [np.unique(score) for score in scores]
                cell = Prepared(stage, comparison, organ, f, positions,
                    np.argsort(-duration,kind="stable"),
                    [np.argsort(p,kind="stable") for p in probs],
                    [np.searchsorted(u,s).astype(np.int64)+1 for u,s in zip(uniques,scores)],
                    [len(u) for u in uniques],
                    [np.where(weight>0,weight*(outcome-p)**2,0.) for p in probs],
                    duration,event,outcome,weight,scores,probs)
                b,a,d = cell.evaluate(np.ones(len(f),dtype=np.int64))
                label = "Direct transfer" if stage=="direct" else "Support aligned"
                for k, metric in enumerate(("harrell_c","ipcw_auc","ipcw_brier")):
                    old = published.loc[published.stage.eq(label)&published.comparison.eq(comparison)&
                                        published.organ_system.eq(organ)&published.metric.eq(metric)].iloc[0]
                    error = max(abs(b[k]-old.base_estimate),abs(a[k]-old.augmented_estimate),
                                abs(d[k]-old.delta))
                    assert error < 2e-12, (stage,comparison,organ,metric,error)
                    parity.append({"stage":stage,"comparison":comparison,"organ_system":organ,
                                   "metric":metric,"max_abs_error":error})
                # Count bootstrap equivalence with the original repeated-row implementation.
                if organ=="cardiovascular":
                    count = master[positions]
                    ix = np.repeat(np.arange(len(f)),count)
                    bb,aa,dd = cell.evaluate(count)
                    for k in range(2):
                        original = np.array([
                            reference["harrell_c"](duration[ix],event[ix],scores[k][ix]),
                            reference["weighted_auc"](outcome[ix],probs[k][ix],weight[ix]),
                            reference["weighted_brier"](outcome[ix],probs[k][ix],weight[ix],
                                                        normalization_n=len(ix))])
                        np.testing.assert_allclose(bb if k==0 else aa,original,atol=2e-12,rtol=0)
                cells.append(cell)
    return cells, universe, fold_indices, pd.DataFrame(parity)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replicates",type=int,default=1000)
    parser.add_argument("--pilot",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    assert output.is_relative_to((ROOT/"Revisions-1030").resolve())
    assert os.environ.get("PROJECT_RUN_DIR") == str(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"analysis_manifest.json").exists():
        raise FileExistsError("Do not overwrite an existing analysis.")
    started=time.monotonic()
    source_hash=digest(__file__)
    records={stage:{"path":str(p),"sha256":digest(p)} for stage,p in INPUTS.items()}
    reference=historical_functions()
    basic_tests(reference)
    cells,universe,folds,parity=prepare(reference)
    parity.to_csv(output/"point_parity.csv",index=False)
    points=[]
    point_array=np.zeros((len(cells),3))
    for i,cell in enumerate(cells):
        b,a,d=cell.evaluate(np.ones(len(cell.duration),dtype=np.int64))
        point_array[i]=d
        for j,metric in enumerate(METRICS):
            points.append({"stage":cell.stage,"comparison":cell.comparison,"organ_system":cell.organ,
                "metric":metric,"n":len(cell.duration),"events":int(cell.event.sum()),
                "base_estimate":b[j],"augmented_estimate":a[j],"point_estimate":d[j]})
    point_frame=pd.DataFrame(points)
    point_frame.to_csv(output/"organ_point_metrics.csv",index=False)
    log("parity_passed",checks=len(parity),max_abs_error=float(parity.max_abs_error.max()),
        participants=len(universe),cells=len(cells))
    write_json(output/"analysis_manifest.json",{
        "status":"BOOTSTRAP_RUNNING","source_sha256":source_hash,"inputs":records,
        "historical_metrics_sha256":digest(SOURCE),"published_results_sha256":digest(PUBLISHED),
        "published_detail_sha256":{k:digest(v) for k,v in PUBLISHED_DETAIL.items()},
        "protocol":{"replicates":args.replicates,"seed":SEED,"horizon_months":12,
                    "bootstrap":"participants within MIMIC hash folds, jointly across organs/routes/references",
                    "aggregation":"pooled OOF metric within organ, then equal mean across seven organs",
                    "conditional_on_fixed_fits":True,"new_neural_inference":False,"new_model_fitting":False},
        "participants":len(universe),"pilot_only":args.pilot})
    rng=np.random.default_rng(SEED)
    samples=np.zeros((args.replicates,len(cells),3))
    for rep in range(args.replicates):
        counts=np.zeros(len(universe),dtype=np.int64)
        for ids in folds:
            counts[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
        for i,cell in enumerate(cells):
            _,_,samples[rep,i]=cell.evaluate(counts[cell.positions])
        assert np.isfinite(samples[rep]).all()
        if (rep+1)%25==0 or rep+1==args.replicates:
            log("bootstrap",replicates_done=rep+1,seconds=time.monotonic()-started)
    ci_rows=[]
    for i,cell in enumerate(cells):
        for j,metric in enumerate(METRICS):
            lo,hi=np.quantile(samples[:,i,j],[.025,.975])
            ci_rows.append({"stage":cell.stage,"comparison":cell.comparison,"organ_system":cell.organ,
                            "metric":metric,"point_estimate":point_array[i,j],
                            "ci95_low":lo,"ci95_high":hi,"replicates":args.replicates})
    pd.DataFrame(ci_rows).to_csv(output/"organ_paired_bootstrap_ci.csv",index=False)
    macro_rows=[]
    replicates=[]
    macro_samples={}
    macro_points={}
    for stage in ("direct","adapted"):
        for comparison in COMPARISONS:
            indices=[i for i,c in enumerate(cells) if c.stage==stage and c.comparison==comparison]
            assert len(indices)==7
            values=samples[:,indices].mean(axis=1)
            point=point_array[indices].mean(axis=0)
            macro_samples[(stage,comparison)]=values
            macro_points[(stage,comparison)]=point
            for j,metric in enumerate(METRICS):
                lo,hi=np.quantile(values[:,j],[.025,.975])
                macro_rows.append({"stage":stage,"comparison":comparison,"metric":metric,
                    "point_estimate":point[j],"ci95_low":lo,"ci95_high":hi,
                    "replicates":args.replicates})
                replicates.extend({"replicate":k,"stage":stage,"comparison":comparison,
                                   "metric":metric,"value":v} for k,v in enumerate(values[:,j]))
    pd.DataFrame(macro_rows).to_csv(output/"macro_paired_bootstrap_ci.csv",index=False)
    pd.DataFrame(replicates).to_csv(output/"macro_bootstrap_replicates.csv",index=False)
    contrast=[]
    for comparison in COMPARISONS:
        values=macro_samples[("adapted",comparison)]-macro_samples[("direct",comparison)]
        point=macro_points[("adapted",comparison)]-macro_points[("direct",comparison)]
        for j,metric in enumerate(METRICS):
            lo,hi=np.quantile(values[:,j],[.025,.975])
            contrast.append({"contrast":"adapted_minus_direct","comparison":comparison,"metric":metric,
                "point_estimate":point[j],"ci95_low":lo,"ci95_high":hi,"replicates":args.replicates})
    pd.DataFrame(contrast).to_csv(output/"adaptation_minus_direct_ci.csv",index=False)
    np.savez_compressed(output/"joint_bootstrap_replicates.npz",samples=samples,
                        cells=np.array([(c.stage,c.comparison,c.organ) for c in cells]))
    assert all(digest(INPUTS[k])==v["sha256"] for k,v in records.items())
    manifest=json.loads((output/"analysis_manifest.json").read_text(encoding="utf-8"))
    manifest.update(status="PILOT_COMPLETE" if args.pilot else "COMPLETE",
                    seconds=time.monotonic()-started,point_parity="PASS",
                    bytes=sum(p.stat().st_size for p in output.glob("*") if p.is_file()),
                    inputs_unchanged=True,source_sha256_unchanged=digest(__file__)==source_hash)
    write_json(output/"analysis_manifest.json",manifest)
    log("complete",**{k:manifest[k] for k in ("status","seconds","bytes")})

if __name__=="__main__":
    main()
