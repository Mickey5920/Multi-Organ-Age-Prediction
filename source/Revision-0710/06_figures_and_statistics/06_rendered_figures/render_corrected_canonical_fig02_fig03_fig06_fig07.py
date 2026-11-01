#!/usr/bin/env python
"""Render corrected-canonical Fig2, Fig3, Fig6, and Fig7 in the 0708 style."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "Revision-0710/06_figures_and_statistics/05_source_data_by_figure/corrected_canonical_0626_v1"
OUTPUT_ROOT = ROOT / "Revision-0710/06_figures_and_statistics/06_rendered_figures/corrected_canonical_0626_v1"
ADAPTER_ROOT = ROOT / "Revision-0710/06_figures_and_statistics/05_source_data_by_figure/corrected_canonical_0626_adapter_v1/formal_source_v1"
AGE_RECORDS = ADAPTER_ROOT / "01_fig02_fig03_fig06_fig07/canonical_age_records.parquet"
CANONICAL_METRICS = ROOT / "Revision-0710/03_ehr_age/05_corrected_primary_v1/01_canonical_5fold/02_metrics"
BASELINE_ROOT = ROOT / "Revision-0710/05_downstream/10_corrected_primary_v1"
MASKS = ROOT / "Revision-0710/01_data_rebuild/07_corrected_primary_v1/02_outcomes_60m/participant_followup_and_evaluation_masks.parquet"
FIG2_REFERENCE = ROOT / "Revision-0626/Final_Figure_Data/scripts/render_clean_ehr_age_fig2_scatter.py"
FIG3_REFERENCE = ROOT / "Revision-0626/Final_Figure_Data/scripts/render_clean_ehr_age_fig3_baselines.py"
FIG67_REFERENCE = ROOT / "Revision-0626/16_final_ehr_age_downstream_20260704/scripts/render_legacy_fig2_to_fig7.py"
FIG7_SPLIT_REFERENCE = ROOT / "Revision-0626/Final_Figure_Data/scripts/render_clean_ehr_age_fig7_split.py"

ORGANS = [
    "cardiovascular", "hepatic", "immune", "metabolic",
    "musculoskeletal", "pulmonary", "renal",
]
ORGAN_LABELS = {
    "cardiovascular": "Cardio.", "hepatic": "Hepatic", "immune": "Immune",
    "metabolic": "Metabolic", "musculoskeletal": "Musculo.",
    "pulmonary": "Pulm.", "renal": "Renal",
}
METHODS = [
    ("retain", "RETAIN", "#FFFFFF", "#777777", "///", "plain temporal baseline"),
    ("behrt_style", "BEHRT-style", "#D0D3DA", "#555555", "", "plain temporal baseline"),
    ("medbert_style", "Med-BERT-style", "#7AA6C2", "#555555", "", "plain temporal baseline"),
    ("retain_organquery", "RETAIN-OQ", "#90BE6D", "#555555", "", "organ-query baseline"),
    ("behrt_style_organquery", "BEHRT-OQ", "#F9C74F", "#555555", "", "organ-query baseline"),
    ("medbert_style_organquery", "Med-BERT-OQ", "#9B5DE5", "#555555", "", "organ-query baseline"),
    ("ehr_age", "EHR-AGE", "#E64B35", "#111111", "", "corrected canonical EHR-AGE"),
]
T_CRIT_DF4_975 = 2.7764451051977987


def load_module(name: str, path: Path) -> ModuleType:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import visual reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def metric_block(frame: pd.DataFrame, actual: str = "actual_age", predicted: str = "predicted_age") -> dict[str, float | int]:
    x = pd.to_numeric(frame[actual], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[predicted], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    err = y - x
    total = float(np.sum((x - x.mean()) ** 2)) if len(x) else np.nan
    return {
        "n": int(len(x)),
        "mae": float(np.mean(np.abs(err))) if len(x) else np.nan,
        "pearson": float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan,
        "r2": float(1.0 - np.sum(err ** 2) / total) if total and total > 0 else np.nan,
    }


def ci95(values: pd.Series) -> tuple[float, float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = T_CRIT_DF4_975 * sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, sd, mean - half, mean + half


def sample_grouped(frame: pd.DataFrame, by: list[str], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, sub in frame.groupby(by, sort=False, observed=True):
        if len(sub) > n:
            parts.append(sub.loc[rng.choice(sub.index.to_numpy(), size=n, replace=False)])
        else:
            parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else frame.head(0).copy()


def render_fig2(core: pd.DataFrame) -> None:
    out = OUTPUT_ROOT / "Fig02"
    src = SOURCE_ROOT / "Fig02"
    out.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    module = load_module("corrected_0708_fig2_visual", FIG2_REFERENCE)
    module.setup_style()
    metrics = []
    samples = []
    image_paths = []
    lower = float(np.nanmin([core.actual_age.min(), core.predicted_age.min()])) - 5.0
    upper = float(np.nanmax([core.actual_age.max(), core.predicted_age.max()])) + 5.0
    limits = (float(np.floor(lower)), float(np.ceil(upper)))
    for organ in ORGANS:
        full = core[core.organ_system.eq(organ)]
        mb = metric_block(full)
        metrics.append({"Organ": organ, "Organ_Label": ORGAN_LABELS[organ], "Pearson_r": mb["pearson"], "MAE": mb["mae"], "R2": mb["r2"], "Sample_Count": mb["n"]})
        sampled = full.sample(n=min(60000, len(full)), random_state=20260706, replace=False).copy()
        sampled["Organ_Label"] = ORGAN_LABELS[organ]
        samples.append(sampled[["participant_id", "organ_system", "Organ_Label", "actual_age", "predicted_age"]])
        image_paths.append(module.draw_joint_like_plot(full, sampled, organ, limits, out))
    sample = pd.concat(samples, ignore_index=True)
    module.stitch_images(image_paths, out)
    violin = sample.rename(columns={"actual_age": "Actual_Age", "predicted_age": "Predicted_Age"})
    module.render_violin(violin, out)
    pd.DataFrame(metrics).to_csv(src / "scatter_plot_metrics.csv", index=False)
    sample.to_csv(src / "scatter_plot_data_sample.csv.gz", index=False, compression="gzip")
    for stem in ("Fig-2-Main-Scatter-Matrix", "Fig-2-Actual-vs-Predicted-Age"):
        png = Image.open(out / f"{stem}.png").convert("RGB")
        fig = plt.figure(figsize=(png.width / 300, png.height / 300), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(png); ax.axis("off")
        fig.savefig(out / f"{stem}.svg", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    write_json(src / "SOURCE_STATUS.json", {"status": "COMPLETE", "full_oof_rows": int(len(core)), "sample_rows": int(len(sample)), "axis_limits": limits})


def baseline_age_tables(mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = mask[mask.strict_healthy_reference_60m.eq(1)][["participant_id", "chronological_age"]].copy()
    strict["participant_id"] = strict.participant_id.astype(str)
    age_map = strict.set_index("participant_id").chronological_age
    strict_ids = set(age_map.index)
    fold_rows: list[dict[str, object]] = []
    organ_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    formal = pd.read_csv(BASELINE_ROOT / "02_baseline_summary/baseline_method_table.csv").set_index("method")
    for method, *_ in METHODS[:-1]:
        eval_dir = BASELINE_ROOT / "01_baseline_evaluation" / method
        manifest = json.loads((eval_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
        registry_path = Path(manifest["prediction_registry"])
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        cox = pd.read_csv(eval_dir / "cox_fold_organ_cells.csv")
        pooled_parts = []
        for item in registry["prediction_files"]:
            if item["split_role"] != "test":
                continue
            fold = int(item["outer_fold"])
            path = (registry_path.parent / item["path"]).resolve()
            pred = pd.read_parquet(path, columns=["participant_id", "organ_system", "actual_age", "predicted_age", "outer_fold", "split_role"])
            pred["participant_id"] = pred.participant_id.astype(str)
            pred = pred[pred.participant_id.isin(strict_ids)].copy()
            pred["actual_age"] = pred.participant_id.map(age_map).astype(float)
            pooled_parts.append(pred[["organ_system", "actual_age", "predicted_age"]])
            mb = metric_block(pred)
            delta = float(cox[cox.fold.eq(fold)].delta_c.mean())
            fold_rows.append({"method": method, "fold": fold, "n_rows": mb["n"], "mae": mb["mae"], "pearson": mb["pearson"], "r2": mb["r2"], "full_risk_macro_delta_c": delta})
            for organ in ORGANS:
                om = metric_block(pred[pred.organ_system.eq(organ)])
                od = float(cox[(cox.fold.eq(fold)) & (cox.organ_system.eq(organ))].delta_c.iloc[0])
                organ_rows.append({"method": method, "fold": fold, "organ_system": organ, "n_rows": om["n"], "mae": om["mae"], "pearson": om["pearson"], "r2": om["r2"], "delta_c": od})
        pooled = metric_block(pd.concat(pooled_parts, ignore_index=True))
        ref = formal.loc[method]
        pooled_rows.append({"method": method, "n_rows": pooled["n"], "mae": pooled["mae"], "pearson": pooled["pearson"], "r2": pooled["r2"], "formal_mae": ref.strict_mae, "formal_pearson": ref.strict_pearson, "formal_r2": ref.strict_r2, "max_abs_difference": max(abs(pooled["mae"]-ref.strict_mae), abs(pooled["pearson"]-ref.strict_pearson), abs(pooled["r2"]-ref.strict_r2))})
    return pd.DataFrame(fold_rows), pd.DataFrame(organ_rows), pd.DataFrame(pooled_rows)


def add_ehr_age_tables(core: pd.DataFrame, fold: pd.DataFrame, organ: pd.DataFrame, pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = core[core.strict_healthy_reference_60m.eq(1)].copy()
    cox = pd.read_csv(CANONICAL_METRICS / "organ_metrics.csv")
    fold_rows = []
    organ_rows = []
    for fold_id in range(5):
        sub = strict[strict.outer_fold.eq(fold_id)]
        mb = metric_block(sub)
        delta = float(cox[cox.outer_fold.eq(fold_id)].age_sex_harrell_delta_c.mean())
        fold_rows.append({"method": "ehr_age", "fold": fold_id, "n_rows": mb["n"], "mae": mb["mae"], "pearson": mb["pearson"], "r2": mb["r2"], "full_risk_macro_delta_c": delta})
        for system in ORGANS:
            om = metric_block(sub[sub.organ_system.eq(system)])
            od = float(cox[(cox.outer_fold.eq(fold_id)) & (cox.organ_system.eq(system))].age_sex_harrell_delta_c.iloc[0])
            organ_rows.append({"method": "ehr_age", "fold": fold_id, "organ_system": system, "n_rows": om["n"], "mae": om["mae"], "pearson": om["pearson"], "r2": om["r2"], "delta_c": od})
    pm = metric_block(strict)
    official = json.loads((CANONICAL_METRICS / "pooled_strict_metrics.json").read_text(encoding="utf-8"))["metrics"]
    pooled_row = {"method": "ehr_age", "n_rows": pm["n"], "mae": pm["mae"], "pearson": pm["pearson"], "r2": pm["r2"], "formal_mae": official["mae"], "formal_pearson": official["pearson"], "formal_r2": official["r2"], "max_abs_difference": max(abs(pm["mae"]-official["mae"]), abs(pm["pearson"]-official["pearson"]), abs(pm["r2"]-official["r2"]))}
    return pd.concat([fold, pd.DataFrame(fold_rows)], ignore_index=True), pd.concat([organ, pd.DataFrame(organ_rows)], ignore_index=True), pd.concat([pooled, pd.DataFrame([pooled_row])], ignore_index=True)


def render_fig3(core: pd.DataFrame) -> None:
    out = OUTPUT_ROOT / "Fig03"
    src = SOURCE_ROOT / "Fig03"
    main_dir, supp_dir = out / "main_figure", out / "supplementary"
    for path in (main_dir, supp_dir, src): path.mkdir(parents=True, exist_ok=True)
    mask = pd.read_parquet(MASKS, columns=["participant_id", "chronological_age", "strict_healthy_reference_60m"])
    fold, organ, pooled = baseline_age_tables(mask)
    fold, organ, pooled = add_ehr_age_tables(core, fold, organ, pooled)
    if len(fold) != 35 or len(organ) != 245 or float(pooled.max_abs_difference.max()) > 1e-10:
        raise RuntimeError("Fig3 five-fold strict metric parity failed")
    meta = pd.DataFrame([{"source_method": m, "label": label, "role": role, "color": color, "edgecolor": edge, "hatch": hatch} for m, label, color, edge, hatch, role in METHODS])
    summaries = []
    for method, sub in fold.groupby("method", sort=False):
        mae = ci95(sub.mae); pear = ci95(sub.pearson); r2 = ci95(sub.r2)
        pool = pooled[pooled.method.eq(method)].iloc[0]
        summaries.append({"source_method": method, "folds": 5, "macro_MAE_mean": mae[0], "macro_MAE_sd": mae[1], "macro_MAE_ci95_low": mae[2], "macro_MAE_ci95_high": mae[3], "macro_pearson_mean": pear[0], "macro_pearson_sd": pear[1], "macro_pearson_ci95_low": pear[2], "macro_pearson_ci95_high": pear[3], "macro_R2_mean": r2[0], "macro_R2_sd": r2[1], "macro_R2_ci95_low": r2[2], "macro_R2_ci95_high": r2[3], "mean_incremental_delta_c_index_vs_age_sex": float(sub.full_risk_macro_delta_c.mean()), "positive_organs_incremental_vs_age_sex": int((organ[organ.method.eq(method)].groupby("organ_system").delta_c.mean() > 0).sum()), "pooled_strict_MAE": pool.mae, "pooled_strict_pearson": pool.pearson, "pooled_strict_R2": pool.r2})
    summary = meta.merge(pd.DataFrame(summaries), on="source_method", how="left")
    organ_summary = organ.groupby(["method", "organ_system"], as_index=False).agg(MAE=("mae", "mean"), pearson_r=("pearson", "mean"), R2=("r2", "mean"), delta_c_index_vs_age_sex=("delta_c", "mean")).rename(columns={"method": "source_method"})
    organ_summary = meta[["source_method", "label", "role"]].merge(organ_summary, on="source_method", how="left")
    organ_summary["organ_label"] = organ_summary.organ_system.map(ORGAN_LABELS)
    module = load_module("corrected_0708_fig3_visual", FIG3_REFERENCE)
    module.MAIN_DIR = main_dir; module.SUPP_DIR = supp_dir
    module.METHODS = meta.to_dict(orient="records")
    def draw_vertical_metric_panel_ci_safe(ax, frame, value_col, lo_col, hi_col, title, ylabel, fmt, zero_baseline=True):
        x = np.arange(len(frame)); values = frame[value_col].to_numpy(dtype=float)
        if lo_col and hi_col:
            lower = frame[lo_col].to_numpy(dtype=float); upper = frame[hi_col].to_numpy(dtype=float)
            yerr = np.vstack([values - lower, upper - values])
        else:
            lower = values; upper = values; yerr = None
        bars = ax.bar(x, values, yerr=yerr, color=frame["color"].tolist(), edgecolor=frame["edgecolor"].tolist(), linewidth=1.1, capsize=3, width=0.72)
        for bar, hatch in zip(bars, frame["hatch"].fillna("")): bar.set_hatch(hatch)
        ax.set_title(title, fontsize=module.MAIN_TITLE_SIZE, fontweight="bold", pad=10); ax.set_ylabel(ylabel, fontsize=module.MAIN_LABEL_SIZE)
        ax.set_xticks(x); ax.set_xticklabels(frame["label"], rotation=36, ha="right", fontsize=module.MAIN_TICK_SIZE)
        ax.grid(axis="y", color="#D0D0D0", linestyle=":", linewidth=0.8); ax.set_axisbelow(True)
        y_min = float(np.nanmin(lower)); y_max = float(np.nanmax(upper)); pad = (y_max-y_min)*0.18 if y_max>y_min else max(abs(y_max)*0.02, 0.001)
        ax.set_ylim(0, y_max+pad) if zero_baseline and y_min >= 0 else ax.set_ylim(y_min-pad, y_max+pad)
        span = ax.get_ylim()[1]-ax.get_ylim()[0]
        for xi,val,high,low in zip(x,values,upper,lower):
            y_text = high + 0.026*span if val >= 0 else low - 0.030*span
            ax.text(xi,y_text,fmt.format(val),ha="center",va="bottom" if val>=0 else "top",fontsize=module.MAIN_VALUE_SIZE,bbox=dict(facecolor="white",edgecolor="none",alpha=0.72,pad=0.6),clip_on=False)
        module.sns.despine(ax=ax, top=True, right=True)
    module.draw_vertical_metric_panel = draw_vertical_metric_panel_ci_safe
    module.setup_style(); module.render_main(summary); module.render_supp_heatmaps(organ_summary)
    fold.to_csv(src / "fig3_strict_fold_metrics.csv", index=False)
    organ.to_csv(src / "fig3_strict_fold_organ_metrics.csv", index=False)
    pooled.to_csv(src / "fig3_pooled_strict_parity.csv", index=False)
    summary.to_csv(src / "fig3_selected_method_summary.csv", index=False)
    organ_summary.to_csv(src / "fig3_organ_metric_source.csv", index=False)
    write_json(src / "SOURCE_STATUS.json", {"status": "COMPLETE", "methods": 7, "fold_rows": 35, "fold_organ_rows": 245, "cohort": "strict_healthy_reference_60m", "max_pooled_parity_abs_difference": float(pooled.max_abs_difference.max())})


def render_fig6(core: pd.DataFrame) -> None:
    out = OUTPUT_ROOT / "Fig06"; src = SOURCE_ROOT / "Fig06"
    out.mkdir(parents=True, exist_ok=True); src.mkdir(parents=True, exist_ok=True)
    module = load_module("corrected_0626_fig67_visual", FIG67_REFERENCE); module.setup_style()
    before = core[["organ_system", "actual_age", "raw_age_gap"]].rename(columns={"organ_system":"Organ", "actual_age":"Actual_Age", "raw_age_gap":"Age_Gap"}); before["Type"]="Before Correction"; before["Correction"]="Before"
    after = core[["organ_system", "actual_age", "corrected_age_gap"]].rename(columns={"organ_system":"Organ", "actual_age":"Actual_Age", "corrected_age_gap":"Age_Gap"}); after["Type"]="After Correction"; after["Correction"]="After"
    gap = pd.concat([before, after], ignore_index=True); gap["AgeGroup"] = pd.Categorical(gap.Actual_Age.map(module.age_bin), categories=module.AGE_ORDER, ordered=True); gap["Error"] = gap.Age_Gap.abs(); gap["Rounded_Age"] = gap.Actual_Age.round().astype(int)
    rows=[]
    for (organ,corr),sub in gap.groupby(["Organ","Correction"],sort=False,observed=True):
        rp,pp=module.corr_stat(sub.Actual_Age,sub.Age_Gap,"pearson"); rs,ps=module.corr_stat(sub.Actual_Age,sub.Age_Gap,"spearman")
        rows.append({"Organ":organ,"Correction":corr,"N":len(sub),"mean_age_gap":sub.Age_Gap.mean(),"sd_age_gap":sub.Age_Gap.std(ddof=1),"median_age_gap":sub.Age_Gap.median(),"mean_abs_error":sub.Error.mean(),"pearson_r_age_gap":rp,"pearson_p_age_gap":pp,"spearman_r_age_gap":rs,"spearman_p_age_gap":ps})
    stats=pd.DataFrame(rows); mae=gap.groupby(["Organ","Correction","Rounded_Age"],as_index=False,observed=True).agg(MAE=("Error","mean"),Count=("Error","size"))
    tables={"gap":gap,"stats":stats,"mae":mae,"box_sample":sample_grouped(gap,["Organ","Correction","AgeGroup"],6000,6101),"scatter_sample":sample_grouped(gap,["Organ","Correction"],25000,6102),"violin_sample":sample_grouped(gap,["Organ","Correction","AgeGroup"],8000,6103)}
    colors=dict(zip(ORGANS,module.sns.color_palette("colorblind",7).as_hex())); dirs={"fig6":out}
    module.render_fig6a_violin(tables,dirs,colors); module.render_fig6b_scatter(tables,dirs,colors); module.render_fig6_combined(tables,dirs,colors)
    stats.to_csv(src/"fig6_bias_correction_summary.csv",index=False); mae.to_csv(src/"AllOrgans_MAE_Grouped_MAE.csv",index=False)
    tables["box_sample"][["Type","Organ","Actual_Age","AgeGroup","Age_Gap"]].to_csv(src/"All_Organs_AgeGap_Boxplot_Data_Long_sample.csv.gz",index=False,compression="gzip")
    tables["scatter_sample"][["Actual_Age","Error","Organ","Correction"]].to_csv(src/"AllOrgans_MAE_Scatter_Long_sample.csv.gz",index=False,compression="gzip")
    tables["violin_sample"][["Actual_Age","Age_Gap","Organ","Type","AgeGroup"]].to_csv(src/"AllOrgans_Violin_Bias_Correction_sample.csv.gz",index=False,compression="gzip")
    write_json(src/"SOURCE_STATUS.json",{"status":"COMPLETE","raw_rows":int(len(core)),"sealed_rows":int(len(core)),"age_correction_refit":False})


def render_fig7() -> None:
    out=OUTPUT_ROOT/"Fig07"; src=SOURCE_ROOT/"Fig07"; main=out/"main_figure"
    for path in (out,src,main): path.mkdir(parents=True,exist_ok=True)
    cols=["participant_id","organ_system","actual_age","corrected_age_gap","Health","Other_Disease",*ORGANS]
    data=pd.read_parquet(AGE_RECORDS,columns=cols)
    base=data[["participant_id","organ_system","actual_age","corrected_age_gap"]].rename(columns={"organ_system":"System","actual_age":"Actual_Age","corrected_age_gap":"Age_Gap"})
    frames=[]
    for flag,group in (("Health","Health"),("Other_Disease","Other_Disease")):
        temp=base[data[flag].to_numpy(dtype=bool)].copy(); temp["Group"]=group; frames.append(temp)
    for organ in ORGANS:
        keep=data.organ_system.eq(organ)&data[organ].astype(bool); temp=base[keep].copy(); temp["Group"]=organ; frames.append(temp)
    plot=pd.concat(frames,ignore_index=True)[["participant_id","Group","System","Actual_Age","Age_Gap"]]
    summary=[]
    legacy=load_module("corrected_0626_fig7_visual",FIG67_REFERENCE); legacy.setup_style(); colors=dict(zip(ORGANS,legacy.sns.color_palette("colorblind",7).as_hex()))
    for group_key in ["Health","target","Other_Disease"]:
        for organ in ORGANS:
            group=organ if group_key=="target" else group_key; sub=plot[(plot.Group.eq(group))&(plot.System.eq(organ))]; r,p=legacy.corr_stat(sub.Actual_Age,sub.Age_Gap,"spearman")
            summary.append({"row_group":group_key,"Group":group,"System":organ,"N":len(sub),"mean":sub.Age_Gap.mean(),"sd":sub.Age_Gap.std(ddof=1),"median":sub.Age_Gap.median(),"spearman_r":r,"spearman_p":p})
    legacy.render_fig7(plot,{"fig7":out},colors)
    split=load_module("corrected_0708_fig7_split_visual",FIG7_SPLIT_REFERENCE); split.MAIN_DIR=main; split.setup_style(); split.render_scatter(plot,colors); split.render_violin(plot,colors)
    plot.to_parquet(src/"AllGroup_7systems_ScatterViolin_6x7_plotdata.parquet",index=False)
    sample_grouped(plot,["Group","System"],6000,7301).to_csv(src/"AllGroup_7systems_ScatterViolin_6x7_plotdata_sample.csv.gz",index=False,compression="gzip")
    pd.DataFrame(summary).to_csv(src/"fig7_group_system_age_gap_summary.csv",index=False)
    write_json(src/"SOURCE_STATUS.json",{"status":"COMPLETE","groups":["Health","target organ disease","Other_Disease"],"plot_rows":int(len(plot)),"age_gap":"sealed corrected_age_gap","age_correction_refit":False})


def build_manifests() -> None:
    rendered=[]; sources=[]
    scoped = ("Fig02", "Fig03", "Fig06", "Fig07")
    for root,target in ((OUTPUT_ROOT,rendered),(SOURCE_ROOT,sources)):
        for figure in scoped:
            for path in sorted((root / figure).rglob("*")):
                if path.is_file():
                    target.append({"relative_path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    key_pngs=[OUTPUT_ROOT/"Fig02/Fig-2-Main-Scatter-Matrix.png",OUTPUT_ROOT/"Fig03/main_figure/Fig-3-Main-Baseline-Comparison.png",OUTPUT_ROOT/"Fig06/Fig-6-Bias-Correction-MAE.png",OUTPUT_ROOT/"Fig07/Fig-7-Scatter-Violin-Age-Gap.png"]
    image_qa=[]
    for path in key_pngs:
        with Image.open(path) as image:
            rgb=np.asarray(image.convert("RGB").resize((128,128)),dtype=np.uint8); nonwhite=float(np.mean(np.any(rgb<250,axis=2)))
            image_qa.append({"path":path.relative_to(OUTPUT_ROOT).as_posix(),"width":image.width,"height":image.height,"nonwhite_fraction":nonwhite,"pass":bool(image.width>500 and image.height>250 and nonwhite>0.02)})
    qa={"status":"PASS" if all(x["pass"] for x in image_qa) else "FAIL","old_numeric_data_read":False,"age_correction_refit":False,"figures_in_scope":["Fig02","Fig03","Fig06","Fig07"],"image_checks":image_qa}
    receipt_prefix = "FIG02_FIG03_FIG06_FIG07"
    write_json(OUTPUT_ROOT/f"{receipt_prefix}_QA_REPORT.json",qa); write_json(SOURCE_ROOT/f"{receipt_prefix}_QA_REPORT.json",qa)
    input_records=[]
    for path in [AGE_RECORDS,ADAPTER_ROOT/"MANIFEST.json",CANONICAL_METRICS/"fold_metrics.csv",CANONICAL_METRICS/"organ_metrics.csv",BASELINE_ROOT/"02_baseline_summary/baseline_method_table.csv"]:
        input_records.append({"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    common={"schema_version":"revision0710.corrected_canonical_0626_fig02_fig03_fig06_fig07.v1","created_at":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"status":"COMPLETE","inputs":input_records,"visual_references":[str(x.relative_to(ROOT)).replace("\\","/") for x in [FIG2_REFERENCE,FIG3_REFERENCE,FIG67_REFERENCE,FIG7_SPLIT_REFERENCE]],"policies":{"old_numeric_data_read":False,"age_correction_refit":False,"training_started":False,"fig04_fig05_touched":False,"fig08_fig12_touched":False}}
    write_json(OUTPUT_ROOT/f"{receipt_prefix}_MANIFEST.json",{**common,"generated_artifacts":rendered}); write_json(SOURCE_ROOT/f"{receipt_prefix}_MANIFEST.json",{**common,"generated_artifacts":sources})
    status={"status":"COMPLETE","completed_figures":["Fig02","Fig03","Fig06","Fig07"],"qa_status":qa["status"]}
    write_json(OUTPUT_ROOT/f"{receipt_prefix}_RUN_STATUS.json",status); write_json(SOURCE_ROOT/f"{receipt_prefix}_RUN_STATUS.json",status)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args()
    required=[AGE_RECORDS,ADAPTER_ROOT/"MANIFEST.json",ADAPTER_ROOT/"RUN_STATUS.json",MASKS,FIG2_REFERENCE,FIG3_REFERENCE,FIG67_REFERENCE,FIG7_SPLIT_REFERENCE]
    missing=[str(x) for x in required if not x.is_file()]
    if missing: raise FileNotFoundError(missing)
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); SOURCE_ROOT.mkdir(parents=True,exist_ok=True)
    core=pd.read_parquet(AGE_RECORDS,columns=["participant_id","outer_fold","organ_system","actual_age","predicted_age","raw_age_gap","corrected_age_gap","strict_healthy_reference_60m"])
    render_fig2(core); render_fig3(core); render_fig6(core); del core
    render_fig7(); build_manifests()
    print(OUTPUT_ROOT); return 0


if __name__ == "__main__":
    raise SystemExit(main())
