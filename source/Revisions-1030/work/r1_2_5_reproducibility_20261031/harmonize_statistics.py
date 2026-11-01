"""Pooled age summaries and stable sensitivity-HR inference from frozen outputs.

No fitting, neural inference, participant-data export, or manuscript modification.
Original inputs are hashed and kept unchanged. New CSVs are aggregate results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
import pandas as pd
from scipy.stats import norm

ORGANS = ("cardiovascular", "hepatic", "immune", "metabolic",
          "musculoskeletal", "pulmonary", "renal")
CASES = ("strict_train", "baseline_train")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def display_log_p(logp):
    if logp < -300 * math.log(10):
        return "<1e-300"
    return f"{math.exp(logp):.8g}"


def holm_log_p(logp):
    """Holm adjustment in log probability, including subnormal probabilities."""
    values = np.asarray(logp, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = -np.inf
    for rank, index in enumerate(order):
        running = max(running, values[index] + math.log(len(values) - rank))
        adjusted[index] = min(0.0, running)
    return adjusted


def moments(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    assert np.isfinite(a).all() and np.isfinite(p).all()
    d = p - a
    return np.array([len(a), a.sum(), p.sum(), np.dot(a, a), np.dot(p, p),
                     np.dot(a, p), np.abs(d).sum(), np.dot(d, d)], float)


def finish_moments(total):
    n, sa, sp, saa, spp, sap, ae, se = total
    va, vp = saa - sa * sa / n, spp - sp * sp / n
    assert va > 0 and vp > 0
    return {"n_prediction_rows": int(n), "mae": ae / n,
            "pearson": (sap - sa * sp / n) / math.sqrt(va * vp),
            "r2": 1 - se / va}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output.resolve()
    assert output.is_relative_to(root / "Revisions-1030")
    assert os.environ.get("PROJECT_RUN_DIR") == str(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def remember(path):
        path = Path(path)
        inputs[str(path)] = sha(path)
        return path

    r14 = root / "Revisions-1030/work/r1_4_reference_20261031"
    outcomes = remember(root / "Revision-0710/01_data_rebuild/09_v2_labels/organ_outcomes_60m.parquet")
    labels = pd.read_parquet(outcomes, columns=["participant_id", "has_prevalent_target",
                                              "strict_healthy_reference_60m"])
    labels["participant_id"] = labels.participant_id.astype(str)
    grouped = labels.groupby("participant_id", sort=False)
    burden = grouped.has_prevalent_target.sum()
    assert grouped.strict_healthy_reference_60m.nunique().eq(1).all()
    strict = grouped.strict_healthy_reference_60m.first().astype(bool)
    context_path = remember(root / "Revision-0710/06_figures_and_statistics/03_reviewer_statistics/03_clinical_utility/01_formal_run/02_input_cache/participant_context_covariates.pkl")
    context = pd.read_pickle(context_path)
    context["participant_id"] = context.participant_id.astype(str)
    baseline = context.set_index("participant_id").has_preindex_icd.eq(0)
    baseline &= burden.reindex(baseline.index).eq(0)
    assert len(strict) == len(baseline) == 456773
    assert int(strict.sum()) == 102129 and int(baseline.sum()) == 175637
    del labels, context

    rows = []
    for case in CASES:
        totals = {(pop, organ): np.zeros(8) for pop in ("baseline_only", "strict_health")
                  for organ in (*ORGANS, "pooled_seven_organs")}
        all_ids = set()
        for fold in range(5):
            folder = r14 / "analysis" / case / "outputs/prepared" / f"outer_fold_{fold}"
            frame = pd.read_pickle(remember(folder / "age_osag.pkl"))
            frame["participant_id"] = frame.participant_id.astype(str)
            assert not frame.duplicated(["participant_id", "organ_system"]).any()
            assert set(frame.organ_system) == set(ORGANS)
            assert frame.groupby("participant_id").size().eq(7).all()
            ids = set(frame.participant_id)
            assert not (ids & all_ids)
            all_ids.update(ids)
            original_fold = pd.read_csv(remember(folder / "age_metrics.csv"))
            for population, membership in (("baseline_only", baseline), ("strict_health", strict)):
                mask = frame.participant_id.map(membership)
                assert mask.notna().all()
                selected = frame.loc[mask.to_numpy(bool)]
                totals[population, "pooled_seven_organs"] += moments(selected.actual_age, selected.predicted_age)
                for organ in ORGANS:
                    part = selected.loc[selected.organ_system.eq(organ)]
                    cell = moments(part.actual_age, part.predicted_age)
                    totals[population, organ] += cell
                    expected = original_fold.loc[original_fold.population.eq(population) &
                                                 original_fold.organ_system.eq(organ)].iloc[0]
                    measured = finish_moments(cell)
                    for metric in ("mae", "pearson", "r2"):
                        np.testing.assert_allclose(measured[metric], expected[metric], rtol=0, atol=2e-11)
            print(json.dumps({"event": "pooled_fold", "case": case, "fold": fold}), flush=True)
        assert len(all_ids) == 456773
        for (population, organ), total in totals.items():
            values = finish_moments(total)
            n = 175637 if population == "baseline_only" else 102129
            assert values["n_prediction_rows"] == n * (7 if organ == "pooled_seven_organs" else 1)
            rows.append({"case": case, "population": population, "organ_system": organ,
                         "n_participants": n, "aggregation": "pooled OOF participant-organ predictions",
                         "raw_age_prediction": True, **values})
    pd.DataFrame(rows).to_csv(output / "pooled_age_metrics.csv", index=False)

    source_hr = remember(r14 / "results/associations/new_reference_detailed_burden_hr.csv")
    hr = pd.read_csv(source_hr)
    assert len(hr) == 7 and set(hr.organ_system) == set(ORGANS)
    beta = hr.gap_beta_per_crossfit_sd.to_numpy(float)
    se = hr.gap_se_model_based.to_numpy(float)
    assert np.isfinite(beta).all() and np.isfinite(se).all() and (se > 0).all()
    z = beta / se
    logp = math.log(2.0) + norm.logsf(np.abs(z))
    log_holm = holm_log_p(logp)
    # Preserve every historical column, including raw floating-point underflow.
    hr["wald_z"] = z
    hr["log10_p_stable"] = logp / math.log(10)
    hr["p_display_stable"] = [display_log_p(v) for v in logp]
    hr["log10_p_holm_7_stable"] = log_holm / math.log(10)
    hr["p_holm_7_display_stable"] = [display_log_p(v) for v in log_holm]
    hr["multiplicity_family"] = "seven corresponding-organ training-reference sensitivity HRs"
    hr["ci_scope"] = "pointwise model-based 95% Wald interval; not multiplicity adjusted"
    np.testing.assert_allclose(np.exp(beta), hr.gap_hr_per_crossfit_sd, rtol=1e-14, atol=0)
    ordinary = hr.gap_p_value.to_numpy(float)
    nonzero = ordinary > 0
    np.testing.assert_allclose(np.exp(logp[nonzero]), ordinary[nonzero], rtol=2e-11, atol=0)
    hr.to_csv(output / "training_reference_hr_stable_inference.csv", index=False)

    manifest = {"status": "COMPLETE", "new_fits": 0, "new_neural_inference": False,
                "age_aggregation": "pooled participant-organ predictions; original fold averages retained",
                "age_populations": {"baseline_only": 175637, "strict_health": 102129},
                "age_rows": len(rows), "hr_rows": len(hr), "underflowed_original_p_values": int((ordinary == 0).sum()),
                "hr_estimates_and_intervals_changed": False,
                "new_adjustment": "Holm within the seven sensitivity HRs; not a new confirmatory family",
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "source_sha256": sha(__file__), "inputs": inputs,
                "inputs_unchanged": all(sha(Path(p)) == h for p, h in inputs.items())}
    assert manifest["inputs_unchanged"]
    (output / "statistics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "inputs"}), flush=True)


if __name__ == "__main__":
    main()
