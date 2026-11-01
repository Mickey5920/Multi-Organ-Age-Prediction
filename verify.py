"""Bounded release checks; never retrains or exports participant-level records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "2")
import _runtime


def static_check(metadata):
    count = 0
    for item in metadata["files"]:
        path = _runtime.SOURCE / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
            compile(_runtime.adapted_tree(path), str(path), "exec")
            count += 1
    for item in metadata.get("delivery_files", []):
        assert hashlib.sha256((_runtime.PACKAGE / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    # Exercise the source binding for an actual study-script subprocess. The
    # child displays help only; no participant input is opened and no fit runs.
    relative = "Revisions-1030/work/r1_2_5_reproducibility_20261031/harmonize_statistics.py"
    child = subprocess.run([sys.executable, "-B", str(_runtime.PROJECT / relative), "--help"],
                           capture_output=True, text=True, check=True)
    assert '"event": "packaged_child"' in child.stdout
    assert "--project-root" in child.stdout
    return {"byte_identical_files": len(metadata["files"]), "compiled_raw_and_adapted_python": count,
            "delivery_file_hashes_checked": len(metadata.get("delivery_files", [])),
            "child_source_binding_help": "PASS"}


def synthetic_check(project):
    """Exercise the published preprocessing contract using invented rows only."""
    import numpy as np
    import pandas as pd
    core = _runtime.load("Revision-0710/01_data_rebuild/scripts/compact_cache_core.py", "release_cache_core")
    values = np.tile(np.array([[1.0], [3.0], [10000.0]], dtype=np.float32), (1, 78))
    missing = np.zeros(values.shape, dtype=bool)
    missing[0, 0] = True
    transform = core.fit_phenotype_transform(values, missing, train_mask=np.array([True, True, False]))
    np.testing.assert_equal(transform["median"][0], 3.0)
    np.testing.assert_equal(transform["median"][1:], 2.0)
    changed = values.copy()
    changed[2] = -10000.0
    for name, vector in core.fit_phenotype_transform(changed, missing, train_mask=np.array([True, True, False])).items():
        np.testing.assert_array_equal(vector, transform[name])
    assert np.isfinite(core.apply_phenotype_transform(values, missing, transform)).all()
    columns = ["participant_id", "event_date", "code", "description", "source", "source_row_number", "relative_days", "attention_mask"]
    icd = pd.DataFrame([["SYNTHETIC_A", pd.Timestamp("2010-01-01"), "I10", "toy diagnosis", "hospital_icd", 2, -10, 1]], columns=columns)
    gp = pd.DataFrame(columns=columns)
    timeline = core.select_unified_timeline(icd, gp, participant_ids=["SYNTHETIC_A", "SYNTHETIC_B"], max_len=32)
    packed = core.pack_compact_sequences(timeline, participant_order=["SYNTHETIC_A", "SYNTHETIC_B"], max_len=32,
                     code_to_id={"I10": 1, "": 0}, text_to_id={"icd_description\x1ftoy diagnosis": 1})
    assert packed["event_code_id"].shape == (2, 32)
    assert packed["event_valid_mask"].sum() == 1
    assert (packed["event_relative_days"][packed["event_valid_mask"]] < 0).all()
    for name in ("event_code_id", "event_text_id", "event_relative_days", "event_type_id"):
        assert (packed[name][~packed["event_valid_mask"]] == 0).all()
    assert packed["event_count"].tolist() == [1, 0]
    previous = dict(_runtime.INPUT_MAP)
    try:
        assert _runtime._logical(project / "invented-inputs/P_DEMO.xpt") == project / "invented-inputs/P_DEMO.xpt"
        _runtime.INPUT_MAP[_runtime._path_key("F:/Data/NHANES/NHANES")] = project / "invented-inputs"
        assert _runtime.input_path("F:/Data/NHANES/NHANES/P_DEMO.xpt") == project / "invented-inputs/P_DEMO.xpt"
    finally:
        _runtime.INPUT_MAP.clear()
        _runtime.INPUT_MAP.update(previous)
    return {"input": "invented in-memory arrays and two synthetic identifiers", "phenotype_slots": 78,
            "training_only_transform": "PASS", "packed_shape": [2, 32], "padding_and_time_contract": "PASS",
            "windows_logical_map_preserves_filename_case": "PASS", "participant_data_read": False,
            "project_root_substitution_idempotent": "PASS",
            "training_steps": 0}


def cox_check(project):
    import numpy as np
    import pandas as pd
    r11 = _runtime.load("Revisions-1030/work/r1_1_information_matched_20261031/run_information_matched.py", "repro_r11")
    clinical, _ = r11.modules()
    folder = project / "Revisions-1030/work/r1_4_reference_20261031/analysis/baseline_train/outputs/folds/outer_fold_4"
    values = pd.read_csv(folder / "hepatic__metrics.csv")
    with np.load(folder / "hepatic__bootstrap.npz") as z:
        frame = pd.DataFrame({"duration": z["duration"], "event": z["event"]})
        observed = {}
        for label, key in (("base", "base_score"), ("all_gaps", "all_gaps_score")):
            actual = float(clinical.harrell_c_index(frame, z[key]))
            expected = float(values.loc[values.augmentation.eq(label), "harrell_c"].iloc[0])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
            observed[label] = {"harrell_c": actual, "max_abs_difference": abs(actual - expected)}
    return {"case": "baseline_train", "fold": 4, "organ": "hepatic", "n": len(frame),
            "published_point_metric_parity": observed, "new_fits": 0}


def covariance_check(project, output):
    """Check SEM dependencies, invented input I/O and saved aggregates; no fit."""
    import importlib.metadata as distribution
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow.parquet as pq
    import semopy
    from PIL import Image

    profile = json.loads((_runtime.PACKAGE / "environments/covariance.json").read_text(encoding="utf-8"))
    versions = {name: distribution.version(name) for name in profile["dependencies"]}
    assert versions == profile["dependencies"], (versions, profile["dependencies"])
    assert sys.version.split()[0] == profile["python"]
    dependency_check = subprocess.run([sys.executable, "-B", "-m", "pip", "check"],
                                      capture_output=True, text=True, check=True)
    assert "No broken requirements found" in dependency_check.stdout

    def reject_fit(*args, **kwargs):
        raise RuntimeError("SEM fitting is forbidden in this bounded verification")

    original_fit = semopy.Model.fit
    semopy.Model.fit = reject_fit
    try:
        folder = "Revision-0730/0-Others/Fig7_Shared_Component_SEM/scripts/"
        atlas = _runtime.load(folder + "build_figure6_shared_component_residual_atlas.py",
                              "build_figure6_shared_component_residual_atlas")
        diagnostic = _runtime.load(folder + "validate_shared_component_sem.py", "release_sem_diagnostic")
        estimator = _runtime.load(folder + "explore_sem_estimators.py", "release_sem_estimators")
        _runtime.load(folder + "export_from_png.py", "export_from_png")
        _runtime.load(folder + "export_figure6_components.py", "release_sem_components")
        organs = tuple(atlas.ORGANS)
        assert tuple(diagnostic.ORGANS) == tuple(estimator.ORGANS) == organs

        fixture = output / "synthetic_fixtures"
        fixture.mkdir(exist_ok=True)
        n = 24
        rows = np.arange(n, dtype=float)
        wide = pd.DataFrame({"participant_id": [f"SYNTHETIC_{i:04d}" for i in range(n)],
                             "outer_fold": np.arange(n) % 5, "actual_age": rows + 40,
                             "sex": np.where(np.arange(n) % 2, "Male", "Female"),
                             "strict_healthy_reference_60m": (np.arange(n) % 3 == 0).astype(np.int8),
                             "phenotype_missing_count": np.arange(n) % 20})
        for k, organ in enumerate(organs):
            wide[f"{organ}__corrected_age_gap"] = np.sin(rows / (k + 2)) + rows * (k + 1) / 100
        wide_path = fixture / "invented_wide.parquet"
        wide.to_parquet(wide_path, index=False, engine="pyarrow")
        pd.testing.assert_frame_equal(pd.read_parquet(wide_path), wide)
        assert pq.read_schema(wide_path).names == list(wide.columns)
        assert pl.read_parquet(wide_path).height == n
        tokens = [row[2] for row in atlas.CLINICAL_GROUPS]
        membership = pd.DataFrame({"participant_id": [wide.participant_id[i] for i in range(len(tokens))],
                                   "analysis_group": tokens})
        membership_path = fixture / "invented_membership.parquet"
        membership.to_parquet(membership_path, index=False, engine="pyarrow")
        atlas.WIDE, atlas.MEMBERSHIP = wide_path, membership_path
        diagnostic.WIDE = estimator.WIDE = wide_path
        loaded, groups = atlas.load_inputs()
        checked = diagnostic.load_wide()
        estimator_input = estimator.load_data()
        assert len(loaded) == len(checked) == len(estimator_input) == n
        assert set(groups.analysis_group) == set(tokens)
        expected = wide[[f"{organ}__corrected_age_gap" for organ in organs]].to_numpy()
        np.testing.assert_array_equal(loaded[list(organs)], expected)
        np.testing.assert_array_equal(checked[list(organs)], expected)
        np.testing.assert_array_equal(estimator_input[list(organs)], expected)
        standardized = atlas.standardize(loaded)
        np.testing.assert_allclose(standardized.mean(), 0, atol=1e-14, rtol=0)
        np.testing.assert_allclose(standardized.std(ddof=0), 1, atol=1e-14, rtol=0)
        duplicate_path = fixture / "invented_duplicate.parquet"
        pd.concat([wide, wide.iloc[:1]], ignore_index=True).to_parquet(duplicate_path, index=False)
        atlas.WIDE = duplicate_path
        try:
            atlas.load_inputs()
        except RuntimeError as error:
            assert "duplicate participant IDs" in str(error)
        else:
            raise AssertionError("duplicate participant input was accepted")
        atlas.WIDE = wide_path
        image_path = fixture / "invented_pixels.png"
        Image.new("RGB", (4, 4), (0, 0, 0)).save(image_path)
        with Image.open(image_path) as handle:
            assert handle.size == (4, 4)

        references = _runtime.PACKAGE / "reference/sem"
        metadata = json.loads((_runtime.PACKAGE / "package.json").read_text(encoding="utf-8"))
        reference_hashes = {}
        for record in metadata["covariance_reference_files"]:
            path = _runtime.PACKAGE / record["path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == record["sha256"], record["path"]
            reference_hashes[record["path"]] = actual
        corr = pd.read_csv(references / "osag_correlation_matrix.csv", index_col=0,
                           float_precision="round_trip").loc[list(organs), list(organs)]
        parameters = pd.read_csv(references / "sem_16group_standardized_parameters.csv",
                                 float_precision="round_trip")
        fits = pd.read_csv(references / "sem_16group_fit_indices.csv", float_precision="round_trip")
        residuals = pd.read_csv(references / "sem_16group_residual_covariance.csv",
                                float_precision="round_trip")
        full = parameters.loc[parameters["sample"].eq("full_cohort")].set_index("organ").loc[list(organs)]
        loadings = full.loading_standardized.to_numpy(float)
        theta = full.residual_variance_standardized.to_numpy(float)
        np.testing.assert_allclose(loadings ** 2, full.shared_variance_fraction, rtol=0, atol=1e-14)
        np.testing.assert_allclose(loadings ** 2 + theta, 1, rtol=0, atol=1e-13)

        # All coefficients are fixed at the saved standardized estimates.
        # Loading their covariance and evaluating Sigma performs no estimation.
        syntax = "Shared =~ " + " + ".join(f"{value:.17g}*{organ}" for organ, value in zip(organs, loadings))
        syntax += "\nShared ~~ 1*Shared\n" + "\n".join(
            f"{organ} ~~ {value:.17g}*{organ}" for organ, value in zip(organs, theta))
        fixed_model = semopy.Model(syntax)
        assert not any(parameter.active for parameter in fixed_model.parameters.values())
        fixed_model.load(cov=corr, n_samples=int(full.n.iloc[0]))
        implied = fixed_model.calc_sigma()[0]
        order = list(fixed_model.vars["observed"])
        expected_implied = pd.DataFrame(np.outer(loadings, loadings) + np.diag(theta), index=organs, columns=organs)
        np.testing.assert_allclose(implied, expected_implied.loc[order, order], rtol=0, atol=1e-13)
        calculated = corr.to_numpy() - np.outer(loadings, loadings)
        full_edges = residuals.loc[residuals["sample"].eq("full_cohort")]
        assert len(full_edges) == 21
        differences = [abs(calculated[organs.index(row.organ_1), organs.index(row.organ_2)] -
                           row.pooled_standardized_covariance_residual) for row in full_edges.itertuples()]
        assert max(differences) < 1e-12
        mask = ~np.eye(len(organs), dtype=bool)
        srmr = float(np.sqrt(np.mean(calculated[mask] ** 2)))
        saved_srmr = float(fits.loc[fits["sample"].eq("full_cohort"), "srmr"].iloc[0])
        np.testing.assert_allclose(srmr, saved_srmr, rtol=0, atol=1e-12)
        health = parameters.loc[parameters["sample"].eq("health")]
        assert len(health) == 7 and health.residual_variance_standardized.min() == 0
        assert fits.loc[fits["sample"].eq("health"), "residual_variance_min"].iloc[0] == 0
        first_component = float(np.linalg.eigvalsh(corr.to_numpy())[-1] / len(organs) * 100)
        assert round(first_component, 1) == 91.2
    finally:
        semopy.Model.fit = original_fit

    return {"dependencies": versions, "pip_check": dependency_check.stdout.strip(),
            "packaged_sem_source_imports": 5, "synthetic_rows": n,
            "parquet_and_image_roundtrip": "PASS", "duplicate_id_rejection": "PASS",
            "fixed_parameter_covariance_evaluation": "PASS", "fixed_check_model_active_parameters": 0,
            "full_cohort_residual_pairs_checked": len(differences),
            "residual_max_abs_difference": max(differences), "srmr": srmr,
            "srmr_abs_difference": abs(srmr - saved_srmr),
            "first_correlation_eigenvalue_variance_percent": first_component,
            "baseline_free_health_zero_residual_boundary_preserved": True,
            "reference_sha256": reference_hashes, "participant_data_read": False,
            "cohort_sem_refits": 0, "synthetic_model_fits": 0,
            "cohort_fit_convergence_retested": False}


def neural_check(project):
    import numpy as np
    import pandas as pd
    import torch
    import gc
    module = _runtime.load("Revisions-1030/work/r1_4_reference_20261031/train_baseline_reference.py", "repro_train_reference")
    labels, reference, _ = module.load_references()
    data, _ = module.train_data(labels, reference, 0)
    raw, model, device, optimizer, scheduler, scaler, edges, anchor = module.make_model(data)
    initial = module.training.trainable_parameter_hash(model)
    del data, optimizer, scheduler, scaler
    gc.collect()
    module.factory.restore_checkpoint(module.original_directory(0) / "checkpoint.pt", model)
    calibration = module.evaluation_data(labels, 0, "calibration")
    prediction = module.training._predict(model=model, batcher=module.batcher(calibration, "calibration"),
                                        device=device, amp=True, max_batches=1)
    ids = module.factory.prediction_ids(calibration, prediction)
    old = pd.read_parquet(project / "Revision-0710/03_ehr_age/06_codebook_v2/01_runs/shared_private_independent_view_huber1_fold0/fold1_calibration_predictions.parquet")
    old["participant_id"] = old.participant_id.astype(str)
    index = pd.MultiIndex.from_product([ids, module.primary.ORGAN_SYSTEMS], names=["participant_id", "organ_system"])
    aligned = old.set_index(["participant_id", "organ_system"]).reindex(index)
    expected = aligned.predicted_age.to_numpy(float).reshape(-1, 7)
    expected_risk = aligned.model_risk_score.to_numpy(float).reshape(-1, 7)
    np.testing.assert_allclose(prediction.predicted_age, expected, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(prediction.risk_score, expected_risk, rtol=1e-6, atol=1e-4)
    return {"participants": len(ids), "organ_outputs": 7, "initial_parameter_sha256": initial,
            "age_max_abs_difference": float(np.max(np.abs(prediction.predicted_age - expected))),
            "risk_max_abs_difference": float(np.max(np.abs(prediction.risk_score - expected_risk))),
            "training_steps": 0, "torch": torch.__version__, "device": str(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("static", "synthetic", "cox", "neural", "covariance"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, project = args.output.resolve(), args.project_root.resolve()
    if not output.is_relative_to(project) or output.is_relative_to(_runtime.PACKAGE.resolve()):
        parser.error("--output must be within --project-root and outside the source package")
    if os.environ.get("PROJECT_RUN_DIR") is not None and Path(os.environ["PROJECT_RUN_DIR"]).resolve() != output:
        parser.error("--output differs from the guarded run directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.scope == "covariance":
        os.environ.setdefault("MPLCONFIGDIR", str(output / "cache/matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(output / "cache"))
    _runtime.activate(project)
    for key in ("NUMBA_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        if "PROJECT_CACHE_DIR" in os.environ:
            os.environ[key] = str(Path(os.environ["PROJECT_CACHE_DIR"]) / key.lower())
    started = time.perf_counter()
    metadata = json.loads((_runtime.PACKAGE / "package.json").read_text(encoding="utf-8"))
    result = (static_check(metadata) if args.scope == "static" else synthetic_check(project) if args.scope == "synthetic"
              else covariance_check(project, output) if args.scope == "covariance"
              else cox_check(project) if args.scope == "cox" else neural_check(project))
    _runtime.assert_source_isolation()
    report = {"status": "PASS", "scope": args.scope, "seconds": time.perf_counter() - started,
              "python": sys.version, "adapter_sha256": hashlib.sha256((_runtime.PACKAGE / "_runtime.py").read_bytes()).hexdigest(),
              "package_sha256": hashlib.sha256((_runtime.PACKAGE / "package.json").read_bytes()).hexdigest(),
              "loaded_packaged_modules": sorted(_runtime.LOADED),
              "loaded_source_sha256": {p: hashlib.sha256((_runtime.SOURCE / p).read_bytes()).hexdigest() for p in _runtime.LOADED},
              "result": result,
              "participant_data_exported": False, "large_scale_training": False}
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("loaded_packaged_modules", "loaded_source_sha256")}), flush=True)


if __name__ == "__main__":
    main()
