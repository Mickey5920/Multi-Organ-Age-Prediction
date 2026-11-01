from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("build_icd_codebook_v1", HERE / "build_codebook.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("icd_codebook_v1")
    MODULE.build(
        output,
        ROOT / "Submit" / "EHR-AGE-Supplement Table.xlsx",
        ROOT
        / "Revision-0710"
        / "01_data_rebuild"
        / "01_index_and_events"
        / "icd_mapping_export.csv",
        ROOT
        / "Revision-0710"
        / "01_data_rebuild"
        / "01_index_and_events"
        / "observed_icd_code_system_matches.csv",
    )
    return output


def test_submission_snapshot_counts(built: Path) -> None:
    raw = pd.read_csv(built / "01_v0_submission" / "s1_system_icd_raw.csv", keep_default_na=False)
    mapping = pd.read_csv(built / "01_v0_submission" / "mapping.csv", keep_default_na=False)
    assert len(raw) == 1818
    assert len(mapping) == 1818
    assert mapping.groupby("organ_system").size().to_dict() == {
        "cardiovascular": 434,
        "hepatic": 71,
        "immune": 130,
        "metabolic": 156,
        "musculoskeletal": 548,
        "pulmonary": 296,
        "renal": 183,
    }


def test_primary_only_repairs_malformed_literal(built: Path) -> None:
    current = pd.read_csv(
        built / "00_source_snapshots" / "current_revision0710_mapping_export.csv",
        keep_default_na=False,
    )
    primary = pd.read_csv(
        built / "02_v1_corrected_primary" / "mapping.csv", keep_default_na=False
    )
    diff = pd.read_csv(built / "04_audit" / "old_new_diff.csv", keep_default_na=False)
    assert len(current) == 1849
    assert len(primary) == 1850
    assert set(map(tuple, diff[["change_type", "organ_system", "icd_prefix_normalized"]].values)) == {
        ("removed", "renal", "N299C64"),
        ("added", "renal", "N299"),
        ("added", "renal", "C64"),
    }
    renal = set(primary.loc[primary["organ_system"].eq("renal"), "icd_prefix_normalized"])
    assert "N299C64" not in renal
    assert {"N299", "C64"}.issubset(renal)


def test_primary_preserves_frozen_assignments(built: Path) -> None:
    primary = pd.read_csv(
        built / "02_v1_corrected_primary" / "mapping.csv", keep_default_na=False
    )

    def systems(prefix: str) -> set[str]:
        return set(primary.loc[primary["icd_prefix_normalized"].eq(prefix), "organ_system"])

    assert systems("C49") == {"cardiovascular", "musculoskeletal"}
    for prefix in ["M80", "M81", "M83"]:
        assert systems(prefix) == {"metabolic", "musculoskeletal"}
    for prefix in ["M05", "M06", "M08", "M30", "M36"]:
        assert systems(prefix) == {"musculoskeletal"}


def test_audits_and_manifest(built: Path) -> None:
    malformed = pd.read_csv(
        built / "04_audit" / "malformed_prefix_audit.csv", keep_default_na=False
    )
    duplicates = pd.read_csv(
        built / "04_audit" / "exact_duplicate_audit.csv", keep_default_na=False
    )
    multimap = pd.read_csv(
        built / "04_audit" / "exact_multimap_audit.csv", keep_default_na=False
    )
    manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    assert malformed.loc[
        malformed["version"].eq("current_revision0710"), "icd_prefix_normalized"
    ].tolist() == ["N299C64"]
    assert not malformed["version"].eq("v1_corrected_primary").any()
    assert duplicates.empty
    assert int(multimap["version"].eq("v1_corrected_primary").sum()) == 31
    assert manifest["status"] == "PASS"
    assert manifest["counts"]["old_new_added_rows"] == 2
    assert manifest["counts"]["old_new_removed_rows"] == 1
    assert "02_v1_corrected_primary/mapping.csv" in manifest["outputs"]


def test_observed_code_delta_restores_c64(built: Path) -> None:
    delta = pd.read_csv(
        built / "04_audit" / "observed_code_classification_delta.csv", keep_default_na=False
    )
    assert delta.to_dict(orient="records") == [
        {
            "icd_code_normalized": "C64",
            "current_mapped_organ_systems": "",
            "primary_mapped_organ_systems": "renal",
            "added_organ_systems": "renal",
            "removed_organ_systems": "",
            "change_type": "newly_mapped",
        }
    ]


def test_manifest_hashes_match_current_files(built: Path) -> None:
    manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in manifest["outputs"])
    for relative_path, identity in manifest["outputs"].items():
        path = built / relative_path
        assert path.stat().st_size == identity["bytes"]
        assert MODULE.sha256_file(path) == identity["sha256"]


def test_sensitivity_is_independent_and_single_assignment(built: Path) -> None:
    primary = pd.read_csv(
        built / "02_v1_corrected_primary" / "mapping.csv", keep_default_na=False
    )
    sensitivity = pd.read_csv(
        built / "03_sensitivity" / "mapping_single_assignment.csv", keep_default_na=False
    )
    excluded = pd.read_csv(
        built / "03_sensitivity" / "exclusions_from_primary.csv", keep_default_na=False
    )
    assert len(primary) == 1850
    assert len(excluded) == 41
    assert len(sensitivity) == 1809
    assert not (
        sensitivity["organ_system"].eq("cardiovascular")
        & sensitivity["icd_prefix_normalized"].str.startswith("C49")
    ).any()
    assert not (
        sensitivity["organ_system"].eq("metabolic")
        & sensitivity["icd_prefix_normalized"].str.startswith(("M80", "M81", "M83"))
    ).any()
    assert set(
        sensitivity.loc[sensitivity["icd_prefix_normalized"].eq("M05"), "organ_system"]
    ) == {"musculoskeletal"}
