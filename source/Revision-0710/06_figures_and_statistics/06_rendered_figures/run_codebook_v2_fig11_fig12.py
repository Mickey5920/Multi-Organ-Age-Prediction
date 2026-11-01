from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "Revision-0710").is_dir())
REVISION = ROOT / "Revision-0710"
OOF = (
    REVISION
    / "03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1/"
    "02_summary/oof_predictions.parquet"
)
PACKAGE = REVISION / "Final_Fig_0716/Codebook_v2_Huber1/Legacy_0626_Package"
INPUT_ROOT = REVISION / "Final_Fig_0716/Codebook_v2_Huber1/00_inputs/fig11_fig12"
SOURCE_OUTPUT = REVISION / "Final_Fig_0716/Codebook_v2_Huber1/Fig11_Fig12_Source"
FINAL_ROOT = REVISION / "Final_Fig_0716/Codebook_v2_Huber1"
SOURCE_SCRIPT = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics/"
    "05_fig11_fig12_marginal_and_review_closure/01_scripts/"
    "run_marginal_gap_associations.py"
)
RENDER_SCRIPT = HERE / "render_fig11_fig12_marginal_0716.py"
CONTEXT = (
    REVISION
    / "05_downstream/06_metric_v2/01_contextual_covariates/"
    "participant_context_covariates.parquet"
)


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_inputs() -> tuple[Path, Path]:
    wide_path = INPUT_ROOT / "canonical_age_gap_wide.parquet"
    membership_path = INPUT_ROOT / "display_group_membership.parquet"
    if not wide_path.is_file():
        oof = pd.read_parquet(
            OOF,
            columns=[
                "participant_id",
                "organ_system",
                "outer_fold",
                "corrected_age_gap",
            ],
        )
        oof["participant_id"] = oof["participant_id"].astype(str)
        participant = oof.drop_duplicates("participant_id")[["participant_id", "outer_fold"]]
        gap = oof.pivot(
            index="participant_id", columns="organ_system", values="corrected_age_gap"
        ).rename(columns=lambda value: f"{value}_linear_corrected_age_gap")
        wide = participant.merge(
            gap.reset_index(), on="participant_id", validate="one_to_one"
        )
        INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(wide_path, index=False)
    if not membership_path.is_file():
        membership = pd.read_csv(
            PACKAGE
            / "00_group_contract/revised_incident_recurrence_group_membership_long.csv.gz",
            usecols=["participant_id", "analysis_group"],
            dtype={"participant_id": str},
        )
        membership.to_parquet(membership_path, index=False)
    return wide_path, membership_path


def main() -> None:
    wide, membership = build_inputs()
    source = load(SOURCE_SCRIPT, "codebook_v2_marginal_source")
    sys.argv = [
        str(SOURCE_SCRIPT),
        "--wide",
        str(wide),
        "--membership",
        str(membership),
        "--context",
        str(CONTEXT),
        "--output-dir",
        str(SOURCE_OUTPUT),
    ]
    source.main()
    renderer = load(RENDER_SCRIPT, "codebook_v2_marginal_renderer")
    renderer.INPUT_ROOT = SOURCE_OUTPUT
    renderer.COEFFICIENTS = SOURCE_OUTPUT / "Fig12/marginal_association_coefficients.csv"
    renderer.MATRIX = SOURCE_OUTPUT / "Fig11/AllGroups_Marginal_Association_Matrix.csv"
    renderer.OUTPUT_ROOT = FINAL_ROOT
    renderer.FIG11 = FINAL_ROOT / "Fig11"
    renderer.FIG12 = FINAL_ROOT / "Fig12"
    renderer.main()


if __name__ == "__main__":
    main()
