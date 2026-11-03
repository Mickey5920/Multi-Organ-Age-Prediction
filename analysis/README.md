# Model comparisons, external validation and statistical analyses

These modules contain the computational routines for the EHR-AGE comparisons and analyses. Inputs and output locations are supplied by the caller. The clinical mapping CSVs describe variables, units and transformations; they contain no participant records.

Run the commands below from the repository root. Use Python 3.10 or 3.11 and install the analysis dependencies:

```bash
python -m pip install -r analysis/requirements.txt
```

For neural models, also install the root `requirements.txt` (PyTorch 2.5.1). `analysis/requirements.txt` pins the direct analysis dependencies; it is not a complete operating-system or CUDA lock. Module checks can use CPU. The reported neural training used CUDA. Numerical results also depend on the source data, participant order, fold assignment, transforms, embeddings and random seeds.

## 1. Model comparisons

`backbones.py` provides RETAIN, BEHRT-style, Med-BERT-style and EHRFormer-style encoders. `comparison_models.py` contains their objective-matched adapter and the MLP/CNN/ResNet/Transformer comparison architectures. `losses.py`, `training.py`, and `schedule.py` contain the joint age/survival objective, training/prediction routines and learning-rate schedule.

The command-line comparator uses the same `data.npz` and `clip_embeddings.npz` formats described in the root README: 78 phenotypes, missingness, sex, 32 pre-index event slots, chronological age, reference membership and seven organ outcome arrays. See `phenotype_columns.csv` for the ordered phenotype names. Keep participant ordering consistent across all arrays.

```bash
python -m analysis.compare --data data/data.npz --embeddings data/clip_embeddings.npz --model medbert_style --outer-fold 0 --output outputs/medbert_fold0
```

Repeat for outer folds 0–4 and the four model names listed above. For outer fold `k`, fold `(k+1)%5` is used for calibration and the other three folds for training. Fit imputation/scaling on training participants only. `--transform data/fold0_transform.npz` reuses an existing training-fitted transform with fields `median`, `mean`, `std` (each length 78), `outer_fold` and `validation_fold`. The fixed-epoch settings and objective weights are in `comparison_config.json`.

`--validate-only --device cpu` checks the supplied inputs and performs a forward pass without training or writing predictions. Normal execution creates a new output directory containing model weights, the phenotype transform, and calibration/test prediction NPZs. Prediction fields are `participant_indices` (row indices into the input data), `chronological_age`, `strict_healthy`, `duration_months`, `event`, `at_risk`, `sex`, `predicted_age` and `risk_score`. Organ arrays have shape `[N,7]` and use the seven-organ order in the root README. Risk scores are the model's direct survival-head outputs, not calibrated event probabilities.

## 2. Clinical/EHR Cox reference

`clinical_reference.py` fits the clinical/EHR reference and the same model with seven OSAGs. Provide one organ and one outer-fold evaluation per call, with separate Parquet tables for calibration participants (`--fit`) and test participants (`--test`).

Required columns:

| Columns | Meaning |
| --- | --- |
| `participant_id`, `organ_system`, `outer_fold` | Unique participant within the selected organ; outer evaluation index 0–4 in both tables |
| `split_role` | `validation` for fitting/calibration rows; `test` for evaluation rows |
| `duration`, `event` | Positive follow-up in months, capped at 60; binary organ event; include at-risk participants only |
| The 21 names in `clinical_reference.CONTEXT` | Age, sex, smoking, education, deprivation, employment, sleep and record-density variables, with their missingness indicators |
| `phenotype_00` … `phenotype_77` | Phenotypes transformed using the corresponding neural training-fold medians, means and scales |
| `phenotype_missing_00` … `phenotype_missing_77` | Original binary missingness indicators |
| `medbert_risk__<organ>` | Seven objective-matched Med-BERT risk scores from the corresponding outer model |
| `gap__<organ>` | Seven EHR-AGE age gaps corrected using calibration participants |

Organ names are `cardiovascular`, `hepatic`, `immune`, `metabolic`, `musculoskeletal`, `pulmonary`, `renal`. Join inputs by participant identifier and preserve the same participants in the reference and augmented analyses. Outcome data are separate from the predictor features.

```bash
python -m analysis.clinical_reference --fit data/calibration.parquet --test data/test.parquet --seed 710 --output outputs/clinical_cox
```

The seed is supplied explicitly; use the original organ/fold seed when reproducing a saved comparison. Three-fold event-stratified cross-validation within fitting participants chooses the Cox penalty from 0.0001, 0.001, 0.01 and 0.1 using mean Harrell C. Ties within 1e-12 choose the larger penalty. Clinical-context medians and Cox standardization are estimated separately in each inner fitting subset, then refitted on the calibration participants. The test participants are used only for evaluation. Outputs include predictions, survival probabilities, selected-penalty CV values and censoring-aware metrics.

`statistics/survival_metrics.py` provides time-dependent AUC, Brier score and integrated Brier score with censoring estimated from fitting participants. `statistics/clinical_utility_metrics.py` provides Harrell C, IPCW reclassification, calibration and decision-curve functions. Requested reporting horizons default to 12, 24 and 60 months; the integrated-Brier grid is monthly. Administrative endpoints use the representable instant immediately before the upper follow-up boundary.

## 3. External preprocessing and evaluation

`external_preprocessing.py` operates on MIMIC-IV hospital tables and NHANES transport files supplied as a dictionary of `Path` objects. `mappings/` contains the 78-slot feature maps, ICD-10 organ definitions and questionnaire outcome rules. The maps specify the exact source variables, units, conversion rules and missingness handling.

MIMIC input dictionary keys are `mimic_patients`, `mimic_admissions`, `mimic_diagnosis_dictionary`, `mimic_labevents`, `mimic_omr`, and, for the prescription sensitivity analysis, `mimic_prescriptions`. Values point to the corresponding `patients.csv.gz`, `admissions.csv.gz`, `d_icd_diagnoses.csv.gz`, `labevents.csv.gz`, `omr.csv.gz`, and `prescriptions.csv.gz` files. The raw diagnosis table `diagnoses_icd.csv.gz` is passed separately to the pre-index filter.

```python
from pathlib import Path
from analysis import external_preprocessing as prep

hospital = Path('data/mimic/hosp')
sources = {key: hospital / filename for key, filename in {
    'mimic_patients': 'patients.csv.gz',
    'mimic_admissions': 'admissions.csv.gz',
    'mimic_diagnosis_dictionary': 'd_icd_diagnoses.csv.gz',
    'mimic_labevents': 'labevents.csv.gz',
    'mimic_omr': 'omr.csv.gz',
}.items()}
cohort, admissions, counts = prep._participant_cohort(sources)
preindex = Path('outputs/preindex_diagnoses.parquet')
prep._materialize_preindex_diagnosis_allowlist(
    cohort, admissions, hospital / 'diagnoses_icd.csv.gz', preindex)
events, event_counts = prep._mimic_diagnosis_events(cohort, admissions, sources, preindex)
raw, missing, raw_unit_sensitivity, missing_unit_sensitivity, coverage = (
    prep._build_mimic_features(cohort, Path('analysis/mappings'), sources))
```

Only completed admissions strictly before the prediction landmark enter the EHR event sequence. Lab and OMR features enforce their measurement/availability timing rules. `external_protocol.prepare_variant_timeline` and `crop_variant_timeline` order, deduplicate and select events before `_pack_events` constructs the left-aligned 32-slot model inputs. `_pack_events` takes a code-to-ID dictionary and a `(text_role, description)`-to-ID dictionary consistent with the frozen embedding tables. Keep the training vocabulary prefix unchanged when appending external terms. Missing phenotype slots use zero in the raw array plus `True` in the missingness mask, then the applicable training-fitted `PhenotypeTransform.apply` fills and scales them.

For NHANES, use `_build_nhanes_features(Path('analysis/mappings'), sources)`. It returns participant metadata, raw phenotypes, missingness and coverage. Dictionary keys `nhanes_p_demo` and `nhanes_l_demo` identify the demographic XPT files for the 2017–March 2020 pre-pandemic and August 2021–August 2023 releases. Additional keys are the `source_file_id` values in `NHANES_78_SLOT_RELEASE_MAP.csv`, and the `source_file_ids` in `NHANES_QUESTIONNAIRE_LABEL_MAP.csv` for outcome construction. For example, `nhanes_p_bmx` identifies the pre-pandemic body-measures file. Supply all mapped files for each release. Use `_build_nhanes_questionnaire_labels` separately from predictor construction; questionnaire outcomes are not model inputs. The NHANES model input has no EHR events, so all event arrays are padding.

`support.apply_support` contains the input-support adaptation used on UKB training/calibration/test data. Its support DataFrame has 78 rows in slot order, with columns `slot_index`, `available` (boolean), `observed_fraction` (0–1), and `event_mode` (`hospital_icd_only` or `none`). Fractions come from external feature availability, without outcomes. For NHANES use the intersection of available features across both releases and their minimum observed fractions. The function acts in place on a role-specific `Fold0DevelopmentData` object; use independent copies for each role. It masks unsupported phenotypes, thins supported observations to the target availability, and removes unsupported event types. It retains the source training-fold phenotype transform. Network training remains on UKB data.

For external predictions, apply each corresponding frozen model and training transform to its external inputs. Use `training._predict` for the prediction interface. `external_protocol.fit_crossfit_age_calibration` returns coefficients for `age ~ 1 + predicted_age + sex`; fit these outside the evaluation fold and apply them to that fold. Preserve participant folds across all compared routes.

`external_protocol.py` supplies endpoint adjudication, Aalen–Johansen incidence, Harrell C, IPCW weights, AUC/Brier and Benjamini–Hochberg adjustment. `external_metrics.py` supplies cause-specific Cox/CIF prediction, weighted age metrics, calibration and NHANES survey-weighted logistic inference. External endpoint times are in days; use `months_to_days` for horizons. Cox evaluation frames use `observed_time_days`, `event_indicator`, `competing_death_indicator` and the requested numeric covariates. NHANES survey inference takes outcome/design arrays, examination weights, strata and PSU labels, plus the full survey-design strata/PSU arrays for domain inference.

## 4. Paired statistical comparisons

`statistics.paired_bootstrap.paired_participant_bootstrap` aligns model predictions by participant/organ/fold, resamples participants jointly within folds, and accepts a metric callback. The same draw is applied to both models. Set `improvement_direction` to `higher` for C/AUC and `lower` for Brier-type errors.

`paired_external.py` performs the MIMIC paired comparison, seven-organ macro averages and adapted-minus-direct contrasts:

```bash
python -m analysis.paired_external --direct data/direct_predictions.parquet --adapted data/adapted_predictions.parquet --output outputs/external_paired
```

Both Parquet files contain one row per participant/organ/comparison at each horizon. Required columns are `subject_id`, `organ_system`, `horizon_months`, `comparison`, `outer_fold`, `observed_time_days`, `event_indicator`, `competing_death_indicator`, `ipcw_outcome`, `ipcw_weight`, `ipcw_observed`, `base_risk`, `augmented_risk`, `base_probability`, and `augmented_probability`. The command selects 12 months. `comparison` identifies the reference covariate set; use identical values in both files. Risk scores support Harrell C; probabilities support IPCW AUC and Brier. IPCW outcome/weights must be computed from the common endpoint definition with censoring fitted outside the evaluation fold. Tables must agree on participants, outcomes, folds and weights. By default, 1,000 participant draws use seed 20261031. Positive Brier improvement means baseline error minus augmented error. CSV outputs report organ estimates, equal-weight seven-organ macro estimates, paired transfer contrasts and 95% percentile intervals.

## 5. Shared-factor SEM

`sem.py` provides the one-factor analysis using semopy 2.3.11:

```python
import pandas as pd
from analysis.sem import fit_one_factor

osags = pd.read_csv('data/osags.csv')
result = fit_one_factor(osags)
```

The input has one participant per row and seven finite numeric columns named by organ, containing corrected OSAGs. `standardize` uses within-group population standard deviations (`ddof=0`). `fit_one_factor` uses DWLS with SLSQP and returns standardized loadings, residual variances/correlations and model-fit statistics. Apply the same function to each prespecified group; `sample_frame` supports a separate `participant_id`, `analysis_group` membership table. This is a descriptive analysis of shared and residual variation.

## Verification and data access

The reusable numerical definitions were checked against the existing analysis source. Packaging checks use synthetic arrays and tables; they do not rerun the study. Access to study data remains subject to the data providers' conditions. Supply authorized data and any required trained checkpoints locally. All example paths above are relative placeholders; local input data, trained weights and generated outputs are not part of this code directory.

The repository's MIT license applies to these additions. Citations and upstream implementation notes are retained in the model code.
