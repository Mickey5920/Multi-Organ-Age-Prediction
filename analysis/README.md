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

## 6. Follow-up functional and biochemical associations

`functional_validation.py` evaluates three follow-up measurements for each of the seven organs (21 measurements), each against the full OSAG and the organ-relative component. It uses saved five-fold participant predictions, adjusts the corresponding baseline measurement, and reports all 42 associations with 95% confidence intervals, Benjamini–Hochberg q values and Holm-adjusted P values. All results, including null and opposite-direction estimates, remain in the output. Related endpoints are not independent replications.

Install the smaller environment for this analysis, or use the matching packages in your existing analysis environment:

```bash
python -m pip install -r analysis/requirements-functional.txt
python -m analysis.functional_validation --self-test
```

The numerical analysis used Python 3.12.4. The dependency file records the versions used for these routines. Only NumPy, pandas, SciPy, statsmodels, PyArrow and threadpoolctl are needed; this analysis does not import PyTorch or train neural models.

### Inputs

Supply eight local files. CSV identifiers are strings named `Participant ID`; Parquet identifiers are strings named `participant_id`. IDs must be unique in participant tables. Dates must be parseable calendar dates. Missing measurements are blank/NaN; they are not replaced with zero. Organ names, in model order, are `cardiovascular`, `hepatic`, `immune`, `metabolic`, `musculoskeletal`, `pulmonary`, and `renal`.

| Flag | File and required contents |
|---|---|
| `--phenotypes` | UK Biobank wide CSV containing the measurement columns listed below, one participant per row. Extra columns are allowed. |
| `--assessment-dates` | CSV with `Participant ID` and `Date of attending assessment centre \| Instance 0`, `Instance 1`, and `Instance 2`. These are visit dates, not assay-processing dates. |
| `--walking-pace` | CSV with `Participant ID` and `Usual walking pace \| Instance 0` and `Instance 2`. Values are `Slow pace`, `Steady average pace`, `Brisk pace`, `None of the above`, `Prefer not to answer`, or missing. |
| `--predictions` | Parquet with one row per participant and organ: `participant_id`, `organ_system`, `actual_age`, `sex`, `outer_fold`, `raw_age_gap`, `corrected_age_gap`, `model_risk_score`. Sex is 0=female/1=male; folds are 0–4. Each person must have all seven organs, with predictions from that person's evaluation fold, not an ensemble that trained on that person. |
| `--covariates` | Participant Parquet: `participant_id`, `fold_id`, `chronological_age`, `sex_male`, `smoking_current`, `smoking_missing`, `phenotype_missing_fraction`, `log1p_preindex_icd_count`, `log1p_preindex_gp_count`, `has_preindex_icd`, `has_preindex_gp`. Binary fields use 0/1. Missing smoking requires `smoking_missing=1`. The five record-density variables use baseline/pre-index records only. |
| `--baseline-labels` | Participant Parquet with `participant_id`, `index_date`, and `prevalent_{organ}_count` plus `has_prevalent_{organ}` for each of the seven organs. Counts must be nonnegative and agree with the corresponding indicators. Baseline burden is derived from these columns. |
| `--cohort` | Participant Parquet with `participant_id` and `index_date`; index dates must match assessment Instance 0. |
| `--folds` | Participant Parquet with `participant_id` and `fold_id` (0–4), agreeing with the covariate table and prediction `outer_fold`. All participant Parquet tables must describe the same cohort. |

For the phenotype CSV, use UK Biobank descriptive headers with the exact pattern `Measure | Instance i`, and, where applicable, `| Array a`:

- Instances 0 and 1: `Albumin`, `Alanine aminotransferase`, `Gamma glutamyltransferase`, `C-reactive protein`, `White blood cell (leukocyte) count`, `Neutrophill count`, `Lymphocyte count`, `Glycated haemoglobin (HbA1c)`, `Triglycerides`, `HDL cholesterol`, `Cystatin C`, `Creatinine`, and `Urea`. The spelling `Neutrophill` follows the source field. The baseline serum albumin header is exactly `Albumin | Instance 0(participant - p30600_i0)`; follow-up is `Albumin | Instance 1`. The similarly named metabolomics field must not be substituted.
- Instances 0 and 2, Arrays 0 and 1: `Systolic blood pressure, automated reading`, `Diastolic blood pressure, automated reading`, and `Pulse rate, automated reading`.
- Instances 0 and 2, Arrays 0–2: `Forced expiratory volume in 1-second (FEV1)` and `Forced vital capacity (FVC)`.
- Instances 0 and 2: `Hand grip strength (left)`, `Hand grip strength (right)`, `Heel bone mineral density (BMD) (left)`, and `Heel bone mineral density (BMD) (right)`.
- Instance 0: `Standing height`.

Units are mmHg for pressure, beats/min for pulse, kg for grip, cm for height, L for raw spirometry, g/cm² for heel BMD, g/L for albumin, U/L for ALT/GGT, mg/L for CRP/cystatin C, 10⁹ cells/L for leukocytes, mmol/mol for HbA1c, mmol/L for triglycerides/HDL/urea, and µmol/L for serum creatinine. No unit conversion is inferred from values. Spirometry effects are reported in mL, and FEV1/FVC differences in percentage points.

### Measurement and analysis rules

The script uses paired Instance 0/1 biochemical measurements and Instance 0/2 physical or walking measurements. It requires two paired blood-pressure readings, both hands for grip, and both heels for ultrasound BMD at each visit. Positive finite laboratory values are retained without winsorization. ALT, GGT, CRP, leukocytes, neutrophil-to-lymphocyte ratio, triglycerides, cystatin C, creatinine and urea are analysed on natural-log scales at both visits; reported effects are `100*(exp(beta)-1)` percent differences in conditional geometric means. Grip retains valid zeros. Slow pace is compared with average/brisk pace using logistic regression, with baseline pace categories included.

For heel BMD, bilateral means at both visits must be strictly within (0.18, 1.06) g/cm² for males and (0.12, 1.025) for females, following the numerical bounds in [Kim, PLoS ONE (2018)](https://doi.org/10.1371/journal.pone.0200785). This is an ultrasound measure and a numerical screen, not DXA or full instrument-quality certification.

The reported spirometry analysis uses numerical repeatability because acceptability field 3061 was unavailable. Explicitly pass `--numeric-repeatability-only`. At both visits it selects positive finite paired FEV1/FVC values with FEV1 ≤ FVC, requires at least two usable blows, and requires the largest two FEV1 and FVC values each to differ by ≤0.15 L (≤0.10 L if maximum selected FVC ≤1 L). Each outcome uses the maximum selected value; the ratio uses the two maxima. These checks do not certify manoeuvre acceptability. See [UK Biobank field 20152](https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=20152).

The model offset is `8*tanh(model_risk_score/2)` in years. The relative component is the target offset minus the mean of the other six offsets. The shared base gap is the mean of `raw_age_gap - offset` across organs. `--offset-cap-years` and `--risk-temperature` must match the model producing the predictions; the defaults reproduce the reported model. Full OSAG uses `corrected_age_gap` directly. Exposure standardization uses the analytic sample's population standard deviation (`ddof=0`). Relative-component models additionally adjust the shared base gap and mean offset.

All models adjust baseline measurement, age, sex, follow-up interval, baseline burden, record density and fold indicators. Pulmonary models also adjust baseline height and current smoking with its missingness indicator. Continuous outcomes use HC3 standard errors; slow walking uses HC0 sandwich standard errors. Confidence intervals are pointwise and condition on the saved neural predictions and correction. Five within-fold fits describe sign consistency; all 42 pooled tests form the multiplicity family. These follow-up associations are not aging-rate estimates or independent validation of seven biological aging clocks.

### Run

```bash
python -m analysis.functional_validation \
  --phenotypes data/phenotypes_all_visits.csv \
  --assessment-dates data/assessment_dates.csv \
  --walking-pace data/walking_pace.csv \
  --predictions data/evaluation_predictions.parquet \
  --covariates data/baseline_covariates.parquet \
  --baseline-labels data/baseline_organ_labels.parquet \
  --cohort data/cohort.parquet --folds data/folds.parquet \
  --numeric-repeatability-only --output outputs/functional_associations
```

Append `--validate-only` to check schema, dates, participant alignment and fold identity without fitting or writing output. Normal execution requires a new or empty output directory and writes aggregate association tables, fold summaries, coefficients, sample flow, exposure scaling and measurement definitions. It does not export participant rows. Use all 42 pooled estimates when reproducing the reported multiple-testing adjustments. Supply authorized data locally under the data providers' access conditions.
