# EHR-AGE: study code and reproducibility guide

This repository contains the EHR-AGE second-round reproducibility source snapshot, version **r2-20261101**. It includes the study implementations, model and analysis configurations, clinical variable mappings, and selected aggregate reference results. The distribution root is the repository root. Git tags and the corresponding exact commit identify source versions; publication availability is determined by the repository's remote refs and Releases, separately from local source verification.

The package does **not** contain UK Biobank or MIMIC participant records, participant-level predictions, trained checkpoints, tensor caches, pretrained CLIP weights, credentials, or access tokens. Researchers must independently obtain the necessary data access and prepare the inputs below. Code access does not confer access to any cohort. NHANES data are obtained separately as well.

**License:** the research code and accompanying documentation in this distribution are licensed under the [MIT License](LICENSE), copyright (c) 2026 EHR-AGE contributors. Dependencies, pretrained models and cohort data retain their own terms; this code license does not grant access to participant data or rights to third-party materials. No third-party Python implementation is vendored; install dependencies separately.

**Verification status:** the source snapshot has been checked for integrity and syntax, and the preprocessing/path interfaces have been exercised with invented rows. The new explicit-Array ICD entry passed 12 synthetic tests and a read-only comparison covering all 502,163 participants and 7,071,243 existing long-table rows, with no diagnosis/description/date differences. It did not write participant data or rebuild the study long table. The complete SEM dependency set was installed in a new isolated Windows environment and checked through packaged-source imports, synthetic file I/O and numerical reconciliation of existing aggregate results, with no SEM fitting. Earlier bounded checks reproduced selected frozen predictions and statistics in the authors' existing environments. A complete raw-data rebuild, five-fold retraining and all downstream analyses in fresh environments have **not** been performed for this release. Windows was used for the local checks. Linux/macOS execution remains unverified.

## Versions and entry points

- `before-round2` identifies the earlier public snapshot at commit `e5613bd9a59d78dd81148f913255605b2623acdd`. Its root-level `Dataloader.py`, `model.py`, `train.py`, `main.py` and `EHR-AGE_architecture.png` are preserved unchanged.
- `r2-20261101` identifies this expanded reproducibility source snapshot. Use `run.py --list` and the `source/` implementations for the study pipeline, plus the standalone preprocessing and figure entries below. The earlier root-level `main.py` is a historical interface, not the entry point for the full revision workflow.
- [CHANGELOG.md](CHANGELOG.md) describes the version scope. Record the full Git commit SHA together with the tag when citing or reproducing a snapshot. The local package SHA256 identity is a separate content identifier and does not substitute for a public commit or release link.

After a version is published, keep its tag fixed. Make later corrections in a new commit and a new version tag. The `.gitattributes` file disables line-ending conversion so Git preserves the byte-identical scientific source and its hash manifest across operating systems; ordinary text diffs remain available.

For an existing Git checkout, resolve the version's exact commit with:

```bash
git rev-parse "r2-20261101^{commit}"
```

Source maintainers can refresh hashes for **already-listed delivery files**, then check the package and staged Git contents:

```bash
python -B tools/manage_release.py --refresh
python -B tools/manage_release.py --check
# After staging the intended files, verify the full index and exact blob bytes:
python -B tools/manage_release.py --check-index
```

Refresh refuses changed retained scientific source and never discovers or adds files automatically. Any intentional new file must be explicitly included in `package.json`. These commands neither commit nor push. Publication and the actual accessible release citation remain separate actions.

## Contents and entry points

| Path | Purpose |
| --- | --- |
| `source/Revision-0710/01_data_rebuild/` | Cohort construction, strictly pre-index events, outcomes, folds, phenotype transforms and compact caches |
| `source/Revision-0710/03_ehr_age/` | Shared/private model, objectives, roles, training, inference, age correction, fixed five-fold model and support adapters |
| `source/Revision-0710/02_baselines_5fold/` | Implemented baseline/comparator training and cache loaders |
| `source/Revision-0710/05_downstream/` | Context covariates, outcome evaluation and baseline registries |
| `source/Revision-0710/06_figures_and_statistics/` | Clinical utility, paired uncertainty, external validation and figure/statistic implementations |
| `source/Revision-0710/Final_Fig_0716/Reviewer_Statistics/` | Recorded disease burden, density, imputation, survival and other sensitivity analyses |
| `source/Revision-0710/Final_Figure_0725/` and `Final_Figure_0728/` | Organ alignment and related descriptive decomposition implementations |
| `source/Revision-0730/0-Others/Fig7_Shared_Component_SEM/scripts/` | Shared-factor covariance/residual atlas, estimator sensitivity, diagnostics and component exports; historical filenames retained |
| `source/Revisions-1030/work/r1_1_information_matched_20261031/` | Information-matched clinical comparison |
| `source/Revisions-1030/work/r1_3_external_20261031/` | External paired participant bootstrap |
| `source/Revisions-1030/work/r1_4_reference_20261031/` | Baseline-only training-reference sensitivity and evaluation |
| `source/Revisions-1030/work/r1_2_5_reproducibility_20261031/harmonize_statistics.py` | Pooled age metrics and numerically stable HR inference |
| `configs/` | Editable examples for private input locations |
| `environments/` | Observed Python/dependency versions and corresponding direct-version requirements |
| `reference/` | Selected aggregate numerical reference tables, including four existing SEM summary tables in `reference/sem`; these are not participant input data |
| `figures/render_figure7.py` | Standalone aggregate-only Figure 7 reconstruction and optional preservation of the submitted vector layout; paired macro intervals are read directly from the formal results |
| `preprocessing/prepare_icd_long.py` | Upstream ICD long-table preparation by participant and original Array index, plus a read-only comparison mode |
| `preprocessing/test_prepare_icd_long.py` | Synthetic regression tests for Array gaps, key/column order, duplicate keys, differences and preservation of existing files |
| `package.json` | Source hashes, delivery-file hashes, entry points and verification boundaries |
| `tools/manage_release.py`, `.gitattributes`, `CHANGELOG.md` | Explicit release-catalog checks, byte-preserving Git settings and version history |
| `run.py`, `_runtime.py`, `_child.py` | Explicit-root launcher and path/import adapter |
| `verify.py` | Static, synthetic and optional restricted-artifact checks |

Historical directory names are retained so source dependencies remain traceable. Historical configurations/manifests describe particular completed runs; their existence in this package does not provide the private data or checkpoints for a new run.

The retained study source and mapping/reference files are copied byte for byte. Portability changes are confined to the delivery adapter and launcher: resolving project roots, locating packaged imports and input paths, choosing subprocess interpreters, and handling the Windows subprocess flag. The additional upstream ICD entry and Figure 7 wrapper are separate delivery files; the ICD entry has been checked against the original study input without replacing it. Model calculations, loss functions, statistical estimators, seeds, scientific gates and original-cohort hash checks have not been changed.

## Environments

Use separate environments. The statistics, Cox, neural and embedding profiles record **observed direct versions**, without a fresh installation test of their complete dependency sets. The covariance profile additionally provides a fully pinned installed-distribution lock tested in a newly created environment. Do not combine all requirement files into one environment. The interpreter must have the Python version appropriate to its profile.

The upstream ICD entry and its tests use only the Python standard library; no pip packages are needed for this stage. They were checked with CPython 3.12.4 on Windows. Use the statistics interpreter for the commands below.

| Profile | Python | Use |
| --- | --- | --- |
| `statistics` | 3.12.4 | Raw manifests, event/outcome/fold construction, non-embedding caches, paired uncertainty, aggregate/statistical summaries |
| `cox` | 3.10.20 | Information-matched Cox models, clinical utility and reference-sensitivity Cox fits |
| `neural` | 3.9.25 | Neural training/inference and UKB support adaptation; recorded PyTorch 2.5.1, CUDA 11.8 |
| `embedding` | 3.10.19 | OpenCLIP text-table generation; open-clip-torch 3.3.0 and PyTorch 2.5.1 |
| `figures` | 3.12.4 | Figure 7 aggregate rendering; observed NumPy 1.26.4, pandas 2.2.2, Matplotlib 3.8.4 and PyMuPDF 1.27.2.3, checked in an existing environment |
| `covariance` | 3.11.11 | Isolated installation with semopy 2.3.11, Polars 1.31.0, PyArrow 21.0.0 and Pillow 11.1.0; packaged-source/I/O and saved-aggregate checks passed, with no cohort fit |

For an already created environment, install that profile's direct requirements, for example:

```bash
/path/to/statistics/python -m pip install -r environments/requirements-statistics.txt
/path/to/cox/python -m pip install -r environments/requirements-cox.txt
/path/to/neural/python -m pip install -r environments/requirements-neural.txt
/path/to/embedding/python -m pip install -r "environments/requirements-embedding-table generation only.txt"
```

The neural profile records the authors' CUDA-enabled PyTorch build. A plain pip installation is not evidence of the same CUDA build, driver, GPU kernels or numerical behavior; check `torch.__version__` and `torch.version.cuda` in the resulting environment. Full training used an NVIDIA RTX 4090. CPU-only inference/training and alternative accelerators have not been validated here. GPU/library changes can prevent bitwise equality even with the same seed.

Figure/Excel-only helpers may additionally import Pillow, openpyxl, seaborn or PyMuPDF. They are separate from the core raw-data and model stages; no third-party implementation is vendored. The covariance profile is a replacement verification environment, not a reconstruction of the historical Python 3.8.19 SEM environment.

The SEM environment is now complete for the supplied covariance sources: **11 direct dependencies** are pinned in `requirements-covariance.txt`; **29 installed distributions**, including transitive dependencies and packaging tools, are pinned in `requirements-covariance.lock.txt`. The existing SEM numerical-library versions and overlapping transitive versions were retained. The newly added data readers are [Polars 1.31.0](https://pypi.org/project/polars/1.31.0/) and [PyArrow 21.0.0](https://pypi.org/project/pyarrow/21.0.0/); Pillow 11.1.0 matches the original environment. The lock describes **CPython 3.11.11, Windows AMD64**, not an asserted cross-platform binary lock or the exact historical Python 3.8.19 environment.

From the distribution root, an equivalent isolated Windows environment can be prepared as follows. Select an existing **Python 3.11.11** interpreter and a new environment destination. Set installation cache/temp locations to your own project directories before installing.

```powershell
$BasePython = "D:/path/to/Python-3.11.11/python.exe"
$SemEnvironment = "D:/Conda/envs/ehr-age-sem"
& $BasePython -m venv $SemEnvironment
$SemPython = "$SemEnvironment/Scripts/python.exe"
& $SemPython -m pip install pip==25.0 setuptools==75.8.0 wheel==0.45.1
& $SemPython -m pip install --no-build-isolation -r environments/requirements-covariance.lock.txt
& $SemPython -m pip check
```

This environment passed installation, dependency consistency, imports of all five supplied SEM/component source files, and the bounded `verify.py --scope covariance` check below. Study-data optimizer convergence, alternative estimator fits and full figure regeneration were not rerun. No packages were added to the original SEM environment.

## Filesystem setup and path mapping

The package location and the private reproduction project must be separate. Use a fresh project/run workspace; avoid pointing any new training command at an original archived run directory.

```text
/path/to/Multi-Organ-Age-Prediction/       # this code package
/data/ehr-age-reproduction/
    private-input-map.json             # edited private copy of the example
    ukb_source_paths.json              # edited private copy of the source-path example
    runs/from_raw/                     # generated study tree, denoted BUILD below
/data/private/ukb/                      # approved exports; not copied into this package
/data/private/mimic-iv-3.1/hosp/         # original MIMIC tables
/data/nhanes/                           # the two specified NHANES releases
```

Copy `configs/ukb_source_paths.example.json` and `configs/input-map.example.json` to your **private project**, then replace every example data location with an absolute path. Use JSON forward slashes on Windows (`E:/...`) or correctly escaped backslashes. Relative mapping keys refer to the selected logical project root; absolute values point to accessible inputs. Prefix mappings apply to reads and use the longest matching prefix. Outputs are not redirected by an input map.

The example `icd_mapping_script` retains an original project path as a **logical alias**. The adapter resolves that alias to the packaged `Cox-0621/0-3-1-Filter-Health and Disease Organ Group.py`; it does not need access to the authors' E: drive. Other original absolute data paths, including the four `Association` files, require the explicit input-map entries. The `--raw-icd` option is supplied explicitly below because the v2 builders have an independent raw-ICD default.

`run.py --list` gives all registered tasks. After `--`, arguments go to the selected study script. A task with `root_mode: output` treats `--output` as the logical workspace root and writes its documented canonical subtrees there. Other tasks read from `--project-root`. Some historical tasks write their fixed canonical output subtree within that root; passing a different launcher output does not relocate all their internal artifacts. Use the canonical destinations shown below.

```bash
python run.py --list
```

The adapter refuses to import a missing project `.py` dependency from an original checkout. An original author-machine `python.exe` subprocess path is resolved to the invoking interpreter unless an explicit input-map entry selects another interpreter. For mixed-environment analyses, use the separate phases and interpreters below instead of the historical all-in-one launcher. No local storage-guard installation is required by this public wrapper. On the authors' machine, its project storage policy still applies independently.

## Figure 7 from public aggregate inputs

`figures/render_figure7.py` reads only the seven tables in `reference/figure7/` and `reference/r1_3_macro_paired_bootstrap_ci.csv`. These contain cohort/organ summaries, not participant rows. Six panel-input tables retain their original bytes. In `MIMIC_12m_delta_c.csv`, the 28 organ C-index intervals and resampling descriptions are updated to the formal joint paired-bootstrap results; all point estimates are unchanged. The four large-marker intervals are read from the **macro** results, never obtained by averaging organ interval limits. They summarize 1,000 shared participant resamples within fixed MIMIC folds, conditional on the frozen UKB, downstream Cox and censoring fits. This rendering command does not fit models, resample participants or recalculate scientific results.

From this package directory, using the observed figure environment:

```bash
/path/to/figure/python -m pip install -r environments/requirements-figures.txt
/path/to/figure/python -B figures/render_figure7.py --output-dir /your/project/outputs/figure7
```

The default outputs are `Figure 7_reconstructed.pdf` and `.svg`; all six panels are rebuilt using the retained renderer and included aggregate inputs. Font availability and rendering-library versions can change geometry without changing plotted values. To retain the original submitted PDF's typography and other panels exactly, additionally supply that **original, pre-interval** vector Figure 7:

```bash
/path/to/figure/python -B figures/render_figure7.py --output-dir /your/project/outputs/figure7 --base-figure '/path/to/original/Figure 7.pdf'
```

This also writes `Figure 7.pdf`, adding the four paired macro intervals as vectors at the original markers. It rejects a different marker geometry; do not supply a PDF already containing the new intervals. The base PDF is optional and is not required for aggregate-only reconstruction. The manuscript uses this layout-preserving output. The original study renderer is retained byte-identically under `source/Revision-0730/0-Others/Fig12_MIMIC_IV_External_Transportability/`; the added wrapper supplies portable aggregate paths and the new interval layer. An optional `--alignment-helper` argument is only for an author's local layout audit and is not a runtime dependency.

## UKB input data: exact export contract

The study manifest/event/cache pipeline consumes the **prepared wide phenotype/assessment exports and combined long ICD export below**. An optional upstream entry now constructs the ICD long format from the specified indexed or order-verified packed diagnosis export and indexed date export. Downloading/extracting arbitrary UKB-RAP tables and selecting the 78-column phenotype export remain input-preparation steps. A complete raw-UKB-field rebuild has not been run for this release. Exact replay requires matching releases, feature order and the documented baseline derivations.

CSV files use a header, comma delimiter and CSV quoting for headers/text containing commas. Use UTF-8; dates should be unambiguous ISO `YYYY-MM-DD` (or an ISO timestamp where applicable). Participant IDs are opaque strings. Preserve leading zeros, remove an export's trailing `.0` only as implemented, and do not parse identifiers through floating point. Wide assessment and phenotype tables must have one nonempty unique ID per participant. Event and registration tables may have repeated participants. Empty numeric cells are missing values, not zero.

### Required UKB sources

| Config key / input | Required columns and types | Role |
| --- | --- | --- |
| `assessment_csv` | `Participant ID` string; `Date of attending assessment centre \| Instance 0` date; `Month of birth` integer 1–12 or English month name/abbreviation; `Year of birth` integer | Baseline index and age |
| `phenotype_csv` | `Participant ID` string; `Sex` (`Female`/`Male`, case-insensitive in the cache builder); `Age at recruitment` numeric; **exactly the 78 numeric columns in the ordered table below** | Raw numeric model slots, sex and missingness |
| `icd_long_csv` | `Participant ID` string; `ICD` string; `Disease_Description_Feature` text; `record_time` date | Diagnosis timeline, pre-existing disease and incident endpoints |
| `death_csv` | `Participant ID(participant - eid)` or `Participant ID(death - eid)` string; `Date of death` date | Earliest recorded death per ID |
| `visit_followup_csv` | `Participant ID` or `eid` string; all available date columns containing `Date of last personal contact with UK Biobank`, `Date of attending assessment centre`, or `Date lost to follow-up` | Observation/censoring sources; preserve instance/array columns rather than supplying one invented date |
| `gp_registration_csv` | `Participant ID` string; `Registration date` date; `Deduction date` date | Earliest registration / latest deduction per participant |
| `gp_prescription_csv` | `Participant ID` string; `Drug name` text; `Date prescription was issued` date | GP prescription timeline, not organ endpoint labels |
| `icd_participant_reference_csv` | Historical source-identity/audit input; the original export also contains the packed `Diagnoses - ICD10` field and indexed dates | May serve as the upstream diagnosis input described below; the primary event builders still consume `icd_long_csv` |

The phenotype reader excludes only the three header names `Participant ID`, `Sex` and `Age at recruitment` when identifying feature columns. Therefore, do **not** add metadata columns to that CSV. The stored `Age at recruitment` is not the chronological-age target used in the model. Birth date is constructed as the **15th day** of the recorded birth month; chronological age is `(index_date - birth_date).days / 365.25`.

The intended source has 502,163 assessment participants. Cohort entry uses assessment/phenotype intersection, valid index/birth dates, age 18–100 years inclusive, and at most **19 missing values out of 78**. It yields 456,773 participants in the reported data. Cohort entry does not require future survival, an incident outcome, or minimum future follow-up. Count mismatch gates are deliberate; relaxing them defines a different-data reconstruction.

### Upstream export preparation boundary

The manifest/event/cache stages implement preprocessing **after** these prepared exports. `preprocessing/prepare_icd_long.py` supplies the upstream ICD pairing step from the exact export schemas below. Its result was compared in memory with every existing 1124 long-table row; no new study table was written. The package still does not perform UKB-RAP download/extraction or a full independent reconstruction of the original 78-slot phenotype export.

- **Baseline phenotype derivations:** the authors confirmed the three Instance 0 formulas and missing-value propagation specified below. These are upstream calculations whose values are already present in the prepared 78-slot export; the cache builder does not recompute them. This author confirmation does not constitute an independent raw-field rebuild. Baseline heel BMD and ankle spacing use the combined Instance 0 columns already present in the source export; do not substitute a left/right average merely because a later-instance script does so. Numeric coercion, training-only transformations and missing masks after this entry boundary are implemented below.
- **Diagnosis/date pairing:** the added entry joins by participant and original Array index, retaining the matching code, description and date. Missing dates remain attached to their original diagnosis slots. It rejects duplicate participant/Array keys, incompatible schemas, unmatched participant keys and dates lacking a diagnosis at the same index. The full read-only check found zero differences against the current 1124 input; aggregate counts and input fingerprints are recorded in `package.json`. The original data also have contiguous nonempty date prefixes, so the historical date-compaction script did not shift diagnoses in this frozen dataset. The new entry does not depend on that contiguity condition.
- **Other sources:** export the required assessment, death, follow-up, registration and prescription columns exactly as specified above. Retain instance/array keys in wide date exports. Do not recompute the chronological-age target from a legacy long-table age column: the included builder derives index age from assessment and birth fields and reads only the four stated ICD-long fields.

These are input-preparation boundaries, not changes to the retained study data or reported results. The public entry-point instructions below reconstruct downstream inputs from matching prepared exports; they are not a claim of a verified raw-field-to-manuscript rebuild.

The confirmed baseline derivations use the following exact source headers. All components come from **Instance 0**; the lung-function ratio uses **Array 0** for both measurements.

| Prepared output header | Formula using source columns |
| --- | --- |
| `Hand grip strength (average) \| Instance 0` | (`Hand grip strength (left) \| Instance 0` + `Hand grip strength (right) \| Instance 0`) / 2 |
| `FEV1-FVC ratio \| Instance 0` | `Forced expiratory volume in 1-second (FEV1) \| Instance 0 \| Array 0` / `Forced vital capacity (FVC) \| Instance 0 \| Array 0` |
| `Waist-hip circumference ratio \| Instance 0` | `Waist circumference \| Instance 0` / `Hip circumference \| Instance 0` |

**Missingness propagates directly:** if either input component is missing, the derived value is missing. Do not average only the observed hand, replace a missing component with zero, or impute components before these derivations. The resulting missing values then enter the existing 78-slot missing-count gate and training-only transformations. The confirmation covers these formulas and missing-value propagation; any additional cleaning of invalid measurements or zero denominators has not been separately verified.

### Ordered phenotype slots and units

Indices are zero based and define the tensor order. Exact header spelling, including historical spelling errors, is intentional. Units are transcribed from the frozen external slot mapping's target units; the UKB builder does not automatically rescale arbitrary input units. Supply hand-grip average, FEV1/FVC and waist/hip ratio in the prepared wide export according to the confirmed Instance 0 formulas and missing-value rules above; this code does not infer derivations from an arbitrary upstream export.

| Slot | Exact phenotype header | Target unit |
| --- | --- | --- |
| 0 | `Hand grip strength (average) \| Instance 0` | kg |
| 1 | `Waist circumference \| Instance 0` | cm |
| 2 | `Hip circumference \| Instance 0` | cm |
| 3 | `Standing height \| Instance 0` | cm |
| 4 | `Pulse rate, automated reading \| Instance 0 \| Array 0` | beats/min |
| 5 | `Forced vital capacity (FVC) \| Instance 0 \| Array 0` | L |
| 6 | `Forced expiratory volume in 1-second (FEV1) \| Instance 0 \| Array 0` | L |
| 7 | `Peak expiratory flow (PEF) \| Instance 0 \| Array 0` | L/min |
| 8 | `Diastolic blood pressure, automated reading \| Instance 0 \| Array 0` | mmHg |
| 9 | `Systolic blood pressure, automated reading \| Instance 0 \| Array 0` | mmHg |
| 10 | `Body mass index (BMI) \| Instance 0` | kg/m2 |
| 11 | `Weight \| Instance 0(participant - p21002_i0)` | kg |
| 12 | `White blood cell (leukocyte) count \| Instance 0` | 10^9/L |
| 13 | `Red blood cell (erythrocyte) count \| Instance 0` | 10^12/L |
| 14 | `Haemoglobin concentration \| Instance 0` | g/dL |
| 15 | `Haematocrit percentage \| Instance 0` | % |
| 16 | `Mean corpuscular volume \| Instance 0` | fL |
| 17 | `Mean corpuscular haemoglobin \| Instance 0` | pg |
| 18 | `Mean corpuscular haemoglobin concentration \| Instance 0` | g/dL |
| 19 | `Red blood cell (erythrocyte) distribution width \| Instance 0` | % |
| 20 | `Platelet count \| Instance 0` | 10^9/L |
| 21 | `Platelet crit \| Instance 0` | % |
| 22 | `Mean platelet (thrombocyte) volume \| Instance 0` | fL |
| 23 | `Platelet distribution width \| Instance 0` | % |
| 24 | `Lymphocyte count \| Instance 0` | 10^9/L |
| 25 | `Monocyte count \| Instance 0` | 10^9/L |
| 26 | `Neutrophill count \| Instance 0` | 10^9/L |
| 27 | `Eosinophill count \| Instance 0` | 10^9/L |
| 28 | `Basophill count \| Instance 0` | 10^9/L |
| 29 | `Nucleated red blood cell count \| Instance 0` | 10^9/L |
| 30 | `Lymphocyte percentage \| Instance 0` | % |
| 31 | `Monocyte percentage \| Instance 0` | % |
| 32 | `Neutrophill percentage \| Instance 0` | % |
| 33 | `Eosinophill percentage \| Instance 0` | % |
| 34 | `Basophill percentage \| Instance 0` | % |
| 35 | `Nucleated red blood cell percentage \| Instance 0` | % |
| 36 | `Reticulocyte percentage \| Instance 0` | % |
| 37 | `Reticulocyte count \| Instance 0` | 10^12/L |
| 38 | `Mean reticulocyte volume \| Instance 0` | fL |
| 39 | `Mean sphered cell volume \| Instance 0` | fL |
| 40 | `Immature reticulocyte fraction \| Instance 0` | % |
| 41 | `High light scatter reticulocyte percentage \| Instance 0` | % |
| 42 | `High light scatter reticulocyte count \| Instance 0` | 10^9/L |
| 43 | `Creatinine (enzymatic) in urine \| Instance 0` | umol/L |
| 44 | `Potassium in urine \| Instance 0` | mmol/L |
| 45 | `Sodium in urine \| Instance 0` | mmol/L |
| 46 | `Albumin \| Instance 0(participant - p30600_i0)` | g/L |
| 47 | `Alkaline phosphatase \| Instance 0` | U/L |
| 48 | `Alanine aminotransferase \| Instance 0` | U/L |
| 49 | `Apolipoprotein A \| Instance 0` | g/L |
| 50 | `Apolipoprotein B \| Instance 0` | g/L |
| 51 | `Aspartate aminotransferase \| Instance 0` | U/L |
| 52 | `Direct bilirubin \| Instance 0` | umol/L |
| 53 | `Urea \| Instance 0` | mmol/L |
| 54 | `Calcium \| Instance 0` | mmol/L |
| 55 | `Cholesterol \| Instance 0` | mmol/L |
| 56 | `Creatinine \| Instance 0` | umol/L |
| 57 | `C-reactive protein \| Instance 0` | mg/L |
| 58 | `Cystatin C \| Instance 0` | mg/L |
| 59 | `Gamma glutamyltransferase \| Instance 0` | U/L |
| 60 | `Glucose \| Instance 0` | mmol/L |
| 61 | `Glycated haemoglobin (HbA1c) \| Instance 0` | mmol/mol |
| 62 | `HDL cholesterol \| Instance 0` | mmol/L |
| 63 | `IGF-1 \| Instance 0` | nmol/L |
| 64 | `LDL direct \| Instance 0` | mmol/L |
| 65 | `Lipoprotein A \| Instance 0` | nmol/L |
| 66 | `Phosphate \| Instance 0` | mmol/L |
| 67 | `SHBG \| Instance 0` | nmol/L |
| 68 | `Total bilirubin \| Instance 0` | umol/L |
| 69 | `Testosterone \| Instance 0` | nmol/L |
| 70 | `Total protein \| Instance 0` | g/L |
| 71 | `Triglycerides \| Instance 0` | mmol/L |
| 72 | `Urate \| Instance 0` | umol/L |
| 73 | `Vitamin D \| Instance 0` | nmol/L |
| 74 | `FEV1-FVC ratio \| Instance 0` | ratio |
| 75 | `Waist-hip circumference ratio \| Instance 0` | ratio |
| 76 | `Heel bone mineral density (BMD) \| Instance 0` | g/cm2 |
| 77 | `Ankle spacing width \| Instance 0` | cm |

The machine-readable ordered list is `source/Revision-0710/01_data_rebuild/01_index_and_events/phenotype_feature_columns.json`. The 78-row `MIMIC_78_SLOT_MAP_V8.csv` also records target units; the NHANES map records release-specific source units and conversions. No phenotype value from an external cohort is an outcome label for neural model fitting.

### Additional baseline covariate exports

Clinical comparisons and reference-membership analyses also require these four files under the mapped `Association` directory. Each contains one row per participant with `Participant ID`. These are separate from the 78-slot model input.

| File | Exact additional columns |
| --- | --- |
| `Associations-1.csv` | `Current tobacco smoking \| Instance 0`; `Qualifications \| Instance 0`; `Townsend deprivation index at recruitment`; `Alcohol drinker status \| Instance 0` |
| `Associations-2.csv` | `Current employment status \| Instance 0`; `Summed MET minutes per week for all activity \| Instance 0` |
| `Associations-4.csv` | `Sleep duration \| Instance 0` |
| `Associations-5.csv` | `Particulate matter air pollution (pm2.5); 2010` |

Text categories are stripped and lowercased. Missing category tokens are empty, `prefer not to answer`, `do not know`, `nan`, `none`, and `<na>`. Multiple qualifications/employment responses are separated by `|`. Unknown tokens raise an error; do not silently recode a new export's numerical category codes.

* Smoking: `yes, on most or all days` and `only occasionally` map to current=1; `no` maps to 0. Missing has an explicit indicator.
* Higher education: any of `college or university degree`, `a levels/as levels or equivalent`, `nvq or hnd or hnc or equivalent`, `other professional qualifications eg: nursing, teaching`. Other recognized education tokens are `o levels/gcses or equivalent`, `cses or equivalent`, `none of the above`.
* Employment groups: employed=`in paid employment or self-employed` or `doing unpaid or voluntary work`; retired=`retired`; unemployed=`unemployed`; other=`unable to work because of sickness or disability`, `looking after home and/or family`, `none of the above`, or `full or part-time student`. Preserve the explicit missing indicator.
* Alcohol recognizes `current`, `previous`, `never`; physical activity and PM2.5 are extended sensitivity covariates. They are not additional columns of the primary C21 reference.
* Continuous covariates retain missingness indicators; fold-specific imputation/scaling is fitted by the downstream fitting code. Do not impute the entire cohort before splitting.

The primary C21 column order is:

```text
chronological_age, sex_male,
smoking_current, smoking_missing, education_higher, education_missing,
townsend, townsend_missing,
employment_employed, employment_retired, employment_unemployed,
employment_other, employment_missing,
sleep_duration, sleep_missing,
phenotype_missing_fraction,
log1p_preindex_icd_count, log1p_preindex_gp_count,
has_preindex_icd, has_preindex_gp, prevalent_organ_count
```

See `05_downstream/06_metric_v2/common/contextual_covariates.py`, its `CANDIDATE_DATA_DICTIONARY.json`, and `scripts/materialize_contextual_covariates.py` for the executable category/range rules. The materialized table adds `participant_id`, `fold_id` and auxiliary/sensitivity columns; it must cover the same cohort without duplicate IDs.

### Upstream ICD Array pairing: exact inputs and output

UKB diagnosis field 41270 and first-diagnosis date field 41280 correspond by Array index ([UKB field definition](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=41280)). The entry accepts two explicitly selected diagnosis representations, both joined to dates by the original index:

| Input | Required schema and interpretation |
| --- | --- |
| Indexed diagnoses, `--code-format indexed` (default) | Unique `Participant ID`; columns `Diagnoses - ICD10 \| Array 0`, `... \| Array 1`, etc. An optional `\| Instance 0` before `\| Array n` is accepted. Code/date header index sets must be identical. Physical column order is irrelevant. |
| Packed diagnoses, `--code-format packed --packed-order-is-array` | Unique `Participant ID`; one `Diagnoses - ICD10` cell containing entries separated by the literal `\|`. Position 0 is Array 0, position 1 is Array 1, etc. Empty interior positions must remain present. **Only use this mode after establishing that this export preserves original Array order.** Never sort or compact the list first. |
| Indexed dates, `--dates-csv` | Unique `Participant ID`; `Date of first in-patient diagnosis - ICD10 \| Array n` columns, optionally with `\| Instance 0`. The current study date export has Array 0–258. Nonzero instances, duplicate logical indices and duplicate participant keys are rejected. |

For the current study, the packed diagnosis source is `3-ICD10-1_participant.csv` and the date source is `Date_ICD_10.csv`. Their participant row orders need not match: the entry uses an ID lookup, not row-position joins. If the diagnosis file also includes indexed dates, values at shared Array keys are cross-checked against the date input. All participants must occur once in each source. Match the data release and record input fingerprints; an ordinal declaration alone is not proof of compatibility between arbitrary exports.

Each nonempty diagnosis cell contains an ICD code followed, where supplied by the original export, by its human-readable description. The first whitespace-delimited token becomes `ICD`, and the remainder becomes `Disease_Description_Feature`; surrounding single quotes are stripped as in the historical preparation. A bare code produces an empty description. Such a code-only export does not reproduce the study's labelled text inputs/embeddings. Retain the matching source descriptions for exact reconstruction.

Dates accept ISO `YYYY-MM-DD` and unambiguous year-first `YYYY/M/D`, optionally with a midnight ` 00:00:00` suffix, and are validated as calendar dates and written as ISO dates. Empty fields denote missing values. A nonempty code with a missing date is retained at the **same Array index** with blank `record_time`; later event-building rules handle undated records. A date with no code at that index is rejected. Repeated diagnosis records are retained rather than deduplicated here. A participant with no codes receives one empty clinical placeholder row.

The output contains exactly these columns:

| Column | Content |
| --- | --- |
| `Participant ID` | Original opaque string identifier |
| `ICD` | Original code token; downstream code normalization remains in the study event builder |
| `Disease_Description_Feature` | Matched original description |
| `record_time` | Matched diagnosis date or missing |
| `source_array_index` | Original numeric Array index; blank only for a diagnosis-free placeholder |

The study event/codebook/label readers select the required clinical columns and ignore the extra trace column. Birthday, assay date and legacy `Actual_Age` are not needed in this ICD output: the index/age builder obtains its own assessment and birth inputs. For a fresh reconstruction, set `icd_long_csv`, the `--raw-icd` arguments and the corresponding private input-map alias consistently to the new long file. Its hash will differ from the historical seven-column file because of the schema/date formatting/order. Record the new provenance identity; an exact frozen replay continues to require the original matching artifacts and hash gates.

### Upstream ICD commands and verification scope

The entry is standalone and does not need the historical path adapter. Set absolute paths, using an authorized private input location and a project output directory separate from this package:

```bash
export PY_STATS=/path/to/python-3.12.4
export PKG=/path/to/Multi-Organ-Age-Prediction
export UKB_CODES=/data/private/ukb/3-ICD10-1_participant.csv
export UKB_DATES=/data/private/ukb/Date_ICD_10.csv
export UKB_LONG=/data/private/ukb/1-combined_ICD_Time_birth_phenotype_age_all-1124.csv

# Read-only comparison against an existing long table; writes no participant data.
"$PY_STATS" -B "$PKG/preprocessing/prepare_icd_long.py" verify \
  --codes-csv "$UKB_CODES" --dates-csv "$UKB_DATES" \
  --code-format packed --packed-order-is-array --existing-long "$UKB_LONG" \
  --expected-existing-sha256 70ae7d04db587783927380cffda826a0960a27e3919f0d2094c0b8aa11f462a6
```

The optional SHA256 argument above identifies the **original study long table**. Supply the appropriate recorded identity for another existing dataset. Comparison keeps all duplicate occurrences, normalizes dates and compares each participant's full multiset of diagnosis/description/date records. The existing table must keep each participant's rows contiguous, as do the study's intermediate and sorted final tables. If it contains `source_array_index`, the comparison checks that column too. Differences return exit status 1; malformed input or a failed identity gate returns status 2. Error/progress messages disclose aggregate counts and schema/row locations, not participant values. Verification keeps the sparse date lookup and participant digests in memory and creates no data/cache files.

For a **new, separately authorized reconstruction**, the build interface is:

```bash
# Create/select an output directory on the intended project disk first.
"$PY_STATS" -B "$PKG/preprocessing/prepare_icd_long.py" build \
  --codes-csv "$UKB_CODES" --dates-csv "$UKB_DATES" \
  --code-format packed --packed-order-is-array \
  --output /data/ehr-age-reproduction/inputs/icd_long_with_array.csv
```

For explicitly indexed diagnosis columns, use `--code-format indexed` and omit `--packed-order-is-array`. Build requires a new destination outside the package and refuses to overwrite an input or an existing file. It validates all rows in a temporary file on the chosen output filesystem before exclusively creating the final file. Budget approximately twice the resulting CSV size for the temporary/final copy at peak. No such full-cohort build was run during this release preparation.

The synthetic tests write only invented fixtures under the explicitly selected test directory:

```bash
"$PY_STATS" -B "$PKG/preprocessing/test_prepare_icd_long.py" \
  --temp-root /data/ehr-age-reproduction/work/icd-array-tests
```

Local checks completed with CPython 3.12.4 / Windows:

| Check | Result |
| --- | --- |
| Synthetic tests: key/column reordering, interior missing dates/codes, packed-order declaration, orphan dates, duplicate keys, malformed dates/schemas, destination preservation, mismatch detection and frozen identity | 12 tests passed |
| Read-only full-cohort comparison using the packaged entry | 502,163 participants; 7,071,243 rows; zero mismatched, missing or unexpected participants |
| Clinical content | 7,015,908 coded rows; 55,335 empty placeholders; zero undated coded rows |
| Original date columns shared by the diagnosis and date exports | 99,428,274 cells checked without a difference after date normalization |
| Formal study input and result changes during verification | None; no participant data written, training or scientific fitting |

The aggregate verification record and all three input SHA256 values are included in `package.json`. This establishes the pairing/content of the current source exports and frozen 1124 long table. It is not a repeated UKB-RAP extraction, phenotype rebuild or end-to-end model-training replication.

## Preprocessing tasks and generated inputs

### Order and leakage boundaries

When starting from indexed or order-verified packed ICD exports, first run the optional upstream entry above in a new reconstruction workspace. The following stages consume its long-table clinical columns, or the matching original prepared input for frozen replay.

1. **Source manifests:** normalize IDs; join assessment and prepared phenotypes; derive baseline age; apply the entry criteria; retain raw phenotype values and a separate missing mask; join follow-up sources. Numeric coercion failures become missing and are subject to the missing-count gate.
2. **ICD events:** normalize codes to uppercase alphanumerics and apply the specified mapping by longest matching prefix. Distinguish all pre-index model events from organ mapping used for endpoint labels. Model inputs require `event_date < index_date`; an index-day diagnosis is prevalent (`<=`) and cannot become an incident (`>`) event.
3. **GP events:** require a valid date, nonempty trimmed drug name and a strictly pre-index prescription. Deduplicate participant/date/description/source, retain stable source row numbers and the latest 64 GP events per participant in the source event table. The intended source scan covers 56,476,345 prescription rows; a truncated `--max-chunks` scan is not a production rebuild.
4. **Outcomes/folds:** construct observation and censoring dates, 60-month cause-specific endpoints and outcome-blind folds. The observation end combines recorded contact, assessments, ICD/GP activity and GP deduction information, capped by earlier loss to follow-up. Event/censoring logic also uses death and the administrative horizon. Do not replace the code with a binary label that treats insufficient observation as an event-free control.
5. **Leakage audit:** check cohort coverage, strictly negative input times, allowed fold columns and outcome consistency.
6. **Corrected-primary rebuild:** use the bundled corrected v1 mapping to rebuild labels and verify that outcome-independent inputs/folds are unchanged. This is a historical regression/audit stage, not an alternative cohort definition.
7. **Codebook v2 and v2 labels:** generate the curated endpoint rules and observed-code audit, then derive v2 participant/organ labels and recurrent sensitivity fields. Preserve separate baseline-chronic and incident-endpoint eligibility. Acute, chronic, secondary and baseline-only rules are explicit in `codebook_v2.csv`.
8. **Compact caches:** build deterministic vocabularies and balanced timelines; store raw phenotype slots and masks; fit median/mean/SD on training folds only. Build CLIP tables separately. Targets/outcomes are kept outside the model-input cache.
9. **Role tables and contextual covariates:** materialize role-specific labels/ages and baseline C21/sensitivity inputs before training/downstream fitting.

Both ICD mapping generations are included: `06_icd_codebook_v1/02_v1_corrected_primary/mapping.csv` and `08_icd_codebook_v2/codebook_v2.csv`. They have different purposes. The observed-code audit is data-derived and is regenerated from the authorized ICD export; it is not a replacement for the curated codebook.

### Example reconstruction commands from prepared exports

These commands are supplied for a researcher preparing a future run; they were **not executed as a new experiment during this release preparation**. Shell examples use Bash; on PowerShell use equivalent absolute paths, `$env:` variables and `&` for an interpreter path. All variables below are chosen by the reproducing researcher.

```bash
export PKG=/path/to/Multi-Organ-Age-Prediction
export PROJECT=/data/ehr-age-reproduction
export BUILD="$PROJECT/runs/from_raw"
export MAP="$PROJECT/private-input-map.json"
export UKB_ICD=/data/private/ukb/combined_icd.csv
export PY_STATS=/path/to/statistics/python
export PY_COX=/path/to/cox/python
export PY_NEURAL=/path/to/neural/python
export PY_EMBED=/path/to/embedding/python
mkdir -p "$BUILD"

"$PY_STATS" "$PKG/run.py" --task source_manifests --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP" -- --hash-sources
"$PY_STATS" "$PKG/run.py" --task icd_events --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP"
"$PY_STATS" "$PKG/run.py" --task gp_events --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP"
"$PY_STATS" "$PKG/run.py" --task outcomes_and_folds --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP"
"$PY_STATS" "$PKG/run.py" --task leakage_audit --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP"
"$PY_STATS" "$PKG/run.py" --task corrected_rebuild --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP"
"$PY_STATS" "$PKG/run.py" --task codebook_v2 --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP" -- --raw-icd "$UKB_ICD"
"$PY_STATS" "$PKG/run.py" --task labels_v2 --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP" -- --raw-icd "$UKB_ICD"
"$PY_STATS" "$PKG/run.py" --task compact_cache --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP" -- --skip-embeddings
"$PY_EMBED" "$PKG/run.py" --task compact_cache --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP" -- --only-embeddings
"$PY_STATS" "$PKG/run.py" --task prepare_roles --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP"
"$PY_STATS" "$PKG/run.py" --task context_covariates --project-root "$PROJECT" --output "$BUILD" --input-map "$MAP"
```

The private source-path config must name the actual `icd_long_csv`; `UKB_ICD` above must resolve to that same file. To rebuild an already completed corrected stage's audit only, its script supports `--skip-build`; it refuses to overwrite a previous corrected label build. Do not use `--allow-count-mismatch`, `--allow-incomplete-source`, `--allow-fail`, chunk limits or participant limits to label a partial run as the reported full-cohort reconstruction.

Primary output subtrees under `BUILD`:

| Subtree | Key outputs |
| --- | --- |
| `Revision-0710/01_data_rebuild/01_index_and_events` | `age_model_cohort.parquet`, `raw_phenotypes.parquet`, `phenotype_missing_mask.parquet`, `followup_source_dates.parquet`, canonical pre-index ICD/GP event tables and participant summaries |
| `.../02_outcomes_60m` | `participant_followup_and_evaluation_masks.parquet`, `organ_outcomes_60m.parquet` and count audits |
| `.../03_folds` | `fold_manifest_5fold.parquet`, `outer_fold_roles.parquet`, `fold0_validation_calibration_screen.parquet` |
| `.../05_leakage_audit` | Leakage gate; inspect failures before downstream work |
| `.../07_corrected_primary_v1` | Corrected index/outcome/fold tables and delta audits |
| `.../08_icd_codebook_v2` | `codebook_v2.csv`, `observed_code_audit.csv` and codebook audit |
| `.../09_v2_labels` | `icd_participant_labels_v2.parquet`, v2 `organ_outcomes_60m.parquet` and group counts |
| `.../04_cache_v1` | Shards, vocabulary, embeddings, fold transforms and cache manifests |
| `Revision-0710/03_ehr_age/06_codebook_v2/00_roles` | Fold0 development/open label and age tables plus role-summary hashes |
| `Revision-0710/05_downstream/06_metric_v2/01_contextual_covariates` | Participant baseline covariates and materialization manifest |

All participant-containing outputs remain private. Source manifests, tensor caches and model outputs are not public artifacts merely because they were generated by public code.

### Intermediate schemas

Parquet dates are datetime values, identifiers are strings, numerical measurements are numeric, and missingness/reference indicators are boolean or explicit integer 0/1. Do not join on row position across independently sorted tables.

| Artifact | Minimum required fields / invariants |
| --- | --- |
| `age_model_cohort.parquet` | `participant_id`, `index_date`, `birth_date`, `chronological_age`, `sex`, `phenotype_missing_count`, `phenotype_missing_fraction`, `age_model_eligible`; one row per participant |
| `raw_phenotypes.parquet` | `participant_id` plus the 78 ordered feature columns; NaN at missing positions |
| `phenotype_missing_mask.parquet` | `participant_id` plus the same 78 columns as booleans; True means missing, not observed |
| Canonical pre-index events | `participant_id`, `event_date`, `code`, `description`, `source`, `source_row_number`, `relative_days`, `attention_mask`; source is `hospital_icd` or `gp_prescription`; active `relative_days < 0`; mask=1; GP code is empty |
| `fold_manifest_5fold.parquet` | `participant_id`, `fold_id` 0–4, baseline age/sex/missingness and derived `sex_norm`, `age_decile`, `missingness_bin`, `stratum`; one row per participant |
| `fold0_validation_calibration_screen.parquet` | `participant_id`, `fold_id`=1, `stratum`, `inner_role`=`calibration` or `screen`; unique participants |
| `organ_outcomes_60m.parquet` v2 | Unique `(participant_id, organ_system)`; seven rows/person; `index_date`, `has_prevalent_target`, `first_incident_target_date`, `event`, `at_risk`, `duration_months`, `strict_healthy_reference_60m`, plus cohort/group and subsequent-event sensitivity fields generated by the label builder |
| Development/open role labels | Exact order: `participant_id`, `fold_id`, `inner_role`, `organ_system`, `duration_months`, `event`, `at_risk`, `strict_healthy_reference_60m` |
| Development/open age targets | Exact order: `participant_id`, `fold_id`, `chronological_age` |
| UKB neural predictions | One row per `(participant_id, organ_system)` per role/fold: `actual_age`, `predicted_age`, `corrected_age_gap`, `model_risk_score`, with the role/fold/outcome fields expected by its evaluation adapter; do not confuse raw gap with corrected gap |

Always align organ axes by **name**. The model/age axis is `cardiovascular, hepatic, immune, metabolic, musculoskeletal, pulmonary, renal`. The ICD label-bitmask and some clinical scripts use `cardiovascular, pulmonary, musculoskeletal, immune, renal, hepatic, metabolic`. A seven-column matrix without its named axis is insufficient input documentation.

The strict reference is a follow-up-defined reference, not the whole age-model cohort. The original data have 102,129 strict-reference participants. The baseline-only reference is defined without future diagnosis, death or follow-up requirements and has 175,637 participants. Both are subsets of the 456,773-person modeling cohort; changing the active reference penalty does not discard other training participants.

### Cache, vocabulary and embedding contract

`batch_manifest.json` lists fold-aligned `.npz` shards, normally `shards/fold{f}_batch{b:03d}.npz`, with at most 8,192 participants/shard. Each shard contains:

| Key | Type / shape |
| --- | --- |
| `participant_id` | Unicode strings `[N]`; unique and aligned with all arrays |
| `participant_row` | int32 `[N]`; index in the cache's canonical participant order |
| `fold_id` | int16 `[N]` |
| `phenotype_raw_zero` | float32 `[N,78]`; missing positions stored as 0 only for storage |
| `phenotype_missing_mask` | bool `[N,78]`; retain this mask when imputing |
| `sex` | int8 `[N]`; female=0, male=1 |
| `event_code_id`, `event_text_id`, `event_relative_days` | int32 `[N,32]` |
| `event_type_id` | uint8 `[N,32]`; padding=0, hospital ICD=1, GP=2 |
| `event_valid_mask` | bool `[N,32]`; True=active |
| `event_count` | int16 `[N]` |
| `storage_event_code_id`, `storage_event_text_id`, `storage_event_relative_days` | int32 `[N,64]` |
| `storage_event_type_id`, `storage_event_valid_mask`, `storage_event_count` | Same respective types as primary arrays, with storage sequence length 64 |

All inactive event positions have code ID, text ID, relative days and event type set to 0 and an inactive mask. Active events have strictly negative relative days. An empty timeline is represented by all padding with `event_count=0`, not by a fabricated event token. Hospital/GP balance reserves up to 16 slots/modality in the storage L64 timeline, then up to 8/modality in primary L32, filling remaining slots with the latest eligible events. Keep deterministic ordering/tie handling in `select_unified_timeline`.

`vocab/event_code_vocab.parquet` and `vocab/event_text_vocab.parquet` map deterministic vocabulary IDs to code/text. Padding ID is 0. Text keys combine role and description with the literal separator `\x1f`: `icd_description\x1f...` or `gp_drug\x1f...`. Vocabularies and text/embedding tables derived from restricted exports are generated privately; the release does not bundle them.

`fold_transforms/outer_fold_{k}.npz` contains float32 `median`, `mean`, `std` vectors of length 78, plus `feature_hash`, `train_id_hash`, `outer_fold`, and `validation_fold`. Training-only observed values define medians; imputed training values define population mean/SD (`ddof=0`); SD below `1e-6` is replaced by 1. Apply `(imputed - mean) / std` to calibration/test. The same mask is also supplied to the model. Never compute these values on all five folds or on an external cohort for frozen transfer.

Text encoding uses **OpenCLIP ViT-B-16, pretrained=`openai`**, not a BiomedCLIP model. Original pretrained weight SHA-256: `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`. It must be acquired separately. Model output dimension is 512; cached code/text embeddings are float16 and not L2-normalized; the padding row is exactly zero. `embedding_tables.npz` provides `code_embedding` and `text_embedding`; `.pt` and `embedding_manifest.json` record the corresponding tables/provenance. Changing text, vocabulary order, tokenizer, pretrained revision, normalization or dtype changes this input representation.

## Splits, training and evaluation

### UKB roles

Participant folds are stratified using baseline sex, age decile and phenotype-missingness bins (`0`, `1–5`, `6–10`, `11–19`) with seed string `revision0710-leakage-safe-v1`. Outcomes do not enter fold assignment. Original fold sizes 0–4 are **91,355; 91,356; 91,353; 91,353; 91,356**.

For outer fold `k`, the test fold is `k`, calibration fold is `(k+1) % 5`, and the other three folds train the neural model. Fold0 has the historical development design: the Fold1 inner split contains **45,599 calibration** and **45,757 screen** participants. Screen participants are excluded from its fitting/calibration role; do not replace calibration by the full 91,356-person Fold1. Fold0 was used for model development/selection. The final architecture/epoch were then frozen for folds 1–4; this is not a claim of five untouched independent development datasets.

The final EHR-AGE variant is `shared_private_independent_view_huber1`. Its recorded configuration is `03_ehr_age/06_codebook_v2/01_runs/shared_private_independent_view_huber1_fold0/run_config.json`. Neural seed=7210; batch=2048; AdamW learning rate=`2e-4`, weight decay=`1e-4`; gradient clip=1; three warmup epochs; cosine schedule with a **60-epoch horizon**; event dropout=0.1; AMP and recorded TF32/determinism settings. Fold0 development ran its specified 60-epoch selection procedure. The frozen final checkpoint epoch is **47**, and fixed outer-fold/reference-sensitivity training uses 47 epochs while retaining the 60-epoch scheduler horizon. Do not change that horizon to 47.

Age correction is fitted using the relevant calibration reference and applied to held-out predictions. Cox standardization, penalties, censoring estimates and any local calibration must also be fitted using their allowed fitting role. OSAG has chronological-age supervision, outcome-related objectives and the reference penalty; it is an outcome-informed age-referenced prognostic representation.

### Neural entry points

The following template assumes `BUILD` has the reconstructed **or authorized archived** study tree, including the required input cache, role tables and baseline evaluation registries/targets. Historical selection/parity gates require the original matching metadata; source/config files alone do not provide missing baseline predictions.

```bash
export PRIMARY="$BUILD/Revision-0710/03_ehr_age/06_codebook_v2"

# Fold0 development/selection; this is a training command, not a smoke test.
"$PY_NEURAL" "$PKG/run.py" --task fold0_model --project-root "$BUILD" --output "$PRIMARY/01_runs/shared_private_independent_view_huber1_fold0" -- --variant shared_private_independent_view_huber1

# Frozen folds 1–4; one writer per fold output directory.
for k in 1 2 3 4; do
  "$PY_NEURAL" "$PKG/run.py" --task frozen_outer_fold --project-root "$BUILD" --output "$PRIMARY/02_fivefold_frozen_huber1/01_outer_folds/outer_fold_$k" -- --outer-fold "$k" --model ehr_age_huber1
done
"$PY_NEURAL" "$PKG/run.py" --task summarize_model --project-root "$BUILD" --output "$PRIMARY/02_fivefold_frozen_huber1/02_summary"
```

The fixed outer-fold runner also implements `--model retain_organquery_joint`. Other matched comparator training is exposed as `matched_comparator`; inspect its `--help` for the implemented method/fold choices. Comparator labels such as Med-BERT-style refer to these study implementations; the package does not supply a third party's pretrained Med-BERT model.

### Descriptive covariance inputs

The retained current covariance implementation is `Revision-0730/0-Others/Fig7_Shared_Component_SEM/scripts/build_figure6_shared_component_residual_atlas.py`; historical figure numbering in its filename is not manuscript numbering. Its tasks are `shared_covariance`, `covariance_diagnostics`, and `covariance_estimators`. These are analysis commands that require private data, **not** help/smoke commands. They write under the selected project's `Revision-0730/0-Others/Fig7_Shared_Component_SEM/{source_data,outputs,qa,development}` as applicable.

The private `Revision-0710/06_figures_and_statistics/02_canonical_analysis_data/02_oof_wide/canonical_age_gap_wide.parquet` has one row/participant, `participant_id`, `outer_fold`, and seven numeric columns named `ORGAN__corrected_age_gap`. The diagnostic reader additionally requires `actual_age`, `sex`, `strict_healthy_reference_60m`, and `phenotype_missing_count` in that wide table. Pivot the named OOF predictions without changing values or fold membership. The separate `Revision-0710/Final_Fig_0716/Codebook_v2_Huber1/00_inputs/fig11_fig12/display_group_membership.parquet` has unique `(participant_id, analysis_group)` rows; groups can overlap. Exact tokens are `Health`, `Other_Disease`, each organ name, and each `ORGAN_Health` token. Preserve the historical display-group definitions, which differ from the strict training reference. Diagnostic scripts additionally read the outcome and contextual-covariate tables specified in their source constants.

The model is the prespecified seven-indicator `Shared =~ ...` model. Estimator sensitivity includes DWLS, ULS, MLW and GLS; report convergence/admissibility and estimator dependence. The displayed residual atlas is observed minus model-implied covariance, not a model with freely estimated organ-to-organ causal paths. A boundary solution must not be relabeled as an admissible independent validation. These data describe shared model-output covariance; the public source does not convert it into independent evidence of biological aging.

### Clinical comparison and round-two analyses

Before clinical fits, materialize C21, then create the private trusted context cache with `clinical_utility ... -- prepare-context`. The context pickle is a locally generated serialization, not a download supplied by this release. Model-specific calibration/test prediction files must already exist in the paths specified by each registry.

The information-matched reference **M** contains C21 + 78 phenotype values + 78 missingness indicators + seven frozen Med-BERT-style direct risk scores = **184 nominal variables**. M+OSAG adds seven corrected organ gaps = **191**. Missing/constant/collinear columns are handled within the fitting code and recorded; nominal counts need not equal retained coefficient counts. These seven risk scores are from the previously fitted comparator for the same outer roles, not refitted on test participants. This is a post hoc information-matched comparison conditional on that comparator choice.

Penalty selection uses three event-stratified folds **within calibration only**, candidates `0.0001, 0.001, 0.01, 0.1`, with larger penalty for a numerical tie. After selection, fit the downstream model on calibration and evaluate on the outer test fold. The three neural training folds, the calibration subset, and the outer test fold therefore have different roles.

```bash
export R11="$BUILD/Revisions-1030/work/r1_1_information_matched_20261031"
"$PY_STATS" "$PKG/run.py" --task information_matched --project-root "$BUILD" --output "$R11" -- prepare
for k in 0 1 2 3 4; do
  "$PY_COX" "$PKG/run.py" --task information_matched --project-root "$BUILD" --output "$R11" -- fit --fold "$k"
  "$PY_STATS" "$PKG/run.py" --task information_matched --project-root "$BUILD" --output "$R11" -- bootstrap --fold "$k" --replicates 1000
done
"$PY_STATS" "$PKG/run.py" --task information_matched --project-root "$BUILD" --output "$R11" -- summarize
"$PY_STATS" "$PKG/run.py" --task information_matched --project-root "$BUILD" --output "$R11" -- verify
```

The baseline-only training-reference sensitivity changes **only** membership in the active organ-offset Huber shrinkage penalty (weight 0.75; beta=1 year): no pre-index ICD and no v2 baseline organ disease, without a future event/death/follow-up requirement. It retains all training participants, inputs, architecture, other losses, seed, folds and fixed epoch. Its entry points are:

| Task / phase | Interpreter | Exact output destination under `BUILD/Revisions-1030/work/r1_4_reference_20261031` |
| --- | --- | --- |
| `training_reference -- --mode train --fold k` | neural | `outputs/folds/outer_fold_k` |
| `reference_evaluation -- --phase prepare --case CASE --fold k` | statistics | `analysis/CASE/outputs/prepared/outer_fold_k` |
| `reference_evaluation -- --phase fit --case CASE --fold k` | cox | `analysis/CASE/outputs/folds/outer_fold_k` |
| `reference_evaluation -- --phase bootstrap --case CASE --fold k --replicates 1000` | statistics | `analysis/CASE/outputs/folds/outer_fold_k` |
| `reference_evaluation -- --phase summary` | statistics | `results` |
| `reference_evaluation -- --phase association` | cox | `results/associations` |

Use `--project-root "$BUILD" --output EXACT_DESTINATION` with each task. `CASE` is `strict_train` or `baseline_train`; run preparation/fitting/bootstrap for both cases and all five folds before summary. The primary training-reference contrast holds **baseline-only age correction fixed** in both cases. Existing strict-training/baseline-only-correction outputs under `Revision-0730/work/p0_core_sensitivity_20260809/outputs` are required by this exact replay. The two retained scientific scripts in that directory document/generate the earlier sensitivity; those private predictions and fitted nuisance objects are not bundled.

`harmonize_statistics` then combines saved age predictions and HR inputs; it does not fit neural models. It requires the completed reference-evaluation results and original reference membership. Use its `--help` for the `--project-root`/`--output` options; output must remain under the selected project's `Revisions-1030` tree.

### Comparison metrics and aggregation

Age outputs: MAE in years, Pearson correlation, and error-based R². A pooled metric concatenates held-out predictions before calculation; an average of foldwise correlations/R² is a different estimand.

UKB risk comparisons: Harrell C and its paired increment; reported IPCW/Uno and time-horizon discrimination where implemented; Brier improvement, IDI/NRI and decision-curve net benefit as specified by the particular analysis. Consult the fitting/evaluation script for the metric's horizon, reference and risk-set restriction. Internal incremental summaries average folds and then organs. Do not swap a 60-month endpoint for the external 12-month endpoint or treat the direct neural risk score as a calibrated survival probability.

Paired intervals use the same participant draws for the relevant models/organs/metrics. The 1,000-replicate analyses condition on the already fitted neural, Cox and censoring models; they do not cover model selection or full refitting variability. Organ-specific intervals are pointwise unless their output explicitly states an adjustment. Preserve sign conventions: an increase is favorable for C/AUC, whereas **Brier improvement = base Brier minus augmented Brier**.

## External data, mapping and preprocessing

The definitive external input registry is:

`source/Revision-0710/06_figures_and_statistics/00_plan/external_validation_prefreeze_v8/artifacts_v8/SOURCE_REGISTRY_V8.json`

It supplies the **complete header list, format, source ID, release, source role and original file fingerprint** for all 42 external source files. Its author-machine absolute paths are logical inputs; map their roots to your own authorized sources. The associated `OUTPUT_SCHEMAS_V8.json` describes the historical table schemas and their conditions; it does not imply that every historical sensitivity is part of the main direct/support-adapted analysis.

### MIMIC-IV 3.1

Use the registered compressed CSV tables under `hosp/`. IDs (`subject_id`, `hadm_id`, `itemid`) are exact integer/string identifiers, not measurements; dates/timestamps retain the original cohort's shifted time system. Empty numeric/date fields are missing. Required read columns include:

| File | Columns consumed by the preprocessing / endpoint pipeline |
| --- | --- |
| `patients.csv.gz` | `subject_id`, `gender` (`M`/`F`), `anchor_age`, `anchor_year`; `dod` for later outcomes |
| `admissions.csv.gz` | `subject_id`, `hadm_id`, `admittime`, `dischtime`, `deathtime` |
| `diagnoses_icd.csv.gz` | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version` (9 or 10) |
| `d_icd_diagnoses.csv.gz` | `icd_code`, `icd_version`, `long_title` |
| `labevents.csv.gz` | `labevent_id`, `subject_id`, `itemid`, `charttime`, `storetime`, `valuenum`, `valueuom` |
| `d_labitems.csv.gz` | `itemid`, `label`, `fluid`, `category` (registered mapping/audit source) |
| `omr.csv.gz` | `subject_id`, `chartdate`, `seq_num`, `result_name`, `result_value` |
| `prescriptions.csv.gz` | Registered full header for the inpatient-prescription proxy sensitivity; primary hospital-ICD transfer does not use this as UKB GP prescribing |

The four bundled mapping tables are in that same `artifacts_v8` directory:

| Mapping | Rows | Contents |
| --- | --- | --- |
| `MIMIC_78_SLOT_MAP_V8.csv` | 78 | Slot index/name, source item/selector, exact units, primary/sensitivity formulas and value guards, target units, lookback, aggregation and tie-break rules |
| `MIMIC_ICD10_ORGAN_MAP_V8.csv` | 22 | External organ endpoint mapping rules |
| `NHANES_78_SLOT_RELEASE_MAP_V8.csv` | 156 | Every phenotype slot in each of two releases; source file/variable/unit, conversion formula, structural missingness and adjudication |
| `NHANES_QUESTIONNAIRE_LABEL_MAP_V8.csv` | 14 | Seven organ-label rows per release, estimability and positive/negative/missing rules; labels are not neural inputs |

MIMIC processing selects the second **valid adult surviving-discharge admission**, after checking time validity/adult age/alive-at-discharge; landmark is its discharge. `anchor_age=91` top-coded ages are excluded. Age is `anchor_age + discharge_year - anchor_year`, preserving integer resolution. Diagnoses available from completed pre-landmark admissions form the hospital-ICD timeline; pre-index laboratory/OMR selection must follow availability-time, lookback, explicit-unit and tie-break rules in the implementation. Do not use a post-landmark lab because its `charttime` looks earlier if availability rules reject it. Do not assume an unlabeled OMR unit for primary input: that is a separate sensitivity.

Primary MIMIC input uses hospital ICD plus explicitly supported phenotype units. It does not equate inpatient prescribing with UKB GP prescribing. Unsupported/unobserved phenotype slots remain missing with their masks. All frozen transformations, vocabulary and text embeddings come from the relevant UKB model/cache. The UKB support-adapted model uses the `mimic_common53` support definition; it is separate from direct transfer and is trained on UKB, without MIMIC gradient updates or MIMIC outcome-label fitting of the neural representation.

External participant folds use the first eight bytes of SHA-256 of `revision0710-external-v3|ID`, interpreted unsigned big-endian, modulo 5. Neural outputs are the frozen five-model UKB ensemble; MIMIC downstream Cox/risk models fit **four target hash folds and evaluate the fifth**, unlike the UKB three-neural-fold/one-calibration-fold design.

### NHANES releases and input types

Only these releases are used: `P_2017_MARCH2020_PREPANDEMIC` (P_* files) and `L_AUG2021_AUG2023` (*_L files). Do not combine DEMO_J with P_* or replace the release-specific survey weights. XPT files are SAS XPORT format, read with Latin-1 string decoding. `SEQN` joins files **within release**; preserve `release_id` and do not assume cross-release records can be pooled as one survey design.

| Component / source ID suffix | Pre-pandemic P file | 2021–2023 L file |
| --- | --- | --- |
| `demo` | `p_demo.xpt` | `DEMO_L.xpt` |
| `bmx` | `p_bmx.xpt` | `BMX_L.xpt` |
| `bpxo` | `p_bpxo.xpt` | `BPXO_L.xpt` |
| `cbc` | `p_cbc.xpt` | `CBC_L.xpt` |
| `alb_cr` | `p_alb_cr.xpt` | `ALB_CR_L.xpt` |
| `biopro` | `p_biopro.xpt` | `BIOPRO_L.xpt` |
| `tchol` | `p_tchol.xpt` | `TCHOL_L.xpt` |
| `hdl` | `p_hdl.xpt` | `HDL_L.xpt` |
| `hscrp` | `p_hscrp.xpt` | `HSCRP_L.xpt` |
| `ghb` | `p_ghb.xpt` | `GHB_L.xpt` |
| `glu` | `p_glu.xpt` | `GLU_L.xpt` |
| `trigly` | `p_trigly.xpt` | `TRIGLY_L.xpt` |
| `bpq` | `p_bpq.xpt` | `BPQ_L.xpt` |
| `mcq` | `p_mcq.xpt` | `MCQ_L.xpt` |
| `diq` | `p_diq.xpt` | `DIQ_L.xpt` |
| `kiq` | `p_kiq_u.xpt` | `KIQ_U_L.xpt` |
| `tst` | Not registered | `TST_L.xpt` |
| `vid` | Not registered | `VID_L.xpt` |

Each phenotype map row names the exact source variable(s), unit and conversion. Absent mappings are structural missingness (stored zero plus missing=True), not an observed zero or a unit guessed from a related variable. Questionnaire rows specify refusal/unknown/negative/positive semantics independently of continuous phenotype missingness.

Demographics use `SEQN`, `RIDAGEYR`, `RIAGENDR` (1=male, 2=female), and `RIDSTATR`. The implemented primary age guard selects examined adults (`RIDSTATR=2`) aged **40–73 inclusive** at integer resolution. Survey analyses retain the broader examined design universe and use 40–73 as a domain flag. They also require `SDMVSTRA`, `SDMVPSU` and the **release/analysis-specific** weights, such as `WTMECPRP`/`WTINTPRP` in P and `WTMEC2YR`/`WTINT2YR` in L; subsample variables have their corresponding fasting/phlebotomy weights. Full source headers are in the registry; the map and source determine which weight/variable applies.

NHANES input is phenotype-only with inactive event arrays. The UKB `nhanes_common_eventless` adapter uses common phenotype support and no event inputs. NHANES local age recalibration is a separate sensitivity using target age labels; it is not zero-shot inference. Questionnaire associations are cross-sectional and are not incident-event validation.

### External execution and paired-bootstrap inputs

`external_validation` wraps `run_external_v2_0717.py`. Its ordered stages are `freeze`, `cache`, `infer`, `seal`, `outcomes`, `basic`, `risk`, `render`, `validate`; inspect `--help` for `--checkpoint-set main`, `mimic_common53`, or `nhanes_common_eventless`. The sequence deliberately creates a label-blind cache and seals neural predictions **before** reading outcomes. Training a support adapter is a separate `support_adapter` task with `--variant` and `--outer-fold`.

The label-blind bridge includes participant tables, pre-index ICD timelines, a separately labeled inpatient-prescription proxy sensitivity, frozen vocabulary and record-density information. Outcomes are separate. The required private UKB artifacts include all five checkpoints, each fold's `run_config.json` and `age_correction.json`, transforms, input/vocabulary/embedding manifests and the baseline comparison summaries used by the historical scientific gates. These files cannot be reconstructed from the public aggregate reference CSVs.

The primary direct/support-aligned output trees are `Revision-0710/06_figures_and_statistics/04_external_validation/core_0717_codebook_v2_huber1` and `core_0717_adapter_mimic_common53`. Set the launcher output to the corresponding canonical work tree. `infer` uses the neural environment; cache/outcome/statistic stages use the appropriate statistics/Cox dependencies. Additional historical prefreeze orchestration routines require their original run metadata and are not standalone public-data reconstruction entry points.

`external_paired` requires both trees' private `shared/06_statistics/mimic_oof_risk_predictions.parquet`. Its input contract is:

```text
subject_id: exact string identifier
organ_system: one of the seven named organs
horizon_months: numeric (analysis selects 12)
comparison: age_sex_plus_gap or age_sex_density_plus_gap
outer_fold: integer 0..4
observed_time_days: positive numeric duration
event_indicator, competing_death_indicator: 0/1
ipcw_outcome, ipcw_weight, ipcw_observed: saved censoring/evaluation components
base_risk, augmented_risk: continuous discrimination scores
base_probability, augmented_probability: horizon probabilities in [0,1]
```

The full key `(subject_id, organ_system, horizon_months, comparison, outer_fold)` is unique. Direct/adapted keys, outcomes and IPCW components must agree before comparison. Saved aggregate parity tables required by the exact replay are included in `source/Revision-0730/4-Data/MIMIC_12m_delta_c.csv` and the two `Fig13D_paired_incremental_risk_metrics.csv` source-data locations. They contain aggregate estimates, not OOF predictions.

```bash
export R13="$BUILD/Revisions-1030/work/r1_3_external_20261031/outputs/formal"
"$PY_STATS" "$PKG/run.py" --task external_paired --project-root "$BUILD" --output "$R13" -- --replicates 1000
```

The 1,000 bootstrap replicates draw participants with replacement **within hash fold**, with one shared set of draws across routes/references/organs. Saved frozen-model point estimates are checked first. External statistics pool OOF predictions within each organ and then average organs. Native external Harrell C excludes equal-duration pairs; the historical internal sksurv event/censor tie treatment differs. Do not silently harmonize them by renaming both columns “C-index”. Brier intervals and discrimination intervals should both be retained, including intervals crossing zero.

## What is reusable and what a reproducer must regenerate

| Item | Publicly included? | Reproduction action |
| --- | --- | --- |
| Study implementations, configurations, clinical slot/endpoint mappings | Yes | Check hashes; retain algorithm/version definitions |
| Selected aggregate point/CI tables | Yes | Compare the same estimand/role/horizon; no individual-level inference can be reconstructed from these alone |
| Authorized prepared UKB/MIMIC source exports | No | Obtain access and supply matching schemas/release; keep privately |
| NHANES XPT source files | No | Obtain the specified releases; verify against the registry and mappings |
| Raw phenotype masks, canonical events, folds, labels, tensors, contextual covariates | No participant data included | Regenerate from source with the above order, or supply an authorized matching archive |
| CLIP pretrained weights | No | Obtain separately, match the frozen revision/hash and tokenizer; then generate private embedding tables |
| Trained UKB EHR-AGE/comparator/support-adapter checkpoints and predictions | No | Retrain under the specified roles/configurations or obtain an appropriately authorized archive; exact frozen replay requires the original matching artifacts |
| Cox/censoring nuisance fits and individual OOF arrays | No | Fit in the allowed roles or supply the matching private frozen inputs |
| Entire original model-development search and historical environment | Not claimed as a fresh reproduction | Final frozen implementation/configuration and historical relevant sources are provided; current checks do not redo the development process |

Original-cohort gates (counts, schemas, hashes and parity checks) are intentional. Parquet writer/serialization versions can change file hashes even when table values match. A fresh-data/new-environment reconstruction must record its own provenance and compare table semantics/estimands; it must not silently replace an original hash or relabel an unmatched artifact as the original frozen replay. Data licenses and access decisions remain independent of any research-code license.

## Verification commands and evidence

Checks write their receipts to a selected private project output, outside the code package:

```bash
"$PY_STATS" "$PKG/verify.py" --scope static --project-root "$PROJECT" --output "$PROJECT/checks/static"
"$PY_STATS" "$PKG/verify.py" --scope synthetic --project-root "$PROJECT" --output "$PROJECT/checks/synthetic"
```

Static verification checks all source/delivery hashes, compiles raw and path-adapted Python, and invokes a packaged subprocess's help. Synthetic verification uses invented arrays/IDs to test the 78-slot training-only transform, L32 event packing, missing/empty-timeline padding, strict negative times and mapped Windows logical paths with filename case preserved. Neither test reads participant data or optimizes a model.

The covariance check is self-contained after installing the pinned SEM environment; it can use an otherwise empty reproduction project and requires no participant records or trained checkpoints:

```bash
export PY_SEM=/path/to/verified-sem-environment/python
"$PY_SEM" "$PKG/verify.py" --scope covariance --project-root "$PROJECT" --output "$PROJECT/checks/covariance"
```

It checks exact direct dependency versions, `pip check`, imports of the five packaged SEM/component sources, and 24 invented rows through the actual Polars/Pandas/Arrow readers. It also tests duplicate-ID rejection and a tiny invented image's Pillow round trip. Synthetic fixtures and the verification receipt are written only under `--output`.

Four existing, unchanged **aggregate-only** CSVs in `reference/sem` allow numerical reconciliation without a new fit. The check fixes every coefficient at its saved standardized estimate and evaluates semopy's implied covariance; `Model.fit` is disabled during verification. This checks library/API compatibility and the stored result calculations, not optimizer convergence or recovered parameter estimates. Local results were:

| Check against saved aggregate results | Result |
| --- | --- |
| Full-cohort off-diagonal residuals | 21 pairs, maximum absolute difference 0 |
| Full-cohort SRMR | 0.042255118396042096; absolute difference 0 |
| Largest correlation eigenvalue / total standardized variance | 91.19960382500301%, consistent with the reported rounded 91.2% |
| Baseline-free Health residual variance boundary | Saved minimum 0 retained; not relabeled as an admissible fit |
| New SEM fits, on either study or synthetic data | 0 |

The original SEM source and scientific values remain unchanged. These checks do not repeat cohort DWLS/ULS/MLW/GLS estimation, standard-error estimation, robustness/outlier analyses, or complete figure exports. Existing convergence and estimator-sensitivity results retain their original evidential scope.

Optional **restricted-artifact** checks are available only when the original private archive is mounted at the selected project root:

```bash
"$PY_COX" "$PKG/verify.py" --scope cox --project-root "$BUILD" --output "$BUILD/checks/cox"
"$PY_NEURAL" "$PKG/verify.py" --scope neural --project-root "$BUILD" --output "$BUILD/checks/neural"
```

They are not self-contained public toy examples. The Cox check recalculates two saved hepatic outer-fold-4 C statistics. The neural check restores the original Fold0 checkpoint and compares one calibration batch; it requires private cache/labels/checkpoint/predictions and performs zero optimization steps.

Previously completed bounded checks on the identical retained scientific source found exact parity for those two C statistics and for 2,048 participants × seven organ outputs in the saved neural batch (maximum absolute age/risk difference 0). External point-metric checks covered 84 values with maximum absolute difference about `5.7e-14`; a five-draw bounded bootstrap replay agreed with the corresponding prefix of the formal saved bootstrap. These are frozen-artifact checks, not independent raw-data/full-training replications. The current release preparation repeated only the affected static/interface checks, not scientific experiments.

Failure interpretation: missing participant data/checkpoints require an authorized input or a newly completed upstream stage; missing source dependencies indicate a packaging defect; a cohort/hash/point-parity mismatch requires investigation of release/roles/schema/version before numerical interpretation. Do not disable a scientific gate to obtain a nominal PASS. A bounded covariance-check PASS is not evidence that study-data SEM convergence has been retested or that biological validation is independent of model outputs.

For citation, use an actual accessible immutable commit/tag/archive after publication. A local `package_id`, directory name or preparation manifest is not a public code-release citation.
