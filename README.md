# EHR-AGE

![EHR-AGE architecture](EHR-AGE_architecture.png)

PyTorch implementation of EHR-AGE for seven age-referenced organ-state markers from baseline phenotypes and longitudinal electronic health records.

## Files

- `prepare_icd_long.py`: pair diagnosis codes and dates by participant and original Array index.
- `Dataloader.py`: input validation, training-fold imputation/standardization, missingness masks and participant-disjoint data loading.
- `model.py`: frozen semantic embeddings, contextual Transformer/BiGRU and reverse event-contribution views, shared age anchor and seven organ-specific query heads.
- `train.py`: joint age/survival objectives, training, prediction and OSAG correction.
- `main.py`: command-line entry point.
- `phenotype_columns.csv`: exact order and units of the 78 phenotype slots.

## Environment

Use Python 3.11 and install the model dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU use, install the PyTorch 2.5.1 build appropriate to your CUDA environment. ICD preparation uses only the Python standard library.

## Data processing

The diagnosis CSV requires `Participant ID` and indexed columns such as `Diagnoses - ICD10 | Array 0`; the date CSV requires the same IDs and `Date of first in-patient diagnosis - ICD10 | Array 0`, etc. Column and participant row order may differ. Each diagnosis cell contains its code followed by the original description. Both files must have unique participant keys and matching Array index sets.

```bash
python prepare_icd_long.py build --codes-csv diagnoses.csv --dates-csv dates.csv --output /your/project/inputs/icd_long.csv
```

For a packed `Diagnoses - ICD10` column containing pipe-separated entries, additionally use `--code-format packed --packed-order-is-array`, after confirming that list positions preserve the original Array indices. Empty interior positions must be retained. Missing dates stay with the corresponding diagnosis; dates without a code at the same index are rejected.

Output columns are `Participant ID`, `ICD`, `Disease_Description_Feature`, `record_time` and `source_array_index`. Dates accept ISO `YYYY-MM-DD` or year-first `YYYY/M/D` and are written as ISO dates. Build requires a new output file outside this code directory and never overwrites an existing file. To compare an existing long table without writing participant data:

```bash
python prepare_icd_long.py verify --codes-csv diagnoses.csv --dates-csv dates.csv --existing-long icd_long.csv
```

For baseline Instance 0 phenotypes, derive hand-grip average as `(left + right) / 2`, FEV1/FVC using Array 0 of both measurements, and waist/hip ratio as waist divided by hip. A missing component propagates to a missing derived value. The ordered model features are listed in `phenotype_columns.csv`.

The model consumes prepared participant arrays. Preparing these arrays additionally requires baseline assessment/birth information, pre-index diagnosis and prescription records, aligned code/text vocabulary IDs, fixed participant folds and organ-specific follow-up labels. ICD conversion alone does not generate all model inputs.

## Model inputs

`data.npz` must contain the following arrays in identical participant order. `N` is the participant count; phenotype values are supplied before standardization, with missing values represented by NaN.

| Array | Shape and meaning |
| --- | --- |
| `phenotype` | Float `[N,78]`, in `phenotype_columns.csv` order |
| `sex` | `[N]` or `[N,1]`; female=0, male=1 |
| `event_code_id`, `event_text_id` | Integer `[N,32]`; vocabulary indices, padding=0 |
| `event_relative_days` | Float `[N,32]`; active events strictly before baseline, in negative days |
| `event_type_id` | Integer `[N,32]`; padding=0, hospital diagnosis=1, GP prescription=2 |
| `event_valid_mask` | Boolean `[N,32]`; active slots left-aligned, then padding |
| `fold` | Integer `[N]`, fixed participant-level assignment 0-4 |
| `chronological_age` | Float `[N]`, baseline age in years |
| `strict_healthy_reference` | Boolean `[N]`, original strict Health reference membership |
| `duration_months` | Float `[N,7]`; at-risk follow-up durations in `(0,60]` months |
| `event`, `at_risk` | Boolean `[N,7]`, observed organ events and risk-set eligibility |

The organ order is cardiovascular, hepatic, immune, metabolic, musculoskeletal, pulmonary and renal. Use the selected 32 pre-index events in chronological order, set padding fields to zero, and use `event_code_id=0` for GP prescription slots. `clip_embeddings.npz` contains finite `code_embedding[V_code,D]` and `text_embedding[V_text,D]` matrices with the same width and padding row 0; their indices must match the event vocabularies. Keep the original code descriptions and drug text when generating the frozen semantic embeddings.

For outer fold `k`, calibration uses `(k+1) % 5`, and training uses the remaining three folds. The loader fits median imputation and standardization only on training participants and applies them to calibration/test data. OSAG age correction is fitted in the separate calibration fold.

## Run

Validate the architecture using invented inputs and no optimization:

```bash
python main.py --smoke-test --validate-only --device cpu --output-dir /your/project/checks/model
```

Train/evaluate one outer fold using authorized prepared inputs:

```bash
python main.py --data data.npz --embeddings clip_embeddings.npz --outer-fold 0 --output-dir /your/project/outputs/fold0
```

Repeat with outer folds 0-4 for the five participant-disjoint rotations. Participant-level cohort data, embedding tables and trained checkpoints are not redistributed. Model outputs are age-referenced, outcome-informed representations; correspondence with biological aging requires independent validation.

Research code and documentation are available under the [MIT License](LICENSE).
