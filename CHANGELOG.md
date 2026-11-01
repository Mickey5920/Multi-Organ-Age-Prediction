# Version history

## r2-20261101 — 1 November 2026

- Add the retained study implementations, configurations and aggregate references for the second-round manuscript revision.
- Document input schemas, phenotype order and units, preprocessing tasks, participant-disjoint roles, training and evaluation commands, dependencies and verification boundaries.
- Add upstream ICD preparation by participant and original Array index, with a read-only comparison mode and synthetic regression tests.
- Include the information-matched clinical comparison, paired external uncertainty analysis, baseline-only training-reference sensitivity and corresponding aggregate reference results.
- Include the tested SEM environment specifications and the aggregate-only Figure 7 renderer with formal paired macro intervals.
- Apply the MIT license to the research code and documentation.
- Preserve the earlier root-level model entry points and architecture image unchanged; the expanded study workflow uses `run.py`, `source/`, and the standalone entries documented in README.

Verification includes 12 ICD synthetic tests and a read-only comparison of all 502,163 participants / 7,071,243 existing long-table rows with zero diagnosis/description/date differences. Existing bounded frozen-artifact and SEM checks retain the scope described in README. A complete raw-data reconstruction or full training replication was not repeated to prepare this source release.

## before-round2 — earlier public snapshot

Preserves public commit [`e5613bd9a59d78dd81148f913255605b2623acdd`](https://github.com/Mickey5920/Multi-Organ-Age-Prediction/commit/e5613bd9a59d78dd81148f913255605b2623acdd), containing `Dataloader.py`, `model.py`, `train.py`, `main.py`, the architecture image and the earlier README. The `before-round2` tag identifies that snapshot without rewriting its history.
