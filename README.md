# Anchor-Aware Effect Calibration (AAEC)

**Public reproducibility preview — 0.1.0-rc1 (2026-11-19)**

This repository provides selected AAEC source code, frozen research records,
figure inputs and outputs, and evidence/provenance audits for the manuscript
**“Anchor-Aware Effect Calibration for Validation-Constrained PEFT Deployment
under Controlled Distribution Shifts.”**

It is a research reproducibility preview, not a turnkey clean-room recreation
of every historical training run. Read `docs/OPEN_ITEMS_AND_LIMITATIONS.md`
before interpreting or reusing the records.

## Contents

- `src/aaec/`: portable reference implementation of the formal joint
  Bicubic/Bilinear validation coefficient selector.
- `tests/`: unit tests for the selector.
- `code/figure_generation/`: scripts and style helpers for regenerating the
  included manuscript figures from frozen figure-data CSVs.
- `data/original_primary/`: frozen 2026-07-28 WP8-A master tables and recovered
  original 15-cell candidate-level records. The candidate-level records are
  aggregate records, not raw per-image predictions.
- `data/revision_c60/`: post-submission Reviewer-2 C60 revision evidence,
  including seed-42 B/C formal evaluation arrays/metadata, corruption subsets,
  and validation-only nine-point sensitivity records. These are not pooled
  with or substituted for the original primary cohort.
- `data/revision_night7/`: Adapter three-seed summaries and external prediction
  inventory metadata. Raw Night7 NPZ bytes are not included.
- `data/figure_data/`, `data/derived_data/`, `figures/`: frozen figure
  inputs, derived summaries, and rendered figures.
- `data/splits/`: recovered CIFAR-100 seed-42 indices for the later C60 cohort
  only; not a complete manifest for every original experiment.
- `docs/`: project scope, evidence hierarchy, selection and implementation
  audits, provenance, and release limitations.

## Evidence hierarchy

1. Original-submission records remain the primary evidence for the paper's
   main claims. The original trained checkpoints were lost when the earlier
   AutoDL instance was released; they are not restored by this package.
2. C60 consists of newly trained post-submission revision runs. It is separate
   supplementary evidence and is not a replacement for original checkpoints.
3. Night7 is run-log plus external-inventory evidence; do not describe it as
   locally independently recomputed per-image evidence.

Do not pool these evidence classes unless a manuscript explicitly identifies
and justifies a mixed-source analysis. See
`docs/DATA_PROVENANCE_AND_EVIDENCE_HIERARCHY.md`.

## Formal coefficient selection

The formal candidate grid is `{0.2, 0.4, 0.6, 0.8, 1.0}`. For each candidate,
Bicubic and Bilinear clean-validation views are assessed separately against
their own grid-best reference. The smallest coefficient jointly satisfying
both retention constraints is selected. Alpha=1 is the fallback only when
the joint feasible set is empty; alpha=1 may also be selected as an ordinary
feasible candidate. The nine-point grid is validation-only sensitivity
analysis, not a replacement formal grid.

The selection audit reports agreement for the recoverable original 15 cells
from saved candidate-level aggregate records and for ten C60 B/C runs from
per-image validation prediction archives plus frozen metadata. It does not
certify byte-identical recovery of the lost original training runners.

## Reproduction

Install the small analysis/test dependencies:

```bash
python -m pip install -r requirements-analysis.txt
python -m pip install -r requirements-training.txt
```

Run the portable selector tests:

```bash
PYTHONPATH=src pytest -q tests/test_selection.py
```

Regenerate the figures from frozen CSV inputs:

```bash
python code/figure_generation/validate_figure_data.py
python code/figure_generation/plot_all_figures.py
```

The recovered training/runtime source snapshots are deliberately excluded
from this public preview pending third-party rights and notice review.
Accordingly, this package does not claim a turnkey rerun of the historical
training jobs. Benchmark images and pretrained model weights are not
redistributed and must be obtained separately under their own terms.

## Licensing

- Original AAEC software in `src/aaec/` and `code/figure_generation/`: MIT
  (`LICENSE`).
- Author-created derived research tables, figure inputs, and manifests:
  CC BY 4.0 (`DATA_LICENSE`).
- Third-party datasets, weights, and any excluded/recovered external source
  snapshots are not relicensed. See `LICENSE_SCOPE.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. Add the final public
repository URL and DOI after the new repository and a versioned archival
release have been created.
