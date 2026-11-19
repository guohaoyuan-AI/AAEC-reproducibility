# Browse and run the AAEC reference selector

This directory exposes the portable selector source directly in GitHub, so a
reader does not need to download the full reproducibility ZIP just to inspect
the formal coefficient-selection implementation.

## Core files

- [`src/aaec/selection.py`](../src/aaec/selection.py): portable reference
  implementation of the formal five-point, joint Bicubic/Bilinear
  clean-validation selector.
- [`tests/test_selection.py`](../tests/test_selection.py): unit tests for
  smallest-jointly-feasible selection, the distinction between a feasible
  alpha=1.0 and empty-set fallback, and input-shape validation.
- [`requirements-analysis.txt`](../requirements-analysis.txt): lightweight
  analysis and test dependencies.
- [`CODE_MIRROR_MANIFEST.csv`](../CODE_MIRROR_MANIFEST.csv): SHA-256 values
  tying these directly browsable files to the corresponding members of the
  public preview ZIP.

## Run the selector tests

From the repository root:

```bash
python -m pip install -r requirements-analysis.txt
PYTHONPATH=src python -m pytest -q tests/test_selection.py
```

## Scope

The selector accepts a matrix with one row per candidate alpha and two
columns ordered as Bicubic and Bilinear validation accuracy. Accuracies are
fractions in [0, 1], while `epsilon_pp` is expressed in percentage points.
It computes each view's own best candidate reference, forms the joint
feasible set, and selects the smallest feasible alpha. If that set is empty,
it returns alpha=1.0 and records the fallback state separately.

This is a portable reference implementation corresponding to the formal
selector documented in the manuscript and archived C60 evaluator. It does
not claim byte-identical recovery of the lost historical training runner,
nor does it provide the complete five-family training environment.

The figure-generation scripts, figure-data CSVs, processed experimental
records, and the full provenance package remain in the downloadable
[reproducibility ZIP](../AAEC_reproducibility_public_preview_0.1.0-rc1_20261119_FIXED.zip).
