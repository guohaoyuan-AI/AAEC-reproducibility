# License scope and third-party materials

This repository uses a dual-license policy, with explicit exclusions.

- **MIT (`LICENSE`)** applies to original AAEC-authored software included in
  `src/aaec/` and `code/figure_generation/`, plus repository-authored utility
  scripts and documentation where applicable.
- **CC BY 4.0 (`DATA_LICENSE`)** applies to author-created derived data,
  processed tables, figure inputs, and author-created manifests, unless a
  file-specific notice says otherwise.
- Neither license relicenses benchmark datasets, pretrained model assets,
  third-party software, or material whose rights belong to others.

## Recovered training/runtime source snapshots

Recovered `tasfd` foundation/runtime source trees and C60 training/evaluation
runner snapshots are intentionally **not included in this public preview**.
Their provenance is documented in the private research audit materials, but
complete upstream copyright/license attribution has not yet been established.
They may be added in a later release after source-by-source rights review.

The package therefore provides the portable AAEC selector and figure-generation
code, processed records, audits, and figures. It is not represented as a
turnkey reproduction environment for the historical training runs.


## Extracted training source

The following files are copied from guohaoyuan-AI/AAEC at commit
e51a778e9d19f503af346e98b60e750b68804f3e and remain under Apache License 2.0;
the license text is preserved in licenses/AAEC-APACHE-2.0.txt:

- src/aaec/backbones/
- src/aaec/config.py
- src/aaec/data/
- src/aaec/evaluate.py
- src/aaec/metrics.py
- src/aaec/seed.py
- tools/run_adapter_adi_v2_any_dataset.py
- tools/run_anchor_peft_adi_v2_any_dataset.py
- configs/
- environment-training.yml

The current portable selector in src/aaec/selection.py and the standalone
selector CLI in scripts/select_formal_alpha.py remain under the repository's
MIT license. The two extracted training scripts are adapted to call that
current selector rather than the older alpha=1-reference rule.

The separate recovered tasfd runtime snapshots containing LoRA/DoRA code
remain excluded pending file-level rights and notice review.
