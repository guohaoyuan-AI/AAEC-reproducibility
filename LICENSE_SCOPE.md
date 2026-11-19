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
