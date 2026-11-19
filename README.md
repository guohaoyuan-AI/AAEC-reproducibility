# Anchor-Aware Effect Calibration (AAEC)

**Public reproducibility preview — 0.1.0-rc1 (2026-11-19)**

This repository hosts the code, derived experimental records, figures, and
provenance audits supporting the manuscript:

**“Anchor-Aware Effect Calibration for Validation-Constrained PEFT Deployment
under Controlled Distribution Shifts.”**

## Download the reproducibility package

The complete preview bundle is stored as a ZIP archive in this repository:

[**AAEC reproducibility preview 0.1.0-rc1 (ZIP)**](AAEC_reproducibility_public_preview_0.1.0-rc1_20261119_FIXED.zip)

- Archive size: 10,427,163 bytes (approximately 9.94 MiB)
- Files in the archive: 822 (plus directory entries)
- NPZ prediction archives: 145
- NPY split-index arrays: 2
- Model-weight files: none detected
- SHA-256: 64881aabcc8bf7d5c34df6765201aa9581474bdde466500f12b1ca1f3855ebd5

The archive contains the release directory
AAEC_public_release_candidate_20261119/. Download and extract the ZIP to
browse the source, data, documentation, and figures. The archive's
SHA256SUMS.csv and scripts/verify_release.py provide file-level integrity
and content checks. The uploaded GitHub object was verified to match the local
archive byte-for-byte using its Git blob SHA-1.

## Package scope

The package includes:

- a portable reference implementation of the formal joint
  Bicubic/Bilinear validation coefficient selector;
- figure-generation scripts and frozen figure-data CSVs;
- original-primary aggregate/candidate-level records;
- separate post-submission C60 revision prediction arrays and metadata;
- Adapter Night7 three-seed summaries and external prediction-inventory
  metadata;
- selection-rule, implementation, code-provenance, and data-provenance audits.

This is a research reproducibility preview, not a turnkey clean-room recreation
of every historical training run. The original trained checkpoints were lost
when the earlier AutoDL instance was released and are not included.

## Evidence boundaries

1. Original-submission records remain the primary evidence for the paper's
   main claims. The recovered original candidate-level records are aggregate
   records, not raw per-image prediction archives.
2. C60 consists of post-submission Reviewer-2 revision experiments. It is
   separate supplementary evidence and must not be pooled with or substituted
   for the original primary cohort.
3. Night7 is run-log plus external-inventory evidence. The raw Night7 NPZ
   bytes are not included, so it is not locally independently recomputed
   per-image evidence.

The package documents deferred provenance items, including complete split
manifests for all original datasets, additional ImageNet-100 seed-43/44
formal records, and source-by-source third-party license review. Do not infer
that these items have been completed. See
AAEC_public_release_candidate_20261119/docs/OPEN_ITEMS_AND_LIMITATIONS.md
inside the archive.

## Reproduction and verification

After extracting the archive, follow its README.md. The portable selector
tests and figure regeneration instructions are documented there.

To verify the archive itself:

    sha256sum AAEC_reproducibility_public_preview_0.1.0-rc1_20261119_FIXED.zip

Expected SHA-256:

    64881aabcc8bf7d5c34df6765201aa9581474bdde466500f12b1ca1f3855ebd5

Then extract the archive and run:

    python AAEC_public_release_candidate_20261119/scripts/verify_release.py

## Licensing

- Portable selector and figure-generation software: MIT (LICENSE).
- Extracted training modules, runners, and configurations: Apache-2.0
  (licenses/AAEC-APACHE-2.0.txt).
- Author-created derived research records and figure data: CC BY 4.0
  (DATA_LICENSE).
- Third-party datasets, pretrained weights, and excluded external source
  snapshots are not relicensed. See LICENSE_SCOPE.md and the archive's
  provenance documents.

## Citation

Citation metadata are provided in CITATION.cff. This repository currently
contains a public preview commit, not a DOI-backed archival release. A
versioned GitHub release and an archival DOI may be added after coauthor
approval and the remaining publication checks.

## Training source and PEFT method notes

The repository exposes the training source recovered from the public AAEC
reference repository. It includes the residual Adapter, SSF, and
LayerNorm-tuning runners, shared ViT/data/evaluation modules, 30 YAML
configurations, and an independently runnable formal selector.

Start with [docs/TRAINING_CODE.md](docs/TRAINING_CODE.md).

### LoRA and DoRA method documentation

[docs/PEFT_METHODS_LORA_DORA.md](docs/PEFT_METHODS_LORA_DORA.md) documents the
core LoRA and DoRA formulations and the implementation semantics recovered
from the project's historical source. It distinguishes the low-rank update
in LoRA from DoRA's direction-plus-magnitude decomposition, and explains the
project implementation's endpoint interpolation semantics.

This is explanatory documentation only. It does not add runnable LoRA/DoRA
source code, runners, configurations, checkpoints, or weights. The public
training-source extraction therefore remains three runnable methods; this
repository must not be described as a five-method executable implementation
or a complete five-method clean-room reproduction.

The ZIP preview remains the frozen research-data/figure package described
above; the training source is exposed as ordinary repository files.
