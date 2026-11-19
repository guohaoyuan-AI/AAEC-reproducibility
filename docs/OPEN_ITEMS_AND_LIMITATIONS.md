# Open items and limitations of the public preview

This release is a transparent research-data/code preview, not a claim that all
historical experiment provenance has been reconstructed. The following items
are intentionally deferred to a later revision.

## 1. Exact dataset split manifests and hashes — deferred

The package includes the recovered CIFAR-100 seed-42 split indices used by the
later C60 revision cohort. It does not include complete exact split manifests
for every original CIFAR-100 run, Tiny-ImageNet-200, or ImageNet-100. The
original primary cohort is supported by frozen aggregate/candidate-level
records, not by a full set of raw per-image predictions and original weights.
Users must obtain benchmark images from their official sources and comply with
their terms. No missing split is reconstructed or inferred in this release.

## 2. ImageNet-100 seed-43/44 formal source records — deferred

The original evidence package has a limited ImageNet-100 pilot scope. This
public preview does not claim complete seed-43/44 formal validation-selection
records for the original cohort. Do not infer multi-seed ImageNet-100
generalization from incomplete records. See the primary data report and
docs/DATA_PROVENANCE_AND_EVIDENCE_HIERARCHY.md.

## 3. Selection-rule and paper-table consistency — audit evidence included

The formal selector and the 2026-11-13 selection audit are included. The
audit reports independent agreement for the recoverable original 15 cells
(candidate-level aggregate reconstruction) and the ten C60 B/C validation
records (per-image prediction archives plus frozen selection metadata). It
also states that the lost original training runners cannot be certified
byte-for-byte and that Night7 selection cannot be independently recomputed
from local per-image predictions because the raw NPZ bytes are absent. These
limitations remain part of the evidence boundary; this release does not claim
that every historical runner was re-executed or byte-identically recovered.

The formal five-point grid is {0.2, 0.4, 0.6, 0.8, 1.0}. The documented
formal rule uses each clean validation view's own grid-best reference, requires
joint feasibility on Bicubic and Bilinear, and selects the smallest feasible
coefficient; alpha=1 is the fallback only when the joint feasible set is empty.
The nine-point grid is validation-only sensitivity evidence and does not alter
the formal five-point protocol.

## 4. Third-party code attribution and PEFT method documentation

Recovered internal tasfd foundation/runtime and C60 training/evaluation
snapshots remain excluded because complete upstream license/notice provenance
has not been established. They are not covered by this repository's MIT
license. Separately, the repository tree includes the public AAEC source
extraction for Adapter, SSF, and LayerNorm tuning under Apache-2.0, with its
license preserved in licenses/AAEC-APACHE-2.0.txt. The standalone selector
and CLI use the current per-view grid-best rule.

A descriptive LoRA/DoRA method note is now included at
docs/PEFT_METHODS_LORA_DORA.md. It summarizes the recovered project method
implementation and historical configuration semantics for documentation and
traceability only. The LoRA/DoRA runtime source, training runner, configurations,
checkpoints, and weights are not included in this public source extraction.
The method note must not be interpreted as runnable LoRA/DoRA support or as
evidence that the repository contains all five PEFT implementations.

## Evidence classes that must not be conflated

- Original primary cohort: frozen historical records; old trained checkpoints
  were lost and are not included.
- C60: later post-submission seed-42 B/C revision experiments; separate from
  the original primary cohort.
- Night7: later Adapter extension supported by logs and external inventory
  hashes/array metadata; underlying NPZ bytes are absent here, so it is not
  locally independently recomputed per-image evidence.

These limitations constrain what can responsibly be claimed from this public package.
