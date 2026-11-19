# Project scope and evidence separation

This repository is the public reproducibility package for the AAEC
Neurocomputing revision. Its contents are limited to AAEC primary records,
AAEC revision experiments, the associated analysis/figure code, and explicitly
labelled evidence audits.

A separate project file, `ADI_v2_next_experiment_plan_zh.md` (2026-07-05),
describes ADI-v2 as follow-up research after the ADI-LoRA/ADI-DoRA submission.
It explicitly requires that Adapter, SSF, LayerNorm tuning, VPT, IA3, and later
ADI-v2 results not be mixed into the original ADI-LoRA/ADI-DoRA submission
evidence chain. That plan is therefore **not included** in this AAEC release.

Recommended handling:

- Keep the AAEC repository cited by the AAEC/Neurocomputing paper.
- Maintain ADI-v2 planning, scripts, configs, and results in a separate
  repository or clearly separate project archive.
- Do not merge the ADI-v2 future experiment plan with AAEC results or imply
  that planned experiments have already been completed.
- If a later paper uses ADI-v2 results, cite the separate versioned ADI-v2
  repository and identify the exact datasets, seeds, and completed phases.
