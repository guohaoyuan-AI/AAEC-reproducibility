# Source attribution

The following source files were extracted from the public repository
https://github.com/guohaoyuan-AI/AAEC at commit
e51a778e9d19f503af346e98b60e750b68804f3e.

The source repository declares Apache License 2.0. Its license text is
preserved at licenses/AAEC-APACHE-2.0.txt.

Extracted source groups:

- src/aaec/backbones/
- src/aaec/config.py
- src/aaec/data/
- src/aaec/evaluate.py
- src/aaec/metrics.py
- src/aaec/seed.py
- tools/run_adapter_adi_v2_any_dataset.py
- tools/run_anchor_peft_adi_v2_any_dataset.py
- configs/ (30 YAML configurations)
- environment-training.yml

The two training entry points were adapted only at the alpha-selection
interface: calls to the historical alpha=1-reference selector were replaced
with calls to the current per-view grid-best selector. This change is
documented in docs/TRAINING_CODE.md.

LoRA/DoRA code is not extracted from this repository because it is not present
in its public tree. No internal runtime snapshot is redistributed here.
