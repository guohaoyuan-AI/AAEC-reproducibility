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

## LoRA/DoRA method-note provenance

The explanatory note in docs/PEFT_METHODS_LORA_DORA.md is based on the
project's recovered implementation file
src/tasfd/models/peft/lora_dora.py (SHA-256:
94742c742e33a9032dc5f46b9a4450a7ea6d13ede945b7db3ad6ba3358dba111)
and the recovered historical ViT-S/16 CIFAR-100 configurations
vits16_cifar100_lora_seed43.yaml and
vits16_cifar100_dora_seed43.yaml.

The method note summarizes the mathematical operations and implementation
semantics for documentation and traceability. It does not copy the LoRA/DoRA
runtime source into this repository. No LoRA/DoRA runner, configuration,
checkpoint, weight, or experiment result is added by this documentation
change. The public runnable training-source extraction remains limited to
Adapter, SSF, and LayerNorm tuning.
