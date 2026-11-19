# Training code included in this repository

## Scope

This source extraction adds the runnable AAEC/ADI-v2 training path that is
available in the public reference repository:
https://github.com/guohaoyuan-AI/AAEC

Source commit:
e51a778e9d19f503af346e98b60e750b68804f3e

The extracted implementation covers three formal AAEC methods:

- residual post-block Adapter;
- Scale-Shift Feature (SSF);
- LayerNorm tuning.

The two training entry points train these methods, save a final checkpoint,
run the five-point Bicubic/Bilinear validation sweep, select alpha using the
current per-view grid-best joint rule, and evaluate the selected alpha and
alpha=1 on held-out interpolation views.

LoRA and DoRA are not present in the referenced public AAEC repository and
are not part of this runnable source extraction. The repository separately
provides [LoRA/DoRA method documentation](PEFT_METHODS_LORA_DORA.md), based
on the project's recovered method implementation and historical
configurations. That document is explanatory only: it does not add LoRA/DoRA
source code, runners, configurations, or checkpoints. Do not describe this
three-method extraction as a complete five-method source release.

## Install

Use Python 3.10.8 as the reference version. The archived environment used
PyTorch 2.1.2, torchvision 0.16.2, timm 1.0.27, and NumPy 1.26.4.

    python -m pip install -r requirements-training.txt
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    export AAEC_DATA_ROOT=/path/to/datasets
    export AAEC_PRETRAINED_ROOT=/path/to/pretrained/checkpoints

The repository does not redistribute benchmark images or pretrained weights.

## Run training

Adapter:

    python tools/run_adapter_adi_v2_any_dataset.py --config configs/vit_b16/cifar100/adapter_seed42.yaml

SSF:

    python tools/run_anchor_peft_adi_v2_any_dataset.py --config configs/vit_b16/cifar100/ssf_seed42.yaml

LayerNorm tuning:

    python tools/run_anchor_peft_adi_v2_any_dataset.py --config configs/vit_b16/cifar100/ln_tuning_seed42.yaml

The configuration directory contains 30 YAML files recovered from the source
repository, spanning ViT-B/16 and ViT-S/16, CIFAR-100, Tiny-ImageNet-200 and
ImageNet-100, and the available Adapter/SSF/LayerNorm seed combinations.
Paths are environment-variable placeholders and must be set locally.

## Run the selector independently

Input CSV columns must include alpha, val_bicubic_acc, and val_bilinear_acc.
Accuracy may be fractional (0-1) or percentage (0-100).

    python scripts/select_formal_alpha.py --input validation_sweep.csv --epsilon-pp 0.5 --output selected_alpha.csv

The selector uses each view's own best accuracy over the supplied alpha grid,
requires joint feasibility within epsilon on both views, and selects the
smallest feasible alpha. Alpha=1.0 is used as fallback only if the joint
feasible set is empty. Nearest and corruption results are not accepted as
selector inputs.

## Important reproducibility limits

- The configs and runners are recovered source artifacts, not proof that every
  historical run used these exact byte-identical files.
- The training scripts now call the public preview's current grid-best
  selector; this is an explicit compatibility adaptation from the older
  alpha=1-reference selector in the source repository.
- Exact original split manifests for every dataset/run are not included.
  CIFAR loaders construct a stratified split from the training set using the
  configured seed and validation ratio.
- The Tiny-ImageNet loader expects an ImageFolder-style train/val-or-test
  directory. Prepare the official validation annotations into class folders
  before running it.
- No checkpoint, pretrained model weight, benchmark image, or training log is
  included.
- The current extraction does not include CIFAR-100-C corruption evaluation.
