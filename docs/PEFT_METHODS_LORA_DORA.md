# LoRA and DoRA: method notes

## Scope

This document summarizes LoRA and DoRA as represented in the project's
recovered PEFT implementation. It makes the method background available
alongside the AAEC public training-source documentation.

This is a **method note only**. It does not add executable LoRA/DoRA source
code, training runners, configurations, model weights, checkpoints, datasets,
or experiment results. The public training-source extraction in this
repository still contains three runnable methods: Adapter, SSF, and LayerNorm
tuning. This document must not be interpreted as a five-method implementation
or a five-method reproduction package.

The equations use column-vector notation. Let a frozen linear layer have
weight \(W_0 \in \mathbb{R}^{d_{out} \times d_{in}}\), bias \(b\), and input
\(x \in \mathbb{R}^{d_{in}}\), so its output is \(y = W_0 x + b\).

## 1. LoRA

### 1.1 Low-rank update

LoRA represents the trainable weight update with two low-rank matrices:

- \(A \in \mathbb{R}^{r \times d_{in}}\);
- \(B \in \mathbb{R}^{d_{out} \times r}\);
- \(r\) is the adapter rank.

The effective update and adapted weight are

$$
\Delta W_{\mathrm{LoRA}} = \frac{\alpha_{\mathrm{L}}}{r} BA,
\qquad
W_{\mathrm{LoRA}} = W_0 + \Delta W_{\mathrm{LoRA}}.
$$

Here, \(\alpha_{\mathrm{L}}\) is the LoRA scaling hyperparameter. The layer
output can be written as

$$
y = W_0 x + b +
\frac{\alpha_{\mathrm{L}}}{r} B(Ax).
$$

The original linear weight and bias remain frozen. The low-rank matrices are
the trainable adapter parameters. In the project's ViT training helper, the
classifier head is also marked trainable.

### 1.2 Initialization

The recovered implementation initializes \(A\) with Kaiming-uniform
initialization and initializes \(B\) to zero. Therefore,

$$
\Delta W_{\mathrm{LoRA}} = 0
$$

at initialization, and wrapping the pretrained layer does not change its
initial function. The original linear layer's parameters are frozen.

The implementation applies dropout, when configured, to the input of the
low-rank branch only. The recovered CIFAR-100 ViT-S/16 configurations use
dropout 0.0.

### 1.3 Effect interpolation

The recovered module exposes a separate delta_scale value. Denoting this
interpolation coefficient by \(\lambda\), the interpolated weight is

$$
W_{\mathrm{LoRA}}(\lambda)
= W_0 + \lambda \Delta W_{\mathrm{LoRA}}.
$$

Thus, \(\lambda=0\) gives the frozen pretrained layer and \(\lambda=1\)
gives the trained LoRA endpoint. Intermediate values scale the complete
low-rank weight delta linearly.

**Parameter naming matters:** lora_alpha controls the internal LoRA scaling
\(\alpha_{\mathrm{L}}/r\), whereas delta_scale controls interpolation
between the pretrained and adapted weights. They are different quantities.
This document uses \(\lambda\) for the interpolation coefficient to avoid
overloading the word “alpha”.

## 2. DoRA

### 2.1 Weight decomposition

DoRA augments a low-rank direction update with a trainable magnitude for each
output row of the linear weight matrix.

First form the low-rank-updated direction tensor

$$
V = W_0 + \Delta W,
\qquad
\Delta W = \frac{\alpha_{\mathrm{L}}}{r} BA.
$$

Let \(\|V\|_{\mathrm{row}}\) denote the vector of row-wise L2 norms. Let
\(m \in \mathbb{R}^{d_{out}}\) contain the trainable row magnitudes. The
effective DoRA weight is

$$
W_{\mathrm{DoRA}}
= \operatorname{diag}(m)
\frac{V}{\|V\|_{\mathrm{row}}}.
$$

The division is performed independently for each output row. Equivalently,
each normalized row direction is multiplied by its corresponding learned
magnitude.

The recovered implementation freezes the original linear weight and trains
\(A\), \(B\), and \(m\). The classifier head is also trainable in the project's
ViT training helper.

### 2.2 Initialization

The implementation initializes \(A\) with Kaiming-uniform initialization,
\(B\) to zero, and \(m\) to the row-wise L2 magnitudes of \(W_0\). Consequently,
the initial effective DoRA weight is the pretrained weight (up to the
implementation's small-norm numerical clamp):

$$
W_{\mathrm{DoRA}} \approx W_0.
$$

The original row magnitudes are retained as a buffer, while the DoRA
magnitudes are trainable parameters.

### 2.3 Effect interpolation and an important distinction

The recovered DoRA module can interpolate between the pretrained weight and
the complete trained DoRA effective weight:

$$
W_{\mathrm{DoRA}}(\lambda)
= W_0 + \lambda
  \left(W_{\mathrm{DoRA}} - W_0\right).
$$

This endpoint interpolation scales the **complete effective-weight delta**,
including both the direction change and the learned magnitude change.

It is not equivalent to scaling only the low-rank direction update before
row-wise normalization, for example by recomputing a normalized weight from
\(W_0 + \lambda\Delta W\). Row-wise normalization is nonlinear, so that
operation does not in general trace the straight line between \(W_0\) and the
trained DoRA endpoint.

The recovered DoRALinear implementation requires dropout 0.0. Its source
comments explain that stochastic dropout in the direction branch cannot be
represented by the same single dense effective-weight expression used for
this exact interpolation.

The module's ability to perform endpoint interpolation should be distinguished
from the settings of an individual historical run: the recovered ViT-S/16
CIFAR-100 seed-43 DoRA configuration sets use_delta_interpolation to false.
The corresponding LoRA configuration sets it to true. This note therefore
documents the implementation capability and the recorded configuration
separately; it does not infer an interpolation or selection result for a run.

## 3. Project implementation summary

The recovered implementation wraps selected PyTorch nn.Linear modules in a
ViT-like model. The default target suffixes are:

- attn.qkv;
- attn.proj.

The injection helper can target all transformer blocks, the last \(K\) blocks,
or an explicit list of block indices. The recovered CIFAR-100 ViT-S/16
seed-43 configurations target both suffixes in all blocks and use:

| Setting | LoRA | DoRA |
|---|---:|---:|
| Rank | 16 | 16 |
| lora_alpha | 16 | 16 |
| Dropout | 0.0 | 0.0 |
| Target modules | attn.qkv, attn.proj | attn.qkv, attn.proj |
| Target blocks | all | all |
| use_delta_interpolation | true | false |

These values describe the recovered historical configuration example. They
are not universal LoRA/DoRA requirements and are not new configurations for
the current AAEC training extraction.

The implementation computes the adapted linear output directly. It does not
require an in-place merge/unmerge cycle during evaluation; changing
delta_scale changes the effective adapter contribution used by the forward
pass.

## 4. LoRA and DoRA compared

| Aspect | LoRA | DoRA |
|---|---|---|
| Adaptation form | Additive low-rank weight update | Low-rank direction update plus trainable row-wise magnitude |
| Trainable PEFT parameters | \(A, B\) | \(A, B, m\) |
| Frozen component | Original linear weight and bias | Original linear weight and bias |
| Initial behavior | Zero low-rank update | Base direction and base row magnitudes |
| Effective-weight expression | \(W_0 + (\alpha_{\mathrm{L}}/r)BA\) | Row-normalized \(W_0 + (\alpha_{\mathrm{L}}/r)BA\), scaled by \(m\) |
| Recovered implementation's dropout | Configurable; recovered configs use 0.0 | Required to be 0.0 |
| Endpoint interpolation | Linear scaling of the low-rank weight delta | Linear scaling of the complete trained DoRA effective-weight delta |

## 5. Provenance and interpretation

The method description is based on the recovered project-authored source file
src/tasfd/models/peft/lora_dora.py (SHA-256:
94742c742e33a9032dc5f46b9a4450a7ea6d13ede945b7db3ad6ba3358dba111) and
the recovered historical configurations:

- vits16_cifar100_lora_seed43.yaml;
- vits16_cifar100_dora_seed43.yaml.

The source file and historical configurations are cited here for
traceability; they are not copied into this repository by this documentation
addition. No training or evaluation results are asserted in this note.

For the scope of the current public AAEC repository, this document is
explanatory method documentation only. The runnable training-source
extraction remains limited to Adapter, SSF, and LayerNorm tuning. See
[Training code included in this repository](TRAINING_CODE.md) and
[Project scope and evidence separation](PROJECT_SCOPE.md).
