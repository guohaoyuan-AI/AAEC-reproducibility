# Adapted from guohaoyuan-AI/AAEC@e51a778e9d19f503af346e98b60e750b68804f3e; uses the current per-view grid-best AAEC selector.
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from aaec.data import build_cifar10_loaders, build_cifar100_loaders
from aaec.evaluate import evaluate
from aaec.config import build_model_from_config, as_bool, load_config
from aaec.selection import FORMAL_RULE, annotate_and_select
from aaec.seed import set_seed


# =============================================================================
# ADI-Adapter for CIFAR-100 interpolation-shift evaluation
# Definition used here:
#   Standard residual adapter:       X_out = X + A(X; phi)
#   ADI residual adapter at eval:    X_out = X + alpha * A(X; phi_tuned)
# where alpha is selected only on validation Bicubic/Bilinear.
# This is feature-delta calibration for residual Adapter modules. It is distinct
# from VPT prompt-parameter interpolation and from LoRA/DoRA weight interpolation.
# =============================================================================


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def batch_to_xy(batch, device: torch.device):
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("images", batch.get("x")))
        y = batch.get("label", batch.get("labels", batch.get("target", batch.get("y"))))
    else:
        x, y = batch[0], batch[1]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def grad_norm_by_name(model: nn.Module, needle: str) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if needle in name and p.grad is not None:
            total += float(p.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


def build_loaders(cfg: dict, seed: int):
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    dataset_name = str(data_cfg.get("name", "CIFAR100")).lower()

    if dataset_name in {"cifar100", "cifar-100"}:
        from aaec.data import build_cifar100_loaders as builder
    elif dataset_name in {"cifar10", "cifar-10"}:
        from aaec.data import build_cifar10_loaders as builder
    elif dataset_name in {"tinyimagenet", "tiny-imagenet", "tiny_imagenet", "tinyimagenet200", "tiny-imagenet-200"}:
        from aaec.data.tiny_imagenet import build_tiny_imagenet_loaders as builder
    elif dataset_name in {"imagenet100", "imagenet-100", "imagefolder"}:
        from aaec.data.imagefolder_datasets import build_imagefolder_classification_loaders as builder
    else:
        raise ValueError(f"Unsupported dataset.name={data_cfg.get('name')}")

    return builder(
        root=data_cfg.get("root", "data/raw"),
        image_size=int(data_cfg.get("image_size", 224)),
        train_interpolation=data_cfg.get("train_interpolation", "bicubic"),
        val_interpolations=data_cfg.get("val_interpolations", ["bicubic", "bilinear"]),
        test_interpolations=data_cfg.get("test_interpolations", ["bicubic", "bilinear", "nearest"]),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        download=as_bool(data_cfg.get("download", False)),
        pin_memory=torch.cuda.is_available(),
    )


class ResidualAdapter(nn.Module):
    """Bottleneck residual Adapter with zero-initialized up-projection.

    The module returns only the residual delta A(x). The caller adds
    alpha * A(x) to the block output, making alpha=1 standard Adapter and
    alpha=0 the no-adapter residual path.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 32,
        dropout: float = 0.0,
        init_std: float = 1e-3,
        use_layernorm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        if activation == "relu":
            self.act = nn.ReLU(inplace=False)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=False)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("activation must be gelu, relu, or silu")

        nn.init.normal_(self.down.weight, std=float(init_std))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.down(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.up(z)
        return z


def parse_adapter_layers(num_blocks: int, spec) -> list[int]:
    if spec is None or str(spec).lower() == "all":
        return list(range(num_blocks))
    s = str(spec).lower()
    if s.startswith("last"):
        n = int(s.replace("last", ""))
        return list(range(max(0, num_blocks - n), num_blocks))
    if s.startswith("first"):
        n = int(s.replace("first", ""))
        return list(range(min(n, num_blocks)))
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    bad = [i for i in parts if i < 0 or i >= num_blocks]
    if bad:
        raise ValueError(f"Invalid adapter layer indices {bad}; num_blocks={num_blocks}")
    return parts


def inject_residual_adapters(model: nn.Module, cfg: dict) -> dict:
    method_cfg = cfg.get("method", {})
    if not hasattr(model, "blocks"):
        raise RuntimeError("Adapter injection expects a timm ViT-like model with model.blocks.")
    if not hasattr(model, "embed_dim"):
        raise RuntimeError("Adapter injection expects model.embed_dim.")

    embed_dim = int(model.embed_dim)
    num_blocks = len(model.blocks)
    bottleneck_dim = int(method_cfg.get("adapter_bottleneck_dim", 32))
    adapter_layers = parse_adapter_layers(num_blocks, method_cfg.get("adapter_layers", "all"))
    dropout = float(method_cfg.get("adapter_dropout", 0.0))
    init_std = float(method_cfg.get("adapter_init_std", 1e-3))
    use_layernorm = bool(method_cfg.get("adapter_use_layernorm", True))
    activation = str(method_cfg.get("adapter_activation", "gelu")).lower()
    location = str(method_cfg.get("adapter_location", "post_block")).lower()
    if location != "post_block":
        raise ValueError("This minimal implementation currently supports adapter_location=post_block only.")

    adapters = nn.ModuleList()
    layer_to_adapter_idx: dict[int, int] = {}
    for layer_idx in adapter_layers:
        layer_to_adapter_idx[layer_idx] = len(adapters)
        adapters.append(
            ResidualAdapter(
                dim=embed_dim,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
                init_std=init_std,
                use_layernorm=use_layernorm,
                activation=activation,
            )
        )
    model.adi_adapters = adapters
    model.adi_adapter_layer_indices = adapter_layers
    model.adi_adapter_delta_scale = 1.0

    for layer_idx, block in enumerate(model.blocks):
        if layer_idx not in layer_to_adapter_idx:
            continue
        adapter_idx = layer_to_adapter_idx[layer_idx]
        original_forward = block.forward

        def forward_with_adapter(self, x, *args, _orig=original_forward, _model=model, _idx=adapter_idx, **kwargs):
            y = _orig(x, *args, **kwargs)
            alpha = float(getattr(_model, "adi_adapter_delta_scale", 1.0))
            if abs(alpha) < 1e-12:
                return y
            delta = _model.adi_adapters[_idx](y)
            return y + alpha * delta

        block.forward = MethodType(forward_with_adapter, block)

    return {
        "peft_type": "residual_adapter",
        "adapter_location": location,
        "adapter_layers": adapter_layers,
        "num_adapter_layers": len(adapter_layers),
        "num_blocks": num_blocks,
        "embed_dim": embed_dim,
        "adapter_bottleneck_dim": bottleneck_dim,
        "adapter_dropout": dropout,
        "adapter_init_std": init_std,
        "adapter_use_layernorm": use_layernorm,
        "adapter_activation": activation,
        "adi_definition": "X_out(alpha) = X_block + alpha * A_tuned(X_block)",
        "alpha_1_meaning": "standard residual Adapter",
        "alpha_0_meaning": "no adapter residual branch; tuned head remains active",
    }


def set_adapter_alpha(model: nn.Module, alpha: float) -> None:
    model.adi_adapter_delta_scale = float(alpha)


@torch.no_grad()
def eval_one(model, loader, device, cfg, split: str, interpolation: str, alpha: float, reference_model=None):
    set_adapter_alpha(model, alpha)
    model.eval()
    return evaluate(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        split=split,
        interpolation=interpolation,
        alpha=alpha,
        reference_model=reference_model,
    )


def select_alpha(alpha_rows: list[dict], cfg: dict) -> tuple[float, str]:
    epsilon_pp = float(cfg.get("delta_interpolation", {}).get("epsilon_pp", 0.5))
    annotated, selected = annotate_and_select(alpha_rows, epsilon_pp=epsilon_pp)
    alpha_rows[:] = annotated
    return float(selected["selected_alpha"]), FORMAL_RULE

def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch: int):
    model.train()
    set_adapter_alpha(model, 1.0)
    amp = bool(cfg.get("train", {}).get("amp", False))
    grad_clip = float(cfg.get("train", {}).get("grad_clip_norm", 1.0))
    max_batches = int(cfg.get("train", {}).get("max_train_batches", 0) or 0)
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    nan_flag = 0

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        x, y = batch_to_xy(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            nan_flag = 1
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_idx}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += float(loss.detach()) * bs
        total_correct += int((logits.detach().argmax(dim=1) == y).sum())
        total_num += bs

    return {
        "epoch": epoch,
        "loss_ce": total_loss / max(total_num, 1),
        "train_acc": total_correct / max(total_num, 1),
        "num_train_samples_seen": total_num,
        "nan_flag": nan_flag,
        "grad_norm_adapter": grad_norm_by_name(model, "adi_adapters"),
        "grad_norm_head": grad_norm_by_name(model, "head"),
    }


def adapter_sanity(model: nn.Module, device: torch.device) -> dict:
    out: dict = {}
    out["has_adi_adapters"] = int(hasattr(model, "adi_adapters"))
    out["num_adapters"] = int(len(model.adi_adapters)) if hasattr(model, "adi_adapters") else 0
    if hasattr(model, "adi_adapters") and len(model.adi_adapters) > 0:
        first = model.adi_adapters[0]
        out["first_adapter_down_requires_grad"] = bool(first.down.weight.requires_grad)
        out["first_adapter_up_requires_grad"] = bool(first.up.weight.requires_grad)
        out["first_adapter_device"] = str(first.up.weight.device)
        out["first_adapter_up_init_norm"] = float(first.up.weight.detach().float().norm().item())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--alphas", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    cfg["seed"] = seed
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))

    method_name = str(cfg.get("method", {}).get("name", "adapter"))
    output_root = Path(cfg.get("project", {}).get("output_root", "outputs/fr_peft_cifar100"))
    if args.out_dir is None:
        run_dir = output_root / f"{method_name}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg, seed)

    model = build_model_from_config(cfg).to(device)
    for p in model.parameters():
        p.requires_grad = False

    report = inject_residual_adapters(model, cfg)
    model = model.to(device)

    # Train only residual adapters and classification head.
    for p in model.adi_adapters.parameters():
        p.requires_grad = True
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    trainable = count_trainable(model)
    total = count_total(model)
    sanity = adapter_sanity(model, device)

    lr = float(cfg.get("train", {}).get("lr_peft", 1e-3))
    wd = float(cfg.get("train", {}).get("weight_decay", 0.05))
    epochs = int(cfg.get("train", {}).get("peft_epochs", 15))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("train", {}).get("amp", False)))

    train_log: list[dict] = []
    for epoch in range(1, epochs + 1):
        row = train_one_epoch(model, loaders.train, optimizer, scaler, device, cfg, epoch)
        scheduler.step()
        val_bicubic = eval_one(model, loaders.val["bicubic"], device, cfg, "val", "bicubic", 1.0)
        val_bilinear = eval_one(model, loaders.val["bilinear"], device, cfg, "val", "bilinear", 1.0)
        row.update({
            "val_bicubic_acc_alpha1": float(val_bicubic["acc"]),
            "val_bilinear_acc_alpha1": float(val_bilinear["acc"]),
            "trainable_params": trainable,
            "total_params": total,
        })
        train_log.append(row)
        write_csv(run_dir / "train_log.csv", train_log)
        print(
            f"[epoch {epoch:02d}] loss={row['loss_ce']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"val_bicubic={row['val_bicubic_acc_alpha1']:.4f} "
            f"val_bilinear={row['val_bilinear_acc_alpha1']:.4f} "
            f"grad_adapter={row['grad_norm_adapter']:.4f}",
            flush=True,
        )

    ckpt_path = run_dir / "checkpoints" / f"final_{method_name}_seed{seed}.pth"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "seed": seed, "injection_report": report}, ckpt_path)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "injection_report.json").write_text(json.dumps({**report, **sanity}, indent=2, ensure_ascii=False), encoding="utf-8")

    reference_model = build_model_from_config(cfg).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    if args.alphas is not None:
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    else:
        alphas = [float(x) for x in cfg.get("delta_interpolation", {}).get("alpha_candidates", [0.2, 0.4, 0.6, 0.8, 1.0])]

    alpha_rows: list[dict] = []
    for alpha in alphas:
        vb = eval_one(model, loaders.val["bicubic"], device, cfg, "val", "bicubic", alpha)
        vl = eval_one(model, loaders.val["bilinear"], device, cfg, "val", "bilinear", alpha)
        alpha_rows.append({
            "seed": seed,
            "method": method_name,
            "alpha": alpha,
            "val_bicubic_acc": float(vb["acc"]),
            "val_bilinear_acc": float(vl["acc"]),
            "val_bicubic_loss": float(vb.get("loss_ce", 0.0)),
            "val_bilinear_loss": float(vl.get("loss_ce", 0.0)),
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_alpha_selection": 0,
        })

    selected_alpha, selection_rule = select_alpha(alpha_rows, cfg)
    write_csv(run_dir / "alpha_selection_same_checkpoint.csv", alpha_rows)

    test_rows: list[dict] = []
    for alpha in sorted(set([1.0, selected_alpha])):
        for interp, loader in loaders.test.items():
            out = eval_one(model, loader, device, cfg, "test", interp, alpha, reference_model=reference_model)
            out.update({
                "seed": seed,
                "method": method_name,
                "alpha": alpha,
                "interpolation": interp,
                "selected_alpha": selected_alpha,
                "selection_rule": selection_rule,
                "nearest_used_for_alpha_selection": 0,
                "nearest_used_for_checkpoint_selection": 0,
                "checkpoint_selection_rule": "final_checkpoint_only",
                "trainable_params": trainable,
                "total_params": total,
            })
            test_rows.append(out)
    write_csv(run_dir / "test_eval_same_checkpoint.csv", test_rows)

    def get_acc(alpha: float, interp: str) -> float:
        for r in test_rows:
            if abs(float(r["alpha"]) - float(alpha)) < 1e-12 and r["interpolation"] == interp:
                return float(r["acc"])
        return math.nan

    base_iid = get_acc(1.0, "bicubic")
    base_bilinear = get_acc(1.0, "bilinear")
    base_nearest = get_acc(1.0, "nearest")
    adi_iid = get_acc(selected_alpha, "bicubic")
    adi_bilinear = get_acc(selected_alpha, "bilinear")
    adi_nearest = get_acc(selected_alpha, "nearest")

    comp = [{
        "seed": seed,
        "method": method_name,
        "selected_alpha": selected_alpha,
        "base_alpha": 1.0,
        "base_iid_acc": base_iid,
        "adi_iid_acc": adi_iid,
        "iid_gain": adi_iid - base_iid,
        "base_bilinear_acc": base_bilinear,
        "adi_bilinear_acc": adi_bilinear,
        "bilinear_gain": adi_bilinear - base_bilinear,
        "base_nearest_acc": base_nearest,
        "adi_nearest_acc": adi_nearest,
        "nearest_gain": adi_nearest - base_nearest,
        "base_nearest_drop": base_iid - base_nearest,
        "adi_nearest_drop": adi_iid - adi_nearest,
        "drop_reduction": (base_iid - base_nearest) - (adi_iid - adi_nearest),
        "nearest_used_for_alpha_selection": 0,
        "nearest_used_for_checkpoint_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "trainable_params": trainable,
        "total_params": total,
        "nan_flag": max(int(r.get("nan_flag", 0)) for r in train_log) if train_log else 0,
        "divergence_flag": 0,
        "adi_definition": report["adi_definition"],
    }]
    write_csv(run_dir / "same_checkpoint_comparison.csv", comp)
    write_csv(run_dir / "summary.csv", comp)

    print("\n[DONE]", run_dir)
    print(json.dumps(comp[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
