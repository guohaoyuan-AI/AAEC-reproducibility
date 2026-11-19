# Adapted from guohaoyuan-AI/AAEC@e51a778e9d19f503af346e98b60e750b68804f3e; uses the current per-view grid-best AAEC selector.
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from aaec.data import build_cifar10_loaders, build_cifar100_loaders
from aaec.evaluate import evaluate
from aaec.config import build_model_from_config, as_bool, load_config
from aaec.selection import FORMAL_RULE, annotate_and_select
from aaec.seed import set_seed


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


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


def grad_norm_trainable(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += float(p.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


def as_float(value, default: float = math.nan) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def metric_mean(row: dict, name: str) -> float:
    values = [
        as_float(row.get(f"val_bicubic_{name}")),
        as_float(row.get(f"val_bilinear_{name}")),
    ]
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


class SSFWrapper(nn.Module):
    """Scale-shift feature modulation with identity anchor."""

    def __init__(self, base: nn.Module, dim: int):
        super().__init__()
        self.base = base
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.delta_scale = 1.0

    def forward(self, x):
        h = self.base(x)
        alpha = float(self.delta_scale)
        gamma_eff = 1.0 + alpha * (self.gamma - 1.0)
        beta_eff = alpha * self.beta
        shape = [1] * h.ndim
        shape[-1] = -1
        return h * gamma_eff.view(*shape) + beta_eff.view(*shape)


class IA3GateWrapper(nn.Module):
    """IA3-style output channel gate with identity anchor."""

    def __init__(self, base: nn.Module, dim: int):
        super().__init__()
        self.base = base
        self.gate = nn.Parameter(torch.ones(dim))
        self.delta_scale = 1.0

    def forward(self, x):
        h = self.base(x)
        alpha = float(self.delta_scale)
        gate_eff = 1.0 + alpha * (self.gate - 1.0)
        shape = [1] * h.ndim
        shape[-1] = -1
        return h * gate_eff.view(*shape)


def inject_ssf(model: nn.Module, cfg: dict) -> dict:
    method_cfg = cfg.get("method", {})
    targets = method_cfg.get("ssf_targets", ["norm1", "norm2"])
    inject_final_norm = bool(method_cfg.get("ssf_final_norm", False))
    report = []
    for block_idx, block in enumerate(model.blocks):
        for target in targets:
            if not hasattr(block, target):
                continue
            base = getattr(block, target)
            if not isinstance(base, nn.LayerNorm):
                continue
            dim = int(base.normalized_shape[0])
            setattr(block, target, SSFWrapper(base, dim))
            report.append(f"blocks.{block_idx}.{target}")
    if inject_final_norm and hasattr(model, "norm") and isinstance(model.norm, nn.LayerNorm):
        dim = int(model.norm.normalized_shape[0])
        model.norm = SSFWrapper(model.norm, dim)
        report.append("norm")
    return {
        "peft_type": "ssf",
        "calibration_object": "feature scale-shift modulation",
        "anchor": "gamma=1, beta=0",
        "adi_definition": "gamma(alpha)=1+alpha*(gamma_tuned-1), beta(alpha)=alpha*beta_tuned",
        "injected_modules": report,
    }


def inject_ia3(model: nn.Module, cfg: dict) -> dict:
    method_cfg = cfg.get("method", {})
    targets = method_cfg.get("ia3_targets", ["attn.proj", "mlp.fc2"])
    report = []
    dim = int(model.embed_dim)
    for block_idx, block in enumerate(model.blocks):
        if "attn.proj" in targets and hasattr(block, "attn") and hasattr(block.attn, "proj"):
            block.attn.proj = IA3GateWrapper(block.attn.proj, dim)
            report.append(f"blocks.{block_idx}.attn.proj")
        if "mlp.fc2" in targets and hasattr(block, "mlp") and hasattr(block.mlp, "fc2"):
            block.mlp.fc2 = IA3GateWrapper(block.mlp.fc2, dim)
            report.append(f"blocks.{block_idx}.mlp.fc2")
    return {
        "peft_type": "ia3",
        "calibration_object": "channel gate",
        "anchor": "gate=1",
        "adi_definition": "gate(alpha)=1+alpha*(gate_tuned-1)",
        "injected_modules": report,
    }


def mark_layernorm_tuning(model: nn.Module) -> dict:
    anchors = {}
    report = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            anchors[name] = {
                "weight": module.weight.detach().clone(),
                "bias": module.bias.detach().clone() if module.bias is not None else None,
            }
            module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
            report.append(name)
    model.ln_tuning_anchors = anchors
    return {
        "peft_type": "ln_tuning",
        "calibration_object": "LayerNorm affine parameter subspace",
        "anchor": "pretrained LayerNorm weight/bias",
        "adi_definition": "theta_LN(alpha)=theta0+alpha*(theta_tuned-theta0)",
        "injected_modules": report,
    }


def snapshot_layernorm_tuned(model: nn.Module) -> None:
    tuned = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and hasattr(model, "ln_tuning_anchors") and name in model.ln_tuning_anchors:
            tuned[name] = {
                "weight": module.weight.detach().clone(),
                "bias": module.bias.detach().clone() if module.bias is not None else None,
            }
    model.ln_tuning_tuned = tuned


@torch.no_grad()
def set_layernorm_alpha(model: nn.Module, alpha: float) -> None:
    if not hasattr(model, "ln_tuning_anchors") or not hasattr(model, "ln_tuning_tuned"):
        return
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm) or name not in model.ln_tuning_anchors:
            continue
        w0 = model.ln_tuning_anchors[name]["weight"].to(module.weight.device)
        wt = model.ln_tuning_tuned[name]["weight"].to(module.weight.device)
        module.weight.copy_(w0 + float(alpha) * (wt - w0))
        if module.bias is not None and model.ln_tuning_anchors[name]["bias"] is not None:
            b0 = model.ln_tuning_anchors[name]["bias"].to(module.bias.device)
            bt = model.ln_tuning_tuned[name]["bias"].to(module.bias.device)
            module.bias.copy_(b0 + float(alpha) * (bt - b0))


def set_anchor_peft_alpha(model: nn.Module, peft_type: str, alpha: float) -> None:
    if peft_type == "ln_tuning":
        set_layernorm_alpha(model, alpha)
        return
    for module in model.modules():
        if isinstance(module, (SSFWrapper, IA3GateWrapper)):
            module.delta_scale = float(alpha)


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


def inject_method(model: nn.Module, cfg: dict) -> dict:
    peft_type = str(cfg.get("method", {}).get("peft_type", "ssf")).lower()
    if peft_type == "ssf":
        return inject_ssf(model, cfg)
    if peft_type == "ia3":
        return inject_ia3(model, cfg)
    if peft_type == "ln_tuning":
        return mark_layernorm_tuning(model)
    raise ValueError(f"Unsupported peft_type={peft_type}")


def mark_trainable(model: nn.Module, peft_type: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if peft_type == "ssf":
        for module in model.modules():
            if isinstance(module, SSFWrapper):
                module.gamma.requires_grad = True
                module.beta.requires_grad = True
    elif peft_type == "ia3":
        for module in model.modules():
            if isinstance(module, IA3GateWrapper):
                module.gate.requires_grad = True
    elif peft_type == "ln_tuning":
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, peft_type: str, epoch: int):
    model.train()
    set_anchor_peft_alpha(model, peft_type, 1.0)
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
        bs = int(y.shape[0])
        total_loss += float(loss.detach()) * bs
        total_correct += int((logits.detach().argmax(dim=1) == y).sum())
        total_num += bs
    return {
        "epoch": epoch,
        "loss_ce": total_loss / max(total_num, 1),
        "train_acc": total_correct / max(total_num, 1),
        "num_train_samples_seen": total_num,
        "nan_flag": nan_flag,
        "grad_norm": grad_norm_trainable(model),
    }


@torch.no_grad()
def eval_one(model, loader, device, cfg, split: str, interpolation: str, alpha: float, peft_type: str, reference_model=None):
    set_anchor_peft_alpha(model, peft_type, alpha)
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


def select_rules(rows: list[dict], epsilon_pp: float, bicubic_guard_pp: float) -> list[dict]:
    del bicubic_guard_pp  # retained in the CLI/config for backward compatibility
    annotated, selected = annotate_and_select(rows, epsilon_pp=epsilon_pp)
    rows[:] = annotated
    return [{
        "selection_rule": FORMAL_RULE,
        "selected_alpha": float(selected["selected_alpha"]),
        "epsilon_pp": float(epsilon_pp),
        "selection_reference": "per_view_grid_best",
        "selected_val_bicubic_acc": as_float(selected["val_bicubic_acc"]),
        "selected_val_bilinear_acc": as_float(selected["val_bilinear_acc"]),
        "selected_bicubic_drop_pp": as_float(selected["bicubic_drop_pp"]),
        "selected_bilinear_drop_pp": as_float(selected["bilinear_drop_pp"]),
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
    }]

def get_acc(rows: list[dict], alpha: float, interp: str) -> float:
    for row in rows:
        if abs(as_float(row["alpha"]) - float(alpha)) < 1e-12 and row["interpolation"] == interp:
            return as_float(row["acc"])
    return math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--alphas", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    cfg["seed"] = seed
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))

    method_cfg = cfg.get("method", {})
    method = str(method_cfg.get("name", method_cfg.get("peft_type", "anchor_peft")))
    peft_type = str(method_cfg.get("peft_type", "ssf")).lower()
    output_root = Path(cfg.get("project", {}).get("output_root", "outputs/anchor_peft_adi_v2"))
    run_dir = Path(args.out_dir) if args.out_dir else output_root / f"{method}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg, seed)
    model = build_model_from_config(cfg).to(device)
    for p in model.parameters():
        p.requires_grad = False
    report = inject_method(model, cfg)
    mark_trainable(model, peft_type)
    model = model.to(device)
    trainable = count_trainable(model)
    total = count_total(model)

    epochs = int(cfg.get("train", {}).get("peft_epochs", 15))
    lr = float(cfg.get("train", {}).get("lr_peft", 1e-3))
    wd = float(cfg.get("train", {}).get("weight_decay", 0.05))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("train", {}).get("amp", False)))

    train_rows = []
    for epoch in range(1, epochs + 1):
        row = train_one_epoch(model, loaders.train, optimizer, scaler, device, cfg, peft_type, epoch)
        scheduler.step()
        vb = eval_one(model, loaders.val["bicubic"], device, cfg, "val", "bicubic", 1.0, peft_type)
        vl = eval_one(model, loaders.val["bilinear"], device, cfg, "val", "bilinear", 1.0, peft_type)
        row.update({
            "seed": seed,
            "method": method,
            "peft_type": peft_type,
            "val_bicubic_acc_alpha1": as_float(vb["acc"]),
            "val_bilinear_acc_alpha1": as_float(vl["acc"]),
            "trainable_params": trainable,
            "total_params": total,
        })
        train_rows.append(row)
        write_csv(run_dir / "train_log.csv", train_rows)
        print(
            f"[epoch {epoch:02d}] loss={row['loss_ce']:.4f} train_acc={row['train_acc']:.4f} "
            f"val_bicubic={row['val_bicubic_acc_alpha1']:.4f} val_bilinear={row['val_bilinear_acc_alpha1']:.4f}",
            flush=True,
        )

    if peft_type == "ln_tuning":
        snapshot_layernorm_tuned(model)

    ckpt_path = ckpt_dir / f"final_{method}_seed{seed}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "seed": seed,
            "injection_report": report,
            "ln_tuning_anchors": getattr(model, "ln_tuning_anchors", None),
            "ln_tuning_tuned": getattr(model, "ln_tuning_tuned", None),
        },
        ckpt_path,
    )
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "injection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    reference_model = build_model_from_config(cfg).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    alphas = [float(x) for x in (args.alphas or ",".join(map(str, cfg.get("delta_interpolation", {}).get("alpha_candidates", [0.2, 0.4, 0.6, 0.8, 1.0])))).split(",") if x.strip()]

    val_rows = []
    for alpha in alphas:
        vb = eval_one(model, loaders.val["bicubic"], device, cfg, "val", "bicubic", alpha, peft_type, reference_model)
        vl = eval_one(model, loaders.val["bilinear"], device, cfg, "val", "bilinear", alpha, peft_type, reference_model)
        row = {
            "seed": seed,
            "method": method,
            "peft_type": peft_type,
            "alpha": alpha,
            "val_bicubic_acc": as_float(vb["acc"]),
            "val_bilinear_acc": as_float(vl["acc"]),
            "val_bicubic_loss": as_float(vb.get("loss_ce")),
            "val_bilinear_loss": as_float(vl.get("loss_ce")),
            "val_bicubic_lfer": as_float(vb.get("lfer")),
            "val_bilinear_lfer": as_float(vl.get("lfer")),
            "val_bicubic_cka_to_frozen": as_float(vb.get("cka_to_frozen")),
            "val_bilinear_cka_to_frozen": as_float(vl.get("cka_to_frozen")),
            "val_bicubic_specdist_to_frozen": as_float(vb.get("specdist_to_frozen")),
            "val_bilinear_specdist_to_frozen": as_float(vl.get("specdist_to_frozen")),
            "val_bicubic_high_specdist_to_frozen": as_float(vb.get("high_specdist_to_frozen")),
            "val_bilinear_high_specdist_to_frozen": as_float(vl.get("high_specdist_to_frozen")),
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_alpha_selection": 0,
            "checkpoint_selection_rule": "final_checkpoint_only",
        }
        val_rows.append(row)
        print(f"[val sweep] alpha={alpha:.2f} bicubic={row['val_bicubic_acc']:.4f} bilinear={row['val_bilinear_acc']:.4f}")

    eps = float(cfg.get("delta_interpolation", {}).get("epsilon_pp", 0.5))
    guard = float(cfg.get("delta_interpolation", {}).get("bicubic_guard_pp", 0.5))
    selections = select_rules(val_rows, eps, guard)
    write_csv(run_dir / "validation_anchor_sweep.csv", val_rows)
    write_csv(run_dir / "selected_alpha_by_rule.csv", selections)

    selected_alphas = sorted({1.0} | {as_float(row["selected_alpha"]) for row in selections})
    rule_by_alpha = {}
    for row in selections:
        rule_by_alpha.setdefault(as_float(row["selected_alpha"]), []).append(row["selection_rule"])

    test_rows = []
    for alpha in selected_alphas:
        for interp, loader in loaders.test.items():
            out = eval_one(model, loader, device, cfg, "test", interp, alpha, peft_type, reference_model)
            test_row = dict(out)
            test_row.update({
                "seed": seed,
                "method": method,
                "peft_type": peft_type,
                "alpha": alpha,
                "interpolation": interp,
                "selected_by_rules": ";".join(rule_by_alpha.get(alpha, ["alpha1_baseline"])),
                "nearest_used_for_alpha_selection": 0,
                "corruption_used_for_alpha_selection": 0,
                "checkpoint_selection_rule": "final_checkpoint_only",
                "trainable_params": trainable,
                "total_params": total,
            })
            test_rows.append(test_row)
            print(f"[test] alpha={alpha:.2f} interp={interp:<8} acc={as_float(test_row['acc']):.4f}", flush=True)
    write_csv(run_dir / "test_selected_alphas.csv", test_rows)

    base_bicubic = get_acc(test_rows, 1.0, "bicubic")
    base_bilinear = get_acc(test_rows, 1.0, "bilinear")
    base_nearest = get_acc(test_rows, 1.0, "nearest")
    summary_rows = []
    for sel in selections:
        alpha = as_float(sel["selected_alpha"])
        bicubic = get_acc(test_rows, alpha, "bicubic")
        bilinear = get_acc(test_rows, alpha, "bilinear")
        nearest = get_acc(test_rows, alpha, "nearest")
        base_drop = base_bicubic - base_nearest
        adi_drop = bicubic - nearest
        summary_rows.append({
            "seed": seed,
            "method": method,
            "peft_type": peft_type,
            "selection_rule": sel["selection_rule"],
            "selected_alpha": alpha,
            "base_bicubic_acc": base_bicubic,
            "adi_bicubic_acc": bicubic,
            "bicubic_gain_pp": (bicubic - base_bicubic) * 100.0,
            "base_bilinear_acc": base_bilinear,
            "adi_bilinear_acc": bilinear,
            "bilinear_gain_pp": (bilinear - base_bilinear) * 100.0,
            "base_nearest_acc": base_nearest,
            "adi_nearest_acc": nearest,
            "nearest_gain_pp": (nearest - base_nearest) * 100.0,
            "base_nearest_drop_pp": base_drop * 100.0,
            "adi_nearest_drop_pp": adi_drop * 100.0,
            "drop_reduction_pp": (base_drop - adi_drop) * 100.0,
            "relative_drop_reduction_pct": ((base_drop - adi_drop) / base_drop * 100.0) if base_drop > 0 else math.nan,
            "go_nearest_ge_1p5pp": int((nearest - base_nearest) * 100.0 >= 1.5),
            "go_clean_drop_le_0p5pp": int((base_bicubic - bicubic) * 100.0 <= 0.5 and (base_bilinear - bilinear) * 100.0 <= 0.5),
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_alpha_selection": 0,
            "checkpoint_selection_rule": "final_checkpoint_only",
            "trainable_params": trainable,
            "total_params": total,
            "checkpoint_path": str(ckpt_path),
        })
    write_csv(run_dir / "summary_by_rule.csv", summary_rows)
    print(f"[done] wrote {run_dir}")


if __name__ == "__main__":
    main()
