"""Portable AAEC joint clean-view coefficient selector.

The formal rule uses each validation view's own best candidate on the supplied
alpha grid, requires joint Bicubic/Bilinear feasibility, and selects the
smallest feasible alpha. If the joint feasible set is empty, alpha=1.0 is the
prespecified fallback. Accuracies may be supplied as fractions or percentages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)
FORMAL_RULE = "per_view_grid_best_joint_smallest_feasible_v1"


@dataclass(frozen=True)
class SelectionResult:
    selected_alpha: float
    feasible_alphas: tuple[float, ...]
    joint_feasible: bool
    fallback_alpha1: bool
    reference_accuracies: tuple[float, float]
    selected_accuracies: tuple[float, float]
    epsilon_pp: float


def select_alpha(
    accuracies: Sequence[Sequence[float]],
    epsilon_pp: float = 0.5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> SelectionResult:
    """Select the smallest alpha satisfying both clean-view retention constraints.

    accuracies has shape (len(alphas), 2), with columns (Bicubic, Bilinear).
    Values must be fractional accuracies in [0, 1]. Epsilon is in percentage
    points. Alpha=1.0 is used only when the joint feasible set is empty.
    """
    a = np.asarray(accuracies, dtype=np.float64)
    alphas = tuple(float(x) for x in alphas)
    if not alphas or a.ndim != 2 or a.shape != (len(alphas), 2):
        raise ValueError(f"expected accuracy shape {(len(alphas), 2)}, got {a.shape}")
    if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("accuracies must be finite fractions in [0, 1]")
    if epsilon_pp < 0:
        raise ValueError("epsilon_pp must be non-negative")
    if tuple(sorted(alphas)) != alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be unique and sorted ascending")
    if 1.0 not in alphas:
        raise ValueError("alpha=1.0 must be present for the prescribed fallback")

    refs = a.max(axis=0)
    eps = float(epsilon_pp) / 100.0
    feasible_idx = np.flatnonzero(np.all(a >= refs[None, :] - eps - 1e-12, axis=1))
    fallback = len(feasible_idx) == 0
    idx = int(feasible_idx[0]) if not fallback else alphas.index(1.0)
    feasible = tuple(alphas[int(i)] for i in feasible_idx)
    return SelectionResult(
        selected_alpha=alphas[idx],
        feasible_alphas=feasible,
        joint_feasible=not fallback,
        fallback_alpha1=fallback,
        reference_accuracies=(float(refs[0]), float(refs[1])),
        selected_accuracies=(float(a[idx, 0]), float(a[idx, 1])),
        epsilon_pp=float(epsilon_pp),
    )


def _find_column(row: Mapping, candidates: Sequence[str]) -> str:
    for key in candidates:
        if key in row:
            return key
    raise KeyError(f"None of the required columns are present: {candidates}")


def annotate_and_select(
    rows: Iterable[Mapping],
    epsilon_pp: float = 0.5,
    reference_alpha: float | None = None,
) -> tuple[list[dict], dict]:
    """Annotate CSV-like validation rows using the current grid-best rule.

    reference_alpha is rejected intentionally: it belongs to the older
    alpha=1-reference selector and must not silently enter the current protocol.
    """
    if reference_alpha is not None:
        raise ValueError(
            "reference_alpha is incompatible with the current AAEC rule; "
            "use per-view grid-best references instead."
        )
    annotated = [dict(row) for row in rows]
    if not annotated:
        raise ValueError("validation sweep is empty")

    alpha_key = _find_column(annotated[0], ("alpha", "coefficient"))
    bicubic_key = _find_column(
        annotated[0],
        ("val_bicubic_acc", "bicubic_validation_accuracy", "val_bicubic_accuracy"),
    )
    bilinear_key = _find_column(
        annotated[0],
        ("val_bilinear_acc", "bilinear_validation_accuracy", "val_bilinear_accuracy"),
    )

    annotated.sort(key=lambda row: float(row[alpha_key]))
    alphas = tuple(float(row[alpha_key]) for row in annotated)
    if len(set(alphas)) != len(alphas):
        raise ValueError("alpha values must be unique")

    raw = np.asarray(
        [[float(row[bicubic_key]), float(row[bilinear_key])] for row in annotated],
        dtype=np.float64,
    )
    if not np.isfinite(raw).all():
        raise ValueError("validation accuracies must be finite")
    scale = 100.0 if float(np.max(np.abs(raw))) > 1.5 else 1.0
    fractional = raw / scale
    result = select_alpha(fractional, epsilon_pp=epsilon_pp, alphas=alphas)
    refs = np.asarray(result.reference_accuracies, dtype=np.float64)

    for idx, row in enumerate(annotated):
        b, l = fractional[idx]
        b_drop = float((refs[0] - b) * 100.0)
        l_drop = float((refs[1] - l) * 100.0)
        b_pass = b_drop <= float(epsilon_pp) + 1e-9
        l_pass = l_drop <= float(epsilon_pp) + 1e-9
        row.update({
            "reference_bicubic_validation_accuracy": float(refs[0] * scale),
            "reference_bilinear_validation_accuracy": float(refs[1] * scale),
            "bicubic_drop_pp": b_drop,
            "bilinear_drop_pp": l_drop,
            "bicubic_pass": int(b_pass),
            "bilinear_pass": int(l_pass),
            "admissible": int(b_pass and l_pass),
            "tolerance_pp": float(epsilon_pp),
            "selection_rule": FORMAL_RULE,
            "selection_reference": "per_view_grid_best",
            "selection_fallback_alpha1": int(result.fallback_alpha1),
            "selected_alpha": float(result.selected_alpha),
            "selected": int(abs(alphas[idx] - result.selected_alpha) < 1e-12),
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_alpha_selection": 0,
        })

    selected = next(row for row in annotated if int(row["selected"]) == 1)
    return annotated, dict(selected)


def select_smallest_admissible(
    rows: Iterable[Mapping],
    epsilon_pp: float = 0.5,
) -> float:
    """Return only the selected coefficient."""
    _, selected = annotate_and_select(rows, epsilon_pp=epsilon_pp)
    return float(selected["selected_alpha"])
