"""Portable reference implementation of the AAEC joint clean-view selector.

Input accuracies are fractions in [0, 1], with rows ordered by alpha and
columns ordered as (Bicubic, Bilinear). Epsilon is expressed in percentage
points. This implementation mirrors the selector in the archived C60
formal evaluator; it is not a claim that the lost original training runner
has been recovered byte-for-byte.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

DEFAULT_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)

@dataclass(frozen=True)
class SelectionResult:
    selected_alpha: float
    feasible_alphas: tuple[float, ...]
    joint_feasible: bool
    fallback_alpha1: bool
    reference_accuracies: tuple[float, float]
    selected_accuracies: tuple[float, float]
    epsilon_pp: float

def select_alpha(accuracies: Sequence[Sequence[float]],
                 epsilon_pp: float = 0.5,
                 alphas: Sequence[float] = DEFAULT_ALPHAS) -> SelectionResult:
    """Select the smallest alpha satisfying both clean-view retention constraints.

    ``accuracies`` must have shape ``(len(alphas), 2)`` and contain
    fractional validation accuracies for Bicubic and Bilinear.
    If the joint feasible set is empty, the prespecified fallback is alpha=1.0.
    """
    a = np.asarray(accuracies, dtype=np.float64)
    alphas = tuple(float(x) for x in alphas)
    if a.ndim != 2 or a.shape != (len(alphas), 2):
        raise ValueError(f"expected accuracy shape {(len(alphas), 2)}, got {a.shape}")
    if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("accuracies must be finite fractions in [0, 1]")
    if epsilon_pp < 0:
        raise ValueError("epsilon_pp must be non-negative")
    if not alphas or tuple(sorted(alphas)) != alphas:
        raise ValueError("alphas must be non-empty and sorted ascending")
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
