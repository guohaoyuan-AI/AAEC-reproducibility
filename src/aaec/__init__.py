"""Anchor-Aware Effect Calibration (AAEC) reference implementation."""

from .selection import (
    DEFAULT_ALPHAS,
    FORMAL_RULE,
    SelectionResult,
    annotate_and_select,
    select_alpha,
    select_smallest_admissible,
)

__all__ = [
    "DEFAULT_ALPHAS",
    "FORMAL_RULE",
    "SelectionResult",
    "annotate_and_select",
    "select_alpha",
    "select_smallest_admissible",
]
