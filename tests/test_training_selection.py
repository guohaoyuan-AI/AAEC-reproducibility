import pytest

from aaec.selection import FORMAL_RULE, annotate_and_select, select_alpha


def test_grid_best_joint_rule_not_alpha1_reference():
    rows = [
        {"alpha": 0.2, "val_bicubic_acc": 0.900, "val_bilinear_acc": 0.890},
        {"alpha": 0.4, "val_bicubic_acc": 0.910, "val_bilinear_acc": 0.905},
        {"alpha": 0.6, "val_bicubic_acc": 0.908, "val_bilinear_acc": 0.904},
        {"alpha": 0.8, "val_bicubic_acc": 0.906, "val_bilinear_acc": 0.903},
        {"alpha": 1.0, "val_bicubic_acc": 0.905, "val_bilinear_acc": 0.902},
    ]
    annotated, selected = annotate_and_select(rows, epsilon_pp=0.5)
    assert selected["selected_alpha"] == 0.4
    assert selected["selection_rule"] == FORMAL_RULE
    assert selected["selection_reference"] == "per_view_grid_best"
    assert next(r for r in annotated if r["alpha"] == 0.2)["admissible"] == 0


def test_alpha1_is_only_fallback_when_joint_set_empty():
    rows = [
        {"alpha": 0.2, "val_bicubic_acc": 0.90, "val_bilinear_acc": 0.80},
        {"alpha": 0.4, "val_bicubic_acc": 0.80, "val_bilinear_acc": 0.90},
        {"alpha": 0.6, "val_bicubic_acc": 0.70, "val_bilinear_acc": 0.70},
        {"alpha": 0.8, "val_bicubic_acc": 0.69, "val_bilinear_acc": 0.69},
        {"alpha": 1.0, "val_bicubic_acc": 0.68, "val_bilinear_acc": 0.68},
    ]
    _, selected = annotate_and_select(rows, epsilon_pp=0.5)
    assert selected["selected_alpha"] == 1.0
    assert selected["selection_fallback_alpha1"] == 1


def test_percentage_input_is_supported():
    rows = [
        {"alpha": 0.2, "val_bicubic_acc": 90.0, "val_bilinear_acc": 89.0},
        {"alpha": 0.4, "val_bicubic_acc": 91.0, "val_bilinear_acc": 90.5},
        {"alpha": 0.6, "val_bicubic_acc": 90.8, "val_bilinear_acc": 90.4},
        {"alpha": 0.8, "val_bicubic_acc": 90.6, "val_bilinear_acc": 90.3},
        {"alpha": 1.0, "val_bicubic_acc": 90.5, "val_bilinear_acc": 90.2},
    ]
    _, selected = annotate_and_select(rows, epsilon_pp=0.5)
    assert selected["selected_alpha"] == 0.4


def test_old_alpha1_reference_argument_is_rejected():
    rows = [
        {"alpha": 0.8, "val_bicubic_acc": 0.9, "val_bilinear_acc": 0.9},
        {"alpha": 1.0, "val_bicubic_acc": 0.9, "val_bilinear_acc": 0.9},
    ]
    with pytest.raises(ValueError, match="incompatible"):
        annotate_and_select(rows, reference_alpha=1.0)


def test_raw_selector():
    result = select_alpha(
        [[0.90, 0.89], [0.91, 0.905], [0.908, 0.904], [0.906, 0.903], [0.905, 0.902]],
        epsilon_pp=0.5,
    )
    assert result.selected_alpha == 0.4
