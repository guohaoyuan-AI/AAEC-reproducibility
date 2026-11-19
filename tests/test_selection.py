import numpy as np
import pytest
from aaec.selection import select_alpha

def test_selects_smallest_jointly_feasible_alpha():
    a = np.array([[.80,.78],[.89,.87],[.90,.88],[.899,.879],[.90,.88]])
    r = select_alpha(a, epsilon_pp=.5)
    assert r.selected_alpha == .6
    assert r.feasible_alphas == (.6,.8,1.0)
    assert not r.fallback_alpha1

def test_fallback_is_distinct_from_feasible_alpha_one():
    feasible = select_alpha(np.array([[.80,.80],[.90,.90],[.91,.91],[.914,.914],[.920,.920]]), .5)
    assert feasible.selected_alpha == 1.0
    assert not feasible.fallback_alpha1
    empty = select_alpha(np.array([[.90,.80],[.80,.90],[.70,.70],[.69,.69],[.68,.68]]), .5)
    assert empty.selected_alpha == 1.0
    assert empty.fallback_alpha1

def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        select_alpha([[.8,.8]], .5)
