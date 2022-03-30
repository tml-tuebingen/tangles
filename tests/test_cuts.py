from multiprocessing.sharedctypes import Value
from tangles.data_types import Cuts
import numpy as np
import pytest


def get_simple_cuts(sorted=False):
    bps = np.array([[0, 0, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    cuts = Cuts(bps)
    if sorted:
        def cost(x): return x.sum()
        cuts.compute_cost_and_order_cuts(cost)
    return cuts


def test_sorting():
    cuts = get_simple_cuts(sorted=True)
    assert np.all(cuts.values == np.array(
        [[0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 1]]))
    assert np.all(cuts.order == np.array([0, 2, 1]))


def test_cut_retrieval():
    cuts = get_simple_cuts(sorted=True)
    assert np.all(cuts.get_cut_at(0, access_sorted=True)
                  == np.array([0, 0, 1, 0]))
    assert np.all(cuts.get_cut_at(1, access_sorted=True)
                  == np.array([0, 0, 1, 1]))
    assert np.all(cuts.get_cut_at(2, access_sorted=True)
                  == np.array([0, 1, 1, 1]))

    assert np.all(cuts.get_cut_at(0, access_sorted=False)
                  == np.array([0, 0, 1, 0]))
    assert np.all(cuts.get_cut_at(1, access_sorted=False)
                  == np.array([0, 1, 1, 1]))
    assert np.all(cuts.get_cut_at(2, access_sorted=False)
                  == np.array([0, 0, 1, 1]))

    cuts = get_simple_cuts(sorted=False)
    assert np.all(cuts.get_cut_at(0, access_sorted=False)
                  == np.array([0, 0, 1, 0]))
    assert np.all(cuts.get_cut_at(1, access_sorted=False)
                  == np.array([0, 1, 1, 1]))
    assert np.all(cuts.get_cut_at(2, access_sorted=False)
                  == np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError):
        c = cuts.get_cut_at(2, access_sorted=True)
