import numpy as np

from tangles.cost_functions import BipartitionSimilarity
from tangles.data_types import Cuts
from tangles.tree_tangles import (ContractedTangleTree,
                                  compute_soft_predictions_children,
                                  tangle_computation)
from tangles.utils import compute_cost_and_order_cuts, compute_hard_predictions, normalize
from tangles.loading import load_GMM
from sklearn.metrics import pairwise_distances


def get_preds(X: np.ndarray, agreement: int):
    bipartitions = Cuts((X == 1).T)
    cost_function = BipartitionSimilarity(
        bipartitions.values.T)
    cuts = compute_cost_and_order_cuts(
        bipartitions, cost_function, verbose=False)

    # Building the tree, contracting and calculating predictions
    tangles_tree = tangle_computation(cuts=cuts,
                                      agreement=agreement,
                                      # print nothing
                                      verbose=0)

    contracted = ContractedTangleTree(tangles_tree)
    contracted.prune(1, verbose=False)

    contracted.calculate_setP()

    # soft predictions
    weight = np.exp(-normalize(cuts.costs))

    bipartitions = Cuts((X == 1).T)

    cost_function = BipartitionSimilarity(
        bipartitions.values.T)
    _ = compute_cost_and_order_cuts(
        bipartitions, cost_function, verbose=False)

    compute_soft_predictions_children(
        node=contracted.root, cuts=bipartitions, weight=weight, verbose=0)
    contracted.processed_soft_predictions = True

    ys_predicted, _ = compute_hard_predictions(
        contracted, verbose=False)

    return ys_predicted


def test_trivial():
    np.random.seed(0)
    X = np.block([
        [np.ones((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.ones((3, 3))]
    ])
    assert (np.array([[0, 0, 0, 1, 1, 1]]) == get_preds(X, 2)).all()
    assert (np.array([[0, 0, 0, 1, 1, 1]]) == get_preds(X, 3)).all()
    assert (np.array([[0, 0, 0, 0, 0, 0]]) == get_preds(X, 4)).all()


def pivot_cut(X, a, b):
    xa = pairwise_distances(a[None, :], X)
    xb = pairwise_distances(b[None, :], X)
    return xa < xb


def test_simple_gaussians_ab():
    """
    AB test to make sure the tangles algorithm still does the exact same thing.
    """
    X, _ = load_GMM([20, 20], np.array([[0, 0], [1, 1]]), [0.7, 0.7], 10)
    idx_a = [0, 1, 5, 12, 20, 23, 34]
    idx_b = [11, 21, 34, 35, 22, 7]
    cuts = []
    for a, b in zip(idx_a, idx_b):
        cut = pivot_cut(X, X[a, :], X[b, :])
        cuts.append(cut)
    pred = get_preds(np.concatenate(cuts).T, 10)
    res_ab = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0,
                       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    assert np.all(pred == res_ab)


def test_simple_gaussians_performance():
    """
    Makes sure that the tangles algorithm achieves 100% performance on a simple gaussian,
    even if the AB test is not passed (so we might have some small implementation change).
    """
    X, ys = load_GMM([20, 20], np.array([[0, 0], [1, 1]]), [0.2, 0.2], 10)
    idx_a = [0, 1, 5, 12, 20, 23, 34]
    idx_b = [11, 21, 34, 35, 22, 7]
    cuts = []
    for a, b in zip(idx_a, idx_b):
        cut = pivot_cut(X, X[a, :], X[b, :])
        cuts.append(cut)
    pred = get_preds(np.concatenate(cuts).T, 10)
    assert np.all(pred == ys)
