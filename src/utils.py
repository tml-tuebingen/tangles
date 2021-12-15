import hashlib
import json

import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors._dist_metrics import DistanceMetric
from tqdm import tqdm
from src.data_types import Cuts

class Orientation(object):

    def __init__(self, direction):
        if type(direction) == int:
            direction = bool(direction)
        self.orientation_bool = direction

        if direction == 'both':
            self.direction = direction
        elif direction is True:
            self.direction = 'left'
        elif direction is False:
            self.direction = 'right'

    def __eq__(self, value):

        if self.direction == value.direction:
            return True

        return False

    def __str__(self):
        if self.direction == 'both':
            return self.direction
        elif self.direction == 'left':
            return 'True'
        elif self.direction == 'right':
            return 'False'


def normalize(array):
    """
    Normalize a 1d numpy array between [0,1]

    Parameters
    ----------
    array: ndarray
        the array to normalize

    Returns
    -------
    ndarray
        the normalized array
    """

    ptp = np.ptp(array)
    if ptp != 0:
        return (array - np.min(array)) / np.ptp(array)
    else:
        return np.ones_like(array)


def matching_items(d1, d2):
    matching_keys = []
    common_keys = d1.keys() & d2.keys()
    for k in common_keys:
        if d1[k] == d2[k]:
            matching_keys.append(k)

    return matching_keys


def merge_dictionaries_with_disagreements(d1, d2):
    merge = {**d1, **d2}
    common_keys = d1.keys() & d2.keys()

    for k in common_keys:
        if d1[k] != d2[k]:
            merge.pop(k)

    return merge


def get_points_to_plot(xs, cs):
    """
    Calculate embedding of points for visualization

    Parameters
    ----------
    xs: ndarray
        the datapoints
    cs: ndarray
        the centers of the clusters if method supports

    Returns
    -------
    xs_embedded:
         embedding of points in two dimensions
    cs_embedded:
         embedding of centers if method supports
    """
    _, nb_features = xs.shape

    nb_centers = None
    cs_embedded = None
    if cs is not None:
        nb_centers, _ = cs.shape

    if nb_features > 2:
        if cs is not None:
            points_to_embed = np.vstack([xs, cs])
            embeds = TSNE(n_components=2, random_state=42).fit_transform(
                points_to_embed)
            xs_embedded, cs_embedded = embeds[:-
                                              nb_centers], embeds[-nb_centers:]
        else:
            xs_embedded = TSNE(
                n_components=2, random_state=42).fit_transform(xs)
    else:
        xs_embedded = xs
        cs_embedded = cs

    if cs is not None:
        return xs_embedded, cs_embedded
    else:
        return xs_embedded, None


def subset(a, b):
    return (a & b).count() == a.count()


def compute_cost_and_order_cuts(bipartitions, cost_functions, verbose=True):
    costs = compute_cost(bipartitions, cost_functions, verbose=verbose)
    return order_cuts(bipartitions, costs)


def compute_cost(bipartitions, cost_function, verbose=True):
    """
    Compute the cost of a series of cuts and returns a cost array.

    Parameters
    ----------
    cuts: Cuts
        where cuts.values has shape (n_questions, n_datapoints)
    cost_function: function
        callable that calculates the cost of a single cut, which is an ndarray of shape
        (n_datapoints)

    Returns
    -------
    cost: ndarray of shape (n_questions) containing the costs of each cut as entries
    """
    if verbose:
        print("Computing costs of cuts...")

    cost_bipartitions = np.zeros(len(bipartitions.values), dtype=float)
    for i_cut, cut in enumerate(tqdm(bipartitions.values, disable=not verbose)):
        cost_bipartitions[i_cut] = cost_function(cut)
    return cost_bipartitions


def order_cuts(bipartitions: Cuts, cost_bipartitions: np.ndarray):
    """
    Orders cuts based on the cost of the cuts.

    bipartitions: Cuts,
    where values contains an ndarray of shape (n_questions, n_datapoints).
    cost_bipartitions: ndarray,
    where values contains an ndarray of shape (n_datapoints). Contains
    the cost of each cut as value.
    """
    idx = np.argsort(cost_bipartitions)

    bipartitions.values = bipartitions.values[idx]
    bipartitions.costs = cost_bipartitions[idx]
    if bipartitions.names is not None:
        bipartitions.names = bipartitions.names[idx]
    if bipartitions.equations is not None:
        bipartitions.equations = bipartitions.equations[idx]

    bipartitions.order = np.argsort(idx)

    return bipartitions


def compute_hard_predictions(condensed_tree, cuts, xs=None, verbose=True):
    if xs is not None:
        cs = []
        nb_cuts = len(cuts.values)

        for leaf in condensed_tree.maximals:
            c = np.full(nb_cuts, 0.5)
            tangle = leaf.tangle
            c[list(tangle.specification.keys())] = np.array(
                list(tangle.specification.values()), dtype=int)
            cs.append(c[cuts.order])

        cs = np.array(cs)

        return compute_mindset_prediciton(xs, cs), cs

    else:
        if not condensed_tree.processed_soft_prediction and verbose:
            print("No probabilities given yet. Calculating soft predictions first!")

        probabilities = []
        for node in condensed_tree.maximals:
            probabilities.append(node.p)

        ys_predicted = np.argmax(probabilities, axis=0)

        return ys_predicted, None


def compute_mindset_prediciton(xs, cs):
    metric = DistanceMetric.get_metric('manhattan')

    distance = metric.pairwise(xs, cs)
    predicted = np.empty(xs.shape[0])

    for i, d in enumerate(distance):
        predicted[i] = np.random.choice(np.flatnonzero(d == d.min()))

    return predicted
