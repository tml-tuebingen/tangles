from enum import Enum, unique
from typing import Optional, Callable
import numpy as np
from tqdm import tqdm


class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@unique
class Dataset(ExtendedEnum):
    breast_cancer_wisconsin = 'BCW'
    gaussian_mixture = 'gau_mix'
    LFR = 'lfr'
    microbiome = 'biome'
    mindsets = 'mind'
    moons = 'moons'
    circles = 'circles'
    SBM = 'sbm'
    questionnaire_likert = 'q_likert'
    retinal = 'retinal'
    wave = 'wave'

    def __str__(self):
        return self.value


@unique
class Preprocessing(ExtendedEnum):
    none = 'none'
    feature_map = 'ftr_map'
    knn_graph = 'knn'
    radius_neighbors_graph = 'rng'
    weighted_knn_graph = 'wknn'

    def __str__(self):
        return self.value


@unique
class CutFinding(ExtendedEnum):
    features = 'fea'
    random_projection = 'rand_proj'
    kmeans = 'kmeans'
    kmodes = 'k_modes'
    q_binning = 'qbin'
    binning = 'bin'
    Kernighan_Lin = 'KL'
    Fiduccia_Mattheyses = 'FM'
    linear = 'lin'
    slice = 'slice'

    def __str__(self):
        return self.value


@unique
class CostFunction(ExtendedEnum):
    euclidean = 'euclidean_sum'
    mean_euclidean = 'euclidean_mean'
    manhattan = 'manhattan_sum'
    mean_manhattan = 'manhattan_mean'
    mean_cut_value = 'cut_mean'
    cut_value = 'cut_sum'

    def __str__(self):
        return self.value


class Data(object):

    def __init__(self, xs=None, ys=None, cs=None, A=None, G=None):

        self.xs = xs
        self.ys = ys
        self.cs = cs
        self.A = A
        self.G = G

        if xs is not None:
            self.original_xs = xs.copy()


class Cuts(object):
    def __init__(self, values: np.ndarray, costs: Optional[np.ndarray] = None, equations=None, names=None):
        """
        Initializes a cut object with the given values.
        A ROW in the given values object represents one cut.
        """
        self.order: Optional[np.ndarray] = None
        self.values = values
        self.values_unsorted = values
        self.costs = costs
        self.names = names
        self.equations = equations

    def get_cut_at(self, id: int, from_unsorted: bool = False):
        """
        Returns the cut at the given ID.

        If from_unsorted is set to True, uses the ID to access the unsorted array.
        """
        if from_unsorted:
            return self.values_unsorted[id]
        return self.values[id, :]

    def unsorted_id(self, id: int):
        """
        Takes in a cut id from this cuts objects and returns the original id, before
        it was sorted.
        """
        if self.order is None:
            return id
        else:
            where_matches_id = np.argwhere(self.order == id)
            if where_matches_id.size != 1:
                print(
                    f"Order of cuts is wrong: {where_matches_id}\nMultiple matches for {id}: {where_matches_id}")
            return where_matches_id[0][0]

    def compute_cost_and_order_cuts(self, cost_function: Callable, verbose=True):
        """
        Computes the costs of this cut and orders the cuts.

        This mutates the cuts object!
        """
        costs = self._compute_cost(cost_function, verbose=verbose)
        self._order_cuts(costs)

    def _compute_cost(self, cost_function, verbose=True):
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

        cost_bipartitions = np.zeros(len(self.values), dtype=float)
        for i_cut, cut in enumerate(tqdm(self.values, disable=not verbose)):
            cost_bipartitions[i_cut] = cost_function(cut)
        return cost_bipartitions

    def _order_cuts(self, cost_bipartitions: np.ndarray):
        """
        Orders cuts based on the cost of the cuts.

        cost_bipartitions: ndarray,
        where values contains an ndarray of shape (n_datapoints). Contains
        the cost of each cut as value.
        """
        idx = np.argsort(cost_bipartitions)

        self.values = self.values[idx]
        self.costs = cost_bipartitions[idx]
        self.order = np.argsort(idx)
        if self.names is not None:
            self.names = self.names[idx]

        return self
