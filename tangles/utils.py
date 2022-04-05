from __future__ import annotations
import numpy as np
from sklearn.manifold import TSNE
from typing import Union, Optional


class Orientation(object):

    def __init__(self, direction: Union[int, bool, str]):
        self.direction: str
        self.orientation_bool: Optional[bool]
        if isinstance(direction, str) and direction == 'both':
            self.direction = direction
            self.orientation_bool = None
            return

        # int case, cast to bool then
        if isinstance(direction, int):
            if direction != 0 and direction != 1:
                raise ValueError('direction must be 0 or 1')
            direction = bool(direction)
        self.orientation_bool = bool(direction)

        # bool case
        if direction is True:
            self.direction = 'left'
        elif direction is False:
            self.direction = 'right'

    def __eq__(self, value: Orientation) -> bool:
        if self.direction == value.direction:
            return True

        return False

    def __str__(self) -> str:
        if self.direction == 'both':
            return self.direction
        elif self.direction == 'left':
            return 'True'
        elif self.direction == 'right':
            return 'False'
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

    def __repr__(self) -> str:
        return self.__str__()

    def orient_cut(self, cut: np.ndarray):
        if self.orientation_bool is None:
            return cut
        elif self.orientation_bool:
            return cut
        else:
            return ~cut


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


def compute_hard_predictions(condensed_tree, verbose=True):
    if not condensed_tree.processed_soft_prediction and verbose:
        print("No probabilities given yet. Calculating soft predictions first!")

    probabilities = []
    for node in condensed_tree.maximals:
        probabilities.append(node.p)

    ys_predicted = np.argmax(probabilities, axis=0)

    return ys_predicted, None
