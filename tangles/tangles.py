from copy import deepcopy
from itertools import combinations
from tangles.utils import subset
from bitarray import bitarray


def pad_bitarray(b, n):
    """
    Pads bitarray b to desired length n with zeros at the end.
    """
    if len(b) < n:
        b.extend(bitarray('0' * (n - len(b))))
    return b


class Tangle(dict):
    """
    This class represents an oriented cut as a couple of lists and a dictionary.
        - cuts contains all the biparitions of the specification defined as binary arrays.
          1 means that that x belongs to the partition and 0 that it does not.
          It is implemented with bitarrays for max speed
        - core contains all the biparitions of the core of the specification defined as binary arrays.
          1 means that that x belongs to the partition and 0 that it does not.
          It is implemented with bitarrays for max speed
        - specification is a dictionary there the key is the index of the cut in the list of all the cuts and
          the value is which orientation of that specification we need to take
    """

    def __str__(self):  # pragma: no cover
        return str(self.specification)

    def __init__(self, cuts=None, core=None, specification=None):
        """
        Initialise a new specification

        Parameters
        ----------
        cuts: list of bitarray
            All the biparitions of the specification
        core: list of bitarray
            All the biparitions of the core of the specification
        specification: dict of bool
            The key is the index of the cut in the list of all the cuts and
            the value is which orientation of that specification we need to take
        """

        super().__init__()
        if core is None:
            core = []
        if cuts is None:
            cuts = []
        if specification is None:
            specification = bitarray()

        self._cuts = cuts
        self._core = core
        self._specification = specification

    def get_cuts(self):
        return self._cuts

    def get_core(self):
        return self._core

    def get_specification(self):
        return self._specification

    def add(self, new_cut, new_cut_id, orientation, min_size):
        """
        Check if new_cut can be added to the current orientation

        Parameters
        ----------
        new_cut: bitarray
            The cut that we need to add as bitarray
        new_cut_id: int
            The index of the cut in the list of all the cuts
        orientation: bool
            The orientation of the cut. True if naturally oriented, False if reversed
        min_size:
            Minimum triplet size that we accept for it to be a tangle

        Returns
        -------
        new_specification: Specification or None
            If it is possible to add we return the new specification otherwise we return None
        """

        core = deepcopy(self._core)
        specification = self._specification.copy()

        pad_bitarray(specification, new_cut_id + 1)

        i_to_remove = []
        for i, core_cut in enumerate(core):
            if subset(core_cut, new_cut):
                specification[new_cut_id] = orientation
                return Tangle(None, core, specification)
            if subset(new_cut, core_cut):
                i_to_remove.append(i)

        for i in i_to_remove[::-1]:
            del core[i]

        # Checking for consistency...
        if len(core) == 0:
            # noinspection PyArgumentList
            if new_cut.count() < min_size:
                return None
        elif len(core) == 1:
            if (core[0] & new_cut).count() < min_size:
                return None
        else:
            for core1, core2 in combinations(core, 2):
                if (core1 & core2 & new_cut).count() < min_size:
                    return None

        core.append(new_cut)
        specification[new_cut_id] = orientation

        return Tangle(None, core, specification)


def core_algorithm(tree, current_cuts, idx_current_cuts):
    """
    Algorithm iteratively adding cuts to the tree

    Parameters
    ----------
    tree: TangleTree
        The binary tree the cut should be added to
    current_cuts: list
        Current cuts to add
    idx_current_cuts: list
        Indices of the cuts giving their layer in the tree (based on the order induced by the cost)

    Returns
    -------
    tree: TangleTree
        We return the new tree with added cuts if it is possible
    """

    for idx_cut, cut in zip(idx_current_cuts, current_cuts):
        could_add = tree.add_cut(cut=cut, cut_id=idx_cut)
        if could_add is False:
            return None

    return tree
