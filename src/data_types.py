from enum import Enum, unique


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

    def __init__(self, values, names=None, equations=None, costs=None):

        self.values = values
        self.names = names
        self.equations = equations
        self.costs = costs
        self.order = None
