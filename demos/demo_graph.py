from functools import partial
from pathlib import Path

from sklearn.metrics import normalized_mutual_info_score, silhouette_score, davies_bouldin_score, adjusted_rand_score

from tangles import cost_functions, data_types, plotting
from tangles.loading import load_GMM, load_SBM
from tangles.cut_finding import a_slice, fid_mat
from tangles import utils

import numpy as np

from tangles.tree_tangles import ContractedTangleTree, tangle_computation, compute_soft_predictions_children

"""
Simple script for exemplary use of the tangle framework.
The execution is divided in the following steps

    1. Load datasets
    2. Find the cuts and compute the costs
    3. For each cut compute the tangles by expanding on the
          previous ones if it is consistent. If its not possible stop
    4. Postprocess in soft and hard clustering
"""

# define your parameters
plot = True
output_directory = Path('output_graph')
seed = 42
agreement = 20

# load your data
print("Loading dataset", flush=True)
A, ys, G = load_SBM(block_sizes=[50, 50, 50, 50], p_in=0.3, p_out=0.1, seed=seed)
data = data_types.Data(A=A, ys=ys, G=G)

print("Preprocessing data", flush=True)

# calculate your bipartitions
print("\tGenerating set of bipartitions", flush=True)
bipartitions = data_types.Cuts(values=fid_mat(xs=data.A,
                                              nb_cuts=20,
                                              lb_f=0.1,
                                              seed=seed,
                                              verbose=3,
                                              early_stopping=5))

print("\tFound {} unique bipartitions".format(len(bipartitions.values)), flush=True)
print("\tCalculating costs of bipartitions", flush=True)
bipartitions = utils.compute_cost_and_order_cuts(bipartitions,
                                                 partial(cost_functions.mean_edges_cut_cost, data.A, None))

print("Tangle algorithm", flush=True)
# calculate the tangle search tree
print("\tBuilding the tangle search tree", flush=True)
tangles_tree = tangle_computation(cuts=bipartitions,
                                  agreement=agreement,
                                  verbose=3  # print everything
                                  )
print("Built tree has {} leaves".format(len(tangles_tree.maximals)), flush=True)
# postprocess tree
print("Postprocessing the tree.", flush=True)
# contract to binary tree
print("\tContracting to binary tree", flush=True)
contracted_tree = ContractedTangleTree(tangles_tree)

# prune short paths
print("\tPruning short paths (length at most 1)", flush=True)
contracted_tree.prune(1)

# calculate
print("\tcalculating set of characterizing bipartitions", flush=True)
contracted_tree.calculate_setP()

# compute soft predictions
# assign weight/ importance to bipartitions
weight = np.exp(-utils.normalize(bipartitions.costs))

# propagate down the tree
print("Calculating soft predictions", flush=True)
compute_soft_predictions_children(node=contracted_tree.root,
                                  cuts=bipartitions,
                                  weight=weight,
                                  verbose=3)

contracted_tree.processed_soft_prediction = True


print("Calculating hard predictions", flush=True)
ys_predicted, _ = utils.compute_hard_predictions(contracted_tree, cuts=bipartitions)

# evaluate hard predictions
if data.ys is not None:
    ARS = adjusted_rand_score(data.ys, ys_predicted)
    NMI = normalized_mutual_info_score(data.ys, ys_predicted)

    print('Found {} clusters.'.format(len(np.unique(ys_predicted))))
    print('Adjusted Rand Score: {}'.format(ARS), flush=True)
    print('Normalized Mutual Information: {}'.format(NMI), flush=True)

if plot:
    print("Plotting the data.", flush=True)

    # the embeddings are calculated using a spring embedding

    output_directory.mkdir(parents=True, exist_ok=True)

    #####
    ##### If you have graphviz installen feel free to uncomment the following lines to also plot and save the trees
    #####
    ## plot the tree
    # tangles_tree.plot_tree(path=output_directory / 'tree.svg')

    ## plot contracted tree
    # contracted_tree.plot_tree(path=output_directory / 'contracted.svg')

    # plot soft predictions
    plotting.plot_soft_predictions(data=data,
                                   contracted_tree=contracted_tree,
                                   eq_cuts=bipartitions.equations,
                                   path=output_directory / 'soft_clustering')

    # plot hard clustering
    plotting.plot_hard_predictions(data=data, ys_predicted=ys_predicted, path=output_directory / 'hard_clustering')

    # plot cuts
    plotting.plot_cuts(data=data, cuts=bipartitions, nb_cuts_to_plot=10, path=output_directory / 'cuts')
