import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tangles.utils import get_points_to_plot

# Standard colors for uniform plots
COLOR_SILVER = '#C0C0C0'
COLOR_SILVER_RGB = (192 / 255, 192 / 255, 192 / 255) + (0.2,)
COLOR_INDIGO_RGB = (55 / 255, 0 / 255, 175 / 255) + (0.5,)
COLOR_CARNATION_RGB = np.array((247 / 255, 96 / 255, 114 / 255, 1)).reshape((1, -1))
CMAP = plt.cm.get_cmap('Blues')

MARKERS = ['o', '^', 'p', 'D', 's', '.', '>', 'x', '-']


def get_nb_points(data):
    if data.xs is not None:
        return len(data.xs)
    elif data.A is not None:
        return len(data.A)
    else:
        raise KeyError('What data are you using?')


def append_to_binary(number, new_digit):
    return int(str(bin(number) + str(new_digit)), 2)


def get_next_id(current_id, direction):
    if current_id == 0:
        if direction == 'left':
            return 1
        else:
            return 2

    level = int(np.ceil(np.log2(current_id)))

    if direction == 'left':
        return current_id + 2 ** level + 1
    else:
        return current_id + 2 ** level + 2


def plot_dataset(data, colors, ax=None, eq_cuts=None, cmap=None, add_colorbar=True, pos=None, gt=None):
    if data.xs is not None:
        ax = plot_dataset_metric(data.xs, data.cs, colors, eq_cuts, ax, cmap, add_colorbar, gt)
    elif data.G is not None:
        ax, pos = plot_dataset_graph(data.G, data.ys, colors, ax, cmap, add_colorbar, pos)

    return ax, pos


def add_colorbar_to_ax(ax, cmap):

    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap),
                      ax=ax, orientation='vertical', shrink=0.7)
    cb.ax.set_title('p', y=-.05)

    return ax


def plot_dataset_graph(G, ys, colors, ax, cmap, add_colorbar, pos):
    if pos is None:
        pos = get_position(G, ys)

    im = nx.draw_networkx(G, pos=pos, ax=ax, node_color=colors, edge_color=COLOR_SILVER, with_labels=False,
                     edgecolors='black')
    if add_colorbar:
        ax = add_colorbar_to_ax(ax, cmap)

    return ax, pos


def plot_dataset_metric(xs, cs, colors, eq_cuts, ax, cmap, add_colorbar, gt):
    plt.style.use('ggplot')
    plt.ioff()

    ax.tick_params(axis='x', colors=(0, 0, 0, 0))
    ax.tick_params(axis='y', colors=(0, 0, 0, 0))
    ax.set_aspect('equal', 'box')
    ax.grid()

    xs_embedded, cs_embedded = get_points_to_plot(xs, cs)

    if gt is not None:
        for i in range(len(np.unique(gt))):
            sc = ax.scatter(xs_embedded[gt == i, 0], xs_embedded[gt == i, 1], color=colors[gt == i], marker=MARKERS[i], cmap=cmap)
    else:
        sc = ax.scatter(xs_embedded[:, 0], xs_embedded[:, 1], color=colors)  # vmin=0, vmax=1, edgecolor='black')

    if add_colorbar:
        ax = add_colorbar_to_ax(ax, cmap)

    return ax


def labels_to_colors(ys, cmap):
    nb_points = len(ys)
    colors = np.zeros((nb_points, 4))
    normalize_ys = mpl.colors.Normalize(vmin=0, vmax=np.max(ys))

    for y in np.unique(ys):
        idx_current = (ys == y).nonzero()[0]
        color = cmap(normalize_ys(y))
        colors[idx_current, :] = np.array(color).reshape((1, -1))

    return colors


def plot_soft_predictions(data, contracted_tree, eq_cuts=None, id_node=0, path=None):
    cmap_groundtruth = plt.cm.get_cmap('autumn')
    cmap_heatmap = plt.cm.get_cmap('Blues')


    if path is not None:
        output_path = path
        output_path.mkdir(parents=True, exist_ok=True)

    if data.ys is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        colors = labels_to_colors(data.ys, cmap=cmap_groundtruth)
        ax, pos = plot_dataset(data, colors, eq_cuts=eq_cuts, ax=ax, add_colorbar=False, gt=data.ys)

        plt.tight_layout()
        if path is not None:
            fig.savefig(output_path / "groundtruth.svg")
            plt.close(fig)
        else:
            plt.show()

    plot_soft_prediction_node(data, contracted_tree.root, eq_cuts=eq_cuts, id_node=0, cmap=cmap_heatmap, path=path,
                              pos=pos)


def plot_soft_prediction_node(data, node, eq_cuts, id_node, cmap, path, pos):
    colors = cmap(node.p)

    if eq_cuts is not None:
        if len(node.characterizing_cuts) != 0:
            id_characterizing_cuts = list(node.characterizing_cuts.keys())
            eq_characterizing_cuts = eq_cuts[id_characterizing_cuts]
        else:
            eq_characterizing_cuts = []
    else:
        eq_characterizing_cuts = []

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plot_dataset(data, colors, eq_cuts=eq_characterizing_cuts, ax=ax, cmap=cmap, pos=pos)
    ax.set_xlabel("Node Id: {}".format(id_node))

    plt.tight_layout()
    if path is not None:
        fig.savefig(path / "node_nb_{}.svg".format(id_node))
        plt.close(fig)
    else:
        plt.show()

    if node.left_child is not None:
        id_left = get_next_id(id_node, 'left')
        plot_soft_prediction_node(data, node.left_child, eq_cuts, id_left, cmap, path, pos=pos)
    if node.right_child is not None:
        id_right = get_next_id(id_node, 'right')
        plot_soft_prediction_node(data, node.right_child, eq_cuts, id_right, cmap, path, pos=pos)


def plot_hard_predictions(data, ys_predicted, path=None):
    cmap_groundtruth = plt.cm.get_cmap('autumn')
    cmap_predictions = plt.cm.get_cmap('cool')

    if path is not None:
        output_path = path
        output_path.mkdir(parents=True, exist_ok=True)

    if data.ys is not None:
        fig, (ax_true, ax_predicted) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        colors_true = labels_to_colors(data.ys, cmap=cmap_groundtruth)
        ax_true.set_title("Ground truth clusters")
        _ = plot_dataset(data, colors_true, ax=ax_true, add_colorbar=False)
    else:
        fig, ax_predicted = plt.subplots(nrows=1, ncols=1, figsize=(10, 50))

    colors_predicted = labels_to_colors(ys_predicted, cmap=cmap_predictions)
    _ = plot_dataset(data, colors_predicted, ax=ax_predicted, add_colorbar=False)
    ax_predicted.set_title("Predicted clusters")

    plt.tight_layout()
    if path is not None:
        fig.savefig(output_path / "hard_clustering.svg")
        plt.close(fig)
    else:
        plt.show()


def get_position(G, ys):
    if ys is not None:
        pos = nx.random_layout(G)
        ncls = np.max(ys) + 1
        xoff = np.sin(2 * np.pi * ys / ncls) * 2
        yoff = np.cos(2 * np.pi * ys / ncls) * 2
        for v in G:
            pos[v][0] += xoff[v]
            pos[v][1] += yoff[v]
        pos = nx.spring_layout(G, pos=pos, iterations=1)
    elif nx.is_connected(G):
        pos = nx.spectral_layout(G)
        pos = nx.spring_layout(G, pos=pos, k=.5, iterations=100)
    else:
        pos = nx.kamada_kawai_layout(G)
        pos = nx.spring_layout(G, pos=pos, k=.5, iterations=100)
    return pos


def plot_cuts(data, cuts, nb_cuts_to_plot, path):
    plt.style.use('ggplot')
    plt.ioff()

    if path is not None:
        path = path / 'cuts'
        path.mkdir(parents=True, exist_ok=True)

    value_cuts = cuts.values
    order_cuts = cuts.costs
    eq_cuts = cuts.equations
    nb_cuts_to_plot = min(nb_cuts_to_plot, len(value_cuts))
    pos = None

    for i in np.arange(nb_cuts_to_plot):
        eq = [eq_cuts[i]] if eq_cuts is not None else None

        fig, pos = plot_cut(data, cut=value_cuts[i], order=order_cuts[i], eq=eq, pos=pos)
        plt.tight_layout()
        if path is not None:
            fig.savefig(path / "cut number {}.png".format(i))
            plt.close(fig)
        else:
            plt.show()


def plot_cut(data, cut, order, eq, pos):
    cmap_groundtruth = plt.cm.get_cmap('tab10')
    cmap_cut = plt.cm.get_cmap('Blues')

    if data.ys is not None:
        fig, (ax_true, ax_cut) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        colors_true = labels_to_colors(data.ys, cmap=cmap_groundtruth)
        ax_true, pos = plot_dataset(data, colors_true, eq_cuts=eq, ax=ax_true, add_colorbar=False, pos=pos)
        ax_true.set_title('Groundtruth')

    else:
        fig, ax_cut = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    ax_cut.set_title('Cut')
    fig.suptitle('Cut of order: {}'.format(order))

    color_cut = labels_to_colors(cut, cmap=cmap_cut)
    ax_cut = plot_dataset(data, color_cut, ax=ax_cut, add_colorbar=False, pos=pos)

    return fig, pos
