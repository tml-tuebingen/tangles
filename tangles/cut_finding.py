import numpy as np
from sklearn.random_projection import GaussianRandomProjection


def a_slice(xs, a, xs_original=None):
    if xs_original is None:
        xs_original = xs

    cut_names = []
    cuts = []

    n, d = np.shape(xs)

    for i in range(d):
        x = xs[:, i]
        x_original = xs_original[:, i]
        # check if i is aready binary
        unique = np.unique(x)
        if len(unique) <= 2:
            if len(unique) == 1:
                continue
            cut = np.zeros_like(x)
            cut[x == unique[0]] = 0
            cut[x == unique[1]] = 1
            cuts.append(cut.astype(bool))
            cut_names.append('{} is True'.format(i))
            continue
        idx_sorted = np.argsort(np.argsort(x))
        j = 1
        while j < n:
            cut = np.array([1] * j + [0] * (n-j))
            cut = cut[idx_sorted].astype(bool)
            cuts.append(cut)
            cut_names.append('{} smaller {}'.format(i, max(x_original[cut])))
            j += (a-1)

    return np.array(cuts), np.array(cut_names)


# ----------------------------------------------------------------------------------------------------------------------
# Kernighan-Lin algorithm
#
# implemented the pseudocode from https://en.wikipedia.org/wiki/Kernighan–Lin_algorithm
# also kept all the variable names
# it's a little difficult to see right away
# This is now fully vectorized. Everything still seems to work. Somebody should review this, though.
# ----------------------------------------------------------------------------------------------------------------------


def initial_partition(xs, fraction=0.5):
    nb_vertices, _ = xs.shape

    partition = int(round(fraction * nb_vertices))

    A = np.zeros(nb_vertices, dtype=bool)
    A[np.random.choice(np.arange(nb_vertices), partition, False)] = True

    B = np.logical_not(A)

    return A, B


def compute_D_values(xs, A, B):
    nb_vertices, _ = xs.shape

    D = np.zeros(nb_vertices)

    A_xs = xs[A, :]
    B_xs = xs[B, :]

    D[A] = np.sum(A_xs[:, B], axis=1) - np.sum(A_xs[:, A], axis=1)
    D[B] = np.sum(B_xs[:, A], axis=1) - np.sum(B_xs[:, B], axis=1)

    return D


def update_D_values(xs, A, B, D):
    nb_vertices, _ = xs.shape

    A_xs = xs[A, :]
    B_xs = xs[B, :]

    D[A] = np.sum(A_xs[:, B], axis=1) - np.sum(A_xs[:, A], axis=1)
    D[B] = np.sum(B_xs[:, A], axis=1) - np.sum(B_xs[:, B], axis=1)

    return D


def maximize(xs, A, B, D):
    Dtiled = np.tile(D, (len(D), 1))
    g = Dtiled + Dtiled.T - 2 * xs

    mask = np.logical_not(np.outer(A, B))
    g = np.ma.masked_array(g, mask)

    (a_res, b_res) = np.unravel_index(np.argmax(g), g.shape)
    g_max = g[a_res, b_res]

    return g_max, a_res, b_res


def kernighan_lin(A, nb_cuts, lb_f, seed, verbose, early_stopping):
    cuts = []
    np.random.seed(seed)

    for i in range(nb_cuts):
        f = np.random.uniform(lb_f, 0.5)
        if verbose >= 3:
            print(f'\tlooking for cut {i + 1}/{nb_cuts} with f={f:.02}')
        cut = kernighan_lin_algorithm(A, early_stopping, f)
        cuts.append(cut)

    cuts = np.array(cuts)

    return cuts


def kernighan_lin_algorithm(xs, early_stopping, fraction):
    nb_vertices, _ = xs.shape

    A, B = initial_partition(xs, fraction)

    i = 0
    while True:
        A_copy = A.copy()
        B_copy = B.copy()
        xs_copy = xs.copy()
        D = compute_D_values(xs, A_copy, B_copy)
        g_max = -np.inf
        g_acc = 0
        swap_max = np.empty_like(A)
        swap_acc = np.zeros_like(A)

        for _ in range(min(sum(A_copy), sum(B_copy))):
            # greedily find best two vertices to swap
            g, a, b = maximize(xs_copy, A_copy, B_copy, D)

            swap_acc[a] = True
            swap_acc[b] = True
            g_acc += g
            if g_acc > g_max:
                g_max = g_acc
                swap_max[:] = swap_acc[:]

            xs_copy[a, :] = 0
            xs_copy[:, a] = 0
            xs_copy[b, :] = 0
            xs_copy[:, b] = 0

            A_copy[a] = False
            B_copy[b] = False

            D = update_D_values(xs_copy, A_copy, B_copy, D)

        if g_max > 0:
            # swapping nodes from initial partition that improve the cut
            np.logical_not(A, out=A, where=swap_max)
            np.logical_not(A, out=B)

        else:
            break

        i += 1

        if early_stopping and i > early_stopping:
            break

    return A


# ----------------------------------------------------------------------------------------------------------------------
# Fiduccia-Mattheyses-Algorithm
#
# ----------------------------------------------------------------------------------------------------------------------

def fid_mat(xs, nb_cuts, lb_f, seed, verbose, early_stopping):
    cuts = []

    np.random.seed(seed)

    for i in range(nb_cuts):
        if verbose >= 3:
            print('\tlooking for cut {}/{}'.format(i+1, nb_cuts))
        cut = fid_mat_algorithm(xs, lb_f, verbose, early_stopping)
        cuts.append(cut)

    cuts = np.array(cuts)

    return cuts


def fid_mat_algorithm(xs, r, verbose, early_stopping):
    r = np.random.uniform(r, 0.5)
    nb_cells, _ = xs.shape
    A, B = initial_partition(xs, np.random.uniform(r, 0.5))

    cell_array = [np.argwhere(row > 0).flatten() for row in xs]

    # while not converged
    i = 0
    while True:
        A_copy = A.copy()
        B_copy = B.copy()
        not_locked = np.full([nb_cells], True)
        gain_list = compute_initial_gains(A, B, cell_array, xs)
        g_max = -np.inf
        g_acc = 0
        move_max = np.empty_like(A)
        move_acc = np.zeros_like(A)

        # iterate over all vertices and move them
        while sum(not_locked > 0):
            base_cell, g = choose_cell_greedy(A_copy, B_copy, gain_list, r, not_locked)

            if base_cell is None:
                break

            g_acc += g
            move_acc[base_cell] = True

            if g_acc > g_max:
                g_max = g_acc
                move_max[:] = move_acc[:]

            if A_copy[base_cell]:
                A_copy, B_copy, gain_list, not_locked = \
                    move_and_update(base_cell, A_copy, B_copy, gain_list, not_locked, cell_array, xs)
            else:
                B_copy, A_copy, gain_list, not_locked = \
                    move_and_update(base_cell, B_copy, A_copy, gain_list, not_locked, cell_array, xs)

        if g_max > 0:
            # moving nodes from initial partition that improve the cut
            np.logical_not(A, out=A, where=move_max)
            np.logical_not(A, out=B)
        else:
            break

        i += 1

        if early_stopping and i > early_stopping:
            break

    if verbose >= 3:
        print("\tfinal ratio: {}".format(np.round(sum(A) / nb_cells, 2)))

    return A


def compute_initial_gains(A, B, cell_array, xs):
    nb_cells = len(cell_array)
    gain_list = np.full(nb_cells, 0)

    for cell_index in range(nb_cells):
        gain = 0
        for adj_cell in cell_array[cell_index]:
            if A[cell_index]:
                gain += compute_gain_for_net(A, B, adj_cell) * xs[cell_index, adj_cell]
            elif B[cell_index]:
                gain += compute_gain_for_net(B, A, adj_cell) * xs[cell_index, adj_cell]

        # add cell to the sorted bucket list
        gain_list[cell_index] = gain

    return gain_list


def compute_gain_for_net(F, T, other_index):
    Fn = F[other_index] + 1
    Tn = T[other_index]

    if Fn == 1:
        return +1
    elif Tn == 0:
        return -1
    else:
        return 0


def choose_cell_greedy(A, B, gain_list, r, not_locked):
    possible_partition = is_balanced(A, r)

    idx = np.arange(len(gain_list))[not_locked]
    cells = gain_list[not_locked]

    # choose cell that is not locked, does not harm the ratio and maximizes the gain
    for cell in idx[np.argsort(-cells)]:
        partition = [A[cell], B[cell]]
        if not_locked[cell] & np.logical_and(partition, possible_partition).any():
            return cell, gain_list[cell]

    return None, 0


def is_balanced(A, r):
    sumA = sum(A)
    cardinalityA_1 = sumA - 1
    cardinalityA_2 = sumA + 1

    W = len(A)

    leftbound = r * W
    rightbound = W - r * W
    return [leftbound <= cardinalityA_1 <= rightbound, leftbound <= cardinalityA_2 <= rightbound]


def move_and_update(base_cell, F, T, gain_list, not_locked, cell_array, xs):
    # lock base cell
    not_locked[base_cell] = False
    # switch block
    F[base_cell] = 0
    T[base_cell] = 1

    # increment or decrement gain of neighbouring cells
    for other_cell in cell_array[base_cell]:
        # check critical nets before move
        Tn = T[other_cell]
        Fn = F[other_cell] + 1
        if not_locked[other_cell]:
            if Tn == 0:
                gain_list = adjust_gain(gain_list, other_cell, +1 * xs[base_cell, other_cell])
            elif Tn == 1:
                gain_list = adjust_gain(gain_list, other_cell, -1 * xs[base_cell, other_cell])

            # # chance net distribution to reflect the move
            Tn += 1
            Fn -= 1
            if Fn == 0:
                gain_list = adjust_gain(gain_list, other_cell, -1 * xs[base_cell, other_cell])
            elif Fn == 1:
                gain_list = adjust_gain(gain_list, other_cell, +1 * xs[base_cell, other_cell])

    return F, T, gain_list, not_locked


def adjust_gain(g_list, cell, value):
    g_list[cell] += value
    return g_list


# ----------------------------------------------------------------------------------------------------------------------
# random projection and 2 means
# ----------------------------------------------------------------------------------------------------------------------

def random_projection_mean(xs, nb_cuts, seed):
    cuts = np.empty([nb_cuts, xs.shape[0]], dtype=bool)

    np.random.seed(seed)

    for c in range(nb_cuts):

        seed = np.random.randint(100)

        projection = GaussianRandomProjection(n_components=1, random_state=seed).fit_transform(xs).reshape(-1)

        cut_value = np.mean(projection)

        cut = projection < cut_value

        cuts[c] = cut.astype(bool)

    return cuts
