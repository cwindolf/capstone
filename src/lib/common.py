import numpy as np
import networkx as nx
from nx_helpers import pairwise_tree_distance
from scipy.special import comb
from scipy.optimize import fsolve
from pgf import single_pgf


def truncated_dot(u, v):
    ''' 'extend' the shorter of u, v with zeros to 'dot' with the longer '''
    l = min(u.shape[0], v.shape[0])
    return np.dot(u[:l], v[:l])


def tv(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def dd_from_ds(ds):
    dd = np.bincount(ds)
    return dd / sum(dd)


def dense_star(k):
    A = np.zeros((k, k))
    A[0, :] = A[:, 0] = 1.0
    A[0, 0] = 0.0
    return A


def choose2(n):
    return comb(n, 2, exact=True)


def pairwise_dists(G):
    '''
    Given adjacency matrix A,
    Return:
        D: np array with same shape as A,
           s.t. D[i, j] is shortest distance from i to j
    '''
    if nx.is_tree(G):
        yield from pairwise_tree_distance(G)
    else:
        for s, tdt in nx.all_pairs_dijkstra_path_length(G).items():
            for t, dt in tdt.items():
                yield (s, t), dt


def gcc(G):
    return max(nx.connected_component_subgraphs(G), key=len)


def F_mr(p):
    '''
    Find expected fraction of nodes in GC for a Molloy&Reed
    graph targeting the degree sequence @p
    '''
    c = sum(k * p_k for k, p_k in enumerate(p))
    r = [k * p_k / c for k, p_k in enumerate(p)][1:] # start at 1
    c_r = sum(k * r_k for k, r_k in enumerate(r))
    # define the pgfs g_p and g_r
    g_p = single_pgf(p)
    g_r = single_pgf(r)
    # solve for lambda_r
    def zero_at_l_eq_g_r_of_l(l):
        return g_r(l) - l
    lambda_r = fsolve(zero_at_l_eq_g_r_of_l, 0.5)[0]
    # and we know F = 1 - g_p(lambda_r)
    return 1 - g_p(lambda_r)


def discrete_def(a):
    n = len(a)
    return lambda x: a[x] if x < n and x >=0 else 0
