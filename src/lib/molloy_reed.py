import networkx as nx
import numpy as np

from .common import gcc, F_mr, dd_from_ds


def mr_from_degree_sequence(target_ds):
    n = len(target_ds)
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    half_edges = np.asarray([i for i, d in enumerate(target_ds)
                             for _ in range(d)])
    np.random.shuffle(half_edges)
    G.add_edges_from(zip(half_edges[:-1:2], half_edges[1::2]))
    return nx.Graph(G)


def mr_from_degree_distribution(n, target_dd):
    target_ds = np.random.choice(len(target_dd), size=n, p=target_dd)
    return mr_from_degree_sequence(target_ds)


def neighbor_deg_dist(p):
    '''
    given the graph degree distribution @p, compute the
    degree distribution @r of a node found by following a random
    edge to one of its endpoints.
    '''
    n = p.shape[0]
    c = np.sum(np.arange(n) * p)
    kp1 = np.arange(1, n)
    return p[1:] * kp1 / c


def gcc_of_mr_targeting(L):
    '''
    Given an input graph @L,
    produce a molloy and reed graph targeting @G's degree
    sequence, and return its gcc.
    Do it so that @gcc is of the same order as @G.
    '''
    p = dd_from_ds(list(L.degree().values())).astype(np.float64)
    F = F_mr(p)
    n = int(len(L) / F)
    # make the graph but only use it's GCC
    return gcc(mr_from_degree_distribution(n, p))
