import numpy as np
import networkx as nx
from numpy.random import rand, randint
from statistics import dd
from common import dense_star


def dense_pref_att_graph(n, m0, m):
    # Start with a star that has m0 + 1 nodes.
    # Center will be node 0.
    # But we're gonna need more nodes...
    A = np.zeros((n, n))
    A[0:m0 + 1, 0:m0 + 1] = dense_star(m0 + 1)
    # Iteratively attach new nodes
    for i in range(m0 + 1, n):
        # Degree distribution is prob. dist. of new attachment
        d = dd(A[0:i, 0:i])
        d /= np.sum(d)
        # get m neighbors from [0,..,i) with p_i = d_i
        n_i = np.random.choice(i, m, replace=False, p=d)
        # Update A
        A[i, n_i] = A[n_i, i] = 1.0
    return A


def dense_pref_att_tree(n, m0):
    # if each node only adds one connection, we'll
    # observe a tree
    return dense_pref_att_graph(n, m0, 1)


def sparse_price_tree(n, alpha):
    lam = alpha / (alpha + 1)
    edges = np.zeros((n, 2), dtype=np.int_)
    for v in range(1, n):
        # get c samples from [0,...,v) wppt d+alpha
        # flip a coin with weight lam
        edges[v, 0] = v
        if rand() < lam:
            # choose uniformly from [0,...,v)
            edges[v, 1] = randint(0, v)
        else:
            # random vertex in randomly chosen edge
            edges[v, 1] = edges[randint(0, v), randint(0, 2)]
    # (Di)Graph will get rid of multi-edges
    G = nx.Graph()
    G.add_edges_from(edges[1:])
    assert(nx.is_tree(G))
    return G


def sparse_pref_att_graph(n, c):
    nc = n * c
    alpha = c
    lam = 0.5 # alpha / (alpha + c)
    edges = np.zeros((nc, 2), dtype=np.int_)
    # start with a star, because we want to be able to make trees
    edges[0:c] = [[0, i] for i in range(c)]
    for v in range(1, n):
        # get c samples froc [0,...,v) wppt d+c
        # flip a coin with weight lam
        vc = v * c
        for i in range(vc, vc + c):
            edges[i, 0] = v
            if rand() < lam:
                # choose uniformly from [0,...,v)
                edges[i, 1] = randint(0, v)
            else:
                # randomly choosen vertex from randomly chosen edge
                edges[i, 1] = edges[randint(0, vc), randint(0, 2)]
    # simple Graph will get rid of multi-edges
    G = nx.Graph()
    G.add_edges_from(edges[1:]) # first one is a self loop
    return G


def sparse_pref_att_tree(n):
    G = sparse_pref_att_graph(n, 1)
    assert(nx.is_tree(G))
    return G
