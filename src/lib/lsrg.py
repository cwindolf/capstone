import networkx as nx
from numpy.random import uniform

from .common import pairwise_dists


def lsrg(L, f, directed=False):
    '''
    Given a latent graph @L, observe a graph where the probability of
    connecting two nodes has some relation `f` to their distance in
    the latent graph.
    Args:
        L: latent graph
        f: function of distance. probability of connecting two nodes
           p_{e_ij} prop. to. f(d_{latent}(i, j)).
           Default to f(x) = 1/x
        directed: whether to treat latent graph as directed, and observe
           a directed graph.
    Return:
        G: the LSRG G_{L,f}
    '''
    n = L.number_of_nodes()
    G_prime = nx.Graph()
    G_prime.add_nodes_from(L)
    for (i, j), dij in pairwise_dists(L):
        if f(dij) >= uniform():
            G_prime.add_edge(i, j)
    return G_prime
