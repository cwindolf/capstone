import numpy as np
import networkx as nx
from plotting import *
from prefatt import sparse_price_tree
from molloy_reed import mr_from_degree_sequence, mr_from_degree_distribution
from latent import d_fuzz
from common import discrete_def, gcc, F_mr, dd_from_ds
from matplotlib import pyplot as plt

f = discrete_def([0.0, 0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])

# choosing alpha=2 for a power-law with neighbor deg dist that
# sums up nice
G = sparse_price_tree(1000, 2)
G_ds = np.asarray(list(G.degree().values()))
p = dd_from_ds(G_ds)
# apply latent graph model using @f
G_prime = d_fuzz(G, f=f)
A_prime = nx.to_scipy_sparse_matrix(G_prime)

# now, make a M&R graph.
# we just want the giant component, but we want that to be
# of the same order as G. so, we need to figure out the
# expected fraction of nodes in the GC of a M&R with ds @p.
F = F_mr(p)
n = int(len(G) / F)
# make the graph but only use it's GCC
G_MR = gcc(mr_from_degree_distribution(n, p))
A_MR = nx.to_scipy_sparse_matrix(G_MR)
# and now apply the latent graph transform with @f
MRG_prime = d_fuzz(G_MR, f=f)
MRA_prime = nx.to_scipy_sparse_matrix(MRG_prime)

print('stats for Price:', 'n', G_prime.number_of_nodes(),
                          'm', G_prime.number_of_edges())
print('stats for M&R:', 'n', MRG_prime.number_of_nodes(),
                        'm', MRG_prime.number_of_edges())

plot_dd_and_jdd(nx.to_scipy_sparse_matrix(G), 'Price Latent Graph')
plot_dd_and_jdd(A_MR, 'M&R Latent GCC')
plot_dd_and_jdd(A_prime, 'From Price')
plot_dd_and_jdd(MRA_prime, 'With Molloy & Reed GCC')
plt.show()