import numpy as np
import networkx as nx
from latent import d_fuzz
from pgf import *
from common import dd_from_ds, discrete_def, F_mr, gcc
from prefatt import sparse_price_tree
from molloy_reed import mr_from_degree_distribution, neighbor_deg_dist
from matplotlib import pyplot as plt

c_G, c_T, c_pred = [], [], []
domain = [2 ** i for i in range(3, 16)]

for N in domain:
    print(N)
    # *************************************************************************** #
    # Empirical stuff *********************************************************** #
    # *************************************************************************** #

    # choosing alpha=2 for a power-law with neighbor deg dist that
    # sums up nice
    T = sparse_price_tree(N, 2)
    ds = np.asarray(list(T.degree().values()))
    p = dd_from_ds(ds).astype(np.float64)

    # Now, make Molloy&Reed graph targeting @ds, but only take its GC
    F = F_mr(p)
    n = int(len(T) / F)
    # make the graph but only use it's GCC
    G = gcc(mr_from_degree_distribution(n, p))

    # now, apply the transformation
    def f(m): return (1 / (2 ** m) if m > 0 else m)
    Tp = d_fuzz(T, f=f)
    Gp = d_fuzz(G, f=f)
    # and get the empirically observed degree distribution
    p_G = dd_from_ds(list(Gp.degree().values()))
    c_G.append(sum(k * p_k for k, p_k in enumerate(p_G)))
    p_T = dd_from_ds(list(Tp.degree().values()))
    c_T.append(sum(k * p_k for k, p_k in enumerate(p_T)))


    # *************************************************************************** #
    # PGF prediction ************************************************************ #
    # *************************************************************************** #

    # the PGF for the degree of a node in @Gp is:
    #              ___
    #     g_D(s) = | | g_p(g_r^i(g_f_i(s)))
    #             i=0:M 
    # where
    M = nx.diameter(T)

    # so we'll need the pgfs g_f_i for i=0:M
    # these are bernoulli, so each takes value 1 wp f(m), 0 wp 1-f(m)
    g_f = [bernoulli_pgf(f(m), ('g_f%d' % m)) for m in range(M + 1)]

    # and we need r and g_r
    r = neighbor_deg_dist(p).astype(np.float64)

    # and do repeated differentiation stuff to get the pmf for D
    # p_prime_predicted = generate_pmf(g_D, n)
    c_pred.append(pgf_predict_mean_degree(p, r, f, M))

    del T, Tp, G, Gp


# *************************************************************************** #
# Comparison **************************************************************** #
# *************************************************************************** #
# fig, ax = plt.figure()
# ax.set_xscale('log', basex=2)
# ax.set_xlabel('n')
# ax.set_ylabel('c')
# plt.scatter(domain, c_G, color='r', label='G')
# plt.scatter(domain, c_T, color='b', label='T')
# plt.scatter(domain, c_pred, color='g', label='Predicted')
# plt.legend()
# plt.show()

# run from ipython

