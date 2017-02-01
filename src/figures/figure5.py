'''
Run PGF estimate for mean degree on LSRGs
made with Price's model trees and M&R graphs
targeting those trees.
Expect more accuracy with the M&R graphs, where
more assumptions hold.
'''
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import pickle

from .figconfig import FIG_PATH, figstyle
from lib import common
from lib.lsrg import lsrg
from lib.pgf import *
from lib.common import dd_from_ds, discrete_def, F_mr, gcc
from lib.prefatt import sparse_price_tree
from lib.molloy_reed import gcc_of_mr_targeting, neighbor_deg_dist

# store observations and predictions here
c_G, c_T, c_pred = [], [], []
# range of graph sizes to test on. maybe it will converge as n -> oo
domain = [2 ** i for i in range(5, 15)]

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
    # we'll sum out to diameter below
    M = nx.diameter(T)

    # Now, make Molloy&Reed graph targeting @ds, but only take its GC
    G = gcc_of_mr_targeting(T)

    # now, apply the transformation
    f = discrete_def([0.0, 0.5, 0.25, 0.125])
    Gp = lsrg(G, f=f)
    Tp = lsrg(T, f=f)
    p_G = dd_from_ds(list(Gp.degree().values()))
    p_T = dd_from_ds(list(Tp.degree().values()))
    del T, G, Gp, Tp # these can get big, we don't need them anymore
    # and get the empirically observed degree distribution
    c_G.append(sum(k * p_k for k, p_k in enumerate(p_G)))
    c_T.append(sum(k * p_k for k, p_k in enumerate(p_T)))


    # *************************************************************************** #
    # PGF prediction ************************************************************ #
    # *************************************************************************** #

    # and we need r and g_r
    r = neighbor_deg_dist(p).astype(np.float64)

    # predict
    c_pred.append(pgf_predict_mean_degree(p, r, f, M))


# *************************************************************************** #
# Comparison **************************************************************** #
# *************************************************************************** #
fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log', basex=2)
ax.set_xlabel('Order')
ax.set_ylabel('Mean Degree')
plt.scatter(domain, c_G, color='k', marker='x', label='Price LSRG')
plt.scatter(domain, c_T, color='k', marker='o', label='M&R LSRG',
            facecolors='none', edgecolors='k',)
plt.scatter(domain, c_pred, marker='+', label='PGF Prediction')
plt.legend(loc='upper left')
plt.show()

# save... these take a while to compute
with open('fig5.dat', 'wb') as f:
    pickle.dump((c_G, c_T, c_pred), f)

