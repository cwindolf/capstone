'''
Run PGF estimate for mean degree on LSRGs
made with Price's model trees and M&R graphs
targeting those trees.
Expect more accuracy with the M&R graphs, where
more assumptions hold.

Uses graphs made by `make_some_graphs`, so run that first!
'''
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import pickle
import os.path

from .make_some_graphs import load_the_graphs
from .figconfig import FIG_PATH, figstyle
from lib.pgf import pgf_predict_mean_degree
from lib.common import dd_from_ds, discrete_def
from lib.molloy_reed import neighbor_deg_dist


f = discrete_def([0.0, 0.5, 0.25, 0.125])
domain, Gs, Ts, Gps, Tps, M = load_the_graphs()
c_T, c_G, c_pred = [], [], []
for N, G, T, Gp, Tp in zip(domain, Gs, Ts, Gps, Tps):
    print(N)
    # *********************************************************************** #
    # Empirical  ************************************************************ #
    # *********************************************************************** #

    # choosing alpha=2 for a power-law with neighbor deg dist that
    # sums up nice
    ds = np.asarray(list(T.degree().values()))
    p = dd_from_ds(ds).astype(np.float64)
    r = neighbor_deg_dist(p).astype(np.float64)

    # and get the empirically observed degree distribution
    p_T_prime_observed = dd_from_ds(list(Tp.degree().values()))
    c_T.append(sum(k * p_k for k, p_k in enumerate(p_T_prime_observed)))
    p_G_prime_observed = dd_from_ds(list(Gp.degree().values()))
    c_G.append(sum(k * p_k for k, p_k in enumerate(p_G_prime_observed)))

    # *********************************************************************** #
    # Estimated ************************************************************* #
    # *********************************************************************** #

    # predict
    c_pred.append(pgf_predict_mean_degree(p, r, f, M))


# *************************************************************************** #
# Comparison **************************************************************** #
# *************************************************************************** #
fig = plt.figure(figsize=(12, 8))
plt.rc('axes', labelsize=16)
ax = plt.gca()
ax.set_xscale('log', basex=2)
ax.set_xlabel('Graph Order')
ax.set_ylabel('Mean Degree')
plt.scatter(domain, c_G, color='r', marker='x', label='M&R LSRG')
plt.scatter(domain, c_T, marker='o', label='Price LSRG',
            facecolors='none', edgecolors='g')
plt.scatter(domain, c_pred, color='b', marker='+', label='PGF Prediction')
plt.legend(loc='upper left')
plt.savefig(FIG_PATH % '5.png', **figstyle)
