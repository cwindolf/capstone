'''
Comparison of distance matrices after LSRG
transformation. Does the M&R look like the Price?
'''
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from .figconfig import FIG_PATH, figstyle
from lib import common
from lib.lsrg import lsrg
from lib.pgf import *
from lib.common import dd_from_ds, discrete_def, F_mr, gcc
from lib.prefatt import sparse_price_tree
from lib.molloy_reed import gcc_of_mr_targeting, neighbor_deg_dist

# choosing alpha=2 for a power-law with neighbor deg dist that
# sums up nice
T = sparse_price_tree(128, 2)
ds = np.asarray(list(T.degree().values()))
p = dd_from_ds(ds).astype(np.float64)

# Now, make Molloy&Reed graph targeting @ds, but only take its GC
G = gcc_of_mr_targeting(T)

# now, apply the transformation
def f(m): return (1 / (2 ** m) if m > 0 else m)
Tp = lsrg(T, f=f)
Gp = lsrg(G, f=f)

names = ['Latent Price tree', 'Latent M&R GCC', 'Price LSRG', 'M&R LSRG']
graphs = [T, G, Tp, Gp]
distance_matrices = map(nx.floyd_warshall_numpy, graphs)
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = [ax for row in axes for ax in row]
plt.tight_layout()
plt.rc('axes', labelsize=36)
for name, D, ax in zip(names, distance_matrices, axes):
    ax.imshow(D, interpolation='nearest', cmap='jet')
    for spine in ax.spines.values():
        spine.set_color('none')
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(name)
plt.savefig(FIG_PATH % '6.png', **figstyle)
