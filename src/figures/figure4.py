'''
Generate bar charts comparing the degree distributions of
LSRGs created from Price trees and M&R graphs targeting
those trees.
'''
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from .figconfig import FIG_PATH, figstyle
from ..lib.latent import lsrg
from ..lib.common import dd_from_ds
from ..lib.prefatt import sparse_price_tree
from ..lib.molloy_reed import gcc_of_mr_targeting, neighbor_deg_dist

# choosing alpha=2 for a power-law with neighbor deg dist that
# sums up nice
T = sparse_price_tree(10000, 2)
ds = np.asarray(list(T.degree().values()))
p = dd_from_ds(ds).astype(np.float64)
r = neighbor_deg_dist(p).astype(np.float64)

# Now, make Molloy&Reed graph targeting @ds, but only take its GC
G = gcc_of_mr_targeting(T)

# now, apply the transformation
def f(m): return int(m == 2)
Gp = lsrg(G, f=f)
Tp = lsrg(T, f=f)


# and get the empirically observed degree distribution
prp = dd_from_ds(list(Tp.degree().values()))
mrp = dd_from_ds(list(Gp.degree().values()))

fig, (ax_P, ax_MR) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
ax_P.bar(range(len(prp)), prp, color='k')
ax_P.set_title('From Price\'s model')
ax_MR.bar(range(len(mrp)), mrp, color='k')
ax_MR.set_title('From Molloy and Reed')
plt.tight_layout(pad=0.1)
plt.savefig(FIG_PATH % '4.png', **figstyle)
