import networkx as nx
import numpy as np
from latent import d_fuzz
from common import dd_from_ds, F_mr, gcc
from prefatt import sparse_price_tree
from matplotlib import pyplot as plt
from molloy_reed import mr_from_degree_distribution, neighbor_deg_dist
from pprint import pprint

# choosing alpha=2 for a power-law with neighbor deg dist that
# sums up nice
T = sparse_price_tree(10000, 2)
ds = np.asarray(list(T.degree().values()))
p = dd_from_ds(ds).astype(np.float64)
r = neighbor_deg_dist(p).astype(np.float64)

# Now, make Molloy&Reed graph targeting @ds, but only take its GC
F = F_mr(p)
n = int(len(T) / F)
# make the graph but only use it's GCC
G = gcc(mr_from_degree_distribution(n, p))

# now, apply the transformation
def f(m): return int(m == 2)
Gp = d_fuzz(G, f=f)
Tp = d_fuzz(T, f=f)


# and get the empirically observed degree distribution
prp = dd_from_ds(list(Tp.degree().values()))
mrp = dd_from_ds(list(Gp.degree().values()))

fig, (ax_P, ax_MR) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
ax_P.bar(range(len(prp)), prp, color='k')
ax_P.set_title('From Price\'s model')
ax_MR.bar(range(len(mrp)), mrp, color='k')
ax_MR.set_title('From Molloy and Reed')
plt.tight_layout(pad=0.1)
plt.savefig('fig/4.png', dpi=300)
