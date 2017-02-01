import networkx as nx
from matplotlib import pyplot as plt

from .figconfig import FIG_PATH, figstyle, graph_style
from ..lib.common import dd_from_ds
from ..lib.lsrg import lsrg
from ..lib.prefatt import sparse_pref_att_tree


def f(m):
    if m == 2:
        return 1.0
    else:
        return 0.0

L = sparse_pref_att_tree(50)
G = lsrg(L, f)

fig, (ax_L, ax_G) = plt.subplots(1, 2, figsize=(10, 4.5))
# figure out tree hierarchy
root_dists = nx.single_source_dijkstra_path_length(L, 0)
max_d = max(root_dists.values())
shells = [[]  for _ in range(max_d + 1)]
for v, d in root_dists.items():
    shells[d].append(v)
pos = nx.shell_layout(L, shells)
nx.draw(L, pos=pos, ax=ax_L, **graph_style)
nx.draw(G, pos=pos, ax=ax_G, **graph_style)
plt.tight_layout(pad=0.0)
plt.savefig(FIG_PATH % '3.png', **figstyle)
