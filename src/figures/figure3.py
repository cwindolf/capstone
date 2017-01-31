import networkx as nx
from latent import d_fuzz
from common import dd_from_ds
from prefatt import sparse_pref_att_tree
from matplotlib import pyplot as plt
from pprint import pprint

def f(m):
    if m == 2:
        return 1.0
    else:
        return 0.0

L = sparse_pref_att_tree(50)
G = d_fuzz(L, f)

fig, (ax_L, ax_G) = plt.subplots(1, 2, figsize=(10, 4.5))
graph_style = { 'node_color': 'k', 'node_size': 30 }
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
plt.savefig('fig/3.png', bbox_inches=0, dpi=300)
