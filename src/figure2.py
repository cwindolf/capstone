import networkx as nx
from latent import d_fuzz
from matplotlib import pyplot as plt

# define the point mass at two
def f(m):
    if m == 2:
        return 1.0
    else:
        return 0.0

# (A): triangle graph
Ltri = nx.cycle_graph(3)

# (B): triangle graph missing an edge
Ltp = nx.path_graph(3)

# (C): square graph
Lsq = nx.cycle_graph(4)

# (D): odd cycle
Loc = nx.cycle_graph(7)

# (E): even cycle
Lec = nx.cycle_graph(8)

# (F): even path
Lep = nx.path_graph(8)

# make LSRGs
names = ['A',  'B', 'C', 'D', 'E', 'F']
Ls    = [Ltri, Ltp, Lsq, Loc, Lec, Lep]
Gs    = [d_fuzz(L, f) for L in Ls]

# save the figure
graph_style = { 'node_color': 'k', 'node_size': 50 }
for (L, G, a) in zip(Ls, Gs, names):
    fig, (ax_L, ax_G) = plt.subplots(1, 2, figsize=(5, 2))
    pos = nx.circular_layout(L)
    nx.draw(L, pos=pos, ax=ax_L, **graph_style)
    nx.draw(G, pos=pos, ax=ax_G, **graph_style)
    plt.tight_layout(pad=0.15)
    plt.savefig('fig/2_%s.png' % a, dpi=300)
