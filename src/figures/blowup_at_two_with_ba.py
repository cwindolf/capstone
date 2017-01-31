import numpy as np
import networkx as nx
from plotting import *
from prefatt import sparse_pref_att_tree
from latent import d_fuzz
from matplotlib import pyplot as plt

def d_2(x):
    return 1.0 if x == 2 else 0.0

G = sparse_pref_att_tree(10000)
A = nx.to_scipy_sparse_matrix(G)
G_prime = d_fuzz(G, f=d_2)
A_prime = nx.to_scipy_sparse_matrix(G_prime)

plot_dd(A)
plot_dd(A_prime)
plot_jdd(A)
plot_jdd(A_prime)
plt.show()