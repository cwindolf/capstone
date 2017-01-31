import numpy as np
from scipy.signal import fftconvolve as convolve
from scipy.sparse import csc_matrix, csr_matrix
# from numpy import convolve
from scipy.misc import comb
from functools import reduce
import networkx as nx
from latent import d_fuzz
from convolution import total_prob
from common import dd_from_ds, discrete_def, F_mr, gcc, truncated_dot
from prefatt import sparse_price_tree
from molloy_reed import mr_from_degree_distribution, neighbor_deg_dist
from tqdm import trange

# *************************************************************************** #
# Empirical stuff *********************************************************** #
# *************************************************************************** #

# choosing alpha=2 for a power-law with neighbor deg dist that
# sums up nice
T = sparse_price_tree(1000, 2)
ds = np.asarray(list(T.degree().values()))
p = dd_from_ds(ds).astype(np.float64)
r = neighbor_deg_dist(p).astype(np.float64)

# Now, make Molloy&Reed graph targeting @ds, but only take its GC
F = F_mr(p)
n = int(len(T) / F)
# make the graph but only use it's GCC
G = gcc(mr_from_degree_distribution(n, p))

# now, apply the transformation
_f = [0.0, 0.8, 0.4, 0.2, 0.1]
f = discrete_def(_f)
Gp = d_fuzz(G, f=f)
# and get the empirically observed degree distribution
p_prime_observed = dd_from_ds(list(Gp.degree().values()))
c_prime_observed = sum(k * p_k for k, p_k in enumerate(p_prime_observed))


# *************************************************************************** #
# Easy estimate ************************************************************* #
# *************************************************************************** #

# precompute R
max_kids_per_node = r.shape[0] - 1
M = min(nx.diameter(G), len(_f) - 1)
# with r^j on columns (j=0:max_kids^M), because generation M might have that
# many kids
# and with r^*(i) on the rows (max_j * max_kids, cuz that's the max sum
# of max_j kids)
max_z_M = max_kids_per_node ** M
# want fast cols here
R = csc_matrix((max_z_M * max_kids_per_node + 1, max_z_M + 1))
R[0, 0] = 1.0 # r^0 is the point pmf at 1
# print(r.shape)
R[:r.shape[0], 1] = r[:, np.newaxis] # r^1 = r
print('rsum')
for j in trange(2, max_z_M + 1):
    max_z_j_prev = R.getcol(j - 1).getnnz()
    R[:max_z_j_prev + max_kids_per_node, j] = convolve(r, R[:max_z_j_prev, j - 1].toarray().flat)[:, np.newaxis]
# now rows
R = R.to_csr()

# here, we just find u_0,...,u_M and then use those to find d_0,...,d_M.
# then we pretend they're independent and sum them all (convolve)
U = np.zeros((M, max_z_M))
U[0, 0] = 1
U[1, :p.shape[0]] = p
for m in range(2, M + 1):
    max_z_m = max_kids_per_node ** m
    for i in range(max_z_m + 1):
        U[m, i] = R[i].dot(U[m - 1])

print('usum')
for u_i in U:
    print(u_i.sum())

# note that P(D_m = d | Z_m = n) is a binomial pmf
# with p=f(m), n=n.
# let's precompute that 3-tensor of binomial pmfs. let's put p on the
# first axis (with length equal to M or the length of f's support, whichever
# is # smaller), then n on the second axis with length r.shape[0] as before,
# and finally the value in question on the third, which will
# also have length n.
# with this setup we can grab the matrix P(D_m = d | Z_m = n) by taking
# m as the first index.
depth = min(M, len(_f))
def binom(i, j, k):
    return comb(i, k) * (f(j) ** k) * ((1 - f(j)) ** (i - k))
binom = np.vectorize(binom)
F = np.fromfunction(binom, (r.shape[0], depth, r.shape[0]), dtype=np.int64)

# now we're ready to find each pmf d_m
dms = [total_prob(F[m], u[m]) for m in range(depth)]
print('d sums')
for dm in dms:
    print(sum(dm))
# and with the fake assumption that the D_ms are independent, we can
# estimate the degree distribution by summing them like they were,
# which means D's pmf is the convolution of all of the d_m
d = reduce(convolve, dms)
print('sumd', sum(d))
# and that's our estimate!
# estimates mean degree at
c = sum(k * d_k for k, d_k in enumerate(d))

# *************************************************************************** #
# Comparison **************************************************************** #
# *************************************************************************** #

print(p_prime_observed)
print(d)
minlen = min(p_prime_observed.shape[0], d.shape[0])
print(np.sum(np.abs(p_prime_observed[:minlen] - d[:minlen])))
print(c_prime_observed, c)


