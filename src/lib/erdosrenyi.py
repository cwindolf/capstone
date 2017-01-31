import numpy as np
from common import choose2


def dense_erdos_renyi_n_m(n, m):
    # choose uniformly from graphs with n nodes, m edges
    A = np.zeros((n, n))
    # choose m random indices uniformly in the upper triangle
    ii, jj = np.triu_indices_from(A, k=1)
    inds = np.random.choice(ii.shape[0], m, replace=False)
    ii, jj = ii[inds], jj[inds]
    # set A symmetrically at those indices
    A[ii, jj] = A[jj, ii] = 1.0
    return A


def dense_erdos_renyi_n_p(n, p):
    # connect nodes randomly wp p
    m = np.random.binomial(choose2(n), p)
    return dense_erdos_renyi_n_m(n, m)


def dense_erdos_renyi_n_c(n, c):
    # parametrize p with c=expected degree
    p = c / (n - 1)
    return dense_erdos_renyi_n_p(n, p)
