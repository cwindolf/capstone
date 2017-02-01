import numpy as np
import numpy.linalg as la

from .common import choose2


def dd(A, dtype=np.float_):
    '''
    Degree distribution from adjacency matrix.
    '''
    return np.sum(A, axis=1, dtype=dtype)


def jdd(A):
    '''
    Joint degree distribution
    '''
    d = dd(A, dtype=np.int_)
    nbins = np.amax(d) + 1
    jdd = np.zeros((nbins, nbins))
    ii, jj = A.nonzero()
    for i, j in zip(ii, jj):
        jdd[d[i], d[j]] += 1.0
    return jdd


def cl(A):
    '''
                  __
     CL(G) = 1/n \   (triangles at i /                          )
                 /__ (              /  connected trips through i)
    '''
    A3 = la.matrix_power(A, 3)
    return (1 / d.shape[0]) * sum(A3[i, i] / choose2(d_i) if d_i > 2 else 0
                                  for i, d_i in enumerate(d))


def clt(A):
    '''
    number of triangles in graph is tr(A^3)/6,
    because we count each triangle three times (once for each vtx),
    and each time, we count it clockwise and counterclockwise
    '''
    A3 = la.matrix_power(A, 3)
    return np.trace(A3) / (2 * np.sum(np.vectorize(choose2)(d)))
