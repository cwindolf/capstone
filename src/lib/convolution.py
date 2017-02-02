import numpy as np
from scipy.signal import fftconvolve as convolve
from .common import discrete_def, binom


def naive_lsrg_mean_degree(p, r, f, M):
    '''
    Use branching process and convlution intuition to estimate
    the mean degree of an LSRG.
    Params:
        @p      np.array
                the latent graph's degree distribution

        @r      np.array
                the latent graph's neighbor-degree distribution

        @f      Py fn
                the function used to produce the lsrg

        @M      int
                the minimum of the latent graph's diameter and the length of
                @f's support. this is how we know where to stop sums.
    '''
    # *********************************************************************** #
    # first things first. let's precompute the r^j for all necessary j.
    # we know that we'll need the max j to be the max number of kids in
    # the penultimate generation, which here is the (@M - 1)th.
    max_kids_per_generation = r.shape[0]
    # first generation has len(@p) kids max, subsequent gens have @mkpg
    max_kids_ever = p.shape[0] * (max_kids_per_generation ** (M - 2))
    # now compute r^j for j=0:max_kids_ever
    # recall r^0 is the point pmf at 0. if we had zero kids, we're going to
    # forever.
    Rs = [np.array([1.0]), r]
    for j in range(2, max_kids_ever + 1):
        Rs.append(convolve(r, Rs[j - 1]))

    # we'll need easy slicing for R
    R = np.zeros((len(Rs), Rs[-1].shape[0]))
    for j, rj in enumerate(Rs):
        R[j,:rj.shape[0]] = rj

    # *********************************************************************** #
    # compute z_m pmfs for m=0:M
    # recall
    #          _∞_
    # z_m(i) = \   r^j(i) z_{m-1}(j)
    #          /__
    #          j=0
    # and that
    # z_0 is the point pmf at 1, and z_1 is p.
    Z = [np.array([0.0, 1.0]), p]
    max_z_m = p.shape[0]
    for m in range(2, M + 1):
        # so, in other words, z_m(i) is the ith column of R dotted with z_{m-1}
        # the ith column of R has length max_kids_ever, which is not completely
        # necessary: we only need len(z_{m-1}) of it.
        # and we need to take the columns i from 0 to the max that z_m could
        # possibly be, which we can compute:
        max_z_m *= max_kids_per_generation
        # and the column dots can be expressed nicely as
        Z.append(np.sum(R[:Z[m-1].shape[0], :max_z_m] 
                        * Z[m - 1][:, np.newaxis],
                        axis=0))

    # *********************************************************************** #
    # compute expected mean degree
    return sum(f(m) 
               * sum(k * z_k for k, z_k in enumerate(Z[m]))
               for m in range(M + 1))


def lsrg_degree_distribution(p, r, f, M):
    '''
    Use branching process and convlution intuition to estimate
    the degree distribution of an LSRG.
    Params:
        @p      np.array
                the latent graph's degree distribution

        @r      np.array
                the latent graph's neighbor-degree distribution

        @f      Py fn
                the function used to produce the lsrg

        @M      int
                the minimum of the latent graph's diameter and the length of
                @f's support. this is how we know where to stop sums.
    '''
    pass

