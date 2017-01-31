import numpy as np
from functools import reduce
from scipy.signal import fftconvolve as convolve


def total_prob(P_x_cond_y, P_y):
    '''
    implementing the law of total probability:
               inf
               __
    P(X = x) = \   P(X=x | Y=j)P(Y=j)
               /_
               j=0
    but here, P(X|Y) is the matrix P_x_cond_y, with row j
    giving the probabilities for X given Y=j,
    and P_y is the pmf vector for Y
    returns X's pmf vector
    '''
    return np.sum(P_x_cond_y[:P_y.shape[0]] * P_y[:, np.newaxis], axis=0)
