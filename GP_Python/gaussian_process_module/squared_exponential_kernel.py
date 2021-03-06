import numpy as np
import math


def squared_exponential_kernel(x1, x2, ell, s_f):
    """!
    The squared exponential kernel is an essential part of the gaussian process.
    Further informations: http://www.gaussianprocess.org/gpml/
    @param x1: np_array: first input vector
    @param x2: np_array: second input vector
    @param ell: float: parameter, defines length_scale
    @param s_f: float: parameter, defines process noise
    @return: np_matrix: covariance between both input vectors
    """

    if x1.ndim >1:
        x1 = np.ndarray.transpose(x1)
    else:
        x1 = np.expand_dims(x1, axis=0)
        x1 = np.ndarray.transpose(x1)

    if x2.ndim >1:
        x2 = np.ndarray.transpose(x2)
    else:
        x2 = np.expand_dims(x2, axis=0)
        x2 = np.ndarray.transpose(x2)

    if x1.ndim > 1:

        s1 = x1.shape
        l1 = s1[1]
    else:
        l1 = 1

    if x2.ndim > 1:

        s2 = x2.shape
        l2 = s2[1]
    else:
        l2 = 1

    index_1 = np.arange(l1)
    index_2 = np.arange(l2)

    k = np.zeros((l1, l2))
    norm_factor = np.zeros((l1,l2))

    for ii in index_1:
        for jj in index_2:

            norm_factor[ii, jj] = np.power(np.linalg.norm(x1[:, ii]-x2[:, jj]), 2)

            k[ii, jj] = s_f*s_f * math.exp( -1/(2*ell*ell) * norm_factor[ii, jj])

    return k


