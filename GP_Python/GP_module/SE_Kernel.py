import numpy as np
import math


def SE_Kernel(x1, x2, ell, s_f):

    x1 = np.transpose(x1)
    x2 = np.transpose(x2)

    s1 = x1.shape
    s2 = x2.shape

    l1 = s1[1]
    l2 = s2[1]

    index_1 = np.arange(l1)
    index_2 = np.arange(l2)

    k = np.zeros((l1, l2))
    norm_factor = np.zeros((l1,l2))

    for ii in index_1:
        for jj in index_2:

            norm_factor[ii,jj] = np.power(np.linalg.norm(x1[:,ii]-x2[:,jj]), 2)

            k[ii, jj] = s_f*s_f * math.exp( -1/(2*ell*ell) * norm_factor[ii, jj])

    return k


# test
#x1 = np.matrix('1.0 2.0; 3.2 4.1')
#x2 = np.matrix('1.1 2.2 4.3 ; 5.1 3.9 4.1')
#print( SE_Kernel(x1, x2,1,1) )


