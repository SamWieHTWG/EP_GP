import numpy as np
import math

def SE_Kernel(x1, x2, ell, s_f):


    s1 = x1.shape
    s2 = x2.shape

    l1 = s1[0]
    l2 = s2[0]

    index_1 = np.arange(l1)
    index_2 = np.arange(l2)

    k = np.zeros((l1, l2))

    # use: np.vectorize
    for ii in index_1:
        for jj in index_2:
            k[ii, jj] = 2 * s_f * math.exp( -1/(2*ell^2) \
                np.linalg.norm()





    return k

x1 = np.matrix('1 2; 3 4; 2 3')

x2 = np.matrix('1 2; 3 4')


print( SE_Kernel(x1,x2,1,1) )


