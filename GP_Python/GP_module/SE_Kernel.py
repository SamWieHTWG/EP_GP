import numpy as np

def SE_Kernel(X1, X2, s_n, ell, s_f):

    y = 0
    l1 = np.size(X1,1)
    l2 = np.size(X2,1)

    index_1 = np.arange(l1)
    index_2 = np.arange(l2)

    # use: np.vectorize
    for ii in index_1:
        for jj in index_2:

            y = y + x1 + x2

    return y

x1 = np.array([2, 1, 4])
x2 = np.array([3, 2, 1])
print( SE_Kernel(x1,x2,0.1,1,1) )


