
import numpy as np
from functions.create_random_PT2 import *
from scipy import signal
from matplotlib import pyplot as p

def test_GP(GP_P, GP_I):

    num, den = create_random_PT2()

    X_test = np.zeros(4)
    X_test[0] = num[0]
    X_test[1] = num[1]
    X_test[2] = den[1]
    X_test[3] = den[2]
    P_test, P_test_Covariance = GP_P.regression(np.matrix(X_test))
    I_test, I_test_Covariance = GP_I.regression(np.matrix(X_test))

    ol_num = np.convolve(num, np.squeeze([P_test, I_test]))
    ol_den = np.convolve(den, [1, 0])
    sys_num = ol_num
    sys_den = [ol_den[0], (ol_num[0] + ol_den[1]), (ol_num[1] + ol_den[2]), (ol_num[2] + ol_den[3])]

    sys = signal.TransferFunction(sys_num, sys_den)
    t, y = signal.step(system=sys, N=200)
    p.plot(t, y)
    p.show()

    pass


