
import numpy as np
from functions.create_random_pt2 import *
from scipy import signal
from matplotlib import pyplot as p


def test_gaussian_process(gaussian_process_p, gaussian_process_i):

    num, den = create_random_pt2()

    x_test = np.zeros(4)
    x_test[0] = num[0]
    x_test[1] = num[1]
    x_test[2] = den[1]
    x_test[3] = den[2]
    p_test, p_test_covariance = gaussian_process_p.regression(np.matrix(x_test))
    i_test, i_test_covariance = gaussian_process_i.regression(np.matrix(x_test))

    ol_num = np.convolve(num, np.squeeze([p_test, i_test]))
    ol_den = np.convolve(den, [1, 0])
    sys_num = ol_num
    sys_den = [ol_den[0], (ol_num[0] + ol_den[1]), (ol_num[1] + ol_den[2]), (ol_num[2] + ol_den[3])]

    sys = signal.TransferFunction(sys_num, sys_den)
    t, y = signal.step(system=sys, N=200)
    p.plot(t, y)
    p.show()

    pass


