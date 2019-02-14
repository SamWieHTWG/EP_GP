
import numpy as np
from functions.create_random_PT2 import *
from scipy import signal
from matplotlib import pyplot as p
import time


def get_rating(GP_P, GP_I, num_iterations, do_plot):

    sum_gp_better_performance = 0

    for index in range(num_iterations):
        iteration_gp_better_performance = score_GP(GP_P, GP_I, do_plot)
        if iteration_gp_better_performance:
            sum_gp_better_performance = sum_gp_better_performance + 1

    return sum_gp_better_performance/num_iterations


def score_GP(GP_P, GP_I, do_plot):

    err_GP = np.Inf
    while err_GP > 10000:

        num, den = create_random_PT2()

        X_test = np.zeros(4)
        X_test[0] = num[0]
        X_test[1] = num[1]
        X_test[2] = den[1]
        X_test[3] = den[2]
        P_test, P_test_Covariance = GP_P.regression(np.matrix(X_test))
        I_test, I_test_Covariance = GP_I.regression(np.matrix(X_test))

        num_of_values = 1000
        t_GP, y_GP = step_response(P_test, I_test, num, den, num_of_values)
        t_ref, y_ref = step_response(1, 5, num, den, num_of_values)

        target_value = np.ones(num_of_values)

        err_GP = np.sum(np.multiply((y_GP-target_value),(y_GP-target_value)))
        err_ref = np.sum(np.multiply((y_ref-target_value),(y_ref-target_value)))

    print(err_GP)
    print(err_ref)

    if do_plot:

        p.plot(t_GP, y_GP)
        p.plot(t_ref, y_ref)
        p.legend('Gaussian Process', 'P = 1, I = 10')
        p.show()
        time.sleep(2)

    if err_GP < err_ref:
        gp_better_performance = True
    else:
        gp_better_performance = False

    return gp_better_performance


def step_response(P, I, num, den, num_of_values):
    ol_num = np.convolve(num, np.squeeze([P, I]))
    ol_den = np.convolve(den, [1, 0])
    sys_num = ol_num
    sys_den = [ol_den[0], (ol_num[0] + ol_den[1]), (ol_num[1] + ol_den[2]), (ol_num[2] + ol_den[3])]

    time = np.linspace(0.0, 5.0, num= num_of_values)
    sys = signal.TransferFunction(sys_num, sys_den)
    t, y = signal.step(system=sys, T=time)
    return t, y