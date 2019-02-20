
import numpy as np
from functions.create_random_pt2 import *
from scipy import signal
from matplotlib import pyplot as p
import time


def get_rating(gp_p, gp_i, num_iterations, do_plot):

    sum_gp_better_performance = 0

    for index in range(num_iterations):
        iteration_gp_better_performance = score_gaussian_process(gp_p, gp_i, do_plot)
        if iteration_gp_better_performance:
            sum_gp_better_performance = sum_gp_better_performance + 1

    return sum_gp_better_performance/num_iterations


def score_gaussian_process(gp_p, gp_i, do_plot):

    error_gaussian_process = np.Inf
    while error_gaussian_process > 10000:

        num, den = create_random_pt2()

        x_test = np.zeros(4)
        x_test[0] = num[0]
        x_test[1] = num[1]
        x_test[2] = den[1]
        x_test[3] = den[2]
        p_test, p_test_covariance = gp_p.regression(np.matrix(x_test))
        i_test, i_test_covariance = gp_i.regression(np.matrix(x_test))

        num_of_values = 1000
        t_gp, y_gp = step_response(p_test, i_test, num, den, num_of_values)
        t_ref, y_ref = step_response(1, 5, num, den, num_of_values)

        target_value = np.ones(num_of_values)

        error_gaussian_process = np.sum(np.multiply((y_gp-target_value), (y_gp-target_value)))
        err_ref = np.sum(np.multiply((y_ref-target_value), (y_ref-target_value)))

    print(error_gaussian_process)
    print(err_ref)

    if do_plot:

        p.plot(t_gp, y_gp)
        p.plot(t_ref, y_ref)
        p.legend('Gaussian Process', 'P = 1, I = 10')
        p.show()
        time.sleep(2)

    if error_gaussian_process < err_ref:
        gp_better_performance = True
    else:
        gp_better_performance = False

    return gp_better_performance


def step_response(p_gain, i_gain, num, den, num_of_values):
    ol_num = np.convolve(num, np.squeeze([p_gain, i_gain]))
    ol_den = np.convolve(den, [1, 0])
    sys_num = ol_num
    sys_den = [ol_den[0], (ol_num[0] + ol_den[1]), (ol_num[1] + ol_den[2]), (ol_num[2] + ol_den[3])]

    time_points = np.linspace(0.0, 5.0, num= num_of_values)
    sys = signal.TransferFunction(sys_num, sys_den)
    t, y = signal.step(system=sys, T=time_points)
    return t, y