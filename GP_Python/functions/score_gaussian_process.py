import warnings
import numpy as np
from functions.create_random_pt2 import *
from scipy import signal
from matplotlib import pyplot as p
import time
import functions.constants


def get_rating(gp_p, gp_i, num_iterations, do_plot):
    """!
    creates rating of the machine learning algorithm by comparing step responses of a pi-controller with fixed
    parameters against a pi-controller with parameter estimated by the algorithm

    @param gp_p: float: p-gain for the controller estimated by gaussian process
    @param gp_i: float: i-gain for the controller estimated by gaussian process
    @param num_iterations: int
    @param do_plot: boolean: true - plot is created by using scipy
    @return: float: score of rating
    """
    sum_gp_better_performance = 0

    for index in range(num_iterations):
        iteration_gp_better_performance = score_gaussian_process(gp_p, gp_i, do_plot)
        if iteration_gp_better_performance:
            sum_gp_better_performance = sum_gp_better_performance + 1

    return sum_gp_better_performance/num_iterations


def score_gaussian_process(gp_p, gp_i, do_plot):
    """!
    this function is used for the rating of the alogrithm. Therfore, a random system is generated, controller parameter
    are estimated by the gaussian proccess and compared to a controller with fixed controller parameter by simulating
    the step response

    @param gp_p: gaussian process object for the p-gain
    @param gp_i: gaussian process object for the i-gain
    @param do_plot: boolean: true - plot is created by using scipy
    @return: boolean: true, if gp gains a better performance
    """
    error_gaussian_process = np.Inf
    while error_gaussian_process > constants.ERROR_LIMIT_FOR_STAB_SYSTEM:

        num, den = create_random_pt2()

        x_test = np.zeros(4)
        x_test[0] = num[0]
        x_test[1] = num[1]
        x_test[2] = den[1]
        x_test[3] = den[2]
        p_test, p_test_covariance = gp_p.regression(np.array(x_test))
        i_test, i_test_covariance = gp_i.regression(np.array(x_test))

        num_of_values = constants.NUMBER_OF_TIME_POINTS
        t_gp, y_gp = step_response(p_test, i_test, num, den, num_of_values)
        t_ref, y_ref = step_response(1, 5, num, den, num_of_values)

        target_value = np.ones(num_of_values)

        with warnings.catch_warnings():  # ignore overflow warnings for this part

            warnings.simplefilter("ignore")
            error_gaussian_process = np.sum(np.multiply((y_gp-target_value), (y_gp-target_value)))
            err_ref = np.sum(np.multiply((y_ref-target_value), (y_ref-target_value)))

    print('\nerror of GP-est:', error_gaussian_process)
    print('error of fixed parameters:', err_ref)

    if do_plot:


        y_step = np.append(np.zeros(1), np.ones(constants.NUMBER_OF_TIME_POINTS-1))
        p.plot(t_gp, y_gp, label="Parameter estimated by Gaussian Process")
        p.plot(t_ref, y_ref, label="P-Gain = 1, I-Gain = 10")
        p.plot(t_ref, y_step, label="Reference")
        # p.legend('Gaussian Process', 'P = 1, I = 10')
        p.legend()
        p.show()
        time.sleep(5)

    if error_gaussian_process < err_ref:
        gp_better_performance = True
    else:
        gp_better_performance = False

    return gp_better_performance


def step_response(p_gain, i_gain, num, den, num_of_values):
    """!
    Simulates step response by using scipy.

    """
    ol_num = np.convolve(num, np.squeeze([p_gain, i_gain]))
    ol_den = np.convolve(den, [1, 0])
    sys_num = ol_num
    sys_den = [ol_den[0], (ol_num[0] + ol_den[1]), (ol_num[1] + ol_den[2]), (ol_num[2] + ol_den[3])]

    time_points = np.linspace(constants.STEP_SIMULATION_START_TIME, constants.STEP_SIMULATION_END_TIME,
                              num=num_of_values)

    sys = signal.TransferFunction(sys_num, sys_den)
    time_points, output = signal.step(system=sys, T=time_points)

    return time_points, output