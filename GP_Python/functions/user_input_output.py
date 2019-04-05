
from gaussian_process_module.gaussian_process import GaussianProcess
import numpy as np
from functions.score_gaussian_process import step_response
import functions.constants as constants
import warnings


def user_input_output(gp_p, gp_i):
    """
    function for user interface - used to achieve control parameters for a specific plant
    @param gp_p: GaussianProcess: gaussian process for p-gain
    @param gp_i: GaussianProcess: gaussian process for i-gain
    """
    print('\n\nplease enter plant parameters as continous '
          'transfer fcn: \n( b1 s + b0 )/(s²+ a1 s + a0)')

    b1_txt = input('\nb1=?')
    b0_txt = input('\nb0=?')
    a1_txt = input('\na1=?')
    a0_txt = input('\na0=?')

    d_cont = np.zeros(3)
    n_cont = np.zeros(2)

    # norm to
    n_cont[0] = float(b1_txt)
    n_cont[1] = float(b0_txt)

    d_cont[0] = 1
    d_cont[1] = float(a1_txt)
    d_cont[2] = float(a0_txt)

    dc_gain = n_cont[1] / d_cont[2]
    n_cont = n_cont / dc_gain

    x_test = np.zeros(4)
    x_test[0] = n_cont[0]
    x_test[1] = n_cont[1]
    x_test[2] = d_cont[1]
    x_test[3] = d_cont[2]

    p_test, __ = gp_p.regression(np.array(x_test))
    i_test, __ = gp_i.regression(np.array(x_test))

    t_gp, y_gp = step_response(p_test, i_test, n_cont, d_cont, constants.NUMBER_OF_TIME_POINTS)

    target_value = np.ones(constants.NUMBER_OF_TIME_POINTS)

    with warnings.catch_warnings():  # ignore overflow warnings for this part
        warnings.simplefilter("ignore")
        error_gaussian_process = np.sum(np.multiply((y_gp - target_value), (y_gp - target_value)))

    if error_gaussian_process > constants.ERROR_LIMIT_FOR_STAB_SYSTEM:
        print('sorry - no stablizing controller found')
    else:
        print('The supposed parameters are' + str(p_test/dc_gain) + ' for p-gain and ' +
              str(i_test) + ' for i-gain')

    pass
