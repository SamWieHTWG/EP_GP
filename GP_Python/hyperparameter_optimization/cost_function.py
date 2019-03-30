from gaussian_process_module.squared_exponential_kernel import  squared_exponential_kernel
import numpy as np


def likelihood_cost_function(x, y, length, sig_f, l, sig_n):
    """!
    Computes the marginal likelihood of the given hyperparameters and input output data. This means
    the probability of the given parameters for the input-output data is determined. Maximizing this likelihood
    leads to the most likely parameters.

    @param x: numpy_matrix: input data
    @param y: numpy_matrix: output data
    @param length: int: length of data set
    @param sig_f: float: parameter defining the process noise
    @param l: float: parameter defining the length scale
    @param sig_n: float: parameter defining the output noise
    @return: probability of the given parameter for the input-output data
    """

    k_xx = squared_exponential_kernel(x, x, l, sig_f)
    k_y = k_xx + sig_n * sig_n * np.eye(length)
    inv_k_y = np.linalg.inv(k_y)

    if np.linalg.det(k_y) == 0:
        return -1 * np.Inf
    else:

        term1 = -1/2 * np.matmul(np.transpose(y), np.matmul(inv_k_y, y))
        term2 = -1/2 * np.log(np.linalg.det(k_y))
        term3 = -1*length/2 * np.log(2*np.pi)

        likelihood = term1 + term2 + term3
        print(likelihood)

        return likelihood

