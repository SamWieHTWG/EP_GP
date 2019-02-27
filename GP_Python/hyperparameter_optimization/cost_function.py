from gaussian_process_module.squared_exponential_kernel import  squared_exponential_kernel
import numpy as np


def likelihood_cost_function(x, y, length, sig_f, l, sig_n):

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

