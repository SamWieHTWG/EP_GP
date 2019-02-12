from GP_module.SE_Kernel import  SE_Kernel
import numpy as np

def likelihood_cost_function(X, Y, length, sig_f, l, sig_n):

    k_XX = SE_Kernel(X, X, l, sig_f)
    k_Y = k_XX + sig_n * sig_n * np.eye(length)
    inv_k_Y = np.linalg.inv(k_Y)

    help_var = inv_k_Y * Y

    term1 = -1/2 * np.transpose(Y) * help_var
    term2 = -1/2 * np.log(np.linalg.det(k_Y))
    term3 = -1*length/2* np.log(2*np.pi)

    likelihood = term1 + term2 + term3

    return likelihood

# Test
X = np.matrix('-1; 0; 1')
Y = np.matrix('1; 0; -1')

print(likelihood_cost_function(X, Y, 3, 1, 1, 0.1))