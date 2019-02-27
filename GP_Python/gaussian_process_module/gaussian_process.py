import numpy as np
from gaussian_process_module.squared_exponential_kernel import squared_exponential_kernel
from functions.data_normalization import *
from gaussian_process_module.check_inputs import validate_gaussian_process_initialization_inputs
from gaussian_process_module.check_inputs import validate_gaussian_process_regression_inputs


class GaussianProcess:

    @validate_gaussian_process_initialization_inputs
    def __init__(self, x_train, y_train, parameter_sig_n, parameter_l, parameter_sig_f):
        # tbd: Konstruktor anpassen an arg* aus Folien, z.B. ob Parameter gegeben

        self.X_train = x_train
        self.y_train, self.y_mean, self.y_compression_fact = train_data_normalization(y_train)
        self.parameter_sig_n = parameter_sig_n
        self.parameter_sig_f = parameter_sig_f
        self.parameter_l = parameter_l

        input_shape = x_train.shape
        self.order = input_shape[0]
        self.k_XX = np.zeros([self.order, self.order])
        self.inv_Cov = np.zeros([self.order, self.order])

        self.train()

    @validate_gaussian_process_regression_inputs
    def regression(self, x):

        k_xX = squared_exponential_kernel(x, self.X_train, self.parameter_l, self.parameter_sig_f)
        k_xx = squared_exponential_kernel(x, x, self.parameter_l, self.parameter_sig_f)

        y_est_normed = np.matmul(k_xX, np.matmul(self.inv_Cov, self.y_train))
        y_estimated = train_data_inv_normalization(y_est_normed, self.y_mean, self.y_compression_fact)

        estimation_deviation = k_xx - np.matmul(k_xX, np.matmul(self.inv_Cov, np.ndarray.transpose(k_xX)))

        return y_estimated, estimation_deviation

    def train(self):
        self.k_XX = squared_exponential_kernel(self.X_train, self.X_train, self.parameter_l, self.parameter_sig_f)
        self.inv_Cov = np.linalg.inv(self.k_XX + self.parameter_sig_n * self.parameter_sig_n * np.eye(self.order))

        pass

    def __eq__(self, other):
        self_str = str(self.__dict__)
        other_str = str(other.__dict__)

        # return self.__dict__ == other.__dict__
        return self_str == other_str

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':  # if this is main file, run fcn
    X = np.array([[1, 1], [0, 0], [-1, -1]])
    print(X)
    y = np.array([[-1], [0], [1]])
    print(y)
    gp = GaussianProcess(x_train=X, y_train=y, parameter_sig_n=0.1, parameter_l=1, parameter_sig_f=1)
    y, C = gp.regression(np.array([[0.5, 0.5]]))
    print(y)
    print(C)

