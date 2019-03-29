# from gaussian_process_module.GaussianProcess import GaussianProcess
import numpy as np


def validate_gaussian_process_initialization_inputs(func):

    def func_wrapper(self, x_train, y_train, sig_n, l, sig_f):
        """Test Documentation - Doxygen
          """

        if sig_n < 0 or sig_f < 0 or l < 0:
            raise ValueError('all Hyperparameters have to be positive'.format(func.__name__))

        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise TypeError('x_train, y_train have to be numpy arrays'.format(func.__name__))

        # Shape testing
        if x_train.ndim < 2:
            x_train = np.expand_dims(x_train, axis=0)
        if y_train.ndim < 2:
            y_train = np.expand_dims(y_train, axis=0)
        if x_train.shape[0] < x_train.shape[1] or y_train.shape[0] < y_train.shape[1]:
            raise ValueError('row vectors expected for y_train, x_train'.format(func.__name__))
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError('different number of x-y values'.format(func.__name__))

        # exectue function
        res = func(self, x_train, y_train, sig_n, l, sig_f)
        return res
    return func_wrapper


def validate_gaussian_process_regression_inputs(func):

    def func_wrapper(self, x):

        if not isinstance(x, np.ndarray):
            raise TypeError('x has to be a numpy ndarray'.format(func.__name__))

        if x.ndim < 2:
            x = np.expand_dims(x, axis=0)
        if x.shape[1] < x.shape[0]:
            raise ValueError('column vector expected for x'.format(func.__name__))
        if x.shape[0] > 1:
            raise ValueError('x has to be a vector'.format(func.__name__))
        if x.shape[1] != self.X_train.shape[1]:
            raise ValueError('wrong input dimension'.format(func.__name__))

        res = func(self, x)
        return res

    return func_wrapper



