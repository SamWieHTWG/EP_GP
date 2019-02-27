import numpy as np


def validate_random_search_init_inputs(func):

    def func_wrapper(self, lower_bound, upper_bound):

        if lower_bound[0] > upper_bound[0]:
            raise ValueError('lower bound must be smaller than upper bound'.format(func.__name__))

        if lower_bound[1] > upper_bound[1]:
            raise ValueError('lower bound must be smaller than upper bound'.format(func.__name__))

        if lower_bound[2] > upper_bound[2]:
            raise ValueError('lower bound must be smaller than upper bound'.format(func.__name__))

        res = func(self, lower_bound, upper_bound)
        return res

    return func_wrapper


def validate_optimize_parameters_inputs(func):

    def func_wrapper(self, num_iterations, optimization_data_x, optimization_data_y_unnormed):

        if optimization_data_x.ndim < 2:
            optimization_data_x = np.expand_dims(optimization_data_x, axis=0)
        if optimization_data_y_unnormed.ndim < 2:
            optimization_data_y_unnormed = np.expand_dims(optimization_data_y_unnormed, axis=0)

        if optimization_data_x.shape[0] < optimization_data_x.shape[1] \
                or optimization_data_y_unnormed.shape[0] < optimization_data_y_unnormed.shape[1]:

            raise ValueError('row vectors expected for y, x'.format(func.__name__))

        if optimization_data_x.shape[0] != optimization_data_y_unnormed.shape[0]:
            raise ValueError('different number of x-y values'.format(func.__name__))

        res = func(self, num_iterations, optimization_data_x, optimization_data_y_unnormed)

        return res

    return func_wrapper
