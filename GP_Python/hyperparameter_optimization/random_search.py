import numpy as np
from hyperparameter_optimization.cost_function import likelihood_cost_function
from functions.data_normalization import *
from hyperparameter_optimization.check_random_search_inputs \
    import validate_random_search_init_inputs
from hyperparameter_optimization.check_random_search_inputs \
    import validate_optimize_parameters_inputs


class RandomSearch:

    @validate_random_search_init_inputs
    def __init__(self, lower_bound, upper_bound):
        """!
        creates a random search object, which determines the most likely hyperparameters regarding given input-output
        data by generating a huge number of random hyperparameter sets and returning the most likely parameters.
        @param lower_bound: list: defines the lower bound for each hyperparameter, at which samples are generated.
        @param upper_bound: list: defines the upper bound for each hyperparameter, at which samples are generated.
        """

        self.lb_sig_n = lower_bound[0]
        self.ub_sig_n = upper_bound[0]

        self.lb_l = lower_bound[1]
        self.ub_l = upper_bound[1]

        self.lb_sig_f = lower_bound[2]
        self.ub_sig_f = upper_bound[2]

        self.opt_parameters = np.zeros((1, 3))
        pass

    @validate_optimize_parameters_inputs
    def optimize_parameters(self, num_iterations, optimization_data_x, optimization_data_y_unnormed):
        """!
        This function generates a huge number of random parameter sets and determines the most likely parameters
        regarding the given input/output data by using the likelihood function.

        @param num_iterations: int: number of random parameter samples
        @param optimization_data_x: numpy_matrix: input data
        @param optimization_data_y_unnormed: numpy_matrix: unnormed output data
        @return: list: most likely parameters
        """

        optimization_data_y, _, _ = train_data_normalization(optimization_data_y_unnormed)

        iteration_cost_result = np.zeros((num_iterations, 1))
        iteration_used_parameters = np.zeros((num_iterations, 3))
        length = len(optimization_data_y)

        for iteration in range(0, num_iterations):

            sig_n_sample = np.random.uniform(self.lb_sig_n, self.ub_sig_n)
            sig_f_sample = np.random.uniform(self.lb_sig_f, self.ub_sig_f)
            l_sample = np.random.uniform(self.lb_l, self.ub_l)

            iteration_used_parameters[iteration, :] = [sig_n_sample, l_sample, sig_f_sample]

            cost = likelihood_cost_function(optimization_data_x, optimization_data_y, length,
                                            sig_f_sample, l_sample, sig_n_sample)

            print('\niteration-cost =\n')
            print(cost)

            iteration_cost_result[iteration, :] = cost

        opt_index = np.argmax(iteration_cost_result)
        opt_par = iteration_used_parameters[opt_index, :]
        self.opt_parameters = opt_par

        pass

    def get_optimal_parameters(self):

        return self.opt_parameters



