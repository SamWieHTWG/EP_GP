import numpy as np
from HypOpt_module.cost_function import likelihood_cost_function
from functions.data_normalization import *

class RandomSearch:

    def __init__(self, lower_bound, upper_bound):

        self.lb_sig_n = lower_bound[0]
        self.ub_sig_n = upper_bound[0]

        self.lb_l = lower_bound[1]
        self.ub_l = upper_bound[1]

        self.lb_sig_f = lower_bound[2]
        self.ub_sig_f = upper_bound[2]

        self.opt_parameters = np.zeros((1, 3))
        pass

    def optimize_parameters(self, num_iterations, optimization_data_X, optimization_data_Y_unnormed):

        optimization_data_Y, _, _ = train_data_normalization(optimization_data_Y_unnormed)

        iteration_cost_result = np.zeros((num_iterations, 1))
        iteration_used_parameters = np.zeros((num_iterations, 3))
        length = len(optimization_data_Y)

        for iteration in range(0, num_iterations):

            sig_n_sample = np.random.uniform(self.lb_sig_n, self.ub_sig_n)
            sig_f_sample = np.random.uniform(self.lb_sig_f, self.ub_sig_f)
            l_sample = np.random.uniform(self.lb_l, self.ub_l)

            iteration_used_parameters[iteration, :] = [sig_n_sample, l_sample, sig_f_sample]

            cost = likelihood_cost_function(optimization_data_X, optimization_data_Y, length, sig_f_sample, l_sample, sig_n_sample)
            iteration_cost_result[iteration, :] = cost
            #print(cost)

        #print(iteration_cost_result)
        opt_index = np.argmax(iteration_cost_result)
        opt_par = iteration_used_parameters[opt_index, :]
        #print(opt_index)
        #print(opt_par)
        self.opt_parameters = opt_par

        pass

    def get_optimal_parameters(self):

        return self.opt_parameters



#X = np.matrix('-1; 0; 1')
#Y = np.matrix('1; 0; -1')
#Search = RandomSearch([0, 0, 0],[1, 1, 1])
#Search.optimize_parameters(10000, X, Y)
#print(Search.get_optimal_parameters())