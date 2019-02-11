import numpy as np

class RandomSearch:

    def __init__(self, lower_bound, upper_bound):

        self.lb_sig_f = lower_bound[0]
        self.ub_sig_f = upper_bound[0]

        self.lb_l = lower_bound[1]
        self.ub_l = upper_bound[1]

        self.lb_sig_n = lower_bound[2]
        self.ub_sig_n = upper_bound[2]

        pass

    def optimize_parameters(self, num_iterations, optimization_data_X, optimization_data_Y):


        pass




