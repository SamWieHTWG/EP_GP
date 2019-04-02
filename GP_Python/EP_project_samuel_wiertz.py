
from functions.read_train_data import read_train_data
from hyperparameter_optimization.random_search import RandomSearch
from gaussian_process_module.gaussian_process import GaussianProcess
from functions.store_gaussian_process import *
from functions.score_gaussian_process import *
from functions.constants import *

# author: Samuel Wiertz, wiertzsamuel@gmail.com
# date: 11.01.2019


# read train data
train_data = read_train_data('data_254')
x_train = np.concatenate((train_data['num'], train_data['den']), axis=1)
y_train_p_gain = train_data['P']
y_train_i_gain = train_data['I']


### create Random search Objects
# Random Search for P-Gain gaussian process
P_Parameter_Search = RandomSearch(lower_bound=HYPERPARAMETER_LOWER_BOUND,
                                  upper_bound=HYPERPARAMETER_UPPER_BOUND)
# Random Search for I-Gain gaussian process
I_Parameter_Search = RandomSearch(lower_bound=HYPERPARAMETER_LOWER_BOUND,
                                  upper_bound=HYPERPARAMETER_UPPER_BOUND)

# do Hyper.-Opt. : true: optimize parameters to loaded training data, false: use previous parameters
if DO_HYERPARAMETER_OPTIMIZATION:
    P_Parameter_Search.optimize_parameters(num_iterations=HYPERPARAMETER_OPTIMIZATION_ITERATIONS,
                                           optimization_data_x=x_train,
                                           optimization_data_y_unnormed= y_train_p_gain)
    I_Parameter_Search.optimize_parameters(num_iterations=HYPERPARAMETER_OPTIMIZATION_ITERATIONS,
                                           optimization_data_x=x_train,
                                           optimization_data_y_unnormed=y_train_i_gain)

    p_gain_optimal_parameters = P_Parameter_Search.get_optimal_parameters()
    i_gain_optimal_parameters = I_Parameter_Search.get_optimal_parameters()

    print(p_gain_optimal_parameters)
    print(i_gain_optimal_parameters)
else:
    p_gain_optimal_parameters = BEST_P_GAIN_HYPERPARAMETERS
    i_gain_optimal_parameters = BEST_I_GAIN_HYPERPARAMETERS


# create Gaussian Processes for I-Part and P-Part
GP_P = GaussianProcess(x_train, y_train_p_gain, p_gain_optimal_parameters[0],
                       p_gain_optimal_parameters[1], p_gain_optimal_parameters[2])
GP_I = GaussianProcess(x_train, y_train_i_gain, i_gain_optimal_parameters[0],
                       i_gain_optimal_parameters[1], i_gain_optimal_parameters[2])

# store created processes
#store_gaussian_process(gaussian_process=GP_P, filename='3640_P')
#store_gaussian_process(gaussian_process=GP_I, filename='3640_I')

# load processes trained by highest amount of training data for rating
GP_P2 = load_gaussian_process('2693_P')
GP_I2 = load_gaussian_process('2693_I')

rating = get_rating(GP_P2, GP_I2, 1000, 1)
print('\nabsolute rating = ')
print(rating)

