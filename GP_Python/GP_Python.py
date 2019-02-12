
from functions.read_Train_Data import read_Train_Data
from HypOpt_module.RandomSearch import RandomSearch
import numpy as np


# author: Samuel Wiertz, wiertzsamuel@gmail.com
# date: 11.01.2019


## read Train Data
train_data = read_Train_Data()
X_train = np.concatenate((train_data['num'], train_data['den']), axis=1)
Y_train_P = train_data['P']
Y_train_I = train_data['I']

## get Parameters for Train Data
P_parameter_lower_bound = [0, 0, 0]
I_parameter_lower_bound = [0, 0, 0]
P_parameter_upper_bound = [0.2, 10, 10]
I_parameter_upper_bound = [0.2, 10, 10]

P_Parameter_Search = RandomSearch(P_parameter_lower_bound, P_parameter_upper_bound)
I_Parameter_Search = RandomSearch(I_parameter_lower_bound, I_parameter_upper_bound)

P_Parameter_Search.optimize_parameters(100, X_train, Y_train_P)
I_Parameter_Search.optimize_parameters(100, X_train, Y_train_I)

P_optimal_parameters = P_Parameter_Search.get_optimal_parameters()
I_optimal_parameters = I_Parameter_Search.get_optimal_parameters()

print(P_optimal_parameters)
print(I_optimal_parameters)

## create Gaussian Processes for I-Part and P-Part





