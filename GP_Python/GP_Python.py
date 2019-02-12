
from functions.read_Train_Data import read_Train_Data
from HypOpt_module.RandomSearch import RandomSearch
from GP_module.GP import GP
import numpy as np
from scipy import signal
from matplotlib import pyplot as p


# author: Samuel Wiertz, wiertzsamuel@gmail.com
# date: 11.01.2019


## read Train Data
train_data = read_Train_Data()
X_train = np.concatenate((train_data['num'], train_data['den']), axis=1)
Y_train_P = train_data['P']
Y_train_I = train_data['I']

# Data Normalization
Y_train_P_mean = np.mean(Y_train_P)
Y_train_I_mean = np.mean(Y_train_I)
Y_train_P_mean_normed = Y_train_P - Y_train_P_mean
Y_train_I_mean_normed = Y_train_I - Y_train_I_mean
Y_train_P_compression_factor = np.max(np.abs(Y_train_P_mean_normed))
Y_train_I_compression_factor = np.max(np.abs(Y_train_I_mean_normed))
Y_train_P_normed = Y_train_P_mean_normed / Y_train_P_compression_factor
Y_train_I_normed = Y_train_I_mean_normed / Y_train_I_compression_factor


## get Parameters for Train Data
P_parameter_lower_bound = [0, 0, 0]
I_parameter_lower_bound = [0, 0, 0]
P_parameter_upper_bound = [0.1, 10, 10]
I_parameter_upper_bound = [0.1, 10, 10]

P_Parameter_Search = RandomSearch(P_parameter_lower_bound, P_parameter_upper_bound)
I_Parameter_Search = RandomSearch(I_parameter_lower_bound, I_parameter_upper_bound)

if 0:
    P_Parameter_Search.optimize_parameters(50, X_train, Y_train_P_normed)
    I_Parameter_Search.optimize_parameters(50, X_train, Y_train_I_normed)

    P_optimal_parameters = P_Parameter_Search.get_optimal_parameters()
    I_optimal_parameters = I_Parameter_Search.get_optimal_parameters()

    print(P_optimal_parameters)
    print(I_optimal_parameters)
else:
    P_optimal_parameters = [0.077, 9.43, 7.5]
    I_optimal_parameters = [0.098, 4, 0.71]

## create Gaussian Processes for I-Part and P-Part
GP_P = GP(X_train, Y_train_P_normed, P_optimal_parameters[0], P_optimal_parameters[1], P_optimal_parameters[2])
GP_I = GP(X_train, Y_train_I_normed, I_optimal_parameters[0], I_optimal_parameters[1], I_optimal_parameters[2])

## GP_Regression
X_test = np.matrix([7.8939, 12.8826, -7.4373, 12.8826])
P_test, P_test_Covariance = GP_P.regression(X_test)
I_test, I_test_Covariance = GP_I.regression(X_test)
# renorm:
P = P_test * Y_train_P_compression_factor + Y_train_P_mean
I = I_test * Y_train_I_compression_factor + Y_train_I_mean

X_test = np.squeeze(X_test)
num = X_test[0,0:1]
den = X_test[2:3]
help =(np.concatenate((P, I), axis=1))
ol_num = np.convolve(num.tolist(), np.squeeze(help.tolist())).tolist()
ol_den = np.convolve(den.tolist(), [1, 0]).tolist()

sys_num = ol_num
sys_den = [ol_den[0], (ol_num[0]+ ol_den[1]), (ol_num[1]+ ol_den[2]), (ol_num[2]+ ol_den[3]) ]

sys = signal.TransferFunction(sys_num, sys_den)
t, y = signal.step(sys)
p.plot(t,y)
p.show()
