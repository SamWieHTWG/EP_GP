
from functions.read_Train_Data import read_Train_Data
from HypOpt_module.RandomSearch import RandomSearch
from GP_module.GP import GP
from functions.test_GP import *
from functions.store_GP import *
from functions.score_GP import *

# author: Samuel Wiertz, wiertzsamuel@gmail.com
# date: 11.01.2019


## read Train Data
train_data = read_Train_Data()
X_train = np.concatenate((train_data['num'], train_data['den']), axis=1)
Y_train_P = train_data['P']
Y_train_I = train_data['I']


## get Parameters for Train Data
P_Parameter_Search = RandomSearch([0, 0, 0], [0.1, 10, 10])
I_Parameter_Search = RandomSearch([0, 0, 0], [0.1, 10, 10])

if 0:
    P_Parameter_Search.optimize_parameters(100, X_train, Y_train_P)
    I_Parameter_Search.optimize_parameters(100, X_train, Y_train_I)

    P_optimal_parameters = P_Parameter_Search.get_optimal_parameters()
    I_optimal_parameters = I_Parameter_Search.get_optimal_parameters()

    print(P_optimal_parameters)
    print(I_optimal_parameters)
else:
    P_optimal_parameters = [0.067, 1.23, 0.528]
    I_optimal_parameters = [0.078, 1.377, 0.469]

## create Gaussian Processes for I-Part and P-Part
GP_P = GP(X_train, Y_train_P, P_optimal_parameters[0], P_optimal_parameters[1], P_optimal_parameters[2])
GP_I = GP(X_train, Y_train_I, I_optimal_parameters[0], I_optimal_parameters[1], I_optimal_parameters[2])
store_GP(GP_P, 'test')
store_GP(GP_I, 'test')
GP_P2 = load_GP('2693')
GP_I2 = load_GP('2693')

rating = get_rating(GP_P2, GP_I, 30, 1)
print(rating)

