import numpy as np
import scipy.io as io


def read_Train_Data():

    filename = '/home/samuel/Documents/EP_GP/GP_Python/train_data/data_254.mat'

    mat_data_dict = io.loadmat(filename)

    mat_data = np.matrix(mat_data_dict['data'])


    train_data = {}
    train_data['P'] = np.matrix(mat_data[:, 1])
    train_data['I'] = np.matrix(mat_data[:, 2])
    num1 = np.matrix(mat_data[:, 3])
    num2 = np.matrix(mat_data[:, 4])
    train_data['num'] = np.concatenate((num1, num2), axis=1)
    den1 = np.matrix(mat_data[:, 5])
    den2 = np.matrix(mat_data[:, 6])
    train_data['den'] = np.concatenate((den1, den2), axis=1)

    return train_data

# test:

#print(read_Train_Data())
