import numpy as np
import scipy.io as io



def read_Train_Data():

    filename = '/home/samuel/Documents/EP_GP/GP_Python/train_data/data.mat'

    mat_data_dict = io.loadmat(filename)

    mat_data = np.matrix(mat_data_dict['data'])


    train_data = {}
    train_data['P'] = mat_data[:, 1];
    train_data['I'] = mat_data[:, 2];
    train_data['num'] = mat_data[:, 3:4];

    return train_data

# test:

print(read_Train_Data())
