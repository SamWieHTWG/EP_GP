import numpy as np
import scipy.io as io


def read_train_data(filename):
    """!
    Reads Data as Matfile

    @param filename str: filename of mat file without .mat extension


    @return np_array: train data as matrix, each line is one train data set
    """

    #filename = '/home/samuel/Documents/EP_GP/GP_Python/train_data/data_2693.mat'
    path = '/home/samuel/Documents/EP_GP/GP_Python/train_data/'

    file = path + filename + '.mat'

    mat_data_dict = io.loadmat(file)

    mat_data = np.array(mat_data_dict['data'])


    train_data = {}
    train_p = mat_data[:, 0]
    train_i = mat_data[:, 1]
    train_data['P'] = np.reshape(train_p, (train_p.size, 1))
    train_data['I'] = np.reshape(train_i, (train_i.size, 1))
    num1 = mat_data[:, 2]
    num2 = mat_data[:, 3]
    num1 = np.reshape(num1, (num1.size, 1))
    num2 = np.reshape(num1, (num2.size, 1))

    train_data['num'] = np.concatenate((num1, num2), axis=1)

    den1 = mat_data[:, 5]
    den2 = mat_data[:, 6]
    den1 = np.reshape(den1,(den1.size, 1))
    den2 = np.reshape(den2, (den2.size, 1))
    train_data['den'] = np.concatenate((den1, den2), axis=1)

    return train_data


# test:
if __name__ == '__main__':  # if this is main file, run fcn
    read_train_data()