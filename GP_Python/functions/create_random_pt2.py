import numpy as np
from functions import constants


def create_random_pt2():

    if np.random.rand() > constants.PROB_FOR_OSCILLATING_SYSTEM:

        real = (np.random.rand()-0.5) * 2 * constants.MAX_REAL_PART
        imag = (np.random.rand()) * constants.MAX_IMAGINARY_PART

        d_cont = np.zeros(3)
        d_cont[0] = 1
        d_cont[1] = -2*real
        d_cont[2] = real*real + imag*imag

        n_cont = np.zeros(2)
        n_cont[0] = np.random.rand()
        n_cont[1] = np.random.rand()

    else:

        pole1 = (np.random.rand() - 0.5) * 2 * constants.MAX_REAL_PART
        pole2 = (np.random.rand() - 0.5) * 2 * constants.MAX_REAL_PART

        d_cont = np.zeros(3)
        d_cont[0] = 1
        d_cont[1] = - pole1 - pole2
        d_cont[2] = -pole1 * -pole2

        n_cont = np.zeros(2)
        n_cont[0] = np.random.rand()
        n_cont[1] = np.random.rand()

    dc_gain = n_cont[1] / d_cont[2]
    n_cont = n_cont / dc_gain

    return n_cont, d_cont


if __name__ == '__main__':
    print('test-create_random_pt2')
    print(create_random_pt2())