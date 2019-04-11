import numpy as np


def train_data_normalization(y_data):
    """!
    remove mean, span data between -1 and 1
    @param y_data: numpy_array: output data to be normed
    @return: numpy_array: normed output data
    """
    y_mean = np.mean(y_data)
    y_mean_normed = y_data - y_mean
    y_compression_factor = np.max(np.abs(y_mean_normed))
    y_normed = y_mean_normed / y_compression_factor

    return y_normed, y_mean, y_compression_factor


def train_data_inv_normalization(y_normed, y_mean, y_compression_factor):
    """!
    inverse of data normalization
    @param y_normed: numpy_array: normed output data
    @param y_mean: float: mean of previous normalization
    @param y_compression_factor: float: compression factor of previous normal
    @return: numpy_array: unnormed data
    """
    y_data = y_normed * y_compression_factor + y_mean

    return y_data

