import numpy as np


def train_data_normalization(y_data):

    y_mean = np.mean(y_data)
    y_mean_normed = y_data - y_mean
    y_compression_factor = np.max(np.abs(y_mean_normed))
    y_normed = y_mean_normed / y_compression_factor

    return y_normed, y_mean, y_compression_factor


def train_data_inv_normalization(y_normed, y_mean, y_compression_factor):

    y_data = y_normed * y_compression_factor + y_mean

    return y_data

