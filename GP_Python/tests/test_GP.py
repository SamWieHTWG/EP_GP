import unittest
import warnings
import numpy as np
from functions.create_random_PT2 import create_random_PT2
from functions.data_normalization import *
from functions.store_GP import store_GP, load_GP
from functions.read_Train_Data import read_Train_Data
from GP_module.GP import GP


class TestGP(unittest.TestCase):  # inherits from unittest.testcase

    def test_CreateRandomPT2(self):
        num, den = create_random_PT2()

        num_ref = np.zeros(2)
        den_ref = np.zeros(3)

        assert type(num) == type(num_ref)  # test return == ndarray
        assert type(den) == type(den_ref)  # test return == ndarray

        self.assertEqual(num.shape, num_ref.shape)  # test shape = expected
        self.assertEqual(den.shape, den_ref.shape)  # test shape = expected

        self.assertAlmostEqual((num[1]/den[2]), 1)  # test dc_gain is zero

    def test_DataNormalization(self):

        y = np.random.rand(100)

        y_normed, y_mean, y_stretch_fact = train_data_normalization(y)
        y_renormed = train_data_inv_normalization(y_normed, y_mean, y_stretch_fact)

        self.assertAlmostEqual(np.max(np.abs(y-y_renormed)), 0, places=4)  # test norm error after renorm

        self.assertAlmostEqual(np.max(np.abs(y_normed)), 1.0, places=4)  # test max abs == 1 ( stretching correct )

    def test_read_Train_Data(self):

        # with warnings.catch_warnings():  # only ignore warnings for this part
        #     warnings.simplefilter("ignore")  # unittest produces warnings when using np.matrix -> ignore

        train_data = read_Train_Data()

        np_array_ref = np.zeros(100)

        X_train = np.concatenate((train_data['num'], train_data['den']), axis=1)
        Y_train_P = train_data['P']
        Y_train_I = train_data['I']

        assert type(X_train) == type(np_array_ref)
        assert type(Y_train_P) == type(np_array_ref) and type(Y_train_I) == type(np_array_ref)

    def test_Store_Load_GP(self):

        # with warnings.catch_warnings():     # only ignore warnings for this part
        #     warnings.simplefilter("ignore")  # unittest produces warnings when using np.matrix -> ignore

        test_GP = GP(np.array([-1, 0, 1]), np.array([1, 0, -1]), 1, 1, 1)

        store_GP(test_GP, 'unit_test')
        loaded_GP = load_GP('unit_test')

        self.assertEqual(test_GP, loaded_GP)

    def test_GP(self):

        # test 1D regression
        # with warnings.catch_warnings():     # only ignore warnings for this part
        #    warnings.simplefilter("ignore")  # unittest produces warnings when using np.matrix -> ignore

        test_GP = GP(np.array([[-1.0], [0.0], [1.0]]), np.array([[1], [0], [-1]]), 0.001, 1.0, 1.0)
        x_test = np.array([1])

        test_result, __ = (test_GP.regression(x_test)) # 1 was also training point -> -1 expected

        self.assertAlmostEqual(float(test_result), -1, 3)

        # test 2D regression
        # with warnings.catch_warnings():     # only ignore warnings for this part
        #     warnings.simplefilter("ignore")  # unittest produces warnings when using np.matrix -> ignore

        test_GP = GP(np.array([[-1, 2], [0, 1], [1, 3]]), np.array([[1], [0], [-1]]), 0.001, 1, 1)
        x_test = np.array([1, 3])

        test_result, __ = (test_GP.regression(x_test)) # 1 was also training point -> -1 expected

        self.assertAlmostEqual(float(test_result), -1, 3)


if __name__ == '__main__':  # if this is main file, run unit test
    unittest.main()
