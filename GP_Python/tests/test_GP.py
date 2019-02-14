import unittest
import numpy as np
from functions.create_random_PT2 import create_random_PT2
from functions.data_normalization import *
from functions.store_GP import store_GP, load_GP
from GP_module.GP import  GP

class TestGP(unittest.TestCase): # inherits from unittest.testcase

    def test_CreateRandomPT2(self):
        num, den = create_random_PT2()

        num_ref = np.zeros(2)
        den_ref = np.zeros(3)

        assert type(num) == type(num_ref) # test return == ndarray
        assert type(den) == type(den_ref) # test return == ndarray

        assert num.shape == num_ref.shape # test shape = expected
        assert den.shape == den_ref.shape # test shape = expected

        self.assertAlmostEqual((num[1]/den[2]), 1) # test dc_gain is zero

    def test_DataNormalization(self):

        y = np.random.rand(100)

        y_normed, y_mean, y_stretch_fact = train_data_normalization(y)
        y_renormed = train_data_inv_normalization(y_normed, y_mean, y_stretch_fact)

        self.assertAlmostEqual(np.max(np.abs(y-y_renormed)), 0, places=4)  # test norm error after renorm

        self.assertAlmostEqual(np.max(np.abs(y_normed)), 1.0, places=4)  # test max abs == 1 ( stretching correct )

    def test_Store_Load_GP(self):

        test_GP = GP(np.matrix([-1, 0, 1]), np.matrix([1, 0, -1]), 1, 1, 1)
        store_GP(test_GP, 'unit_test')
        loaded_GP = load_GP('unit_test')

        print(test_GP == loaded_GP)
        #self.assertEqual(test_GP, loaded_GP)



if __name__ == '__main__': #if this is main file, run unit test
    unittest.main()