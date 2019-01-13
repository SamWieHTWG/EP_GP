import numpy as np
from GP_module.SE_Kernel import SE_Kernel

class GP:

    sig_f = 1
    sig_n = 0.1
    ell = 1

    x1 = 1;
    x2 = 2;

    # tbd: GP-Reg
    def regression(self,x):
        pass

    # tbd: GP-Train
    def train(selfself, X, Y):
        pass






x1 = np.matrix('1')
x2 = np.matrix('2')
# test
SE_Kernel(x1,x2,1,1)