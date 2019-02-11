import numpy as np
from GP_module.SE_Kernel import SE_Kernel

class GP:

    def __init__(self, X_train, y_train, sig_n, l, sig_f): # tbd: Konstruktor anpassen an arg* aus Folien, z.B. ob Parameter gegeben
        #tbd: Ausnahmen werfen bei Fehler
        # numpy lagert Funktionen aus an in C integrierte Funktionen!! Effizient

        # Konstruktor
        #
        self.X_train = X_train
        self.y_train = y_train
        self.sig_n = sig_n
        self.sig_f = sig_f
        self.l = l

        input_shape = X_train.shape
        self.order = input_shape[0]
        self.k_XX = np.zeros([self.order, self.order])
        self.inv_Cov = np.zeros([self.order, self.order])

        self.train()

    # tbd: GP-Reg
    def regression(self, x):
        k_xX = SE_Kernel(x,self.X_train, self.l, self.sig_f)
        k_xx = SE_Kernel(x, x, self.l, self.sig_f)
        #print(k_xX)
        #print(self.k_XX)
        #print(self.inv_Cov)
        y_help = self.inv_Cov * self.y_train
        y_estimated = k_xX * y_help

        estimation_deviation = k_xx - k_xX * self.inv_Cov * np.transpose(k_xX)

        return y_estimated, estimation_deviation

    # tbd: Moduldatei als Python Skript ausführbar machen zum Testen!
    # Unterscheidung im Modul: if _name__ = __main__ abfragbar, ob über skript ausgeführt

    # tbd: GP-Train
    def train(self):
        self.k_XX = SE_Kernel(self.X_train, self.X_train, self.l, self.sig_f)
        self.inv_Cov = np.linalg.inv(self.k_XX + self.sig_n * self.sig_n * np.eye(self.order))

        pass






X = np.matrix('1 1; 0 0; -1 -1')
y = np.matrix('-1; 0 ; 1')
gp = GP(X_train=X, y_train=y, sig_n=0.1, l=1, sig_f=1)
y,C = gp.regression(np.matrix('0.5 0.5'))
print(y)