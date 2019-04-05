import numpy as np

import time


size = 5000
X = np.random.rand(size, size)
print('random matrix created')
start = time.time()


Y = np.linalg.inv(X)

end = time.time()
print(end - start)
