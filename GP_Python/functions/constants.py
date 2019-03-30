### max imag part of the random system-pole. Part of oscillation is defined by this.
MAX_IMAGINARY_PART = 5

### max real part of random system-pole. Defines, how fast system is corresponding.
MAX_REAL_PART = 10

### probability, that the randomly generated system is an oscillating system
PROB_FOR_OSCILLATING_SYSTEM = 0.8


### limit, for which a system is marked as unstable
ERROR_LIMIT_FOR_STAB_SYSTEM = 10000


### number of time points used for simulation during scoring
NUMBER_OF_TIME_POINTS = 1000


STEP_SIMULATION_START_TIME = 0
STEP_SIMULATION_END_TIME = 5

### true-> hyperparameter optimization using Grid Search is processed. no -> previous values are used
DO_HYERPARAMETER_OPTIMIZATION = 0

HYPERPARAMETER_OPTIMIZATION_ITERATIONS = 100
HYPERPARAMETER_LOWER_BOUND = [0, 0, 0]
HYPERPARAMETER_UPPER_BOUND = [0.1, 10, 10]

### previously calculated hyperparameters for p-gain regression
BEST_P_GAIN_HYPERPARAMETERS = [0.067, 1.23, 0.528]

### previously calculated hyperparameters for i-gain regression
BEST_I_GAIN_HYPERPARAMETERS = [0.078, 1.377, 0.469]

### number of iterations used for scoring
NUMBER_OF_RATING_ITERATIONS = 10
