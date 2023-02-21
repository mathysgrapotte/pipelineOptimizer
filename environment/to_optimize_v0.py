# In this sandbox, we are building a pipeline to be optimized. 

import numpy as np 

granularity = 50

parameters_range = {
    "p1":np.linspace(start=0, stop=1, num=granularity, dtype=float),
    "p2":np.linspace(start=0, stop=100, num=granularity, dtype=int),
    "p3":np.linspace(start=0, stop=50, num=granularity, dtype=int)
}

# Main loop 
def main_loop(parameters, start):
    computed = (start ** parameters["p1"] + parameters["p2"]) * 2**parameters["p3"]
    # hard coded correct parameters
    correct = (start**0.47 + 38) * 2**12
    return computed, correct
