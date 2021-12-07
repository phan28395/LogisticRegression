import math
import numpy as np
def sigmoid(z):
    list_result_g = []
    if isinstance(z, int) or isinstance(z, float):
        g = 1 / (1 + math.exp(-z))
        return g
    else:
        for element in z:
            list_result_g.append(1 / (1 + math.exp(-element)))
        return np.array(list_result_g)
