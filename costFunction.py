import math
import numpy as np
import pandas as pd
import sigmoid

# column_name = ['X1', 'X2', 'y']
# df = pd.read_csv('ex2data1.txt', names=column_name, header=None)
# X = df.iloc[:, 0:2].values
# y = df.iloc[:, 2]. values
# m, n = X.shape
# intercept_array = np.ones(m,)
# X = np.insert(X, 0, intercept_array, axis=1)
#
# initial_theta = np.zeros((n + 1, 1))
# initial_theta = np.array([-24, 0.2, 0.2])

def cost_function_calculate(theta, X, y):
    sum_cost = 0
    for i in range(len(y)):
        #x_vector = np.array([X[i][0], X[i][1], X[i][2]])
        z = np.dot(theta.transpose(), X[i])
        first_part = (-y[i] * np.log(sigmoid.sigmoid(z)))
        second_part = ((1 - y[i]) * np.log(1 - sigmoid.sigmoid(z)))
        sum_cost += (first_part - second_part)

    cost_function = (1/len(y)) * sum_cost
    return cost_function

def gradient_descent_calculate(theta, X, y):
    sum_gradient = 0
    list_theta = []
    for i in range(n + 1):
        for j in range(len(y)):
            x_vector = np.array([X[j][0], X[j][1], X[j][2]])
            z = np.dot(theta.transpose(), X[j])
            sum_gradient += (sigmoid.sigmoid(z) - y[j]) * X[j][i]
        list_theta.append((1/len(y) * sum_gradient))
        sum_gradient = 0
    return np.array(list_theta)

if __name__ == "__main__":
    print(cost_function_calculate(initial_theta, X, y))
    print(gradient_descent_calculate(initial_theta, X, y))
