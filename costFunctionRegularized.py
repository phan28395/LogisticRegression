import costFunction
import mapFeature
import data2
import numpy as np

X = data2.X
y = data2.y
X = mapFeature.mapFeature(X[:, 0], X[:, 1])
theta = np.zeros(X[0].shape)

def cost_function_regularized_calculate(X, y, theta):

    cost_function_sum = costFunction.cost_function_calculate(X, y, theta)
    lambda_cost = 1
    m = X.shape[0]
    sum_theta = 0
    for i in range(len(theta)):
        sum_theta += theta[i + 1]
    sum_theta_lambda = ((lambda_cost) / (2 * m)) * sum_theta
    return cost_function_sum + sum_theta_lambda

print(cost_function_regularized_calculate(X, y, theta))

