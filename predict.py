import costFunction
import sigmoid
import numpy as np
theta = np.array([-25.161, 0.206, 0.201])
X = costFunction.X
y = costFunction.y
def predict(theta, X):
    """
       Predict whether the label is 0 or 1 using learned logistic regression.
       Computes the predictions for X using a threshold at 0.5
       (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

       Parameters
       ----------
       theta : array_like
           Parameters for logistic regression. A vecotor of shape (n+1, ).

       X : array_like
           The data to use for computing predictions. The rows is the number
           of points to compute predictions, and columns is the number of
           features.

       Returns
       -------

       p : array_like
           Predictions and 0 or 1 for each row in X.

       Instructions
       ------------
       Complete the following code to make predictions using your learned
       logistic regression parameters.You should set p to a vector of 0's and 1's
       """
    m = X.shape[0]
    p = np.zeros(m)

    p = np.round(sigmoid.sigmoid(X.dot(theta.T)))
    print('probability:', sigmoid.sigmoid(X.dot(theta.T)))
    return p


def accuracy_checking(p, y):
    """Calculate the accuracy of the model (percentage)"""
    comparision = p == y
    count = 0
    for element in comparision:
        if element == True:
            count += 1
    probability = (count / len(y)) * 100
    return probability


print("The model with calculated theta have the accuracy:" ,accuracy_checking(predict(theta, X), y))