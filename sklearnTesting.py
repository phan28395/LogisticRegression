from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
df = pd.read_csv('ex2data1.txt')
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
pos = y == 1
neg = y == 0
plt.plot(X[pos, 0], X[pos, 1], 'k*')
plt.plot(X[neg, 0], X[neg, 1], 'ro')
plt.show()
log_reg = LogisticRegression()
log_reg.fit(X, y)
print(log_reg.predict(np.array([[40, 30]])))
print(log_reg.coef_)

def prob(X, theta):
    z = np.dot(X, theta.transpose())
    return 1 / (1 + math.exp(-z))

print(prob(np.array([[40, 30]]), log_reg.coef_))