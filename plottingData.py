import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):

    pos = y == 1
    neg = y == 0
    print(X[pos, 0])
    plt.plot(X[pos, 0], X[pos, 1], 'k*')
    plt.plot(X[neg, 0], X[neg, 1], 'ro')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()

column_name = ['X1', 'X2', 'y']
df = pd.read_csv('ex2data1.txt', names=column_name, header=None)
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2]. values
plotData(X, y)
