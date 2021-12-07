import pandas as pd
import plottingData
import mapFeature

data_chip = pd.read_csv('ex2data2.txt')
X = data_chip.iloc[:, 0:2].values
y = data_chip.iloc[:, 2].values

plottingData.plotData(X, y)


