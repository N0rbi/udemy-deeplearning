import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the som

from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=len(X[0]), random_seed=22222)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualize
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, colors[y[i]]+markers[y[i]], markersize=10, linewidth=2, markerfacecolor='None')
#show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8, 2)], mappings[(8, 3)], mappings[(7, 3)]), axis=0)
frauds = sc.inverse_transform(frauds)
print('')