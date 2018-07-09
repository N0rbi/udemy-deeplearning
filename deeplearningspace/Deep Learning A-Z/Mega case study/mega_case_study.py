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

som = MiniSom(x=10, y=10, input_len=len(X[0]), random_seed=12345678)
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
frauds = np.concatenate((mappings[(3,7)], mappings[(6,1)]), axis=0)
frauds = sc.inverse_transform(frauds)
print(frauds.shape)


######
#   ANN
######

customers = dataset.iloc[:, 1:].values

# dependent var

is_fraud = np.zeros(len(dataset))
for i in range(len(is_fraud)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Code from the ANN tutorial

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(customers)
y_train = is_fraud

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer = 'uniform', activation = 'relu', input_dim=len(X_train[0])))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=2)

# Part 3 - Making predictions and evaluating the model

# Predicting probabilities
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]


print("")