# -*- coding: utf-8 -*-

# Data preprocessing

import numpy as np
import pandas as pd

# Importing
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:-1].values
Y = data.iloc[:, -1:].values

# Create dummies
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

x_encoder1 = LabelEncoder()
X[:, 1] = x_encoder1.fit_transform(X[:, 1])
x_encoder2 = LabelEncoder()
X[:, 2] = x_encoder2.fit_transform(X[:, 2])

onehot = OneHotEncoder(categorical_features=[1])

X = onehot.fit_transform(X).toarray()

# Dummy var trap !!!
X = X[:, 1:]

# Scaling
from sklearn.model_selection import train_test_split


# Split data
X_tr, X_t, y_tr, y_t = train_test_split(X, Y)

from sklearn.preprocessing import StandardScaler

# scale after splitting, because otherwise the bias would be too big

x_scale = StandardScaler()

X_tr = x_scale.fit_transform(X_tr)

X_t = x_scale.transform(X_t)

### ANN ###
#Keras libs:
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(6,)))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid', input_shape=(6,) ))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], )

classifier.fit(X_tr, y_tr, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_t)
y_pred = np.round(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_t, y_pred)

hw = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 500000 ]])

hw[:,1] = x_encoder1.transform(hw[:,1])
hw[:,2] = x_encoder2.transform(hw[:,2])
hw = onehot.transform(hw).toarray()

hw = hw[:, 1:]
classifier.predict(x_scale.transform(hw))