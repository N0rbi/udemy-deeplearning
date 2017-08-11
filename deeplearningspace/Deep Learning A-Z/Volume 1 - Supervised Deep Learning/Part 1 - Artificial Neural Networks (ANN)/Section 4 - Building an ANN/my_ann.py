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

x_encoder = LabelEncoder()
X[:, 1] = x_encoder.fit_transform(X[:, 1])
X[:, 2] = x_encoder.fit_transform(X[:, 2])

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
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11)))