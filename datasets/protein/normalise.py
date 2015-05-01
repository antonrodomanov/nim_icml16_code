#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import MinMaxScaler

print('Loading train...')
A_train = np.loadtxt('bio_train.dat', usecols=np.arange(2, 77))
y_train = A_train[:, 0]
X_train = A_train[:, 1:]

print('Loading test...')
X_test = np.loadtxt('bio_test.dat', usecols=np.arange(3, 77))

print('Normalising...')
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

print('Saving result...')
np.savetxt('bio_train_scaled.dat', X_train_scaled)
np.savetxt('bio_train_scaled.lab', y_train, fmt='%d')
np.savetxt('bio_test_scaled.dat', X_test_scaled)
