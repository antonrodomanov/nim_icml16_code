#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import MinMaxScaler

##########################################################################

print('Loading train...')
X_train = np.loadtxt('alpha_train.dat')

print('Normalising train...')
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = min_max_scaler.fit_transform(X_train)

print('Saving train...')
np.savetxt('alpha_train_scaled.dat', X_train_scaled)

print('Deleting train...')
del(X_train, X_train_scaled)

##########################################################################

print('Loading test...')
X_test = np.loadtxt('alpha_test.dat')

print('Normalising test...')
X_test_scaled = min_max_scaler.transform(X_test)

print('Saving test...')
np.savetxt('alpha_test_scaled.dat', X_test_scaled)

print('Deleting test...')
del(X_test, X_test_scaled)

##########################################################################

print('Loading validation...')
X_val = np.loadtxt('alpha_val.dat')

print('Normalising validation...')
X_val_scaled = min_max_scaler.transform(X_val)

print('Saving validation...')
np.savetxt('alpha_val_scaled.dat', X_val_scaled)

print('Deleting validation...')
del(X_val, X_val_scaled)
