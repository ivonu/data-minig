#!/usr/bin/env python2.7

import numpy as np

# Load the file stored in NPZ format
data = np.load("../data/tiny_subset.zip")['arr_0']

# To store the file in CSV format
np.savetxt('../data/training.csv', data)

# Load CSV file
#data = np.loadtxt('training.csv')
