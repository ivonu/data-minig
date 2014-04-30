#!/usr/bin/env python2.7

import numpy as np

import mapper


if __name__ == "__main__":
    # Load the file stored in NPZ format
    data = np.load("../data/tiny_subset.zip")['arr_0']

    mu = np.random.rand(750, 200)
    t = 0
    eta = 0.5

    # or eta = 1.0 / t in for loop
    # or
    # eta = min (1, C/t) in for loop

    for v in data:
        t += 1
        mapper.updateMu(v, mu, eta)  # To store the file in CSV format
    np.savetxt('training.csv', mu)

