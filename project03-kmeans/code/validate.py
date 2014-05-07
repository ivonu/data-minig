#!/usr/bin/env python2.7

import sys

import numpy as np
import mapper as m
import reducer_old as r

data_size = 100000
data_train_size = 60000
data_validate_size = data_size - data_train_size
buckets = 10
bucket_size = data_train_size/buckets


def train (data_train):

    # mappers
    S = []
    ts = []
    for i in range(buckets):
        print '%i. bucket...' % i
        mu1 = np.random.randn(200, 750) / 42
        t = np.zeros(200)

        for x_t in data_train[i*bucket_size:i*bucket_size+bucket_size]:
            m.updateMu(x_t, mu1, t)
        S.append(mu1)
        ts.append(t)


    # reducer
    final_mu = np.random.randn(200, 750) / 42
    t = np.zeros(200)

    for c, mu in enumerate(S):
        weights = ts[c]

        for c2, x_t in enumerate(mu):
            r.updateMu(x_t, final_mu, t, weights[c2])

    return final_mu


def validate(data_validate, mu):

    total_dist = 0
    for x_t in data_validate:
        mindist = sys.float_info.max
        for j, mu_j in enumerate(mu):
            dist = np.sum(np.square(x_t - mu_j))
            if dist < mindist:
                mindist = dist
        total_dist += (mindist/data_validate_size)

    return total_dist


if __name__ == "__main__":
    data = np.load("../data/tiny_subset.zip")['arr_0']

    data_train = data[:data_train_size]
    data_validate = data[data_train_size+1:]

    print "train..."
    mu = train(data_train)
    print "validate..."
    error = validate(data_validate, mu)

    print error