#!/usr/bin/env python2.7
import sys

import numpy as np

if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')


def updateMu(x_t, mu, t, weight):
    # c = argmin_j || mu_j - x_t ||_2
    c = 0

    mindist = sys.float_info.max
    for j, mu_j in enumerate(mu):
        dist = np.linalg.norm(x_t - mu_j)
        if dist < mindist:
            mindist = dist
            c = j

    # update mu_c
    t[c] += 1
    eta = np.min([0.01, 1.0 / t[c]])

    for i in range(weight):
        mu[c] += eta * (x_t - mu[c])


if __name__ == "__main__":

    # init first 200
    mu = np.random.randn(200, 750) / 100
    t = np.zeros(200)

    # count = 0
    # for line in sys.stdin:
    #     line = line[2:]
    #     split = line.split('\t', 1)
    #     line = split[1]
    #     line = line.strip()
    #     #parse a line
    #     x_t = np.fromstring(line, sep=" ")
    #     mu[count] = x_t
    #     count += 1
    #     if count == 200:
    #         break

    for line in sys.stdin:
        line = line[2:]
        split = line.split('\t', 1)
        weight = int(split[0])
        line = split[1]
        line = line.strip()
        #parse a line
        x_t = np.fromstring(line, sep=" ")
        updateMu(x_t, mu, t, weight)

    for mu_i in mu:
        print_string = " ".join([repr(s) for s in mu_i])
        print '%s' % print_string
