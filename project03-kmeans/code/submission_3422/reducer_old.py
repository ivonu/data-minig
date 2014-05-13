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
        dist = np.sum(np.square(x_t - mu_j))
        if dist < mindist:
            mindist = dist
            c = j

    # update mu_c
    t[c] += 1
    eta = np.min([0.05, 1.0 / t[c]])

    for i in range(int(weight / 3)):
        mu[c] += eta * (x_t - mu[c])


if __name__ == "__main__":

    np.set_printoptions(linewidth=1000000000000)
    np.set_printoptions(precision=100)

    # init first 200
    mu = np.random.randn(200, 750) / 30
    t = np.zeros(200)

    for line in sys.stdin:
        line = line[2:]
        split = line.split('\t', 1)
        weight = int(split[0])
        line = split[1]
        line = line.strip()
        #parse a line
        x_t = np.fromstring(line, sep=" ")
        updateMu(x_t, mu, t, weight)

    #t *= -1
    #t = np.sort(t)
    #t *= -1
    for c in range(200):
        #print_string = " ".join([repr(s) for s in mu[c]])
        print_string = np.array_str(mu[c])[1:-1]
        print '%s' % print_string
