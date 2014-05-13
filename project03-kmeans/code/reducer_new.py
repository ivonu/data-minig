#!/usr/bin/env python2.7
import sys

import numpy as np

def updateMu(x_t, mu, t):
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
    eta = np.min([0.05, 3.0 / t[c]])
    mu[c] += eta * (x_t - mu[c])

if __name__ == "__main__":

    np.set_printoptions(linewidth=1000000000000)
    np.set_printoptions(precision=100)

    mu = np.random.randn(200, 750) / 30
    t = np.zeros(200)

    co = 0
    for line in sys.stdin:
        line = line[2:]
        line = line.strip()
        #parse a line
        x_t = np.fromstring(line, sep=" ")
        updateMu(x_t, mu, t)

    for mu_i in mu:
        #print_string = " ".join([repr(s) for s in mu_i])
        #print '%s' % print_string
        print_string = np.array_str(mu_i)[1:-1]
        print '%s' % print_string
