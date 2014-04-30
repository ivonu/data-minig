#!/usr/bin/env python2.7
import sys

import numpy as np


def updateMu(x_t, mu, eta):
    # c = argmin_j || mu_j - x_t ||_2
    c = 0

    mindist = sys.float_info.max
    for j, mu_j in enumerate(mu):
        dist = np.linalg.norm(x_t - mu_j)
        if dist < mindist:
            mindist = dist
            c = j

    # update mu_c
    mu[c] += eta * (x_t - mu[c])


if __name__ == "__main__":

    # init first 200
    mu = np.zeros(200, 750)

    t = 0
    for line in sys.stdin:
        key, line = line.split("\t")
        line = line.strip()
        #parse a line
        x_t = np.fromstring(sep=" ")
        mu[t] = x_t
        t += 1
        if t == 200:
            break
    t = 0
    for line in sys.stdin:
        key, line = line.split("\t")
        #parse a line
        x_t = np.fromstring(line, sep=" ")
        t += 1
        eta = 1.0 / t
        updateMu(x_t, mu, eta)

    for mu_i in mu:
        print_string = " ".join([repr(s) for s in mu_i])
        print '%s' % print_string
