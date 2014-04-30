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
    mu = np.zeros([200, 750])
    t = 0

    count = 0
    for line in sys.stdin:
        line = line[2:]
        line = line.strip()
        #parse a line
        x_t = np.fromstring(line, sep=" ")
        mu[count] = x_t
        count += 1
        if count == 200:
            break

    t = 0
    for line in sys.stdin:
        line = line[2:]
        line = line.strip()
        #parse a line
        x_t = np.fromstring(line, sep=" ")
        t += 1
        eta = 1 / t
        updateMu(x_t, mu, eta)

    for mu_i in mu:
        print_string = " ".join([repr(s) for s in mu_i])
        print '%s' % print_string
