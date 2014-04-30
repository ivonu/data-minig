#!/usr/bin/env python2.7

import sys

import numpy as np




# from dm-08-unsupervised-annotated.pdf SLIDE 25
#


def updateMu(x_t, mu, eta):
    # c = argmin_j || mu_j - x_t ||_2
    c = 0

    mindist = sys.float_info.max
    for j, mu_j in enumerate(mu.T):
        dist = np.linalg.norm(x_t - mu_j)
        if dist < mindist:
            mindist = dist
            c = j

    # update mu_c
    mu.T[c] -= eta * (x_t - mu.T[c])


if __name__ == "__main__":
    mu = np.random.rand(750, 200)
    t = 0
    for line in sys.stdin:
        line = line.strip()
        #parse a line
        x_t = np.fromstring(sep=" ")
        t += 1
        eta = 1.0 / t
        updateMu(x_t, mu, eta)
    for mu_i in mu.T:
        print_string = " ".join([repr(s) for s in mu_i])
    print '1 %s' % print_string

