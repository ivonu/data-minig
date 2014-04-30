#!/usr/bin/env python2.7

import sys

import numpy as np




# from dm-08-unsupervised-annotated.pdf SLIDE 25
#

if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')


def updateMu(x_t, mu, t):
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
    eta = np.min([0.01, 1.0 / t[c]]);
    mu[c] += eta * (x_t - mu[c])


if __name__ == "__main__":
    mu = np.random.randn(200, 750) / 100
    #mu = (np.random.rand(200, 750) * 2) - 1
    t = np.zeros(200)
    for line in sys.stdin:
        line = line.strip()
        #parse a line
        x_t = np.fromstring(line, sep=" ")
        updateMu(x_t, mu, t)
    for c, mu_i in enumerate(mu):
        print_string = " ".join([repr(s) for s in mu_i])
        print '1\t%i\t%s' % (t[c], print_string)

