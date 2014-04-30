#!/usr/bin/env python2.7

import sys

import numpy as np




# from dm-08-unsupervised-annotated.pdf SLIDE 25
#
def runMapper(stream):
    t = 0
    mu = np.random.rand(750, 200)
    for line in stream:
        line = line.strip()

        t += 1
        eta = 1.0 / t
        # or
        # eta = min (1, C/t)

        #parse a line
        x_t = np.fromstring(line[2:], sep=" ")


        # c = argmin_j || mu_j - x_t ||_2
        mindist = sys.float_info.max
        c = 0

        for j, mu_j in enumerate(mu.T):
            dist = np.linalg.norm(x_t - mu_j)
            if dist < mindist:
                mindist = dist
                c = j

        # update mu_c
        mu.T[c] -= eta * (x_t - mu.T[c])

    for mu_i in mu.T:
        print_string = " ".join([repr(s) for s in mu_i])
        print '1 %s' % print_string


if __name__ == "__main__":
    runMapper(sys.stdin)

