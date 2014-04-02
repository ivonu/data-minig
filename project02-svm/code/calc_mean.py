#!/usr/bin/env python2.7

import sys
import fileinput
import numpy as np

TRAINING_TXT = "../data/training.txt"
MEAN_TXT = "../data/means.txt"

if __name__ == "__main__":

    means = np.zeros(400)
    count = 0
    with open(TRAINING_TXT) as f:
        for line in f:
            count += 1
            line = line.strip()
            x_i = np.fromstring(line[2:], sep=" ")
            means += x_i
            if count % 10000 == 0:
                print count

    means /= count

    vars = np.zeros(400)
    count = 0
    with open(TRAINING_TXT) as f:
        for line in f:
            count += 1
            line = line.strip()
            x_i = np.fromstring(line[2:], sep=" ")
            vars += np.power(x_i - means, 2)
            if count % 10000 == 0:
                print count

    vars /= count

    with open(MEAN_TXT, 'w') as o:
        means_string = " ".join([repr(s) for s in means])
        vars_string = " ".join([repr(s) for s in vars])
        o.write(means_string)
        o.write("\n")
        o.write(vars_string)



