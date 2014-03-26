#!/usr/bin/env python2.7

import sys

import numpy as np

import mapper as map

if __name__ == "__main__":

    # read w from reducer_output
    with open(sys.argv[1]) as f:
        w = np.fromstring(f.readline(), sep=" ")

    # read x
    count = 0;
    for line in open(sys.argv[2]):

        x = np.fromstring(line[2:], sep=" ")
        print "%s" % '1' if np.dot(w, map.transform(x)) > 0 else '-1'

        count += 1
        if count >= 150:
            break
