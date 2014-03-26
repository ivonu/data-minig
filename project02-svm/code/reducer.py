#!/usr/bin/env python2.7

import sys

import numpy as np


# DEBUG: read from file in argument instead of stdin
if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')

if __name__ == "__main__":


    count = 1

    # read first line
    first = sys.stdin.readline()

    #TODO case where first is eof

    w = np.fromstring(first[2:], sep=" ")

    for line in sys.stdin:
        line = line.strip()
        count += 1

        w += np.fromstring(line[2:], sep=" ")

    w /= count
    w_string = " ".join([str(s) for s in w])
    print '%s' % (w_string)
