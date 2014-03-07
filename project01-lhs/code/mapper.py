#!/usr/bin/env python

import numpy as np
import sys


# for local debugging purpouses
sys.stdin = open("../data/train.txt", 'r')

# DEBUG: read from file in argument instead of stdin
if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')

#######################
# Parameters
#######################

# number of permutation hash functions
k = 5


# total number of shingles possible
c = 1000000


# this is a function definition
# for each video we need to generate a column in our signature matrix
def partition(video_id, shingles):

    # pass is a null operation
    pass


# create k
def initPermutationhashes(numHashes):

    for i in range(numHashes):
        #TODO generate hash funcitions h(x) = ax + b mod c
        print i


if __name__ == "__main__":
    # Very important. Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)

    initPermutationhashes()

    for line in sys.stdin:
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")
        partition(video_id, shingles)


