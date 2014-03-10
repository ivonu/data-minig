#!/usr/bin/env python2

import sys

import numpy as np
from numpy.random import rand



# for local debugging purpouses
#sys.stdin = open("../data/train.txt", 'r')

# DEBUG: read from file in argument instead of stdin
if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')

#######################
# Parameters
#######################


# r: band width
# b: nr of bands
# k: nr of  permutation-hash functions
# TODO: n: number of buckets ?
# c: total number of shingles possible
r = 16
b = 16
k = r * b
n = 1000
c = 10000


def hash_shingle(hash_fn, shingle):
    return (hash_fn[0] * shingle + hash_fn[1]) % c


def hash_band(hash_fns, vector):
    bucket_nr = 0
    for i in range(r):
        bucket_nr += (hash_fns[i][0] * vector[i] * hash_fns[i][1]) % n
    return int(bucket_nr % n)


def partition(video_id, shingles, perm_hash_fns, band_hash_fns):
    signature = np.ones((k, 1))
    signature[:] = 10001

    for shingle in shingles:
        for i in range(k):
            signature[i] = min(hash_shingle(perm_hash_fns[i], shingle), signature[i])

    # split signature column into bands
    # foreach band_i in bands{
    #       bucket-nr = h_band( signature in band_i)
    #       idea: sort shingles to easier find false positives in reducer
    #       emit key:bucket-nr + band-nr value: movie + all its shigles
    for band in range(b):
        bucket_nr = hash_band(band_hash_fns, signature[band * r:band * r + r])
        print '%s:%s\t%s' % (bucket_nr, band, video_id)


# create k [a b] tuples
# a from 0 to 999
# b from 0 to 9999
# returns [a_0 b_0]
#         [a_1 b_1]
#         [....]
#         [a_k-1 b_k-1]
def init_permutation_hashes(num_hashes):
    a = rand(num_hashes, 1) * 1000
    b = rand(num_hashes, 1) * 10000
    return np.floor(np.hstack([a, b]))


# TODO maybe we need som other constants than 1000 and 1000
def init_band_hashes(num_hashes):
    a = rand(num_hashes, 1) * 1000
    b = rand(num_hashes, 1) * 1000
    return np.floor(np.hstack([a, b]))


if __name__ == "__main__":
    # Very important. Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)

    perm_hash_fns = init_permutation_hashes(k)
    band_hash_fns = init_band_hashes(r)

    for line in sys.stdin:
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], sep=" ")
        partition(video_id, shingles, perm_hash_fns, band_hash_fns)


