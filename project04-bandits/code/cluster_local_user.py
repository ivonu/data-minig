#!/usr/bin/env python2.7
import sys
import numpy as np
from scipy.cluster.vq import kmeans2


def cluster_users(path, k):

    users = []

    with file(path) as inf:
        for i, line in enumerate(inf):

            # Parsing the log line.
            log_line = line.strip().split()
            user = [float(x) for x in log_line[1:7]]
            users.append(user)

    centroids, labels = kmeans2(np.array(users), k)
    return centroids


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: ./cluster_local_user.py log"
        sys.exit(-1)

    centroids = cluster_users(sys.argv[1], 20)

    for cent in centroids:
        print cent
