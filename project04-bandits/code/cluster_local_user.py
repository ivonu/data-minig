#!/usr/bin/env python2.7
import sys

import numpy as np
from scipy.cluster.vq import kmeans


def cluster_users(path, k):
    users = []

    with file(path) as inf:
        for i, line in enumerate(inf):
            # Parsing the log line.
            log_line = line.strip().split()
            user = [float(x) for x in log_line[1:7]]
            users.append(user)

    centroids, labels = kmeans(np.array(users), k)
    return centroids


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000, precision=20)
    if len(sys.argv) != 2:
        print "Usage: ./cluster_local_user.py log"
        sys.exit(-1)

    centroids = cluster_users(sys.argv[1], 20)
    print "centroids = ["
    for cent in centroids:
        print "[%s,%s,%s,%s,%s,%s]," % (cent[0], cent[1], cent[2], cent[3], cent[4], cent[5])

    print "]"
