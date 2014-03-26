from __future__ import print_function


#split data to 10 buckets
with open("../data/training.txt") as f:
    for fcnt in range(10):
        with open("../data/training" + str(fcnt) + ".txt", 'w') as o:
            for lncnt in range(10000):
                data = f.readline()
                o.write(data)
