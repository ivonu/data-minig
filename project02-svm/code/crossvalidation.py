#!/usr/bin/env python2.7

import sys
import fileinput

import numpy as np

import mapper as map


pegasos_k = 1000
k = 10
bucket_size = 10000
errmin = sys.float_info.max
best_lambda = 0
lambdas = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
bucket_files = []


def calculate_err_on(test_file, w_hat):
    wrong_classified = 0
    linecount = 0
    with open(test_file) as test_input:
        for line in test_input:
            linecount += 1
            y_i = 1. if line[0:1] == '+' else -1.
            x_i = map.transform(np.fromstring(line[2:], sep=" "))
            dot_prod = np.dot(w_hat, x_i)
            if (dot_prod < 0 and y_i == 1) or (dot_prod > 0 and y_i == -1):
                wrong_classified += 1

    return float(wrong_classified) / float(linecount)


def call_pegasos(bucket_no, _lambda):
    # calculate training set (all but set bucket_no)
    training_set = []
    test_file = None
    for i, bucket in enumerate(bucket_files):
        if i == bucket_no:
            test_file = bucket
        else:
            training_set.append(bucket)

    # use training set as input
    inputstream = fileinput.input(training_set)
    w_hat = map.pegasos(_lambda, pegasos_k, inputstream)

    return calculate_err_on(test_file, w_hat)


def svm_cross(_lambda):
    mean_err = 0
    for bucket_no in range(k):
        err = call_pegasos(bucket_no, _lambda)
        mean_err += err / k
    return mean_err


if __name__ == "__main__":
    #split data to k buckets
    with open("../data/training.txt") as f:
        for fcnt in range(k):
            bucket_file = "../data/training" + str(fcnt)
            bucket_files.append(bucket_file)
            with open(bucket_file, 'w') as o:
                for lncnt in range(bucket_size):
                    data = f.readline()
                    o.write(data)

    for _lambda in lambdas:
        err = svm_cross(_lambda)
        print "lambda %f has error %f" % (_lambda, err )
        if (err < errmin):
            errmin = err
            best_lambda = _lambda

    print 'lambda %f is best, with error %f' % (best_lambda, errmin)

