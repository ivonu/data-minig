# !/usr/bin/env python2.7

from datetime import datetime

import sys

import numpy as np
import numpy.linalg as linalg


# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.


# number of user features
d = 6

alpha = .2
beta = 0.2
As = {}
AInvs = {}
bs = {}
thetas = {}
articles = {}
current_art_id = 0
current_user_features = np.zeros(6)


timestamps = {}

def set_articles(art):
    global d
    global As
    global AInvs
    global bs
    global thetas
    global articles
    global timestamps

    articles = art
    for article_id in art:
        AInvs[article_id] = As[article_id] = np.identity(d, dtype=np.float64)
        bs[article_id] = np.zeros(d, dtype=np.float64)
        thetas[article_id] = np.zeros(d, dtype=np.float64)
        timestamps[article_id] = sys.maxint


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global As
    global AInvs
    global bs
    global thetas
    global current_art_id
    global current_user_features

    if reward == -1:
        return

    As[current_art_id] += np.outer(current_user_features, current_user_features)
    AInvs[current_art_id] = linalg.inv(As[current_art_id])

    bs[current_art_id] += reward * current_user_features
    thetas[current_art_id] = AInvs[current_art_id].dot(bs[current_art_id])


def reccomend(timestamp, user_features, art_ids):
    global alpha
    global beta
    global current_art_id
    global current_user_features
    global As
    global AInvs
    global bs
    global thetas
    global articles
    global timestamps

    user_features = np.array(user_features, dtype=np.float64)

    # create new user feature "time"
    dt = datetime.fromtimestamp(timestamp)
    new_feature = np.min([dt.hour / 24.0, (24.0 - dt.hour) / 24.0])
    new_feature2 = dt.isoweekday() / 7.0
    user_features = np.array(user_features + [new_feature] + [new_feature2])

    max_ucb = sys.float_info.min
    for art_id in art_ids:
        if timestamp < timestamps[art_id]:
            timestamps[art_id] = timestamp

        A_inv = AInvs[art_id]
        theta_a = thetas[art_id]
        a = theta_a.dot(user_features)
        b = alpha * np.sqrt(user_features.dot(A_inv).dot(user_features))
        c = beta * np.exp(-(timestamp - timestamps[art_id]))
        ucb = a + b + c

        if ucb > max_ucb:
            max_ucb = ucb
            current_art_id = art_id
            current_user_features = user_features

    return current_art_id
