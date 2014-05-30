#!/usr/bin/env python2.7

import sys
import datetime

import numpy as np
import numpy.linalg as linalg









# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.


# number of user features
d = 7

alpha = 0.35
As = {}
AInvs = {}
bs = {}
thetas = {}
articles = {}
current_art_id = 0
current_user_features = np.zeros(6)


def set_articles(art):
    global d
    global As
    global AInvs
    global bs
    global thetas
    global articles
    articles = art
    for article_id in art:
        AInvs[article_id] = As[article_id] = np.identity(d)
        bs[article_id] = np.zeros(d)
        thetas[article_id] = np.zeros(d)


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
    global current_art_id
    global current_user_features
    global As
    global AInvs
    global bs
    global thetas
    global articles

    # create new user feature "time"
    dt = datetime.datetime.fromtimestamp(timestamp)

    new_feature = np.min([dt.hour / 24.0, (24.0 - dt.hour) / 24.0])
    # new_feature2 = dt.isoweekday() / 7.0
    # user_features = np.array(user_features[1:] + [new_feature, new_feature2])
    user_features = np.array(user_features + [new_feature])

    max_ucb = sys.float_info.min
    for art_id in art_ids:
        A_inv = AInvs[art_id]
        theta_a = thetas[art_id]
        features = user_features
        a = theta_a.dot(features)
        b = alpha * np.sqrt(features.dot(A_inv).dot(features))
        ucb = a + b

        if ucb > max_ucb:
            max_ucb = ucb
            current_art_id = art_id
            current_user_features = features

    return current_art_id
