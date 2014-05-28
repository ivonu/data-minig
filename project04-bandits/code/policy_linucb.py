#!/usr/bin/env python2.7

import sys

import numpy as np
import numpy.linalg as linalg



# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.


delta = .95
#alpha = 1 + np.sqrt(np.log(2.0 / delta) / 2)
alpha = 1.2
As = {}
AInvs = {}
bs = {}
thetas = {}
articles = {}
current_art_id = 0
current_user_features = np.zeros(6)


def set_articles(art):
    global As
    global AInvs
    global bs
    global thetas
    global articles
    articles = art
    for article_id in art:
        AInvs[article_id] = As[article_id] = np.identity(len(art[article_id]))
        bs[article_id] = np.zeros(len(art[article_id]))
        thetas[article_id] = np.zeros(len(art[article_id]))


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
    As[current_art_id] += current_user_features.dot(current_user_features)
    AInvs[current_art_id] = linalg.inv(As[current_art_id])

    bs[current_art_id] += reward * current_user_features
    thetas[current_art_id] = AInvs[current_art_id].dot(bs[current_art_id])


def reccomend(timestamp, user_features, articles):
    global alpha
    global current_art_id
    global current_user_features
    global As
    global AInvs
    global bs
    global thetas

    user_features = np.array(user_features)

    max_ucb = sys.float_info.min
    for art_id in articles:
        A_inv = AInvs[art_id]
        theta_a = thetas[art_id]
        ucb = theta_a.dot(user_features) + np.sqrt(user_features.dot(A_inv).dot(user_features))

        if ucb > max_ucb:
            max_ucb = ucb
            current_art_id = art_id

    current_user_features = user_features
    return current_art_id
