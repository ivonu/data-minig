#!/usr/bin/env python2.7

import sys

import numpy as np
from numpy.linalg import inv



# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.

alpha = 0.5

#number of user features
k = 6
#number of article features
d = 6
A_0 = np.identity(k)
b_0 = np.zeros(k)

As = {}
Bs = {}
bs = {}

articles = {}
current_art_id = 0
current_user_features = np.zeros(6)


def set_articles(art):
    global As
    global Bs
    global bs
    global articles

    for article_id in art:
        articles[article_id] = np.array(art[article_id])
        As[article_id] = np.identity(d)
        Bs[article_id] = np.zeros((d, k))
        bs[article_id] = np.zeros(d)


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global As
    global Bs
    global bs
    global A_0
    global b_0
    global current_user_features
    global current_art_id

    if reward == -1:
        return

    A_0 += Bs[current_art_id].T.dot(inv(As[current_art_id])).dot(Bs[current_art_id])
    b_0 += Bs[current_art_id].T.dot(inv(As[current_art_id])).dot(bs[current_art_id])
    As[current_art_id] += np.outer(articles[current_art_id], articles[current_art_id])
    Bs[current_art_id] += np.outer(articles[current_art_id], current_user_features)

    bs[current_art_id] += reward * articles[current_art_id]

    A_0 += np.outer(current_user_features, current_user_features) - Bs[current_art_id].T.dot(
        inv(As[current_art_id])).dot(Bs[current_art_id])
    b_0 += reward * current_user_features - Bs[current_art_id].T.dot(inv(As[current_art_id])).dot(bs[current_art_id])


def reccomend(timestamp, user_features, article_ids):
    global As
    global Bs
    global bs
    global A_0
    global b_0
    global current_user_features
    global current_art_id

    user_features = np.array(user_features)
    # beta = kx1
    invA_0 = inv(A_0)
    beta = invA_0.dot(b_0)
    max_ucb = sys.float_info.min
    for art_id in article_ids:
        invA_a = inv(As[art_id])
        B_a = Bs[art_id]
        theta_a = invA_a.dot(( bs[art_id] - B_a.dot(beta)))
        z_a_dot_inv_A_0 = user_features.dot(invA_0)
        t1 = z_a_dot_inv_A_0.dot(user_features)
        x_a = articles[art_id]
        t2 = 2 * z_a_dot_inv_A_0.dot(B_a.T).dot(invA_a).dot(x_a)
        x_a_dot_invA_a = x_a.dot(invA_a)
        t3 = x_a_dot_invA_a.dot(x_a)
        t4 = x_a_dot_invA_a.dot(B_a).dot(invA_0).dot(B_a.T).dot(invA_a).dot(x_a)

        s_a = t1 - t2 + t3 + t4
        ucb = user_features.dot(beta) + x_a.dot(theta_a) + alpha * np.sqrt(s_a)

        if ucb > max_ucb:
            max_ucb = ucb
            current_art_id = art_id

    current_user_features = user_features

    return current_art_id




