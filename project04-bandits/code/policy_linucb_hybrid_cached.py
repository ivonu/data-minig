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

invA_as = {}
x_a_dot_invA_as = {}
B_aT_dot_invA_a_dot_x_as = {}
x_a_dot_invA_a_dot_x_as = {}
x_a_dot_invA_a_dot_B_as = {}
invA_0 = np.identity(k)
beta = np.zeros(k)


def set_articles(art):
    global As
    global Bs
    global bs
    global articles

    global invA_as
    global x_a_dot_invA_as
    global B_aT_dot_invA_a_dot_x_as
    global x_a_dot_invA_a_dot_x_as
    global x_a_dot_invA_a_dot_B_as

    for article_id in art:
        articles[article_id] = np.array(art[article_id])
        As[article_id] = np.identity(d)
        Bs[article_id] = np.zeros((d, k))
        bs[article_id] = np.zeros(d)

        invA_as[article_id] = np.identity(d)
        x_a_dot_invA_as[article_id] = articles[article_id].dot(invA_as[article_id])
        B_aT_dot_invA_a_dot_x_as[article_id] = Bs[article_id].T.dot(invA_as[article_id]).dot(articles[article_id])
        x_a_dot_invA_a_dot_x_as[article_id] = x_a_dot_invA_as[article_id].dot(articles[article_id])
        x_a_dot_invA_a_dot_B_as[article_id] = x_a_dot_invA_as[article_id].dot(Bs[article_id])



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

    global invA_as
    global invA_0
    global beta
    global x_a_dot_invA_as
    global B_aT_dot_invA_a_dot_x_as
    global x_a_dot_invA_a_dot_x_as
    global x_a_dot_invA_a_dot_B_as

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

    # beta = kx1
    invA_0 = inv(A_0)
    beta = invA_0.dot(b_0)

    invA_as[current_art_id] = inv(As[current_art_id])
    x_a_dot_invA_as[current_art_id] = articles[current_art_id].dot(invA_as[current_art_id])
    B_aT_dot_invA_a_dot_x_as[current_art_id] = Bs[current_art_id].T.dot(invA_as[current_art_id]).dot(articles[current_art_id])
    x_a_dot_invA_a_dot_x_as[current_art_id] = x_a_dot_invA_as[current_art_id].dot(articles[current_art_id])
    x_a_dot_invA_a_dot_B_as[current_art_id] = x_a_dot_invA_as[current_art_id].dot(Bs[current_art_id])


def reccomend(timestamp, user_features, article_ids):
    global As
    global Bs
    global bs
    global A_0
    global b_0
    global current_user_features
    global current_art_id

    global invA_as
    global invA_0
    global beta
    global x_a_dot_invA_as
    global B_aT_dot_invA_a_dot_x_as
    global x_a_dot_invA_a_dot_x_as
    global x_a_dot_invA_a_dot_B_as

    z_a = np.array(user_features)

    max_ucb = sys.float_info.min
    for art_id in article_ids:

        B_a = Bs[art_id]
        theta_a = invA_as[art_id].dot((bs[art_id] - B_a.dot(beta)))

        x_a = articles[art_id]
        z_a_dot_inv_A_0 = z_a.dot(invA_0)
        x_a_dot_invA_a_dot_x_a = x_a_dot_invA_a_dot_x_as[art_id]
        B_aT_dot_invA_a_dot_x_a = B_aT_dot_invA_a_dot_x_as[art_id]
        x_a_dot_invA_a_dot_B_a = x_a_dot_invA_a_dot_B_as[art_id]

        t1 = z_a_dot_inv_A_0.dot(z_a)
        t2 = 2 * z_a_dot_inv_A_0.dot(B_aT_dot_invA_a_dot_x_a)
        t3 = x_a_dot_invA_a_dot_x_a
        t4 = x_a_dot_invA_a_dot_B_a.dot(invA_0).dot(B_aT_dot_invA_a_dot_x_a)

        s_a = t1 - t2 + t3 + t4
        ucb = z_a.dot(beta) + x_a.dot(theta_a) + alpha * np.sqrt(s_a)

        if ucb > max_ucb:
            max_ucb = ucb
            current_art_id = art_id

    current_user_features = z_a

    return current_art_id




