#!/usr/bin/env python2.7

import sys

import numpy as np


# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.

mu = {}

n = {}

arts = {}

t = 1

max_art_id = 0


def set_articles(art):
    global mu
    global n
    global arts

    arts = art
    for article in art:
        mu[article] = float(0)
        n[article] = 0


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global mu
    global n
    global arts
    global t

    n[max_art_id] += 1
    mu[max_art_id] += (reward - mu[max_art_id]) / n[max_art_id]


# This function will be called by the evaluator.
# Check task description for details.
def UCB(art_id):
    global mu
    global n
    global arts
    global t

    return mu[art_id] + np.sqrt(2 * np.log(t) / n[art_id])


def reccomend(timestamp, user_features, articles):
    global max_art_id

    max_ucb = sys.float_info.min
    max_art_id = articles[0]
    for art in articles:
        if n[art] == 0:
            return art
    for art_id in articles:
        ucb = UCB(art_id)
        if ucb > max_ucb:
            max_ucb = ucb
            max_art_id = art_id

    return max_art_id
