#!/usr/bin/env python2.7

import sys

import numpy as np


# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.

arts = {}
arts_to_index = {}

t = 1

current_centroid_idx = 0

current_art_id = 0


########## with clustering
# number of user groups
k = 5

success_counts = np.zeros(0)

total_counts = np.zeros(0)


def set_articles(art):
    global arts
    global k

    global success_counts
    global total_counts
    global arts_to_index

    success_counts = np.zeros((k, len(art)))
    total_counts = np.zeros((k, len(art)))

    arts = art

    for k, article in enumerate(art):
        arts_to_index[article] = k


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global success_counts
    global total_counts
    global current_art_id
    global current_centroid_idx
    global arts_to_index
    global t

    if reward == -1:
        return

    t += 1
    current_art_index = arts_to_index[current_art_id]
    total_counts[current_centroid_idx][current_art_index] += 1
    success_counts[current_centroid_idx][current_art_index] += reward


# This function will be called by the evaluator.
# Check task description for details.
def UCB(centroid_idx, art_index):
    global total_counts
    global success_counts

    tot = total_counts[centroid_idx][art_index]
    suc = success_counts[centroid_idx][art_index]

    mu = suc / tot
    return mu + np.sqrt(2 * np.log(t) / tot)


def reccomend(timestamp, user_features, articles):
    global max_art_id
    global current_centroid_idx
    global current_art_id

    current_centroid_idx = np.argmax(user_features[1:])

    minpullthreshold = 1

    max_ucb = sys.float_info.min
    max_art_id = articles[0]

    for art_id in articles:
        art_index = arts_to_index[art_id]

        if total_counts[current_centroid_idx][art_index] <= minpullthreshold:
            # make sure we pull arms that are only rarely selected
            current_art_id = art_id
            return art_id

        ucb = UCB(current_centroid_idx, art_index)
        if ucb > max_ucb:
            max_ucb = ucb
            current_art_id = art_id

    return current_art_id
