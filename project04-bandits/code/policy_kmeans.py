#!/usr/bin/env python2.7

import sys

import numpy as np


# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.

centroids = [
    [1.0, 0.175731, 0.0958882682927, 0.144383146341, 0.548856182927, 0.0351414634146],
    [1.0, 0.0323260704225, 0.000202647887324, 0.90642543662, 0.038903028169, 0.0221426619718],
    [1.0, 0.0504162424242, 0.0620485151515, 0.0633262878788, 0.818385848485, 0.00582298484848],
    [1.0, 0.212260981481, 0.00306168518519, 0.589915111111, 0.1741525, 0.0206097037037],
    [1.0, 0.0265272613636, 0.00462709090909, 0.0129328977273, 0.0264492840909, 0.929463386364],
    [1.0, 0.0633119411765, 0.000237411764706, 0.389842294118, 0.0312064705882, 0.515401529412],
    [1.0, 0.675108423729, 0.0685630169492, 0.131388389831, 0.111560084746, 0.0133800677966],
    [1.0, 0.0557522888889, 0.0389461555556, 0.0418158222222, 0.0811834222222, 0.7823024],
    [1.0, 0.00246385714286, 0.000238867109635, 0.00128027242525, 0.00230186378738, 0.99371517608],
    [1.0, 0.119223336957, 0.726456771739, 0.00186136956522, 0.12866498913, 0.0237934456522],
    [1.0, 0.14445515, 0.16092955, 0.02676485, 0.15268185, 0.51516865],
    [1.0, 0.507936361905, 0.0923565238095, 0.0780322, 0.308406028571, 0.0132689619048],
]

arts = {}
arts_to_index = {}

t = 1

current_centroid_idx = 0

current_art_id = 0



########## with clustering
# number of user groups
k = len(centroids)

success_counts = np.zeros(0)

total_counts = np.zeros(0)

user_cluster_centroids = np.zeros(0)

user_cluster_touch_counts = np.zeros(0)


def set_articles(art):
    global arts
    global k

    global success_counts
    global total_counts
    global user_cluster_centroids
    global user_cluster_touch_counts
    global arts_to_index
    global centroids

    success_counts = np.zeros((k, len(art)))
    total_counts = np.zeros((k, len(art)))

    # better initialization required!
    user_cluster_centroids = np.array(centroids)
    #user_cluster_centroids = np.random.random((k, 6))
    user_cluster_touch_counts = np.zeros(k);
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


# online k-means for user centroids
def updateMu(x_t):
    # c = argmin_j || mu_j - x_t ||_2
    c = 0
    min_dist = sys.float_info.max
    for j, mu_j in enumerate(user_cluster_centroids):
        dist = np.sum(np.square(x_t - mu_j))
        if dist < min_dist:
            min_dist = dist
            c = j

    # update mu_c
    user_cluster_touch_counts[c] += 1
    eta = np.min([0.05, 1.0 / user_cluster_touch_counts[c]]);
    user_cluster_centroids[c] += eta * (x_t - user_cluster_centroids[c])

    return c


def reccomend(timestamp, user_features, articles):
    global max_art_id
    global current_centroid_idx
    global current_art_id

    current_centroid_idx = updateMu(user_features)

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
