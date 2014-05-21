#!/usr/bin/env python2.7

import sys

import numpy as np


# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.

centroids = [
    [1.0, 0.0799611428571, 0.00234807142857, 0.557770857143, 0.350938142857, 0.008982],
    [1.0, 0.273980636364, 0.00251745454545, 0.633479818182, 0.068857, 0.0211650454545],
    [1.0, 0.129997418605, 0.162582372093, 0.0374430465116, 0.61611827907, 0.0538588837209],
    [1.0, 0.1437156875, 0.1990529375, 0.0040978125, 0.153567375, 0.4995661875],
    [1.0, 0.200421073171, 0.553122121951, 0.00392117073171, 0.207473853659, 0.035061804878],
    [1.0, 0.0247240428571, 0.0034959, 0.0106372142857, 0.0240757857143, 0.937066985714],
    [1.0, 0.0211835490196, 6.39215686275e-05, 0.948882411765, 0.0213407058824, 0.00852911764706],
    [1.0, 0.750657814815, 0.0788379259259, 0.0823902962963, 0.0691904074074, 0.0189234444444],
    [1.0, 0.000904995689655, 8.32198275862e-05, 0.000542931034483, 0.000775525862069, 0.997693306034],
    [1.0, 0.565841819149, 0.0918654148936, 0.0647296702128, 0.266057553191, 0.0115056382979],
    [1.0, 0.485204625, 0.00750691666667, 0.365224, 0.128711625, 0.0133528333333],
    [1.0, 0.00832434666667, 0.00106586666667, 0.00445617333333, 0.00750141333333, 0.978652413333],
    [1.0, 0.0408278846154, 0.0361180384615, 0.0666147692308, 0.850098692308, 0.00634051923077],
    [1.0, 0.09824675, 0.02901345, 0.0976169, 0.1014491, 0.6736736],
    [1.0, 0.0608222666667, 0.0002198, 0.4112028, 0.0282382, 0.499516666667],
    [1.0, 0.0708686, 0.831845018182, 0.000803254545455, 0.0826729272727, 0.0138100545455],
    [1.0, 0.100137862069, 0.0100780689655, 0.285888965517, 0.58427337931, 0.0196218275862],
    [1.0, 0.042060255814, 0.0312073023256, 0.0273735348837, 0.0679147209302, 0.831444302326],
    [1.0, 0.0733979310345, 0.000765103448276, 0.771379068966, 0.104581, 0.0498770344828],
    [1.0, 0.363296083333, 0.0847013333333, 0.10821925, 0.43284275, 0.0109405]
]

arts = {}
arts_to_index = {}

t = 1

current_centroid_idx = 0

current_art_id = 0



########## with clustering
# number of user groups
k = 20

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

    success_counts = np.zeros((k, len(art)))
    total_counts = np.zeros((k, len(art)))

    # better initialization required!
    user_cluster_centroids = np.array(centroids)
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
