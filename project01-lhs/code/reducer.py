#!/usr/bin/env python2

import sys

import numpy as np


last_key = None
key_count = 0
duplicates = []

if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')


vid_to_shingle_dict = dict();


def print_duplicates(duplicates, vid_to_shingle_dict):

    for video_id_l in duplicates:
            for video_id_r in duplicates:
                if video_id_l == video_id_r:
                    continue

                if video_id_l > video_id_r:
                    continue

                s_l = vid_to_shingle_dict[video_id_l]
                s_r = vid_to_shingle_dict[video_id_r]

                s_intersect = s_l.intersection(s_r)

                assert isinstance(s_intersect, set)
                num_elem_common = len(s_intersect)
                num_elem_l = len(s_l)
                num_elem_r = len(s_r)

                if float(num_elem_common) / float(num_elem_l) > 0.85:
                    print "%d\t%d" % (video_id_l, video_id_r)




for line in sys.stdin:
    line = line.strip()
    key, video_shingle_string = line.split("\t")
    video_id = int(video_shingle_string[0:9])
    shingles = np.fromstring(video_shingle_string[10:], sep="-")

    vid_to_shingle_dict[video_id] = set(shingles);

    if last_key is None:
        last_key = key

    if key == last_key:
        duplicates.append(int(video_id))
    else:
        # Key changed (previous line was k=x, this line is k=y)

        duplicates = set(duplicates)

        # remove false positives
        print_duplicates(duplicates, vid_to_shingle_dict);

        duplicates = [int(video_id)]
        last_key = key


