"""
audio_recognize.py
This file contains the basic functions used for audio clustering and recognition

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans


def cluster_syllables(syllables, specgram, sp_freq, f_low, f_high, win):
    features = []

    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    for syl in syllables:
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        cur_image = specgram[start:end, f1:f2]
        mean = np.mean(cur_image)
        std = np.std(cur_image)
        duration = end - start
        features.append([duration * win, mean, std])
    features = np.array(features)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)
    print(y_kmeans)

    return y_kmeans