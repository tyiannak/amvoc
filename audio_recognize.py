"""
audio_recognize.py
This file contains the basic functions used for audio clustering and recognition

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVR

def cluster_syllables(syllables, specgram, sp_freq,
                      f_low, f_high, win):
    """
    TODO
    :param syllables:
    :param specgram:
    :param sp_freq:
    :param f_low:
    :param f_high:
    :param win:
    :return:
    """

    features = []

    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))
    images = []

    countour_points = []

    for syl in syllables:
        # for each detected syllable (vocalization)

        # A. get the spectrogram area in the defined frequency range
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        cur_image = specgram[start:end, f1:f2]

        # B. perform frequency contour detection through linear regression

        max_pos = np.argmax(cur_image, axis=1)
        max_vals = np.max(cur_image, axis=1)
        mean_max = np.mean(max_vals)

        interpoints_x = []
        interpoints_y = []
        for ip in range(1, len(max_vals)-1):
            if max_vals[ip] > mean_max:
                interpoints_x.append(ip)
                interpoints_y.append(max_pos[ip])
        svr_poly = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_poly = svr_poly.fit(np.array(interpoints_x).reshape(-1, 1),
                                np.array(interpoints_y))
        x_new = list(range(min(interpoints_x), max(interpoints_x)+1))
        y_new = svr_poly.predict(np.array(x_new).reshape(-1, 1))
        points_t = [j * win + syl[0] for j in x_new]
        points_f = [sp_freq[int(j + f1)] for j in y_new]
        countour_points.append([points_t, points_f])
        duration = end - start
        cur_features = [duration,
                        np.max(points_f) - np.min(points_f),
                        points_f[0] - points_f[-1]]
        features.append(cur_features)
    features = np.array(features)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)

    return y_kmeans, countour_points, features