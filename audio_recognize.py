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

    features, countour_points, init_points = [], [], []
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))


    for syl in syllables:
        # for each detected syllable (vocalization)

        # A. get the spectrogram area in the defined frequency range
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        cur_image = specgram[start:end, f1:f2]

        # B. perform frequency contour detection through SVM regression

        # B1. get the positions and values of the maximum frequencies 
        # per time window:
        max_pos = np.argmax(cur_image, axis=1)
        max_vals = np.max(cur_image, axis=1)
        threshold = np.percentile(max_vals, 20)

        point_time, point_freq = [], []
        # B2. keep only the points where the frequencies are larger than the 
        # lower 20% percentile of the highest frequencies of each frame
        # (thresholding)
        for ip in range(1, len(max_vals)-1):
            if max_vals[ip] > threshold:
                point_time.append(ip)
                point_freq.append(max_pos[ip])
                
        # B3. train a regression SVM to map time coordinates to frequency values
        svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr = svr.fit(np.array(point_time).reshape(-1, 1), np.array(point_freq))

        # B4. predict the frequencies for the same time range
        x_new = list(range(min(point_time), max(point_time)+1))
        y_new = svr.predict(np.array(x_new).reshape(-1, 1))
        points_t = [j * win + syl[0] for j in x_new]
        points_f = [sp_freq[int(j + f1)] for j in y_new]
        points_t_init = [j * win + syl[0] for j in point_time]
        points_f_init = [sp_freq[int(j + f1)] for j in point_freq]

        init_points.append([points_t_init, points_f_init])
        countour_points.append([points_t, points_f])

        # C. Extract features based on the frequency contour
        delta = np.diff(points_f)
        delta_2 = np.diff(delta)
        duration = end - start
        max_freq = np.max(points_f)
        min_freq = np.min(points_f)
        mean_freq = np.mean(points_f)
        max_freq_change = np.max(np.abs(delta))
        min_freq_change = np.min(np.abs(delta))
        delta_mean = np.mean(delta)
        delta_std = np.std(delta)
        delta2_mean = np.mean(delta_2)
        delta2_std = np.std(delta_2)
        freq_start = points_f[0]
        freq_end = points_f[-1]

        cur_features = [duration,
                        min_freq, max_freq, mean_freq,
                        max_freq_change, min_freq_change,
                        delta_mean, delta_std,
                        delta2_mean, delta2_std,
                        freq_start, freq_end]

        features.append(cur_features)

    feature_names = ["duration",
                    "min_freq", "max_freq", "mean_freq",
                    "max_freq_change", "min_freq_change",
                    "delta_mean", "delta_std",
                    "delta2_mean", "delta2_std",
                    "freq_start", "freq_end"]

    features = np.array(features)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)

    return y_kmeans, countour_points, init_points, features, feature_names