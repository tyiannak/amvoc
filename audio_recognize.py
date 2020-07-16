"""
audio_recognize.py
This file contains the basic functions used for audio clustering and recognition

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVR
import plotly.graph_objs as go

def util_generate_cluster_graphs(list_contour, cluster_ids):
    clusters = np.unique(cluster_ids)
    cluster_plots = [[] for c in range(len(clusters))]
    for il, l in enumerate(list_contour):
        t = l[0]
        y = l[1]
        t = t - np.min(t)
        if len(cluster_plots[cluster_ids[il]]) == 0:
            cluster_plots[cluster_ids[il]] = ([[t.tolist(), y]])
        else:
            cluster_plots[cluster_ids[il]].append([t.tolist(), y])

    scatter_plots = [[] for c in range(len(clusters))]
    for c in range(len(clusters)):
        L = len(cluster_plots[c])
        required = 10
        perms = np.random.permutation(L)
        for i in perms[0:required]:
            x = cluster_plots[c][i][0]
            y = cluster_plots[c][i][1]
            scatter_plots[c].append(go.Scatter(x=x, y=y,
                                               name="F_{0:d}".format(i)))

    return scatter_plots


def util_generate_cluster_images(list_of_img, cluster_ids):
    clusters = np.unique(cluster_ids)
    cluster_images = []
    for c in clusters:
        cluster_image_width = 500
        time_total = 0
        n_freqs = list_of_img[0].shape[1]

        for i, im in enumerate(list_of_img):
            if cluster_ids[i] == c:
                time_total += im.shape[0]

        n_rows = int(time_total / cluster_image_width) + 1
        cluster_image = np.zeros((n_rows * n_freqs, cluster_image_width))

        count_t = 0
        count_row = 0
        for i, im in enumerate(list_of_img):
            if cluster_ids[i] == c:
                t = im.shape[0]
                if count_t + t > cluster_image_width:
                    count_row +=1
                    count_t = 0
                # append current spectrogram image to larger map:
                cluster_image[count_row * n_freqs: count_row * n_freqs + n_freqs,
                count_t: count_t + t] = im.T
                # create "grid"
                cluster_image[count_row * n_freqs, count_t: count_t + t] = 0.1
                cluster_image[count_row * n_freqs: count_row * n_freqs + n_freqs,
                count_t] = 0.1
                cluster_image[count_row * n_freqs: count_row * n_freqs + n_freqs,
                count_t + t] = 0.1
                count_t += t
        cluster_images.append(cluster_image)
    return(cluster_images)
#        plt.imshow(cluster_image)
#        plt.show()

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
    images = []

    for syl in syllables:
        # for each detected syllable (vocalization)

        # A. get the spectrogram area in the defined frequency range
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        cur_image = specgram[start:end, f1:f2]
        images.append(cur_image)

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
        y_new[y_new + f1 > sp_freq.shape[0]] = sp_freq.shape[0] - f1 - 1
        points_t = [j * win + syl[0] for j in x_new]
        points_f = [sp_freq[int(j + f1)] for j in y_new]
        points_t_init = [j * win + syl[0] for j in point_time]
        points_f_init = [sp_freq[int(j + f1)] for j in point_freq]

        init_points.append([points_t_init, points_f_init])
        countour_points.append([points_t, points_f])

        # C. Extract features based on the frequency contour
        delta = np.diff(points_f)
        delta_2 = np.diff(delta)
        duration = points_t[-1] - points_t[0]
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
        pos_min_freq = (points_t[np.argmin(points_f)] - points_t[0]) / duration
        pos_max_freq = (points_t[np.argmax(points_f)] - points_t[0]) / duration

        cur_features = [duration,
                        min_freq, max_freq, mean_freq,
                        max_freq_change, min_freq_change,
                        delta_mean, delta_std,
                        delta2_mean, delta2_std,
                        freq_start, freq_end]
        """
        cur_features = [pos_min_freq,
                        pos_max_freq,
                        (freq_start - freq_end) / mean_freq]
        """
        """
        import scipy.signal
        desired = 100
        ratio = int(100 * (desired / len(points_f)))
        cur_features = scipy.signal.resample_poly(points_f, up=ratio, down=100)
        cur_features = cur_features[0:desired - 10].tolist()
        """
        features.append(cur_features)

    feature_names = ["duration",
                    "min_freq", "max_freq", "mean_freq",
                    "max_freq_change", "min_freq_change",
                    "delta_mean", "delta_std",
                    "delta2_mean", "delta2_std",
                    "freq_start", "freq_end"]

    features = np.array(features)
    from sklearn import preprocessing
    features = preprocessing.scale(features)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)

    return y_kmeans, images, countour_points, \
           init_points, features, feature_names