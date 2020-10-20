"""
audio_recognize.py
This file contains the basic functions used for audio clustering and recognition

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, AffinityPropagation, Birch, MiniBatchKMeans, OPTICS, SpectralClustering,estimate_bandwidth
from sklearn.svm import SVR
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning


def util_generate_cluster_graphs(list_contour, cluster_ids):
    clusters = np.unique(cluster_ids)
    cluster_plots = [[] for c in range(len(clusters))]
    print(list_contour)
    for il, l in enumerate(list_contour):
        t = l[0]
        print ('t = {}'.format(t))
        y = l[1]
        print('y = {}'.format(y))
        t = t - np.min(t)
        print('new t = {}'.format(t))
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

def metrics(X, y):
    silhouette_avg = silhouette_score(X, y)
    ch_score = calinski_harabasz_score(X,y)
    db_score = davies_bouldin_score(X,y)     
    return silhouette_avg, ch_score, db_score

def cluster_plot(features, y, centers):
    # Tune T-SNE
    # feats = []
    # kl = []
    # iterations = [500, 1000, 2000, 5000]
    # for i in range (4):
    #     tsne = TSNE(n_components=2, perplexity = 30, n_iter = iterations[i])
    #     feats_2d = tsne.fit_transform(features)
    #     feats.append(feats_2d)
    #     kl.append(tsne.kl_divergence_)
    # index = np.argmin(np.array(kl))
    # print(iterations[index])
    # feats_2d = feats[index]

    #Dimension reduction for plotting
    tsne = TSNE(n_components=2, perplexity = 50, n_iter = 5000, random_state = 1)
    features = np.append(features, centers, axis = 0)
    n_clusters = len(np.unique(y))
    feats_2 = tsne.fit_transform(features)
    feats_2d = feats_2[:-n_clusters, :]
    centers_2d = feats_2[-n_clusters:,:]
    fig = go.Figure(layout = go.Layout(title = 'Clustered syllables', xaxis = dict(title = 'x'), yaxis = dict(title = 'y')))
    fig.add_trace(go.Scatter(x = feats_2d[:, 0], y = feats_2d[:, 1], name='',
                     mode='markers',
                     marker=go.scatter.Marker(color=y),
                     showlegend=False))
    
    # Draw Xs at cluster centers
    fig.add_trace(go.Scatter(x = centers_2d[:, 0], y = centers_2d[:, 1], name='',
                     mode='markers',
                     marker=go.scatter.Marker(symbol='x',
                                       size=20,
                                       color=np.arange(len(np.unique(y)))),
                     showlegend=False))
    return fig


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
    warnings.filterwarnings('ignore', category=ConvergenceWarning) 
    features, countour_points, init_points = [], [], []
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))
    freqs = [f1,f2]
    np.save('freqs.npy', freqs)
    images = []
    segments = []
    for syl in syllables:
        # for each detected syllable (vocalization)

        # A. get the spectrogram area in the defined frequency range
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        segments.append([start,end])
        np.save('segments.npy', segments)
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

        feature_mode = 2
        if feature_mode == 1:
            cur_features = [duration,
                            min_freq, max_freq, mean_freq,
                            max_freq_change, min_freq_change,
                            delta_mean, delta_std,
                            delta2_mean, delta2_std,
                            freq_start, freq_end]
        elif feature_mode == 2:
            cur_features = [duration,
                            pos_min_freq,
                            pos_max_freq,
                            (freq_start - freq_end) / mean_freq]
        else:
            import scipy.signal
            desired = 100
            ratio = int(100 * (desired / len(points_f)))
            cur_features = scipy.signal.resample_poly(points_f, up=ratio, down=100)
            cur_features = cur_features[0:desired - 10].tolist()

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
    
    return images, countour_points, \
           init_points, features, feature_names


def center_detection(n_clusters, features, y):
    centroids = [[] for i in range(n_clusters)]
    for i in range(len(y)):
        centroids[y[i]].append(features[i].tolist())
    centers = np.zeros((n_clusters,len(features[0])))
    for i in range (n_clusters):
        centers[i, :] = np.mean(np.array(centroids[i]), axis=0)
    return centers


def cluster_help(clusterer, centers, features, n_clusters):
    y = clusterer.fit_predict(features)
    #In case of Birch clustering, if threshold is too big for generating n_clusters clusters
    if len(np.unique(y))< n_clusters:
        return y,[],[]
    if centers == 1:
        centers = clusterer.cluster_centers_
    else:
        centers = center_detection(n_clusters, features,y)
    scores = metrics(features, y)
    return y, centers, scores


def clustering(method, n_clusters, features):
    if method == 'agg':
        clusterer =  AgglomerativeClustering(n_clusters=n_clusters)
        y, centers, scores = cluster_help(clusterer, 0, features, n_clusters)
    elif method == 'birch':
        thresholds = np.arange(0.1,2.1,0.2)
        sil_scores, ch_scores, db_scores = [], [], []
        #Choosing the best threshold based on metrics results
        for thres in thresholds:
            clusterer = Birch(threshold = thres, n_clusters=n_clusters)
            y, centers, scores = cluster_help(clusterer, 0, features, n_clusters)
            #Stop checking bigger values of threshold
            if len(np.unique(y)) < n_clusters:
                break
            sil_scores.append(scores[0])
            ch_scores.append(scores[1])
            db_scores.append(scores[2])

        sil_ind = np.argsort(np.argsort(sil_scores))
        ch_ind = np.argsort(np.argsort(ch_scores))
        db_ind = np.argsort(np.argsort(db_scores))
        sum = sil_ind + ch_ind- db_ind
        thres = thresholds[np.argmax(sum)]
        scores = [sil_scores[np.argmax(sum)], ch_scores[np.argmax(sum)], db_scores[np.argmax(sum)]]
        clusterer = Birch(threshold = thres, n_clusters = n_clusters)
        y, centers, _ = cluster_help(clusterer,0, features, n_clusters)
    elif method == 'gmm':
        clusterer = GaussianMixture(n_components=n_clusters)
        y, centers, scores = cluster_help(clusterer, 0, features, n_clusters)
    elif method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters)
        y, centers, scores = cluster_help(clusterer, 1, features, n_clusters)
    elif method == 'mbkmeans':
        clusterer = MiniBatchKMeans(n_clusters = n_clusters)
        y, centers, scores = cluster_help(clusterer, 1, features, n_clusters)
    elif method == 'spec':
        clusterer = SpectralClustering(n_clusters = n_clusters)
        y, centers, scores = cluster_help(clusterer, 0, features, n_clusters)
    return y, centers, scores 