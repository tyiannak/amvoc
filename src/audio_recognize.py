"""
audio_recognize.py
This file contains the basic functions used for audio clustering and recognition

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MiniBatchKMeans, estimate_bandwidth
from sklearn.svm import SVR
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from PIL import Image
import statistics

def metrics(X, y):
    silhouette_avg = silhouette_score(X, y)
    ch_score = calinski_harabasz_score(X,y)
    db_score = davies_bouldin_score(X,y)     
    return silhouette_avg, ch_score, db_score


def cluster_syllables(syllables, specgram, sp_freq,
                      f_low, f_high, win, train = False, model_name = '\.model_test', comp=False):
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
    features_s, countour_points, init_points = [], [], []
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))
    freqs = [f1,f2]
    images = []
    segments = []
    max_dur = 0
    test = []
    syllables_final = []

    vec1 = []
    for syl in syllables:
        # for each detected syllable (vocalization)

        # A. get the spectrogram area in the defined frequency range
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        
        cur_image = specgram[start:end, f1:f2]

        if train:
            images.append(cur_image)
            continue
        
        images.append(cur_image)
        segments.append([start,end])
        syllables_final.append(syl)
        if cur_image.shape[0] > max_dur:
            max_dur = cur_image.shape[0]
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
            cur_features = [duration, min_freq, max_freq, mean_freq,
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

        features_s.append(cur_features)

    if train or comp:
        return images

    features_s = StandardScaler().fit_transform(features_s)  

    feature_names = ["duration",
                    "min_freq", "max_freq", "mean_freq",
                    "max_freq_change", "min_freq_change",
                    "delta_mean", "delta_std",
                    "delta2_mean", "delta2_std",
                    "freq_start", "freq_end"]

    init_images = np.array(images, dtype = object)
    
    #crop and normalize images
    time_limit = 64

    for i in range(len(images)):
        if len(images[i])>time_limit:
            images[i] = images[i][int((len(images[i])-time_limit)/2):int((len(images[i])-time_limit)/2)+time_limit,:]/np.amax(images[i])
        elif len(images[i])<time_limit:
            images[i] = np.pad(images[i]/np.amax(images[i]), ((int((time_limit-images[i].shape[0])/2), (time_limit-images[i].shape[0]) - int((time_limit-images[i].shape[0])/2)),(0,0)))
        else:
            images[i] = images[i]/np.amax(images[i])
    
    specs = np.array(images)
    specs = specs.reshape(specs.shape[0], 1, specs.shape[1], specs.shape[2])
    model = torch.load(model_name, map_location=torch.device('cpu'))
    dataset = TensorDataset(torch.tensor(specs, dtype = torch.float))
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            outputs += model(data[0], False)

    for i in range(len(outputs)):
        outputs[i] = outputs[i].detach().numpy().flatten()

    outputs=np.array(outputs)
    features = outputs
    selector = VarianceThreshold(threshold=(1.2*np.mean(np.var(features, axis = 0))))

    features = selector.fit_transform(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    test = min(100,features.shape[0], features.shape[1])
    n_comp = 0
    while (1):
        pca = PCA(n_components=test, random_state=9)
        pca.fit(features)
        evar = pca.explained_variance_ratio_
        cum_evar = np.cumsum(evar)
        n_comp = np.where(cum_evar >= 0.95)
        if not list(n_comp[0]):
            test = test + 50
        else:
            n_comp = n_comp[0][0] + 1
            break
    pca = PCA(n_components=n_comp, random_state=9)
    features = pca.fit_transform(features)
    features_d = features
    
    return images, list(init_images), countour_points, \
           init_points, [features_s, features_d, outputs], feature_names, freqs, segments, syllables_final,\
           selector, scaler, pca


def cluster_help(clusterer, features, n_clusters):
    y = clusterer.fit_predict(features)
    #In case of Birch clustering, if threshold is too big for generating n_clusters clusters
    if len(np.unique(y))< n_clusters:
        return y,[]
    scores = metrics(features, y)
    return y, scores


def clustering(method, n_clusters, features):
    if method == 'agg':
        clusterer =  AgglomerativeClustering(n_clusters=n_clusters)
        y, scores = cluster_help(clusterer, features, n_clusters)
    elif method == 'birch':
        clusterer = Birch(n_clusters = n_clusters)
        y, scores = cluster_help(clusterer,features, n_clusters)
    elif method == 'gmm':
        clusterer = GaussianMixture(n_components=n_clusters, random_state=9)
        y, scores = cluster_help(clusterer,features, n_clusters)
    elif method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=9)
        y, scores = cluster_help(clusterer, features, n_clusters)
    elif method == 'mbkmeans':
        clusterer = MiniBatchKMeans(n_clusters = n_clusters, random_state=9)
        y, scores = cluster_help(clusterer, features, n_clusters)
    return y, scores 