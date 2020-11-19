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
from sklearn.feature_selection import VarianceThreshold
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning
import torch
import torch.nn as nn
from helper import ConvAutoencoder
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader



def metrics(X, y):
    silhouette_avg = silhouette_score(X, y)
    ch_score = calinski_harabasz_score(X,y)
    db_score = davies_bouldin_score(X,y)     
    return silhouette_avg, ch_score, db_score


def cluster_syllables(syllables, specgram, sp_freq,
                      f_low, f_high, win, train = False):
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
    # np.save('freqs.npy', freqs)
    images = []
    segments = []
    max_dur = 0
    # i=0
    test = []
    for syl in syllables:
        # for each detected syllable (vocalization)

        # A. get the spectrogram area in the defined frequency range
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        segments.append([start,end])
        # np.save('segments.npy', segments)
        cur_image = specgram[start:end, f1:f2]
        images.append(cur_image)
        if cur_image.shape[0] > max_dur:
            max_dur = cur_image.shape[0]
        if train:
        #     test.append([np.mean(cur_image/np.amax(cur_image)),np.var(cur_image), np.mean(cur_image-np.amax(cur_image))])
            continue
        # if i<50:
        #     print(np.mean(cur_image-np.amax(cur_image)))
        
        # # variance.append(np.var(cur_image))
            # fig = plt.figure()
            # plt.imshow(cur_image)
            # plt.show()
        # test.append([np.mean(cur_image/np.amax(cur_image)),np.var(cur_image), np.mean(cur_image-np.amax(cur_image))])
        # B. perform frequency contour detection through SVM regression
        # i+=1
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
        # print("Length:{}".format(len(points_t)))
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
    if train:
        # clusterer = KMeans(n_clusters=2)
        # y = clusterer.fit_predict(test)
        # im = np.array(images)
        # im1 = im[y==0]
        # # print(im1.shape)
        # im2 = im[y==1]
        # # print(im2.shape)
        # if np.var(im1[0])< np.var(im2[0]):
        #     im = im2
        # else:
        #     im= im1
        # return list(im)
        return images
    
    # for image in images:
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.show()
            

    feature_names = ["duration",
                    "min_freq", "max_freq", "mean_freq",
                    "max_freq_change", "min_freq_change",
                    "delta_mean", "delta_std",
                    "delta2_mean", "delta2_std",
                    "freq_start", "freq_end"]

    init_images = np.array(images)
    # images = []
    max_dur = ((max_dur + 7) & (-8)) 
    print(max_dur)
    time_limit = max_dur
    for i in range(len(images)):
        if len(images[i])>time_limit:
            images[i] = images[i][int((len(images[i])-time_limit)/2):int((len(images[i])-time_limit)/2)+time_limit,:]/np.amax(images[i])
        elif len(images[i])<time_limit:
            images[i] = np.pad(images[i]/np.amax(images[i]), ((int((time_limit-images[i].shape[0])/2), (time_limit-images[i].shape[0]) - int((time_limit-images[i].shape[0])/2)),(0,0)))
        else:
            images[i] = images[i]/np.amax(images[i])
    specs = np.array(images)
    specs = specs.reshape(specs.shape[0], 1, specs.shape[1], specs.shape[2])
    model = torch.load('./model_1')
    dataset = TensorDataset(torch.tensor(specs, dtype = torch.float))
    # specs  = torch.from_numpy(specs)
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # print(data[0])
            outputs += model(data[0])

    # outputs = model(images.float())
    # outputs = outputs.detach().numpy()
    for i in range(len(outputs)):
        outputs[i] = outputs[i].detach().numpy().flatten()

    outputs=np.array(outputs)
    # print(outputs.shape)
    # print(np.var(outputs, axis = 0))
    # print(max(np.var(outputs, axis = 0)))
    selector = VarianceThreshold(threshold=0.2)
    outputs = selector.fit_transform(outputs)
    # from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    # features = preprocessing.scale(features)
    features = MinMaxScaler().fit_transform(features)

    # outputs = np.mean(outputs, axis=1)
    # outputs = outputs.reshape(outputs.shape[0], -1)
    # print(outputs.shape)
    pca = PCA(n_components=10)
    feats = pca.fit_transform(outputs)
    feats = outputs
    print(feats[0])
    # print(outputs.reshape(outputs.shape[0], -1).shape)
    features = np.array(features)
    # features = feats
    # print(features.shape)
    
    # features = np.array(features)
    
    
    return list(init_images), countour_points, \
           init_points, features, feature_names, freqs, segments


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
        thresholds = np.arange(0.1,2.1,0.2)
        sil_scores, ch_scores, db_scores = [], [], []
        #Choosing the best threshold based on metrics results
        for thres in thresholds:
            clusterer = Birch(threshold = thres, n_clusters=n_clusters)
            y, scores = cluster_help(clusterer,features, n_clusters)
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
        y, scores = cluster_help(clusterer,features, n_clusters)
    elif method == 'gmm':
        clusterer = GaussianMixture(n_components=n_clusters)
        y, scores = cluster_help(clusterer,features, n_clusters)
    elif method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters)
        y, scores = cluster_help(clusterer, features, n_clusters)
    elif method == 'mbkmeans':
        clusterer = MiniBatchKMeans(n_clusters = n_clusters)
        y, scores = cluster_help(clusterer, features, n_clusters)
    elif method == 'spec':
        clusterer = SpectralClustering(n_clusters = n_clusters)
        y, scores = cluster_help(clusterer, features, n_clusters)

    return y, scores 