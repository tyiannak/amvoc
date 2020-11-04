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
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, (3,3), padding =(1,1))  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 8, (3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(8, 8, (3,3), padding=(1,1))
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool1 = nn.MaxPool2d((2,4), 2)
        self.pool2 = nn.MaxPool2d((2,2), 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 1, 2, stride=(2,2))

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # compressed representation
        x = F.relu(self.conv3(x))
        x=self.pool2(x)
        if self.training:
            ## decode ##
            # add transpose conv layers, with relu activation function
            x = F.relu(self.t_conv1(x))
            # # output layer (with sigmoid for scaling from 0 to 1)
            x = F.relu(self.t_conv2(x))
            # print(x.shape)
            x = F.sigmoid(self.t_conv3(x))                
        return x


def metrics(X, y):
    silhouette_avg = silhouette_score(X, y)
    ch_score = calinski_harabasz_score(X,y)
    db_score = davies_bouldin_score(X,y)     
    return silhouette_avg, ch_score, db_score


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
    # np.save('freqs.npy', freqs)
    images = []
    segments = []
    max_dur = 0
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

    '''
    train_data = []
    max_dur = ((max_dur + 7) & (-8)) 
    for i in range(len(images)):
        train_data.append(np.pad(images[i]/np.amax(cur_image), ((int((max_dur-images[i].shape[0])/2), (max_dur-images[i].shape[0]) - int((max_dur-images[i].shape[0])/2)),(0,0))))

    train_data=np.array(train_data)
    a = np.array(train_data)
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
    train_data  = torch.from_numpy(train_data)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    model = ConvAutoencoder()
    # specify loss function
    criterion = nn.BCELoss()
    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # number of epochs to train the model
    n_epochs = 200
    train_data = torch.autograd.Variable(train_data)
    model = model.float()
    train_loss = 10.0
    epoch = 0
    # print(train_data)
    while (train_loss >= 0.001):
    # train_loss = 0.0
    # epoch = 0
    # while(train_loss >= 0):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        
        # _ stands in for labels, here
        model.train()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(train_data.float())
        # print(outputs)
        # print(train_data.float())
        # calculate the loss
        loss = criterion(outputs, train_data.float())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss = loss.item()
        epoch+=1
        # # print avg training statistics 
        # train_loss = train_loss/len(train_data)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
    print(outputs.shape)
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(a[1].T)
    fig.add_subplot(1,2,2)
    outputs = outputs.detach().numpy()
    plt.imshow(outputs.reshape(outputs.shape[0], outputs.shape[2],outputs.shape[3])[1].T)
    plt.show()
    model.eval()
    outputs = model(train_data.float())
    outputs = outputs.detach().numpy()
    print(outputs.shape)
    outputs = np.mean(outputs, axis=1)
    outputs = outputs.reshape(outputs.shape[0], -1)
    pca = PCA(n_components=0.8)
    feats = pca.fit_transform(outputs)
    print(feats[0])
    # print(outputs.reshape(outputs.shape[0], -1).shape)
    features = np.array(features)
    # features = feats
    # print(features.shape)
    '''
    features = np.array(features)
    from sklearn import preprocessing
    features = preprocessing.scale(features)
    
    return images, countour_points, \
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