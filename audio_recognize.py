"""
audio_recognize.py
This file contains the basic functions used for audio clustering and recognition

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans


def blockshaped(arr):
    blocks = []
    blocks.append(arr[0:int(arr.shape[0]/2), 0:int(arr.shape[1]/2)])
    blocks.append(arr[0:int(arr.shape[0]/2), int(arr.shape[1]/2):])
    blocks.append(arr[int(arr.shape[0]/2):, 0:int(arr.shape[1]/2)])
    blocks.append(arr[int(arr.shape[0]/2):, int(arr.shape[1]/2):])
    for b in blocks:
        print(b.shape)
    return blocks



def cluster_syllables(syllables, specgram, sp_freq,
                      f_low, f_high, win):
    features = []

    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    for syl in syllables:
        start = int(syl[0] / win)
        end = int(syl[1] / win)
        cur_image = specgram[start:end, f1:f2]
        print(cur_image.shape)
        cur_blocks = blockshaped(cur_image)
        duration = end - start
        cur_features = [duration]
        for b in cur_blocks:
            cur_features.append(np.mean(b))
            cur_features.append(np.std(b))
        features.append(cur_features)
    features = np.array(features)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)
    print(y_kmeans)

    return y_kmeans