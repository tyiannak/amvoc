"""
audio_process.py
This file contains the basic functions used for analyizing audio information

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-
from pyAudioAnalysis import audioBasicIO as io
from pyAudioAnalysis import ShortTermFeatures as sF
import numpy as np
import os


def get_spectrogram(path, win, step):
    """
    get_spectrogram() is a wrapper to
    pyAudioAnalysis.ShortTermFeatures.spectrogram() with a caching functionality

    :param path: path of the WAV file to analyze
    :param win: short-term window to be used in spectrogram calculation
    :param step: short-term step to be used in spectrogram calculation
    :return: spectrogram matrix, time array, freq array and sampling freq
    """
    fs, s = io.read_audio_file(path)
    cache_name = path + "_{0:.6f}_{1:.6f}.npz".format(win, step)
    if os.path.isfile(cache_name):
        print("LOAD")
        npzfile = np.load(cache_name)
        spec_val = npzfile["arr_0"]
        spec_time = npzfile["arr_1"]
        spec_freq = npzfile["arr_2"]
    else:
        print("COMPUTING SPECTROGRAM")
        spec_val, spec_time, spec_freq = sF.spectrogram(s, fs,
                                                        round(fs * win),
                                                        round(fs * step),
                                                        False, True)
        np.savez(cache_name, spec_val, spec_time, spec_freq)
        print("DONE")
    #    f, f_n  = sF.feature_extraction(s, fs, win * fs / 1000.0,
    #                                    step * fs / 1000.0, deltas=True)
    return spec_val, np.array(spec_time), np.array(spec_freq), fs


def get_syllables(feature_sequence, win_step, threshold=40,
                  min_duration=0.05):

    threshold = np.percentile(feature_sequence, threshold)
    indices = np.where(feature_sequence > threshold)[0]

    # get the indices of the frames that satisfy the thresholding
    index, seg_limits, time_clusters = 0, [], []

    # group frame indices to onset segments
    while index < len(indices):
        # for each of the detected onset indices
        cur_cluster = [indices[index]]
        if index == len(indices)-1:
            break
        while indices[index+1] - cur_cluster[-1] <= 2:
            cur_cluster.append(indices[index+1])
            index += 1
            if index == len(indices)-1:
                break
        index += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * win_step,
                           cur_cluster[-1] * win_step])

    # post process: remove very small segments:
    seg_limits_2 = []
    for s_lim in seg_limits:
        if s_lim[1] - s_lim[0] > min_duration:
            seg_limits_2.append(s_lim)

    return seg_limits_2

