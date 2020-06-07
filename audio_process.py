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


def get_syllables(spectral_en, total_en, win_step, threshold_per=40,
                  min_duration=0.02):
    """
    The basic vocalization (syllable) detection method
    :param spectral_en: the input feature sequence (spectral energy)
    :param total_en: the secondary feature sequence
           (i.e. the total spectral energy)
    :param win_step: window step (in msecs) used in feature extraction
    :param threshold_per: threshold parameter (percentage)
    :param min_duration: minimum vocalization duration
    :return:
      1) segment limits of detected syllables
      2) dynamic threshold sequence
    """
    # Step 1: dynamic threshold computation (threshold is a sequence):
    global_mean = np.mean(spectral_en)
    filter_size = int(2 / win_step)
    smooth_filter = np.ones(filter_size) / filter_size
    threshold = threshold_per * (0.5 * np.convolve(spectral_en,
                                                   smooth_filter, mode="same") +
                                 0.5 * global_mean) / 100.0

    # spectral energy ratio-related threshold
    C = 0.01
    filter_size = 5
    smooth_filter = np.ones(filter_size) / filter_size
    spectral_ratio = (spectral_en + C) / (total_en + C)
    spectral_ratio = np.convolve(spectral_ratio, smooth_filter, mode="same")

    # Step 2: run the thresholding
    # (get the indices of the frames that satisfy the thresholding criteria:
    # (a) spectral energy is higher than the dynamic threshold and
    # (b) spectral ratio (spectral energy by total energy is > 0.6)
    indices1 = ((spectral_en > threshold) & (spectral_ratio > 0.6))
    indices1 = np.convolve(indices1, smooth_filter, mode="same")

    indices = np.where(indices1)[0]
    # smooth decisions:
    indices_temp = []
    for i, iv in enumerate(indices):
        if ((iv + 1) in indices) or ((iv - 1)  in indices):
            indices_temp.append(iv)
    indices = indices_temp


    # Step 3: window indices to segment limits
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
        seg_limits.append([cur_cluster[0] * win_step - win_step,
                           cur_cluster[-1] * win_step + win_step])

    # Step 4: post process (remove very small segments)
    seg_limits_2 = []
    for s_lim in seg_limits:
        if s_lim[1] - s_lim[0] > min_duration:
            seg_limits_2.append(s_lim)

    return seg_limits_2, threshold, spectral_ratio
