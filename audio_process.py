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
from scipy import ndimage
import matplotlib.pyplot as plt


def get_spectrogram(path, win, step, disable_caching=True, smooth=True):
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
    if not disable_caching and os.path.isfile(cache_name):
        print("Loading cached spectrogram")
        npzfile = np.load(cache_name)
        spec_val = npzfile["arr_0"]
        spec_time = npzfile["arr_1"]
        spec_freq = npzfile["arr_2"]
    else:
        print("Computing spectrogram")
        spec_val, spec_time, spec_freq = sF.spectrogram(s, fs,
                                                        round(fs * win),
                                                        round(fs * step),
                                                        False, True)
        if not disable_caching:
            np.savez(cache_name, spec_val, spec_time, spec_freq)
    #    f, f_n  = sF.feature_extraction(s, fs, win * fs / 1000.0,
    #                                    step * fs / 1000.0, deltas=True)
    if smooth:
        spec_val = ndimage.median_filter(spec_val, (2, 3))

    return spec_val, np.array(spec_time), np.array(spec_freq), fs


def get_spectrogram_buffer(s, fs, win, step, smooth=True):
    """
    get_spectrogram_buffer() same as get_spectrogram() but input is an audio
    buffer, instead of an audio file
    """
    spec_val, spec_time, spec_freq = sF.spectrogram(s, fs,
                                                    round(fs * win),
                                                    round(fs * step),
                                                    False, True)
    if smooth:
        spec_val = ndimage.median_filter(spec_val, (2, 3))

    return spec_val, np.array(spec_time), np.array(spec_freq), fs



def clean_spectrogram(spectrogram):
    """
    returns a normalized spectrogram where peaks are more clear
    (for visualization purposes)
    :param spectrogram:
    :return:
    """
    new_spectrogram = np.copy(spectrogram)
    for i in range(new_spectrogram.shape[0]):
        new_spectrogram[i, :] /= sum(new_spectrogram[i, :])
    return new_spectrogram

def prepare_features(spectrogram):
    """
    Calculates useful features for the vocalization detection
    :param spectrogram: the spectrogram for the vocalization detection, only the frequency range of interest
    :return 
        1) spectral_energy
        2) means: mean spectral energy of a range of 60 kHz around the maximum frequency in each time frame
        3) maximum energy value for each time frame 
    """
    # 1.calculate spectral energy 
    spectral_energy = spectrogram.sum(axis=1)

    # find frequency of maximum value for each time frame
    indices = np.argmax(spectrogram, axis=1)
    # keep a frequency window of 60 kHz around the frequency corrsponding to max 
    ind_down = np.argmax(spectrogram, axis=1) - 30
    ind_up = np.argmax(spectrogram, axis=1) + 30

    # remove points that are out of frequency range 
    change = np.where(ind_down < 0)[0]
    ind_down[change]=0
    
    change = np.where(ind_up > spectrogram.shape[1]-1)[0]
    ind_up[change]=spectrogram.shape[1]-1

    # 2.calculate the mean of the spectrogram values in the defined frequency window for each time frame 
    means = []
    for i in range(spectral_energy.shape[0]):
        means.append(np.mean(spectrogram[i,ind_down[i]:ind_up[i]]))
    means = np.array(means)

    # 3.calculate the maximum values of each time frame
    max_values = np.amax(spectrogram, axis=1)

    return spectral_energy, means, max_values


def get_syllables(spectral_en, means, max_values, win_step, threshold_per=40, factor=3.5,
                  min_duration=0.02, threshold_buf=None):
    """
    The basic vocalization (syllable) detection method
    :param spectral_en: the input feature sequence (spectral energy)
    :param means: mean spectral energy of a range of 60 kHz around the maximum frequency in each time frame
    :param max_values: maximum energy value for each time frame 
    :param win_step: window step (in msecs) used in feature extraction
    :param threshold_per: threshold parameter (percentage)
    :param min_duration: minimum vocalization duration
    :param threshold_buf: this optional argument is the buffer of the previous
    average spectral energy values. It is to be used in online mode (where
    information on the whole recording is not available and therefore threshold
    calculation must be incremental)
    :return:
      1) segment limits of detected syllables
      2) dynamic threshold sequence
    """
    # Step 1: dynamic threshold computation (threshold is a sequence) for 
    # spectral energy:
    if threshold_buf is None:
        global_mean = np.mean(spectral_en)
        filter_size = int(2 / win_step)
        moving_avg_filter = np.ones(filter_size) / filter_size
        threshold = threshold_per * (0.5 * np.convolve(spectral_en,
                                                       moving_avg_filter,
                                                       mode="same") +
                                     0.5 * global_mean) / 100.0
    else:
        threshold = threshold_per * (0.3 * np.mean(threshold_buf) + 
                                    0.7 * np.mean(spectral_en)) / 100.0

    # Step 2: define a smoothing filter 
    filter_size_smooth = int(0.02 / win_step)
    smooth_filter = np.ones(filter_size_smooth) / filter_size_smooth

    # Step 3: thresholding
    # (get the indices of the frames that satisfy the thresholding criteria:
    # (a) spectral energy is higher than the dynamic threshold and
    # (b) maximum energy is higher than mean energy by a factor of 3.5

    is_vocal = ((spectral_en > threshold)&
                (max_values/means > factor))

    # Step 4: smooth 
    is_vocal = np.convolve(is_vocal, smooth_filter, mode="same")
    indices = np.where(is_vocal)[0]
    
    # Step 5: window indices to segment limits
    index, seg_limits, time_clusters = 0, [], []

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
        if cur_cluster[0]==0:
            seg_limits[-1][0]=0
    
    # Step 6: post process (remove very small segments)
    seg_limits_2 = []
    for i, s_lim in enumerate(seg_limits):
        if s_lim[1] - s_lim[0] > min_duration:
            # merge successive vocalizations with time difference less than 11 ms
            if i>0 and s_lim[0]-seg_limits[i-1][1]<=0.011:
                seg_limits_2[-1][1] = s_lim[1]
            else:
                seg_limits_2.append(s_lim)

    return seg_limits_2, threshold
