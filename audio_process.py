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


def get_syllables(spectral_en, total_en,  win_step,crit=0, threshold_per=40,
                  min_duration=0.02, threshold_buf=None, prev_time_frames=None):
    """
    The basic vocalization (syllable) detection method
    :param spectral_en: the input feature sequence (spectral energy)
    :param total_en: the secondary feature sequence
           (i.e. the total spectral energy)
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
        if prev_time_frames:
            threshold = threshold_per * (0.3 * np.mean(threshold_buf[-prev_time_frames:]) + 
                                        0.7 * np.mean(spectral_en)) / 100.0
        else:
            threshold = threshold_per * (0.3 * np.mean(threshold_buf) + 
                                        0.7 * np.mean(spectral_en)) / 100.0

    # Step 2: spectral energy ratio computation:
    C = 0.01
    filter_size_smooth = int(0.02 / win_step)
    smooth_filter = np.ones(filter_size_smooth) / filter_size_smooth
    spectral_ratio = (spectral_en + C) / (total_en + C)
    spectral_ratio = np.convolve(spectral_ratio, smooth_filter, mode="same")
    # print(np.where(spectral_en-np.mean(spectral_en) > np.sqrt(np.var(spectral_en))))
    # plt.subplot(1,2,1)
    # plt.plot(spectral_en)
    # plt.plot(np.mean(spectral_en)*np.ones(spectral_en.shape))
    # plt.subplot(1,2,2)
    # # plt.plot(crit, '-o')
    # plt.show()
    # Step 3: thresholding
    # (get the indices of the frames that satisfy the thresholding criteria:
    # (a) spectral energy is higher than the dynamic threshold and
    # (b) spectral ratio (spectral energy by total energy is
    #     larger than the mean spectral ratio)
    # mean_spectral_ratio = spectral_ratio.mean()
    # TODO: This value is optimized for F1, F2 = 30 - 110 KHz and should be
    # TODO: recalculated if changed !
    mean_spectral_ratio = 0.69
    # ratio = np.var(crit[2]/np.amax(crit[2]), axis=1)
    
    # /np.mean(crit[2]/np.amax(crit[2]), axis=1)
    # print(np.mean(spectral_en/np.amax(spectral_en)))
    # print(np.var(spectral_en/np.amax(spectral_en)))
    indices = np.argmax(crit[2], axis=1)
    ind_down = np.argmax(crit[2], axis=1) - 30
    change = np.where(ind_down<0)[0]
    ind_down[change]=0
    ind_up = np.argmax(crit[2], axis=1) + 30
    change = np.where(ind_up>crit[2].shape[1]-1)[0]
    ind_up[change]=crit[2].shape[1]-1
    means = []
    for i in range(spectral_en.shape[0]):
        means.append(np.mean(crit[2][i,ind_down[i]:ind_up[i]]))
    means = np.array(means)
    # print(np.mean(crit[2])-np.amax(crit[2]))
    # plt.plot(np.amax(crit[2], axis=1)- means)
    # plt.show()
    is_vocal = ((spectral_en > threshold)&
                ((np.amax(crit[2], axis=1)/np.mean(crit[2], axis=1) > 4.25)&
                (np.amax(crit[2], axis=1)/means > 3))
                # ())
    # (spectral_ratio > mean_spectral_ratio))
                # ((spectral_en/np.amax(spectral_en))-np.mean(spectral_en/np.amax(spectral_en)) > 1.5*(np.var(spectral_en/np.amax(spectral_en)))))
                # (ratio>10**(-6) ))
                
                # (spectral_en-np.mean(spectral_en) > np.var(spectral_en)))
                # (crit>0))
    
    # Step 4: smooth 
    is_vocal = np.convolve(is_vocal, smooth_filter, mode="same")
    indices = np.where(is_vocal)[0]
    # print(var[indices])
    
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
        # if (np.var(crit[2][cur_cluster]/np.amax(crit[2][cur_cluster]))<0.05) or (np.var(crit[0][cur_cluster]-np.mean(crit[0][cur_cluster]))>3500):
        #     continue
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * win_step - win_step,
                           cur_cluster[-1] * win_step + win_step])
    # print(time_clusters)
    # print(crit)
    # for cluster in time_clusters:
    #     print(np.var(crit[2][cluster]/np.amax(crit[2][cluster]))/np.mean(crit[2][cluster]/np.amax(crit[2][cluster])))
    #     print(np.var(np.argmax(crit[2][cluster], axis =0)))
    #     temp = np.where(spectral_en[time_clusters[i]]>np.mean(spectral_en[time_clusters[i]]))[0]
    #     print(np.var(crit[0][cluster])/crit[0][cluster].shape[0])
        # print(np.var(crit[0][cluster]-np.mean(crit[cluster])))
        # print('Time mean: {} var: {}, Freq mean: {} var: {}'.format(np.mean(crit[1][cluster]/np.amax(crit[1][cluster])), np.var(crit[1][cluster]/np.amax(crit[1][cluster])), np.mean(crit[2][cluster]/np.amax(crit[2][cluster])), np.var(crit[2][cluster]/np.amax(crit[2][cluster]))))
    # Step 6: post process (remove very small segments)
    seg_limits_2 = []
    for i, s_lim in enumerate(seg_limits):
        if s_lim[1] - s_lim[0] > min_duration:
            # temp = np.where(spectral_en[time_clusters[i]]>np.mean(spectral_en[time_clusters[i]]))[0]
            # ratio = np.var(crit[2][time_clusters[i]]/np.amax(crit[2][time_clusters[i]]))/np.mean(crit[2][time_clusters[i]]/np.amax(crit[2][time_clusters[i]]))
            # total_ratio = np.var(crit[2]/np.amax(crit[2]))/np.mean(crit[2]/np.amax(crit[2]))
            # print(ratio)
            # if ratio > 0.03:
                # print('Ratio {}'.format(np.var(crit[2][time_clusters[i]]/np.amax(crit[2][time_clusters[i]]))/np.mean(crit[2][time_clusters[i]]/np.amax(crit[2][time_clusters[i]]))))
                seg_limits_2.append(s_lim)

    return seg_limits_2, threshold, spectral_ratio
