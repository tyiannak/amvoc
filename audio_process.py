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
    fs, s = io.read_audio_file(path)
    print(round(fs * win))
    print(round(fs * step))
    cache_name = path + "_{0:.6f}_{1:.6f}.npz".format(win, step)
    if os.path.isfile(cache_name):
        print("LOAD")
        npzfile = np.load(cache_name)
        print(npzfile.files)
        spec_val = npzfile["arr_0"]
        spec_time = npzfile["arr_1"]
        spec_freq = npzfile["arr_2"]
    else:
        spec_val, spec_time, spec_freq = sF.spectrogram(s, fs,
                                                        round(fs * win),
                                                        round(fs * step),
                                                        False, True)
        np.savez(cache_name, spec_val, spec_time, spec_freq)
        print("DONE")
    #    f, f_n  = sF.feature_extraction(s, fs, win * fs / 1000.0,
    #                                    step * fs / 1000.0, deltas=True)
    return spec_val, np.array(spec_time), np.array(spec_freq)
