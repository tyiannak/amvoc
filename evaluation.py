# -*- coding: utf-8 -*-
import argparse
import numpy as np
import plotly.graph_objs as go
import audio_process as ap
import json
import csv


def read_ground_truth(filename):
    """
    :param filename: the path of the CSV segment file
    :return:
      - seg_start:  a np array of segments' start positions
      - seg_end:    a np array of segments' ending positions
    """
    with open(filename, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter=',')
        segs = []
        labels = []
        for ir, row in enumerate(reader):
            if ir > 0:
                if len(row) == 6:
                    segs.append([float(row[3]), float(row[4])])
                    labels.append(int(row[5]))
    return segs, labels


ST_WIN = 0.001   # short-term window
ST_STEP = 0.002  # short-term step
MIN_VOC_DUR = 0.005

# The frequencies used for spectral energy calculation (Fs/2 normalized):
F1 = 0.3
F2 = 0.8


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="Audio file")
    parser.add_argument("-g", "--ground_truth_file", required=True, nargs=None,
                        help="Ground truth file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # feature (spectrogram) extraction:
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                           ST_WIN, ST_STEP)

    f_low, f_high = F1 * fs / 2.0, F2 * fs / 2.0

    print(f_low, f_high)
    #    spectrogram = spectrogram[0::5, 0::5]
    #    spectrogram_time = spectrogram_time[0::5]
    #    spectrogram_freq = spectrogram_freq[0::5]
    #    st_step = ST_STEP / 5

    # define feature sequence for vocalization detection
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    spectral_energy_1 = spectrogram.sum(axis=1)
    spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)

    thres = 1.1
    segs, thres_sm = ap.get_syllables(spectral_energy_2, ST_STEP,
                                            threshold_per=thres * 100,
                                            min_duration=MIN_VOC_DUR)

    segs_gt, f_gt = read_ground_truth(args.ground_truth_file)
    
