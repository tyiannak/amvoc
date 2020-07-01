# -*- coding: utf-8 -*-
import argparse
import numpy as np
import plotly
import plotly.graph_objs as go
import audio_process as ap
import csv
import os

# Global params
MIN_VOC_DUR = 0.005
# The frequencies used for spectral energy calculation (Hz)
F1 = 30000
F2 = 110000

def read_ground_truth(filename, offset=0):
    """
    :param filename: the path of the CSV segment file
    :return:
      - seg_start:  a np array of segments' start positions
      - seg_end:    a np array of segments' ending positions
    """
    extension = (os.path.splitext(filename)[1])
    if extension==".mupet":
        with open(filename, 'rt') as f_handle:
            reader = csv.reader(f_handle, delimiter=',')
            segs = []
            labels = []
            for ir, row in enumerate(reader):
                if ir > 0:
                    if len(row) == 6:
                        segs.append([float(row[3]) - offset,
                                     float(row[4]) - offset])
                        labels.append(int(row[5]))
    elif extension==".msa":
        with open(filename, 'rt') as f_handle:
            reader = csv.reader(f_handle, delimiter=',')
            segs = []
            labels = []
            for ir, row in enumerate(reader):
                if ir > 0:
                    if len(row) == 16:
                        segs.append([float(row[14]) - 0.06,
                                     float(row[15]) - 0.06])
                        labels.append((row[1]))
    return segs, labels


def temporal_evaluation(s1, s2, duration):
    """
    Temporal evaluation of agreement between sequences s1 and s2
    :param s1: sequence of decisions s1
    :param s2: sequence of decisions s2
    :param duration: duration of each sequence element (in seconds)
    :return: f1 metric of the 2nd class
    """
    time_resolution = 0.001
    t = 0
    cm = np.zeros((2, 2))
    while t < duration:
        found_f1 = 0
        for i1 in range(len(s1)):
            if (t <= s1[i1][1]) and (t > s1[i1][0]):
                found_f1 = 1
                break
        found_f2 = 0
        for i2 in range(len(s2)):
            if (t <= s2[i2][1]) and (t > s2[i2][0]):
                found_f2 = 1
                break
        t += time_resolution
        cm[found_f1, found_f2] += 1
    print(cm)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    f1 = 2 * recall * precision / (recall + precision)
    return f1


def event_evaluation(s1, s2):
    """
    Event-level evaluation of agreement between sequences s1 and s2
    :param s1: sequence of decisions s1
    :param s2: sequence of decisions s2
    :return: event-level f1 metric
    """
    used_1 = [0] * len(s1)
    found_1 = [0] * len(s1)
    used_2 = [0] * len(s2)
    found_2 = [0] * len(s2)

    for i in range(len(s1)):
        for j in range(len(s2)):
            if not used_2[j]:
                if (s2[j][0] >= s1[i][0] and s2[j][0] <= s1[i][1]) or \
                   (s2[j][1] >= s1[i][0] and s2[j][1] <= s1[i][1]) or \
                   (s2[j][0] <  s1[i][0] and s2[j][1] > s1[i][1]):
                    found_1[i] = 1
                    used_2[j] = 1

    for i in range(len(s2)):
        for j in range(len(s1)):
            if not used_1[j]:
                if (s1[j][0] >= s2[i][0] and s1[j][0] <= s2[i][1]) or \
                   (s1[j][1] >= s2[i][0] and s1[j][1] <= s2[i][1]) or \
                   (s1[j][0] <  s2[i][0] and s1[j][1] > s2[i][1]):
                    found_2[i] = 1
                    used_1[j] = 1

    correct1 = (sum(found_1) / len(found_1))
    correct2 = (sum(found_2) / len(found_2))
    harmonic_mean = 2 * correct1 * correct2 / (correct1 + correct2)
    return harmonic_mean


def restricted_float_short_term_window(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal"
                                         % (x,))
    if x < 0.0001 or x > 0.1:
        raise argparse.ArgumentTypeError("%r not in range [0.0001, 0.1]"%(x,))
    return x


def restricted_float_threshold(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" %
                                         (x,))
    if x < 0.5 or x > 2.0:
        raise argparse.ArgumentTypeError("%r not in range [0.5, 2.0]"%(x,))
    return x


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="Audio file")
    parser.add_argument("-w", "--win", type=restricted_float_short_term_window,
                        help="Short-term window size (for spectrogram)",
                        default=0.002)
    parser.add_argument("-s", "--step", type=restricted_float_short_term_window,
                        help="Short-term window step (for spectrogram)",
                        default=0.002)
    parser.add_argument("-t", "--threshold",
                        type=restricted_float_threshold,
                        help="Threshold factor",
                        default=1)
    parser.add_argument("--resize_freq",
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        type=int,
                        help="Resize factor in the frequency domain " +
                             "(for lighter visualization of the spectrogram)",
                        default=1)
    parser.add_argument("--resize_time",
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        type=int,
                        help="Resize factor in the time domain " +
                             "(for lighter visualization of the spectrogram)",
                        default=1)

    parser.add_argument("-g", "--ground_truth_file", required=True, nargs=None,
                        help="Ground truth file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    thres = args.threshold
    spec_resize_ratio_freq = args.resize_freq
    spec_resize_ratio_time = args.resize_time

    # feature (spectrogram) extraction:
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                           args.win, args.step)
    duration = spectrogram.shape[0] * args.step

    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

    # define feature sequence for vocalization detection
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    spectral_energy_1 = spectrogram.sum(axis=1)
    spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)

    segs, thres_sm, spectral_ratio = ap.get_syllables(spectral_energy_2,
                                                      spectral_energy_1,
                                                      args.step,
                                                      threshold_per=thres * 100,
                                                      min_duration=MIN_VOC_DUR)

    segs_gt, f_gt = read_ground_truth(args.ground_truth_file)

    shapes, shapes2, shapes_gt, shapes_gt2 = [], [], [], []

    for s in segs:
        s1 = {
            'type': 'rect',
            'x0': s[0],
            'y0': f_low,
            'x1': s[1],
            'y1': f_high,
            'line': {'color': 'rgba(50, 50, 255, 1)', 'width': 2},
            'fillcolor': 'rgba(50, 50, 255, 0.1)'}
        s2 = {
            'type': 'rect',
            'x0': s[0],
            'y0': -0.05,
            'x1': s[1],
            'y1': 0.0,
            'line': {'color': 'rgba(50, 50, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(50, 50, 128, 0.1)'}
        shapes.append(s1)
        shapes2.append(s2)
    for s in segs_gt:
        s1 = {
            'type': 'rect',
            'x0': s[0],
            'y0': (f_low-1000),
            'x1': s[1],
            'y1': f_low,
            'line': {'color': 'rgba(128, 50, 50, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 50, 50, 0.4)'}
        s2 = {
            'type': 'rect',
            'x0': s[0],
            'y0': -0.1,
            'x1': s[1],
            'y1': -0.05,
            'line': {'color': 'rgba(128, 50, 50, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 50, 50, 0.4)'}

        shapes_gt.append(s1)
        shapes_gt2.append(s2)
    clean_spectrogram = ap.clean_spectrogram(spectrogram)
    heatmap = go.Heatmap(z=clean_spectrogram[::spec_resize_ratio_time,
                           ::spec_resize_ratio_freq].T,
                         y=sp_freq[::spec_resize_ratio_freq],
                         x=sp_time[::spec_resize_ratio_time],
                         colorscale='Jet')
    layout = go.Layout(title='Evaluation', xaxis=dict(title='time (sec)', ),
                       yaxis=dict(title='Freqs (Hz)'))

    fig = go.Figure(data=[heatmap], layout=layout)
    fig.update_layout(shapes=shapes + shapes_gt)
    out_file = args.input_file + "{0:.4f}_{1:.4f}_{2:.2f}".format(args.win,
                                                                  args.step,
                                                                  thres)
    plotly.offline.plot(fig, filename=out_file + "_spec.html",
                        auto_open=True)

    fig2 = go.Figure(data=[go.Scatter(x=sp_time[::spec_resize_ratio_time],
                                      y=spectral_energy_1[::spec_resize_ratio_time],
                                      name="Energy"),
                     go.Scatter(x=sp_time[::spec_resize_ratio_time],
                                y=spectral_energy_2[::spec_resize_ratio_time],
                                name="Spectral Energy"),
                     go.Scatter(x=sp_time[::spec_resize_ratio_time],
                                y=spectral_ratio[::spec_resize_ratio_time],
                                name="Spectral Ratio"),
                     go.Scatter(x=sp_time[::spec_resize_ratio_time],
                                y=thres_sm[::spec_resize_ratio_time],
                                name="Threshold")],
                     layout=layout)
    fig2.update_layout(shapes=shapes2 + shapes_gt2)
    plotly.offline.plot(fig2, filename=out_file + "_plots.html",
                        auto_open=True)

    accuracy_temporal = temporal_evaluation(segs_gt, segs, duration)
    accuracy_event = event_evaluation(segs_gt, segs)

    print(thres, MIN_VOC_DUR, accuracy_temporal, accuracy_event)