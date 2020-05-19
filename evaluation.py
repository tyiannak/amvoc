# -*- coding: utf-8 -*-
import argparse
import numpy as np
import plotly
import plotly.graph_objs as go
import audio_process as ap
import csv


def read_ground_truth(filename, offset=10):
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
                    segs.append([float(row[3]) - offset,
                                 float(row[4]) - offset])
                    labels.append(int(row[5]))
    return segs, labels


def temporal_evaluation(s1, s2, duration):
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
    print(recall, precision)

ST_WIN = 0.001   # short-term window
ST_STEP = 0.002  # short-term step
MIN_VOC_DUR = 0.005

# The frequencies used for spectral energy calculation (Hz)
F1 = 30000
F2 = 100000


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
    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

    # define feature sequence for vocalization detection
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    spectral_energy_1 = spectrogram.sum(axis=1)
    spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)

    thres = 1.2
    segs, thres_sm = ap.get_syllables(spectral_energy_2, ST_STEP,
                                      threshold_per=thres * 100,
                                      min_duration=MIN_VOC_DUR)

    segs_gt, f_gt = read_ground_truth(args.ground_truth_file)

    shapes, shapes_gt = [], []
    for s in segs:
        s1 = {
            'type': 'rect', 'x0': s[0], 'y0': f_low, 'x1': s[1], 'y1': f_high,
            'line': {'color': 'rgba(50, 50, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(50, 50, 128, 0.1)'}
        shapes.append(s1)
    for s in segs_gt:
        s1 = {
            'type': 'rect', 'x0': s[0], 'y0': f_low-1000, 'x1': s[1], 'y1': f_low,
            'line': {'color': 'rgba(128, 50, 50, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 50, 50, 0.4)'}
        shapes_gt.append(s1)
    heatmap = go.Heatmap(z=spectrogram.T, y=sp_freq, x=sp_time,  colorscale='Jet')
    layout = go.Layout(title='Evaluation', xaxis=dict(title='time (sec)', ),
                       yaxis=dict(title='Freqs (Hz)'))
    fig = go.Figure(data=[heatmap], layout=layout)
    fig.update_layout(shapes=shapes + shapes_gt)
    plotly.offline.plot(fig, filename="temp.html", auto_open=True)

    temporal_evaluation(segs_gt, segs, 10)