"""


Instructions:

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-
import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
from pyAudioAnalysis import audioBasicIO as io
from pyAudioAnalysis import ShortTermFeatures as sF
import numpy as np
import plotly.graph_objs as go
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


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


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Filler detection demo")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    sp, sp_time, sp_freq = get_spectrogram(args.input_file, 0.005, 0.002)
    print(sp.shape)
    print(sp_time.shape)
    print(sp_freq.shape)
    f1 = np.argmin(np.abs(sp_freq - 25000))
    f2 = np.argmin(np.abs(sp_freq - 85000))
    print(f1, f2)
    spectral_energy_1 = sp.sum(axis=1)
    spectral_energy_2 = sp[:, f1:f2].sum(axis=1)
    threshold = np.percentile(spectral_energy_2, 50)
    indices = np.where(spectral_energy_2 > threshold)[0]

    # get the indices of the frames that satisfy the thresholding
    index = 0
    seg_limits = []
    time_clusters = []

    # Step 4B: group frame indices to onset segments
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
        seg_limits.append([cur_cluster[0] * 0.002,
                           cur_cluster[-1] * 0.002])
    print(seg_limits)
    # Step 5: Post process: remove very small segments:
    min_duration = 0.05
    seg_limits_2 = []
    for s_lim in seg_limits:
        if s_lim[1] - s_lim[0] > min_duration:
            seg_limits_2.append(s_lim)
    print(seg_limits_2)
    seg_limits = seg_limits_2

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='AMVOC'),
        html.Div(children='''Analysis of Mouse Vocal Communication'''),
        dcc.Graph(
            id='heatmap1',
            figure={
                'data': [go.Heatmap(x=sp_time, y=sp_freq, z=sp.T,
                                    name='F', colorscale='Jet',
                                    showscale=False)],
                'layout': go.Layout(
                     xaxis=dict(title='Time (Sec)'),
                     yaxis=dict(title='Freq (Hz)')
                )}),
        dcc.Graph(
            id='energy',
            figure={
                'data': [go.Scatter(x=sp_time, y=spectral_energy_1),
                         go.Scatter(x=sp_time, y=spectral_energy_2)],
                'layout': go.Layout(
                    xaxis=dict(title='Time (Sec)'),
                    yaxis=dict(title='Energy'))})
    ])
    app.run_server(debug=True)
