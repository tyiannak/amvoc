"""


Instructions:

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-
import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import audio_process as ap

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Filler detection demo")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    sp, sp_time, sp_freq = ap.get_spectrogram(args.input_file, 0.005, 0.002)
    print(sp.shape)
    print(sp_time.shape)
    print(sp_freq.shape)
    f1 = np.argmin(np.abs(sp_freq - 25000))
    f2 = np.argmin(np.abs(sp_freq - 85000))
    print(f1, f2)
    spectral_energy_1 = sp.sum(axis=1)
    spectral_energy_2 = sp[:, f1:f2].sum(axis=1)
    threshold = np.percentile(spectral_energy_2, 40)
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
    max_e = spectral_energy_1.max()
    seg_limits = seg_limits_2

    shapes1, shapes2 = [], []
    for s in seg_limits_2:
        s1 = {
            'type': 'rect', 'x0': s[0], 'y0': 0, 'x1': s[1], 'y1': max_e,
            'line': {'color': 'rgba(128, 0, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 0, 128, 0.4)'}
        shapes1.append(s1)
        s2 = {
            'type': 'rect', 'x0': s[0], 'y0': 20000, 'x1': s[1], 'y1': 100000,
            'line': {'color': 'rgba(128, 0, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 0, 128, 0.1)'}
        shapes2.append(s2)

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    colors = colors = {
    'background': '#111111',
    'text': '#7FDBFF'
    }

    app.layout = html.Div(children=[
        html.H1(children='AMVOC',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div(children='Analysis of Mouse Vocal Communication',
                 style={
                     'textAlign': 'center',
                     'color': colors['text']
                 }),

        html.Label('Slider'),
        dcc.Slider(
            id="slider_thres",
            min=0,
            max=9,
            marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in
                   range(1, 6)},
            value=5,
        ),
        html.Div(id='thres_text'),

        html.Div([
        dcc.Graph(
            id='heatmap1',
            figure={
                'data': [go.Heatmap(x=sp_time, y=sp_freq, z=sp.T,
                                    name='F', colorscale='Jet',
                                    showscale=False)],
                'layout': go.Layout(
                     xaxis=dict(title='Time (Sec)'),
                     yaxis=dict(title='Freq (Hz)'),
                     shapes=shapes2
                )}),
        dcc.Graph(
            id='energy',
            figure={
                'data': [go.Scatter(x=sp_time, y=spectral_energy_1),
                         go.Scatter(x=sp_time, y=spectral_energy_2)],
                'layout': go.Layout(
                    xaxis=dict(title='Time (Sec)'),
                    yaxis=dict(title='Energy'), showlegend=False,
                    shapes=shapes1)})])
        ])


    @app.callback(
        Output('thres_text', component_property='children'),
        [Input('slider_thres', 'value')])
    def update_thres(val):
        return val

    app.run_server(debug=True)
