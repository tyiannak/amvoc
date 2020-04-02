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
colors = {'background': '#111111', 'text': '#7FDBFF'}

ST_WIN = 0.005  # short-term window
ST_STEP = 0.002  # short-term step
#st_step_vis = ST_STEP * 5

F1 = 0.2
F2 = 0.8


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    return parser.parse_args()


def get_shapes(segments, freq1, freq2, max_t):
    # create rectangles to draw syllables
    shapes1, shapes2 = [], []
    for s in segments:
        s1 = {
            'type': 'rect', 'x0': s[0], 'y0': freq1, 'x1': s[1], 'y1': freq2,
            'line': {'color': 'rgba(128, 0, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 0, 128, 0.1)'}
        shapes1.append(s1)
        s2 = {
            'type': 'rect', 'x0': s[0], 'y0': 0, 'x1': s[1], 'y1': max_t,
            'line': {'color': 'rgba(128, 0, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 0, 128, 0.4)'}
        shapes2.append(s2)
    return shapes1, shapes2


def get_layout():
    thres = 0.4
    seg_limits = ap.get_syllables(spectral_energy_2, ST_STEP,
                                  threshold=thres * 100,
                                  min_duration=0.05)
    shapes1, shapes2 = get_shapes(seg_limits, f_low, f_high,
                                  spectral_energy_1.max())

    layout = html.Div(children=[
        html.H1(children='AMVOC', style={'textAlign': 'center',
                                         'color': colors['text']}),
        html.Div(children='Analysis of Mouse Vocal Communication',
                 style={'textAlign': 'center', 'color': colors['text']}),

        dcc.Slider(
            id="slider_thres", min=0.3, step=0.05, max=0.7,
            marks={i: str(i) for i in [0.3, 0.4, 0.5, 0.6, 0.7]},
            value=0.4,
        ),
        html.Div(id='thres_text'),

        html.Div([
            dcc.Graph(
                id='heatmap1',
                figure={
                    'data': [go.Heatmap(x=sp_time, y=sp_freq, z=spectrogram.T,
                                        name='F', colorscale='Jet',
                                        showscale=False)],
                    'layout': go.Layout(
                        xaxis=dict(title='Time (Sec)'),
                        yaxis=dict(title='Freq (Hz)'),
                        shapes=shapes1)
                }),
            dcc.Graph(
                id='energy',
                figure={
                    'data': [go.Scatter(x=sp_time, y=spectral_energy_1),
                             go.Scatter(x=sp_time, y=spectral_energy_2)],
                    'layout': go.Layout(
                        xaxis=dict(title='Time (Sec)'),
                        yaxis=dict(title='Energy'), showlegend=False,
                        shapes=shapes2)
                }
            )]),
        html.Div(id='intermediate-value', style={'display': 'none'})
    ])

    return layout


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

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = get_layout()

    @app.callback(
        Output('intermediate-value', component_property='children'),
        [Input('slider_thres', 'value')])
    def update_thres(val):
        return val


    @app.callback([Output('heatmap1', 'figure'),
                   Output('energy', 'figure')],
                  [Input('intermediate-value', 'children')])
    def update_graph(val):
        # get vocalization syllables from thresholding of the feature sequence
        seg_limits = ap.get_syllables(spectral_energy_2, ST_STEP,
                                      threshold=val*100, min_duration=0.05)
        shapes1, shapes2 = get_shapes(seg_limits, f_low, f_high,
                                      spectral_energy_1.max())

        fig1 = {
            'data': [go.Heatmap(x=sp_time, y=sp_freq, z=spectrogram.T,
                                name='F', colorscale='Jet',
                                showscale=False)],
            'layout': go.Layout(
                xaxis=dict(title='Time (Sec)'), yaxis=dict(title='Freq (Hz)'),
                shapes=shapes1)
        }

        fig2 = {
            'data': [go.Scatter(x=sp_time, y=spectral_energy_1),
                     go.Scatter(x=sp_time, y=spectral_energy_2)],
            'layout': go.Layout(
                xaxis=dict(title='Time (Sec)'),
                yaxis=dict(title='Energy'), showlegend=False,
                shapes=shapes2)
        }
        return fig1, fig2


    app.run_server(debug=True)
