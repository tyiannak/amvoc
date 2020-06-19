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
from dash.dependencies import Input, Output, State
import audio_process as ap
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {'background': '#111111', 'text': '#7FDBFF'}

# These values are selected based on the evaluation.py script and the results
# obtained from running these scripts on two long annotated recordings
ST_WIN = 0.004   # short-term window
ST_STEP = 0.004  # short-term step
MIN_VOC_DUR = 0.005

# The frequencies used for spectral energy calculation (Hz)
F1 = 30000
F2 = 100000


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    return parser.parse_args()


def get_shapes(segments, freq1, freq2):
    # create rectangles to draw syllables
    shapes1, shapes2 = [], []
    for s in segments:
        s1 = {
            'type': 'rect', 'x0': s[0], 'y0': freq1, 'x1': s[1], 'y1': freq2,
            'line': {'color': 'rgba(128, 0, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 0, 128, 0.1)'}
        shapes1.append(s1)
    return shapes1


def get_layout():
    # Also, default thres value is set to 1.2 (this is the optimal based on
    # the same evaluation that led to the parameter set of the
    # short-term window and step
    thres = 1.2
    seg_limits, thres_sm, _ = ap.get_syllables(spectral_energy_2,
                                               spectral_energy_1,
                                               ST_STEP,
                                               threshold_per=thres * 100,
                                               min_duration=MIN_VOC_DUR)
    shapes1 = get_shapes(seg_limits, f_low, f_high)


    layout = html.Div(children=[
        html.H2(children='AMVOC', style={'textAlign': 'center',
                                         'color': colors['text']}),

        html.Div([
            html.Div([
                html.Div(children='Thres = 1.2',
                         style={'textAlign': 'center',
                                'color': colors['text']},
                         id="label_thres"),
                dcc.Slider(
                    id="slider_thres", min=1.1, step=0.05, max=1.5,
                    marks={i: str(i) for i in [1.1, 1.2, 1.3, 1.4, 1.5]},
                    value=1.2)], className="two columns")
            ]),

        html.Div([
            html.Div([
                html.Label(id="label_sel_start", children="Selected start",
                           style={'textAlign': 'center',
                                  'color': colors['text']}),
                html.Label(id="label_sel_end", children="Selected end",
                           style={'textAlign': 'center',
                                  'color': colors['text']}),
                dcc.Dropdown(
                    id='dropdown_class',
                    options=[
                        {'label': 'no-class', 'value': 'no'},
                        {'label': 'class1', 'value': 'c1'},
                        {'label': 'class2', 'value': 'c2'},
                        {'label': 'class3', 'value': 'c3'},
                        {'label': 'class4', 'value': 'c4'},
                    ], value='no'
                ),
            ], className="two columns")
        ], className="row"),

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
                })]),
        # these are intermediate values to be used for sharing content
        # between callbacks
        # (see here https://dash.plotly.com/sharing-data-between-callbacks)
        html.Div(id='intermediate_val_thres', style={'display': 'none'}),
        html.Div(id='intermediate_val_syllables', style={'display': 'none'}),
        html.Div(id='intermediate_val_selected_syllable',
                 style={'display': 'none'})
    ])

    return layout


if __name__ == "__main__":
    args = parse_arguments()

    # feature (spectrogram) extraction:
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                           ST_WIN, ST_STEP)
    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

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
        Output('intermediate_val_thres', component_property='children'),
        [Input('slider_thres', 'value')])
    def update_thres(val):
        return val


    @app.callback([Output('heatmap1', 'figure'),
                   Output('label_thres', 'children')],
                  [Input('intermediate_val_thres', 'children')])
    def update_graph(val):
        # get vocalization syllables from thresholding of the feature sequence
        seg_limits, sm, _ = ap.get_syllables(spectral_energy_2, spectral_energy_1,
                                             ST_STEP,
                                             threshold_per=val*100,
                                             min_duration=MIN_VOC_DUR)
        syllables = [{"st": s[0], "et": s[1], "label": ""} for s in seg_limits]

        with open('annotations.json', 'w') as outfile:
            json.dump(syllables, outfile)

        shapes1 = get_shapes(seg_limits, f_low, f_high)

        fig1 = {
            'data': [go.Heatmap(x=sp_time, y=sp_freq, z=spectrogram.T,
                                name='F', colorscale='Jet',
                                showscale=False)],
            'layout': go.Layout(
                xaxis=dict(title='Time (Sec)'), yaxis=dict(title='Freq (Hz)'),
                shapes=shapes1)
        }

        return fig1, "Thres = {0:.2f}".format(val)


    """
    On-spectrogram-click callback: 
        1) load syllables from json file
        2) find if selected point belongs in segment and 
           "select" segment (syllable)
        3) return:
            3.1) start and end time of selected syllable (update text boxes)
            3.2) update intermediate variable for selected syllable ID
            3.3) read class label of selected syllable 
                 and update class dropdown menu of class name
    """
    @app.callback(
        [Output('label_sel_start', 'children'),
         Output('label_sel_end', 'children'),
         Output('intermediate_val_selected_syllable', 'children'),
         Output('dropdown_class', 'value')],
        [Input('heatmap1', 'clickData')])
    def display_click_data(click_data):
        with open('annotations.json') as json_file:
            syllables = json.load(json_file)
        t1, t2 = 0.0, 0.0
        i_s = -1
        found = False
        if click_data:
            if len(click_data["points"]) > 0:
                t = click_data["points"][0]["x"]
                for i_s, s in enumerate(syllables):
                    if s["st"] < t and s["et"] > t:
                        t1 = s["st"]
                        t2 = s["et"]
                        syllable_label = s["label"]
                        found = True
                        break
        if not found:
            i_s = -1
            syllable_label = ""
        return "{0:.2f}".format(t1), "{0:.2f}".format(t2), \
               "{0:d}".format(i_s), syllable_label



    @app.callback(
        Output('intermediate_val_syllables', 'children'),
        [Input('dropdown_class', 'value'),
         Input('intermediate_val_selected_syllable', 'children')])
    def update_annotations(dropdown_class, selected):
        with open('annotations.json') as json_file:
            syllables = json.load(json_file)
        if dropdown_class and selected:
            syllables[int(selected)]["label"] = dropdown_class
            with open('annotations.json', 'w') as outfile:
                json.dump(syllables, outfile)
        return "{}"


    app.run_server(debug=True)
