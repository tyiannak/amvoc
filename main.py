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
import audio_recognize as ar
import json
import dash_bootstrap_components as dbc
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F


colors = {'background': '#111111', 'text': '#7FDBFF'}

# These values are selected based on the evaluation.py script and the results
# obtained from running these scripts on two long annotated recordings
ST_WIN = 0.002    # short-term window
ST_STEP = 0.002   # short-term step
MIN_VOC_DUR = 0.005

# The frequencies used for spectral energy calculation (Hz)
F1 = 30000
F2 = 110000

# Also, default thres value is set to 1.3 (this is the optimal based on
# the same evaluation that led to the parameter set of the
# short-term window and step
thres = 1.

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 64), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 64, 3, padding =1)  
        # conv layer (depth from 64 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        # conv layer (depth from 32 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 8, 3, padding=1)

        self.pool = nn.MaxPool2d((2,2), 2)

        ## decoder layers ##
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x) # compressed representation

        if self.training:
            ## decode ##
            # add transpose conv layers, with relu activation function
            x = F.relu(self.t_conv1(x))        
            x = F.relu(self.t_conv2(x))
            # output layer (with sigmoid for scaling from 0 to 1)
            x = F.sigmoid(self.t_conv3(x))      
        return x

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    return parser.parse_args()


def get_shapes(segments, freq1, freq2):
    # create rectangles to draw syllables
    shapes1 = []
    for s in segments:
        s1 = {
            'type': 'rect', 'x0': s[0], 'y0': freq1, 'x1': s[1], 'y1': freq2,
            'line': {'color': 'rgba(128, 0, 128, 1)', 'width': 2},
            'fillcolor': 'rgba(128, 0, 128, 0.1)'}
        shapes1.append(s1)
    return shapes1


def get_layout():

    global list_contour, segments, images, f1, f2, feats_simple, feats_deep, feats_2d_s, feats_2d_d, seg_limits, syllables
    seg_limits, thres_sm, _ = ap.get_syllables(spectral_energy_2,
                                               spectral_energy_1,
                                               ST_STEP,
                                               threshold_per=thres * 100,
                                               min_duration=MIN_VOC_DUR)
    images, f_points, f_points_init, \
    [feats_simple, feats_deep], feat_names, [f1, f2], segments, seg_limits = ar.cluster_syllables(seg_limits, spectrogram,
                                             sp_freq, f_low, f_high,  ST_STEP)

    #Dimension reduction for plotting
    # Tune T-SNE
    # feats = []
    # kl = []
    # iterations = [500, 1000, 2000, 5000]
    # for i in range (4):
    #     tsne = TSNE(n_components=2, perplexity = 30, n_iter = iterations[i])
    #     feats_2d = tsne.fit_transform(features)
    #     feats.append(feats_2d)
    #     kl.append(tsne.kl_divergence_)
    # index = np.argmin(np.array(kl))
    # print(iterations[index])

    tsne = TSNE(n_components=2, perplexity = 50, n_iter = 5000, random_state = 1)
    feats_2d_s = tsne.fit_transform(feats_simple)
    tsne = TSNE(n_components=2, perplexity = 50, n_iter = 5000, random_state = 1)
    feats_2d_d = tsne.fit_transform(feats_deep)
    list_contour = np.array(f_points, dtype=object)
    images = np.array(images, dtype=object) 
    f_points_all, f_points_init_all = [[], []], [[], []]
    

    for iS in range(len(seg_limits)):
        f_points_all[0] += f_points[iS][0]
        f_points_all[1] += f_points[iS][1]
        f_points_init_all[0] += f_points_init[iS][0]
        f_points_init_all[1] += f_points_init[iS][1]
    shapes2, shapes3 = [], []
    for x, y in zip(f_points_all[0], f_points_all[1]):
        s1 = {
            'type': 'rect',
            'x0': x - ST_STEP / 5,
            'y0': y - 1000,
            'x1': x + ST_STEP / 5,
            'y1': y + 1000,
            'line': {'color': 'rgba(128, 0, 0, 1)', 'width': 1},
            'fillcolor': 'rgba(128, 0, 0, 1)'}
        shapes2.append(s1)

    for x, y in zip(f_points_init_all[0], f_points_init_all[1]):
        s1 = {
            'type': 'rect',
            'x0': x - ST_STEP / 15,
            'y0': y - 200,
            'x1': x + ST_STEP / 15,
            'y1': y + 200,
            'line': {'color': 'rgba(0, 128, 0, 1)', 'width': 1},
            'fillcolor': 'rgba(128, 128, 0, 1)'}
        shapes3.append(s1)
    
    
    syllables = [{"st": s[0], "et": s[1]}
                 for iS, s in enumerate(seg_limits)]

    shapes1 = get_shapes(seg_limits, f_low, f_high)
    layout = dbc.Container([
        # Title
         dbc.Row(dbc.Col(html.H2("AMVOC", style={'textAlign': 'center',
                                        'color': colors['text'], 'marginBottom': 30, 'marginTop':30}))),

        # # Selected segment controls
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             html.Label(id="label_sel_start", children="Selected start",
        #                        style={'textAlign': 'center',
        #                                'color': colors['text']}),
        #             width=1,
        #         ),
        #         dbc.Col
        #             (
        #             html.Label(id="label_sel_end", children="Selected end",
        #                        style={'textAlign': 'center',
        #                               'color': colors['text']}),
        #             width=1,
        #         ),
        #         dbc.Col(
        #             html.Label(
        #                 id='label_class',
        #                 children="Class",
        #                        style={'textAlign': 'center',
        #                                'color': colors['text']}
        #             ),
        #             width=1,
        #         )
        #     ], className="h-5"),

        # # Main heatmap
        # dbc.Row(dbc.Col(
        #     dcc.Graph(
        #         id='heatmap1',
        #         figure={
        #             'data': [go.Heatmap(x=sp_time[::spec_resize_ratio_time],
        #                                 y=sp_freq[::spec_resize_ratio_freq],
        #                                 z=clean_spectrogram[::spec_resize_ratio_time,
        #                                   ::spec_resize_ratio_freq].T,
        #                                 name='F', colorscale='Jet',
        #                                 showscale=False)],
        #             'layout': go.Layout(
        #                 title = 'Spectrogram of the signal',
        #                 margin=dict(l=55, r=20, b=120, t=40, pad=4),
        #                 xaxis=dict(title='Time (Sec)'),
        #                 yaxis=dict(title='Freq (Hz)'),
        #                 shapes=shapes1 + shapes2 + shapes3)
        #         }), width=12,
        #     style={"height": "100%", "background-color": "white"}),
        #     className="h-50",
        # ),
        dbc.Row([dbc.Col(
                dcc.Dropdown(
                    id='dropdown_cluster',
                    options=[
                        {'label': 'Agglomerative', 'value': 'agg'},
                        {'label': 'Birch', 'value': 'birch'},
                        {'label': 'Gaussian Mixture', 'value': 'gmm'},
                        {'label': 'K-Means', 'value': 'kmeans'},
                        {'label': 'Mini-Batch K-Means', 'value': 'mbkmeans'},
                        {'label': 'Spectral', 'value': 'spec'},
                    ], value='agg'
                ),
                width=2,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id='dropdown_n_clusters',
                    options=[{'label': i, 'value': i} for i in range(2,11)
                    ], value=2
                ),
                width=2,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id='dropdown_feats_type',
                    options=[
                        {'label': 'Simple', 'value': 'simple'},
                        {'label': 'Deep', 'value': 'deep'},
                    ], value='simple'
                ),
                width=2,
            ),
            html.Table([
            html.Tr([html.Td(['Silhouette score']), html.Td(id='silhouette')]),
            html.Tr([html.Td(['Calinski-Harabasz score']), html.Td(id='cal-har')]),
            html.Tr([html.Td(['Davies-Bouldin score']), html.Td(id='dav-bould')]),
        ]),]),
        dbc.Row(dbc.Col(
            dcc.Graph(id='cluster_graph')
        )),
        dbc.Row([dbc.Col(
        dcc.Graph(id='spectrogram', hoverData = {'points': [{'pointIndex': 0}]}),width = 6
        
        ), 
        dbc.Col(
            dcc.Graph(id='contour_plot', hoverData = {'points': [{'pointIndex': 0}]}), width = 6
        )]),
    ], style={"height": "100vh"})
    return layout


if __name__ == "__main__":
    args = parse_arguments()
    global sp_time, sp_freq
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                           ST_WIN, ST_STEP)


    # These should change depending on the signal's size
    spec_resize_ratio_freq = 4
    spec_resize_ratio_time = 4

    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

    # define feature sequence for vocalization detection
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    spectral_energy_1 = spectrogram.sum(axis=1)
    spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)
    app = dash.Dash(
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    clean_spectrogram = ap.clean_spectrogram(spectrogram)
    app.layout = get_layout()

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
    # @app.callback(
    #     [Output('label_sel_start', 'children'),
    #      Output('label_sel_end', 'children'),
    #      Output('label_class', 'children')
    #      ],
    #     [Input('heatmap1', 'clickData')])
    # def display_click_data(click_data):
    #     t1, t2 = 0.0, 0.0
    #     i_s = -1
    #     found = False
    #     if click_data:
    #         if len(click_data["points"]) > 0:
    #             t = click_data["points"][0]["x"]
    #             for i_s, s in enumerate(syllables):
    #                 if s["st"] < t and s["et"] > t:
    #                     t1 = s["st"]
    #                     t2 = s["et"]
    #                     syllable_label = 'class {}'.format(labels[i_s])
    #                     found = True
    #                     break
    #     if not found:
    #         i_s = -1
    #         syllable_label = ""
    #     return "{0:.2f}".format(t1), "{0:.2f}".format(t2), syllable_label
               

    @app.callback(
        [Output('cluster_graph', 'figure'),
         Output('silhouette', 'children'),
         Output('cal-har', 'children'),
         Output('dav-bould', 'children')],
        [Input('dropdown_cluster', 'value'),
         Input('dropdown_n_clusters', 'value'),
         Input('dropdown_feats_type', 'value')])
    def update_cluster_graph(method, n_clusters, feats_type):
        global labels
        if feats_type == 'simple':
            y, scores = ar.clustering(method, n_clusters, feats_simple)
            labels = y
            fig = go.Figure(data = go.Scatter(x = feats_2d_s[:, 0], y = feats_2d_s[:, 1], name='',
                        mode='markers',
                        marker=go.scatter.Marker(color=y),
                        showlegend=False),layout = go.Layout(title = 'Clustered syllables', xaxis = dict(title = 'x'), yaxis = dict(title = 'y')))
        elif feats_type == 'deep':
            y, scores = ar.clustering(method, n_clusters, feats_deep)
            labels = y
            fig = go.Figure(data = go.Scatter(x = feats_2d_d[:, 0], y = feats_2d_d[:, 1], name='',
                        mode='markers',
                        marker=go.scatter.Marker(color=y),
                        showlegend=False),layout = go.Layout(title = 'Clustered syllables', xaxis = dict(title = 'x'), yaxis = dict(title = 'y')))
        return fig, round(scores[0],3), round(scores[1]), round(scores[2],3)

    @app.callback(
        Output('spectrogram', 'figure'),
        [Input('cluster_graph', 'hoverData')]
    )
    def display_hover_data(hoverData):
        if hoverData:
            index = hoverData['points'][0]['pointIndex']
        else:
            index = 0
        fig = go.Figure(data = go.Heatmap(x =sp_time[segments[index][0]:segments[index][1]], y=sp_freq[f1:f2], z = images[index].T, zmin = np.amin(images[index]), 
                                          zmax = np.amax(images[index])+(np.amax(images[index])-np.amin(images[index])), showscale=False),
                        layout = go.Layout(title = 'Spectrogram of syllable', margin={'l': 0, 'b': 40, 't': 40, 'r': 0}, 
                                           xaxis = dict(range=[(sp_time[segments[index][0]]+ sp_time[segments[index][1]])/2-0.1, (sp_time[segments[index][0]]+ sp_time[segments[index][1]])/2+0.1], 
                                                        title = 'Time (Sec)'), yaxis=dict(range = [sp_freq[0], sp_freq[-1]], title='Freq (Hz)')))
        return fig

    @app.callback(
        Output('contour_plot', 'figure'),
        [Input('cluster_graph', 'hoverData')]
    )
    def display_hover_data(hoverData):
        if hoverData:
            index = hoverData['points'][0]['pointIndex']
        else:
            index = 0
        fig = go.Figure(data = go.Scatter(x = list_contour[index][0], y=list_contour[index][1], mode='lines+markers'), 
                        layout = go.Layout(title = 'Points of max frequency per time window of syllable', margin=dict(l=0, r=0, b=40, t=40, pad=4), 
                                           xaxis=dict(visible=True, title = 'Time (Sec)'), yaxis=dict(visible=True, autorange=False, range=[sp_freq[0], sp_freq[-1]], title='Freq (Hz)')))

        return fig

    app.run_server()
