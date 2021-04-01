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
from dash_table import DataTable
import audio_process as ap
import audio_recognize as ar
import utils
import dash_bootstrap_components as dbc
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sys import exit
import json
import csv

colors = {'background': '#111111', 'text': '#7FDBFF'}

config_data = utils.load_config("config.json")
ST_WIN = config_data['params']['ST_WIN']
ST_STEP = config_data['params']['ST_STEP']
MIN_VOC_DUR = config_data['params']['MIN_VOC_DUR']
F1 = config_data['params']['F1']
F2 = config_data['params']['F2']
thres = config_data['params']['thres']
factor = config_data['params']['factor']


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 64), 3x3 kernels
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
    parser.add_argument("-c", "--continue_", required=True, nargs=None,
                        help="Decision")
    parser.add_argument("-s", "--spectrogram", required=False, nargs=None,
                        help="Condition")
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


def get_layout(spec=False):

    global list_contour, segments, images, f1, f2, feats_simple, feats_deep, feats_2d_s, feats_2d_d, seg_limits, syllables
    seg_limits, thres_sm = ap.get_syllables(spectral_energy,
                                               means,
                                               max_values,
                                               ST_STEP,
                                               threshold_per=thres * 100,
                                               factor=factor,
                                               min_duration=MIN_VOC_DUR)
    time_end = time.time()
    print("Time needed for vocalizations detection: {} s".format(round(time_end-time_start, 1)))
    continue_ = args.continue_
    with open('offline_vocalizations.csv', 'w') as fp:
            for iS, s in enumerate(seg_limits):
                fp.write(f'{s[0]},'
                        f'{s[1]}\n')   
    if continue_=="n":
        exit()       

    images, f_points, f_points_init, \
    [feats_simple, feats_deep], feat_names, [f1, f2], segments, seg_limits = ar.cluster_syllables(seg_limits, spectrogram,
                                             sp_freq, f_low, f_high,  ST_STEP)

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
    if spec:
        layout = dbc.Container([
            # Title
            dbc.Row(dbc.Col(html.H2("AMVOC", style={'textAlign': 'center',
                                            'color': colors['text'], 'marginBottom': 30, 'marginTop':30}))),

            # Selected segment controls
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(id="label_sel_start", children="Selected start",
                                style={'textAlign': 'center',
                                        'color': colors['text']}),
                        width=1,
                    ),
                    dbc.Col
                        (
                        html.Label(id="label_sel_end", children="Selected end",
                                style={'textAlign': 'center',
                                        'color': colors['text']}),
                        width=1,
                    ),
                    dbc.Col(
                        html.Label(
                            id='label_class',
                            children="Class",
                                style={'textAlign': 'center',
                                        'color': colors['text']}
                        ),
                        width=1,
                    )
                ], className="h-5"),

            # Main heatmap
            dbc.Row(dbc.Col(
                dcc.Graph(
                    id='heatmap1',
                    figure={
                        'data': [go.Heatmap(x=sp_time[::spec_resize_ratio_time],
                                            y=sp_freq[::spec_resize_ratio_freq],
                                            z=clean_spectrogram[::spec_resize_ratio_time,
                                            ::spec_resize_ratio_freq].T,
                                            name='F', colorscale='Jet',
                                            showscale=False)],
                        'layout': go.Layout(
                            title = 'Spectrogram of the signal',
                            margin=dict(l=55, r=20, b=120, t=40, pad=4),
                            xaxis=dict(title='Time (Sec)'),
                            yaxis=dict(title='Freq (Hz)'),
                            shapes=shapes1 + shapes2 + shapes3)
                    }), width=12,
                style={"height": "100%", "background-color": "white"}),
                className="h-50",
            ),
            dbc.Row([dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_cluster',
                        options=[
                            {'label': 'Agglomerative', 'value': 'agg'},
                            {'label': 'Birch', 'value': 'birch'},
                            {'label': 'Gaussian Mixture', 'value': 'gmm'},
                            {'label': 'K-Means', 'value': 'kmeans'},
                            {'label': 'Mini-Batch K-Means', 'value': 'mbkmeans'},
                            # {'label': 'Spectral', 'value': 'spec'},
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
                            {'label': 'Method 1', 'value': 'deep'},
                            {'label': 'Method 2', 'value': 'simple'},
                        ], value='deep'
                    ),
                    width=2,
                ),
                html.Table([
                html.Tr([html.Td(['Silhouette score']), html.Td(id='silhouette')]),
                html.Tr([html.Td(['Calinski-Harabasz score']), html.Td(id='cal-har')]),
                html.Tr([html.Td(['Davies-Bouldin score']), html.Td(id='dav-bould')]),
            ]),
            ]),
            dbc.Row([dbc.Col(html.Div(children = "Global cluster annotations"), width=3, style={'marginBottom': 20, 'marginTop': 20}),
                    dbc.Col(html.Div(children = "Specific cluster annotations"), width=3, style={'marginBottom': 20, 'marginTop': 20}),
                    dbc.Col(html.Div(children = "Point annotations"), width=3, style={'marginBottom': 20, 'marginTop': 20}),
            ]),
            dbc.Row([
            dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_total_cluster_annotation',
                        options=[
                            {'label': 'No Validation', 'value': 'no'}]+
                            [{'label': i, 'value': i} for i in np.arange(1,6)], value = 'no'
                    ),
                    width=2,
                    style={'display': 'block'}
                ),
            dbc.Col(     
                html.Button('Submit', id='btn_3', n_clicks=0),  
                    width=1,
                    style={'display': 'block'}
            ),
            
            dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_cluster_annotation',
                        options=[
                            {'label': 'No Validation', 'value': 'no'}]+
                            [{'label': i, 'value': i} for i in np.arange(1,6)], value = 'no'
                    ),
                    width=2,
                    style={'display': 'block'}
                ),
            dbc.Col(     
                html.Button('Submit', id='btn_2', n_clicks=0),  
                    width=1,
                    style={'display': 'block'}
            ),
            dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_point_annotation',
                        options=[
                            {'label': 'No Validation', 'value': 'no'},
                            {'label': 'Approve', 'value': 'approve'},
                            {'label': 'Reject', 'value': 'reject'},
                        ], value = 'no'
                    ),
                    width=2,
                    style={'display': 'block'}
                ),
            dbc.Col(     
                html.Button('Submit', id='btn_1', n_clicks=0),  
                    width=1,
                    style={'display': 'block'}
            ),
            ]),
            dbc.Row([dbc.Col(
                dcc.Graph(id='cluster_graph'), width = 9, md = 8, style={'marginLeft': 0}),
                 dbc.Col(
                    html.Div(children=[html.Div([ DataTable(id='total_annotation', style_cell={'whiteSpace': 'normal','height': 'auto','width': 100},
                    columns = [{'id': 'Global annotation', 'name': 'Global annotation'} ])],style={'marginBottom':10}),
                    DataTable(id='cluster_table', style_cell={'whiteSpace': 'normal','height': 'auto','width': 100},columns = [{'id': column, 'name': column} for column in ['Clusters', 'Cluster annotation', 'Annotated points']])]),
                    style = {'marginTop':10, 'marginLeft': 5, 'marginRight':0}, width ='25%', 
            ), 
            ],justify='start'),
            dbc.Row([dbc.Col(
            dcc.Graph(id='spectrogram', hoverData = {'points': [{'pointIndex': 0}]}),width = 6, style= {'marginTop': 30}
            
            ), 
            dbc.Col(
                dcc.Graph(id='contour_plot', hoverData = {'points': [{'pointIndex': 0}]}), width = 6, style={'marginTop': 30}
            )]),
            # these are intermediate values to be used for sharing content
            # between callbacks
            # (see here https://dash.plotly.com/sharing-data-between-callbacks)
            dbc.Row(id='intermediate_val_syllables', style={'display': 'none'}),
            dbc.Row(id='intermediate_val_total_clusters', style={'display': 'none'}),
            dbc.Row(id='intermediate_val_clusters', style={'display': 'none'}),
            dbc.Row(id='clustering_info',
                     style={'display': 'none'})
        ], style={"height": "100vh"})
    else:
        layout = dbc.Container([
            # Title
            dbc.Row(dbc.Col(html.H2("AMVOC", style={'textAlign': 'center',
                                            'color': colors['text'], 'marginBottom': 30, 'marginTop':30}))),
            dbc.Row([dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_cluster',
                        options=[
                            {'label': 'Agglomerative', 'value': 'agg'},
                            {'label': 'Birch', 'value': 'birch'},
                            {'label': 'Gaussian Mixture', 'value': 'gmm'},
                            {'label': 'K-Means', 'value': 'kmeans'},
                            {'label': 'Mini-Batch K-Means', 'value': 'mbkmeans'},
                            # {'label': 'Spectral', 'value': 'spec'},
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
                            {'label': 'Method 1', 'value': 'deep'},
                            {'label': 'Method 2', 'value': 'simple'},
                        ], value='deep'
                    ),
                    width=2,
                ),
                html.Table([
                html.Tr([html.Td(['Silhouette score']), html.Td(id='silhouette')]),
                html.Tr([html.Td(['Calinski-Harabasz score']), html.Td(id='cal-har')]),
                html.Tr([html.Td(['Davies-Bouldin score']), html.Td(id='dav-bould')]),
            ]),
            ]),
            dbc.Row([dbc.Col(html.Div(children = "Global cluster annotations"), width=3, style={'marginBottom': 20, 'marginTop': 20}),
                    dbc.Col(html.Div(children = "Specific cluster annotations"), width=3, style={'marginBottom': 20, 'marginTop': 20}),
                    dbc.Col(html.Div(children = "Point annotations"), width=3, style={'marginBottom': 20, 'marginTop': 20}),
            ]),
            dbc.Row([
            dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_total_cluster_annotation',
                        options=[
                            {'label': 'No Validation', 'value': 'no'}]+
                            [{'label': i, 'value': i} for i in np.arange(1,6)], value = 'no'
                    ),
                    width=2,
                    style={'display': 'block'}
                ),
            dbc.Col(     
                html.Button('Submit', id='btn_3', n_clicks=0),  
                    width=1,
                    style={'display': 'block'}
            ),
            
            dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_cluster_annotation',
                        options=[
                            {'label': 'No Validation', 'value': 'no'}]+
                            [{'label': i, 'value': i} for i in np.arange(1,6)], value = 'no'
                    ),
                    width=2,
                    style={'display': 'block'}
                ),
            dbc.Col(     
                html.Button('Submit', id='btn_2', n_clicks=0),  
                    width=1,
                    style={'display': 'block'}
            ),
            dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_point_annotation',
                        options=[
                            {'label': 'No Validation', 'value': 'no'},
                            {'label': 'Approve', 'value': 'approve'},
                            {'label': 'Reject', 'value': 'reject'},
                        ], value = 'no'
                    ),
                    width=2,
                    style={'display': 'block'}
                ),
            dbc.Col(     
                html.Button('Submit', id='btn_1', n_clicks=0),  
                    width=1,
                    style={'display': 'block'}
            ),
            ]),
            dbc.Row([dbc.Col(
                dcc.Graph(id='cluster_graph'), width = 9, md = 8, style={'marginLeft': 0}),
                 dbc.Col(
                    html.Div(children=[html.Div([ DataTable(id='total_annotation', style_cell={'whiteSpace': 'normal','height': 'auto','width': 100},
                    columns = [{'id': 'Global annotation', 'name': 'Global annotation'} ])],style={'marginBottom':10}),
                    DataTable(id='cluster_table', style_cell={'whiteSpace': 'normal','height': 'auto','width': 100},columns = [{'id': column, 'name': column} for column in ['Clusters', 'Cluster annotation', 'Annotated points']])]),
                    style = {'marginTop':10, 'marginLeft': 5, 'marginRight': 0}, width ='25%', 
            ), 
            ],justify='start'),
            dbc.Row([dbc.Col(
            dcc.Graph(id='spectrogram', hoverData = {'points': [{'pointIndex': 0}]}),width = 6, style={'marginTop': 20}
            ), 
            dbc.Col(
                dcc.Graph(id='contour_plot', hoverData = {'points': [{'pointIndex': 0}]}), width = 6, style={'marginTop': 20}
            )]),
            # these are intermediate values to be used for sharing content
            # between callbacks
            # (see here https://dash.plotly.com/sharing-data-between-callbacks)
            dbc.Row(id='intermediate_val_syllables', style={'display': 'none'}),
            dbc.Row(id='intermediate_val_total_clusters', style={'display': 'none'}),
            dbc.Row(id='intermediate_val_clusters', style={'display': 'none'}),
            dbc.Row(id='clustering_info',
                     style={'display': 'none'})
        ], style={"height": "100vh"})
    return layout


if __name__ == "__main__":
    args = parse_arguments()
    global sp_time, sp_freq, moves, click_index
    click_index =-1
    time_start = time.time()
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                           ST_WIN, ST_STEP)

    with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'w') as outfile:
        x = json.dumps({'input_file': args.input_file.split('/')[-1], 'total_cluster_annotations': [], 'cluster_annotations': [], 'point_annotations': []}, indent=4)
        outfile.write(x)
    # These should change depending on the signal's size
    spec_resize_ratio_freq = 4
    spec_resize_ratio_time = 4

    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

    # define feature sequence for vocalization detection
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    spectral_energy, means, max_values = ap.prepare_features(spectrogram[:, f1:f2])
    
    app = dash.Dash(
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    clean_spectrogram = ap.clean_spectrogram(spectrogram)
    
    if args.spectrogram=='True' or args.spectrogram=='true' or args.spectrogram=='1':
        app.layout = get_layout(True)
    else:
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
    @app.callback(
        [Output('label_sel_start', 'children'),
         Output('label_sel_end', 'children'),
         Output('label_class', 'children')
         ],
        [Input('heatmap1', 'clickData')])
    def display_click_data(click_data):
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
                        syllable_label = 'class {}'.format(labels[i_s])
                        found = True
                        break
        if not found:
            i_s = -1
            syllable_label = ""
        return "{0:.2f}".format(t1), "{0:.2f}".format(t2), syllable_label
               

    @app.callback(
        [Output('cluster_graph', 'figure'),
         Output('silhouette', 'children'),
         Output('cal-har', 'children'),
         Output('dav-bould', 'children'),
         Output('clustering_info', 'children'),],
        [Input('dropdown_cluster', 'value'),
         Input('dropdown_n_clusters', 'value'),
         Input('dropdown_feats_type', 'value'), 
         Input('intermediate_val_syllables', 'children'),],
        [State('silhouette', 'children'),
         State('cal-har', 'children'),
         State('dav-bould', 'children'),
         State('clustering_info', 'children'),
         State('cluster_graph', 'clickData'),
         State('cluster_graph', 'figure'),
        ])
    def update_cluster_graph(method, n_clusters, feats_type, n_clicks_3, sil, cal_har, dav_bould, clust_info, click_data, fig):
        global labels,click_index
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'intermediate_val_syllables.children' in changed_id:
            if click_data and (n_clicks_3=='approve' or n_clicks_3=='reject'):
                index=click_data['points'][0]['pointIndex']
                fig['data'][0]['marker']['size'][index]=10
                if n_clicks_3 == 'approve':
                    fig['data'][0]['marker']['line']['color'][index]='Green'
                else:
                    fig['data'][0]['marker']['line']['color'][index]='Red'
                click_index = -1
                return fig, sil, cal_har, dav_bould, clust_info

            elif click_data:
                index=click_data['points'][0]['pointIndex']
                fig['data'][0]['marker']['size'][index]=10
                if click_index != -1 and click_index != index:
                    fig['data'][0]['marker']['size'][click_index]=7.5
                click_index = index

                return fig, sil, cal_har, dav_bould, clust_info
        else:
            if feats_type == 'simple':
                y, scores = ar.clustering(method, n_clusters, feats_simple)
                labels = y
                fig = go.Figure(data = go.Scatter(x = feats_2d_s[:, 0], y = feats_2d_s[:, 1], name='',
                            mode='markers',
                            marker=go.scatter.Marker(color=y, size=[7.5 for i in range(len(y))], line=dict(width=2,
                                        color=['White' for i in range(len(y))]), opacity=1.),text = ['cluster {}'.format(y[i]) for i in range (len(y))],
                            showlegend=False),layout = go.Layout(title = 'Clustered syllables', xaxis = dict(title = 'x'), yaxis = dict(title = 'y'), 
                            margin=dict(l=0, r=5), ))
            elif feats_type == 'deep':
                y, scores = ar.clustering(method, n_clusters, feats_deep)
                labels = y
                fig = go.Figure(data = go.Scatter(x = feats_2d_d[:, 0], y = feats_2d_d[:, 1], name='',
                            mode='markers',
                            marker=go.scatter.Marker(color=y, size=[7.5 for i in range(len(y))],  line=dict(width=2,
                                        color=['White' for i in range(len(y))]), opacity=1.),text = ['cluster {}'.format(y[i]) for i in range (len(y))],
                            showlegend=False),layout = go.Layout(title = 'Clustered syllables', xaxis = dict(title = 'x'), yaxis = dict(title = 'y'),
                            margin=dict(l=0, r=5), ))
            data = {
                "method": method,
                "number_of_clusters": n_clusters,
                "features_type": feats_type,
                # "clustering": labels 
            }
            fig = fig.to_dict()
            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'r') as infile:
                loaded_data=json.load(infile)
            for annotation in loaded_data['point_annotations']:
                if annotation['method'] == method and annotation['number_of_clusters'] == n_clusters and annotation['features_type']==feats_type:
                    index=annotation['index']
                    fig['data'][0]['marker']['size'][index]=10
                    if annotation['annotation'] == 'approve':
                        fig['data'][0]['marker']['line']['color'][index]='Green'
                    else:
                        fig['data'][0]['marker']['line']['color'][index]='Red'
        print(scores)
        return fig, round(scores[0],3), round(scores[1]), round(scores[2],3), data

    @app.callback(
        [Output('cluster_table', 'data'),
         Output('total_annotation', 'data')],
        [Input('dropdown_cluster', 'value'),
         Input('dropdown_n_clusters', 'value'),
         Input('dropdown_feats_type', 'value'), 
         Input('intermediate_val_total_clusters', 'children'),
         Input('intermediate_val_clusters', 'children'),
         Input('intermediate_val_syllables', 'children'),],
        [State('cluster_graph', 'clickData'),
         State('cluster_table', 'data'),
         State('total_annotation', 'data')
        ])
    def update_cluster_table(method, n_clusters, feats_type, n_clicks_1, n_clicks_2, n_clicks_3, click_data, table, total):
        global labels
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'intermediate_val_syllables.children' in changed_id:
            if click_data and  n_clicks_3!='{}':
                index=click_data['points'][0]['pointIndex']
                table[int(labels[index])]['Annotated points'] +=1
                return table, total
            elif click_data:
                return table, total
        elif 'intermediate_val_clusters.children' in changed_id:
            if click_data and  n_clicks_2!='{}':
                index=click_data['points'][0]['pointIndex']
                table[int(labels[index])]['Cluster annotation'] = int(n_clicks_2)
                return table, total
            elif click_data:
                return table, total
        elif 'intermediate_val_total_clusters.children' in changed_id:
            if n_clicks_1!='{}':
                return table, [{'Global annotation': int(n_clicks_1)}]
            else:
                return table, total
        else:
            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'r') as infile:
                loaded_data=json.load(infile)
            cnt = [0 for i in range(n_clusters)]
            for annotation in loaded_data['point_annotations']:
                if annotation['method'] == method and annotation['number_of_clusters'] == n_clusters and annotation['features_type']==feats_type:
                    cnt[int(annotation['class'])] += 1
            cluster_score = ['-' for i in range(n_clusters)]
            for annotation in loaded_data['cluster_annotations']:
                if annotation['method'] == method and annotation['number_of_clusters'] == n_clusters and annotation['features_type']==feats_type:
                    cluster_score[int(annotation['class'])] = annotation['annotation']

            total = '-'
            for annotation in loaded_data['total_cluster_annotations']:
                if annotation['method'] == method and annotation['number_of_clusters'] == n_clusters and annotation['features_type']==feats_type:
                    total = annotation['annotation']
                    break
            table = [{'Clusters':'cluster {}'.format(i), 'Cluster annotation': cluster_score[i], 'Annotated points': cnt[i]} for i in range (n_clusters)]
            total = [{'Global annotation': total}]
        return table, total

    
    @app.callback(
        Output('intermediate_val_syllables', 'children'),
        [Input('cluster_graph', 'clickData'),
         Input('dropdown_point_annotation', 'value'),
         Input('clustering_info', 'children'),
         Input('btn_1', 'n_clicks')])
         
    def point_annotation(click_data, val, info, n_clicks):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if click_data and (val=='approve' or val=='reject') and 'btn_1' in changed_id:

            point_info = {'index': click_data['points'][0]['pointIndex'] , 
                          'class': int(labels[click_data['points'][0]['pointIndex']]), 
                          'start time': syllables[click_data['points'][0]['pointIndex']]['st'], 'end time': syllables[click_data['points'][0]['pointIndex']]['et'],
                          'annotation': val}
            point_info = {**point_info, **info}

            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'r') as infile:
                data=json.load(infile)

            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'w') as outfile:
                ready = False
                for i, point_ann in enumerate(data['point_annotations']):    
                    shared_items = {k: point_info[k] for k in point_info if k in point_ann and point_info[k] == point_ann[k]}
                    if len(shared_items)==len(point_info)-1:
                        data['point_annotations'][i]['annotation'] = val
                        ready = True
                        break
                if ready==False:
                    data['point_annotations'].append(point_info)
                x = json.dumps(data, indent=2)
                outfile.write(x)
            return val

        return '{}'

    @app.callback(
        Output('intermediate_val_clusters', 'children'),
        [Input('cluster_graph', 'clickData'),
         Input('dropdown_cluster_annotation', 'value'),
         Input('clustering_info', 'children'),
         Input('btn_2', 'n_clicks')])
         
    def cluster_annotation(click_data, val, info, n_clicks):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if click_data and val and val!='no' and 'btn_2' in changed_id:

            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'r') as infile:
                data=json.load(infile)
            
            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'w') as outfile:
                ready = False
                info['class'] = int(labels[click_data['points'][0]['pointIndex']])
                for i, cluster_ann in enumerate(data['cluster_annotations']):    
                    shared_items = {k: info[k] for k in info if k in cluster_ann and info[k] == cluster_ann[k]}
                    if len(shared_items)==len(info):
                        data['cluster_annotations'][i]['annotation'] = val
                        ready = True
                        break
                if ready==False:
                    info['annotation'] = val 
                    data['cluster_annotations'].append(info)
                x = json.dumps(data, indent=2)
                outfile.write(x)
            return val
        return '{}'    

    @app.callback(
        Output('intermediate_val_total_clusters', 'children'),
        [Input('dropdown_total_cluster_annotation', 'value'),
         Input('clustering_info', 'children'),
         Input('btn_3', 'n_clicks')])
         
    def total_cluster_annotation(val, info, n_clicks):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if val and val!='no' and 'btn_3' in changed_id:

            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'r') as infile:
                data=json.load(infile)
            
            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'w') as outfile:
                ready = False
                for i, total_cluster_ann in enumerate(data['total_cluster_annotations']):    
                    shared_items = {k: info[k] for k in info if k in total_cluster_ann and info[k] == total_cluster_ann[k]}
                    if len(shared_items)==len(info):
                        data['total_cluster_annotations'][i]['annotation'] = val
                        ready = True
                        break
                if ready==False:
                    info['annotation'] = val 
                    data['total_cluster_annotations'].append(info)
                x = json.dumps(data, indent=2)
                outfile.write(x)
            return val
        return '{}'    

    @app.callback(
        Output('spectrogram', 'figure'),
        [Input('cluster_graph', 'hoverData')],
    )
    def display_hover_data(hoverData):
        if hoverData:
            index = hoverData['points'][0]['pointIndex']
        else:
            index = 0
        fig = go.Figure(data = go.Heatmap(x =sp_time[segments[index][0]:segments[index][1]], y=sp_freq[f1:f2], z = (images[index].T)/np.amax(images[index]), 
        showscale=False),
                        layout = go.Layout(title = 'Spectrogram of syllable', margin={'l': 0, 'b': 40, 't': 40, 'r': 0}, 
                                        xaxis = dict(range=[(sp_time[segments[index][0]]+ sp_time[segments[index][1]])/2-0.1, (sp_time[segments[index][0]]+ sp_time[segments[index][1]])/2+0.1], 
                                                        title = 'Time (Sec)'), yaxis=dict(range = [sp_freq[0], sp_freq[-1]], title='Freq (Hz)')))
        return fig

    @app.callback(
        Output('contour_plot', 'figure'),
        [Input('cluster_graph', 'hoverData')],
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
