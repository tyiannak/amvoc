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
from plotly.subplots import make_subplots
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash_table import DataTable
import audio_process as ap
import audio_recognize as ar
import utils
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from sklearn.manifold import TSNE
from scipy.spatial import distance
from scipy.special import erf, erfc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sys import exit
import json
import csv
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pickle
import training_task as tr_t
import umap
import joblib
import os
import torch
from conv_autoencoder import ConvAutoencoder

colors = {'background': '#111111', 'text': '#7FDBFF'}

config_data = utils.load_config("config.json")
ST_WIN = config_data['params']['ST_WIN']
ST_STEP = config_data['params']['ST_STEP']
MIN_VOC_DUR = config_data['params']['MIN_VOC_DUR']
F1 = config_data['params']['F1']
F2 = config_data['params']['F2']
thres = config_data['params']['thres']
factor = config_data['params']['factor']
gamma_1 = config_data['params']['gamma_1']
gamma_2 = config_data['params']['gamma_2']
gamma_3 = config_data['params']['gamma_3']
model_name = config_data['params']['model']
num_ann_per_batch = config_data['params']['num_ann_per_batch']




def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    parser.add_argument("-c", "--continue_", required=True, nargs=None,
                        help="Decision")
    # parser.add_argument("-s", "--spectrogram", required=False, nargs=None,
    #                     help="Condition")
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


def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir / 'checkpoint.pt'
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def get_layout():

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
    global voc_name, bn
    bn=0
    voc_name = 'offline_{}.csv'.format((args.input_file.split('/')[-1]).split('.')[0])
    with open(voc_name, 'w') as fp:
            writer=csv.writer(fp)
            writer.writerow(["Start time", "End time"])
            for iS, s in enumerate(seg_limits):
                fp.write(f'{round(s[0],3)},'
                        f'{round(s[1],3)}\n')   
    if continue_=="n":
        exit()       

    specs, images, f_points, f_points_init, \
    [feats_simple, feats_deep, outputs_deep], feat_names, [f1, f2], segments, seg_limits, selector, scaler, pca = ar.cluster_syllables(seg_limits, spectrogram,
                                             sp_freq, f_low, f_high,  ST_STEP, model_name=model_name)
    reducer = umap.UMAP(random_state=1)
    # tsne = TSNE(n_components=2, perplexity = 50, n_iter = 5000, random_state = 1)
    feats_2d_s = reducer.fit_transform(feats_simple)
    # tsne = TSNE(n_components=2, perplexity = 50, n_iter = 5000, random_state = 1)
    feats_2d_d = reducer.fit_transform(feats_deep)
    list_contour = np.array(f_points, dtype=object)
    images = np.array(images, dtype=object) 
    f_points_all, f_points_init_all = [[], []], [[], []]
    

    shapes1 = get_shapes(seg_limits, f_low, f_high)
    shapes2, shapes3 = [], []
    
    for iS in range(len(seg_limits)):
        f_points_all[0] += f_points[iS][0]
        f_points_all[1] += f_points[iS][1]
        f_points_init_all[0] += f_points_init[iS][0]
        f_points_init_all[1] += f_points_init[iS][1]
    
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

    #save necessary 
    np.save('./dash/list_contour.npy', list_contour)
    np.save('./dash/segments.npy', segments)
    np.save('./dash/images.npy', images, allow_pickle=True)
    np.save('./dash/f1.npy', f1)
    np.save('./dash/f2.npy', f2)
    np.save('./dash/feats_simple.npy', feats_simple)
    np.save('./dash/feats_deep.npy', feats_deep)
    np.save('./dash/outputs_deep.npy', outputs_deep)
    np.save('./dash/feats_2d_s.npy', feats_2d_s)
    np.save('./dash/feats_2d_d.npy', feats_2d_d)
    np.save('./dash/seg_limits.npy', seg_limits)
    np.save('./dash/syllables.npy', syllables)
    np.save('./dash/specs.npy', specs)
    joblib.dump(selector, './dash/vt_selector.bin', compress=True)
    joblib.dump(scaler, './dash/std_scaler.bin', compress=True)
    joblib.dump(pca, './dash/pca.bin', compress=True)
    

    @app.callback(
        [
            Output("heatmap1", "figure"),
            Output("spectrogram_collapse", "is_open")
        ],
        [Input("show_spect", "n_clicks")],
        [State("spectrogram_collapse", "is_open")],
    )
    def show_spectrogram_panel(n_clicks, is_open):
        figure = {}
        if n_clicks:
            # plot figure every time show spectrogram button is pressed
            if not is_open:
                figure = {
                        'data': [go.Heatmap(x=sp_time[::spec_resize_ratio_time],
                                            y=sp_freq[::spec_resize_ratio_freq],
                                            z=clean_spectrogram[::spec_resize_ratio_time, ::spec_resize_ratio_freq].T,
                                            name='F', colorscale='Jet',
                                            showscale=False)],
                        'layout': go.Layout(
                            title='Spectrogram of the signal',
                            margin=dict(l=55, r=20, b=120, t=40, pad=4),
                            xaxis=dict(title='Time (Sec)'),
                            yaxis=dict(title='Freq (Hz)'),
                            shapes=shapes1 + shapes2 + shapes3)
                        }
            return figure, not is_open
        return figure, is_open



    layout = dbc.Container([
        # Title
        dbc.Row(
            dbc.Col(
                html.H2("AMVOC", style={
                                        'textAlign': 'center',
                                        'color': colors['text'],
                                        'marginBottom': 30,
                                        'marginTop':30
                                        }
                        )
            )
        ),

        dbc.Row(
                dbc.Col(
                    html.Button('Show Spectrogram', id='show_spect', n_clicks=0, style={'marginBottom': 30}),
                    width=2,
                    style={'display': 'block'}
                ) 
        ),
        # full spectrogram of the signal revealed on button press
        dcc.Loading(
            dbc.Collapse(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label(
                                    id="label_sel_start",
                                    children="Selected start",
                                    style={'textAlign': 'center',
                                           'color': colors['text']}),
                                width=1,
                            ),
                            dbc.Col(
                                html.Label(
                                    id="label_sel_end",
                                    children="Selected end",
                                    style={'textAlign': 'center',
                                           'color': colors['text']}),
                                width=1,
                            ),
                            dbc.Col(
                                html.Label(
                                    id='label_class',
                                    children="Class",
                                    style={'textAlign': 'center',
                                           'color': colors['text']}),
                                width=1,
                            )
                        ], className="h-10"),

                    # Main heatmap
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(id='heatmap1'),
                            width=12,
                            style={"height": "100%", "background-color": "white"}
                        ),
                    )
                ],
                is_open=False,
                id="spectrogram_collapse"
            ),
            id="loading_spect",
            type="default"

        )
        ,
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
                        {'label': 'Deep', 'value': 'deep'},
                        {'label': 'Simple', 'value': 'simple'},
                    ], value='deep'
                ),
                width=2,
            ),
            dbc.Col(
                html.Button('Save clustering', id='btn_f', n_clicks=0),  
                width=2,
                style={'display': 'block'}
            ),
            dbc.Col(
                html.Button('Retrain model', id='btn_r', n_clicks=0),  
                width=2,
                style={'display': 'block'}
            ),
            dbc.Col(
                html.Button('Update', id='btn_s', n_clicks=0),  
                width=2,
                style={'display': 'block'}
            ),
        #     html.Table([
        #     html.Tr([html.Td(['Silhouette score']), html.Td(id='silhouette')]),
        #     html.Tr([html.Td(['Calinski-Harabasz score']), html.Td(id='cal-har')]),
        #     html.Tr([html.Td(['Davies-Bouldin score']), html.Td(id='dav-bould')]),
        # ]),
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
        dcc.Store(id='clustering_info'),
        dbc.Row(id='save_clustering', style={'display':'none'}),
        dbc.Row(id='retrain_model', style={'display':'none'}),
        dbc.Row(id='retrain_const', style={'display':'none'}),
        dbc.Row(id='retrain_batch', style={'display':'none'}),
        dbc.Row(id='train_after_stop', style={'display':'none'}),
        dbc.Row(id='update', style={'display':'none'}),
        dbc.Row(id='pairs', style={'display':'none'}),
        dbc.Row(html.Div(
                    [
                        # dbc.Button("Retrain model", id="btn_r", n_clicks=0),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Should the two vocalizations belong to the same cluster?"),
                                dbc.ModalBody(
                                    # dbc.Col(
                                    dcc.Graph(id='pw_specs')),
                                    # dcc.Graph(id='pw_specs'), width = 9, md = 8, style={'marginLeft': 0})),
                                dbc.ModalFooter(
                                    dbc.Row([
                                        dbc.Col(html.Button("Yes", id="b_y", className="ml-auto", n_clicks=0)),
                                        dbc.Col(html.Button("No", id="b_n", className="ml-auto", n_clicks=0)),
                                        dbc.Col(html.Button("Stop", id="b_stop", className="ml-auto", n_clicks=0)),
                                        dbc.Col(html.Button("Cancel", id="b_cancel", className="ml-auto", n_clicks=0))]
                                    )   
                                    ),
                            ],
                            id="modal",
                            backdrop='static',
                            is_open=False,
                        ),
                    ]
                ),
            style={'display': 'none'}
            )
    ], style={"height": "100vh"})

    return layout


if __name__ == "__main__":
    args = parse_arguments()
    click_index =-1
    time_start = time.time()
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                           ST_WIN, ST_STEP)
    if not os.path.exists('dash'):
        os.mkdir('dash')
    # save necessary
    np.save('./dash/sp_time.npy', sp_time)
    np.save('./dash/sp_freq.npy', sp_freq)
    np.save('./dash/click_index.npy', click_index)
    np.save('./dash/spectrogram.npy', spectrogram)

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
        syllables = np.load('./dash/syllables.npy',allow_pickle=True)
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
                        labels = np.load('./dash/labels.npy')
                        syllable_label = 'class {}'.format(labels[i_s])
                        found = True
                        break
        if not found:
            i_s = -1
            syllable_label = ""
        return "{0:.2f}".format(t1), "{0:.2f}".format(t2), syllable_label
               
    @app.callback(
        [Output('cluster_graph', 'figure'),
         Output('clustering_info', 'data'),],
        [Input('dropdown_cluster', 'value'),
         Input('dropdown_n_clusters', 'value'),
         Input('dropdown_feats_type', 'value'), 
         Input('intermediate_val_syllables', 'children'),
         Input('update', 'children')],
        [State('clustering_info', 'data'),
         State('cluster_graph', 'clickData'),
         State('cluster_graph', 'figure'),
        ])
    def update_cluster_graph(method, n_clusters, feats_type, n_clicks_3, update,
                            clust_info, click_data, fig):
       
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        click_index = np.load('./dash/click_index.npy')

        if 'intermediate_val_syllables.children' in changed_id:
            if click_data and (n_clicks_3[0]=='approve' or n_clicks_3[0]=='reject'):
                index=click_data['points'][0]['pointIndex']
                fig['data'][0]['marker']['size'][index]=10
                if n_clicks_3[0] == 'approve':
                    fig['data'][0]['marker']['line']['color'][index]='Green'
                else:
                    fig['data'][0]['marker']['line']['color'][index]='Red'
                click_index = -1
                np.save('./dash/click_index.npy', click_index)
                return fig, clust_info

            elif click_data:
                index=click_data['points'][0]['pointIndex']
                fig['data'][0]['marker']['size'][index]=10
                if click_index != -1 and click_index != index:
                    fig['data'][0]['marker']['size'][click_index]=7.5
                click_index = index 
                np.save('./dash/click_index.npy', click_index)
                return fig, clust_info
        specs= np.load('./dash/specs.npy')
        # pairwise_constraints = np.zeros((len(specs), len(specs)))
        # np.save('./dash/pw.npy', pairwise_constraints)
        if feats_type == 'simple':
            feats = np.load('./dash/feats_simple.npy')
            feats_2d = np.load('./dash/feats_2d_s.npy')
        elif feats_type== 'deep':
            feats = np.load('./dash/feats_deep.npy')
            feats_2d = np.load('./dash/feats_2d_d.npy')
        y, scores = ar.clustering(method, n_clusters, feats)
            # labels = y
        np.save('./dash/labels.npy', y)
        # np.save('./dash/centers.npy', centers)
        fig = go.Figure(data = go.Scatter(x = feats_2d[:, 0],
                                            y = feats_2d[:, 1], name='',
                    mode='markers',
                    marker=go.scatter.Marker(color=y,
                                                size=[7.5
                                                    for i in range(len(y))],
                                                line=dict(width=2,
                                color=['White' for i in range(len(y))]),
                                                opacity=1.),
                                            text =
                                            ['cluster {}'.format(y[i])
                                            for i in range (len(y))],
                    showlegend=False),
                        layout = go.Layout(title = 'Clustered syllables',
                                            xaxis = dict(title = 'x'),
                                            yaxis = dict(title = 'y'),
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
        return fig, data

    @app.callback(
        Output('save_clustering', 'children'),
        [Input('dropdown_cluster', 'value'),
         Input('dropdown_n_clusters', 'value'),
         Input('dropdown_feats_type', 'value'), 
         Input('btn_f', 'n_clicks'),
         Input('retrain_model', 'children')]
    )
    def save(method, n_clusters, feats_type, n_clicks_f, retrained):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]        
        if n_clicks_f and n_clicks_f!='no' and 'btn_f' in changed_id:
            clf=SVC()
            labels = np.load('./dash/labels.npy')
            np.save('labels_{}.npy'.format((args.input_file.split('/')[-1]).split('.')[0]), labels)
            syllables = np.load('./dash/syllables.npy', allow_pickle=True)
            with open(voc_name, 'w') as fp:
                writer=csv.writer(fp)
                writer.writerow(["Start time", "End time", "Cluster"])
                for iS, s in enumerate(syllables):
                    fp.write(f'{round(s["st"],3)},'
                            f'{round(s["et"],3)},'
                            f'{labels[iS]}\n')   
            if feats_type=='simple':
                feats = np.load('./dash/feats_simple.npy')
            else:
                feats = np.load('./dash/feats_deep.npy')
            clf.fit(feats,labels)
            if retrained:
                model = torch.load("./dash/model_{}_{}_{}_{}".format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type), map_location=torch.device('cpu'))
                torch.save(model, "model_{}_{}_{}_{}".format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type))
            # centers = np.load('./dash/centers.npy')
            # np.save('centers_{}_{}_{}_{}.npy'.format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type), centers)
            pickle.dump(clf, open('clf_{}_{}_{}_{}.sav'.format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type), 'wb'))
            joblib.dump(joblib.load('./dash/vt_selector.bin'), 'vt_selector_{}_{}_{}_{}.bin'.format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type),compress=True)
            joblib.dump(joblib.load('./dash/std_scaler.bin'),'std_scaler_{}_{}_{}_{}.bin'.format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type),compress=True)
            joblib.dump(joblib.load('./dash/pca.bin'),'pca_{}_{}_{}_{}.bin'.format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type),compress=True)
            print("SAVED")
        return '{}'

    @app.callback(
        Output('retrain_model', 'children'),
        [Input('dropdown_cluster', 'value'),
        Input('dropdown_n_clusters', 'value'),
        Input('dropdown_feats_type', 'value'),
        Input('btn_r', 'n_clicks')]
    )
    def retrain(method, n_clusters, feats_type, n_clicks_r):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]   
        retrained=False     

        setup = [method, n_clusters, feats_type]
        with open("./dash/setup.txt", "wb") as fp:   #Pickling
               pickle.dump(setup, fp)
        if n_clicks_r and n_clicks_r!='no' and 'btn_r' in changed_id:
            specs=np.load('./dash/specs.npy')
            train_loader = tr_t.data_prep(specs)
            torch.save(train_loader, './dash/trainloader')

            spectrogram = np.load('./dash/images.npy', allow_pickle=True)
            outputs_init = np.load('./dash/outputs_deep.npy')

            model = torch.load(model_name)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            ckp_path = "./dash/checkpoint.pt"

            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, './dash/checkpoint.pt')
            clusterer = KMeans(n_clusters=n_clusters, random_state=9)

            y_ = clusterer.fit_predict(np.array(outputs_init,dtype=object))
            kmeans_centers = clusterer.cluster_centers_
            np.save('./dash/kmeans_centers.npy', kmeans_centers)
            pairwise_constraints = np.zeros((len(spectrogram), len(spectrogram)))
            np.save('./dash/pwc.npy', pairwise_constraints)

            bn, cnt, x, y, dist = 0, 0, -1, -1, 0
            
            batches = []
            for i,feats in enumerate(train_loader):
                batches.append(feats[0])
            batch_size = batches[0].shape[0]
            help_ = [bn, cnt, x, y, dist, batch_size]
            np.save('./dash/help.npy', np.array(help_))
            
            with open("./dash/batches.txt", "wb") as fp:   #Pickling
               pickle.dump(batches, fp)
            
            retrained=True

        return retrained
    @app.callback(
        Output('retrain_const', 'children'),
        [Input('retrain_model', 'children'),
         Input('retrain_batch', 'children'),
         Input('btn_r', 'n_clicks')]
    )
    def retrain_set_const(retrain_model, retrain_batch, btn_r):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]   
        if (retrain_model and 'btn_r' in changed_id) or retrain_batch:  
            train_loader = torch.load('./dash/trainloader')
            kmeans_centers = np.load('./dash/kmeans_centers.npy')
            pairwise_constraints = np.load('./dash/pwc.npy')

            help_=np.load('./dash/help.npy')
            cnt= int(help_[1])
            batch_size=int(help_[-1])
            bn = int(help_[0])
            with open("./dash/batches.txt", "rb") as fp:
                batches = pickle.load(fp)
            model = ConvAutoencoder()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            ckp_path = "./dash/checkpoint.pt"
            model, optimizer = load_ckp(ckp_path, model, optimizer)

            x,y, dist = tr_t.set_constraints(train_loader, model, batches[bn], kmeans_centers, cnt) 
            spectrogram = np.load('./dash/images.npy', allow_pickle=True)

            image_1 = spectrogram[bn*batch_size+x].T
            image_2 = spectrogram[bn*batch_size+y].T
            cnt+=1

            help_=[help_[0], cnt, x, y, dist, batch_size]
            np.save('./dash/help.npy', np.array(help_))
            np.save('./dash/image_1.npy', image_1)
            np.save('./dash/image_2.npy', image_2)
            return True
        else:
            return False
    
    @app.callback(
        [Output("modal", "is_open"),
        Output('pw_specs', 'figure')],
        [Input('b_y', 'n_clicks'),
        Input('b_n', 'n_clicks'),
        Input('b_stop', 'n_clicks'),
        Input('b_cancel', 'n_clicks'),
        Input("retrain_const", "children")],
        [State("modal", "is_open"),
        ],
    )
    def toggle_modal(b_y, b_n, b_stop, b_cancel, images, is_open):

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  

        if (b_y and 'b_y' in changed_id)  or (b_n and 'b_n' in changed_id) or b_stop or (b_cancel and 'b_cancel' in changed_id):
            return False, {}
        if images:

            image_1 = np.load('./dash/image_1.npy')
            image_2 = np.load('./dash/image_2.npy')
            
            sp_time = np.load('./dash/sp_time.npy')
            sp_freq = np.load('./dash/sp_freq.npy')
            images = np.load('./dash/images.npy', allow_pickle=True)
            f1 = np.load('./dash/f1.npy')
            f2 = np.load('./dash/f2.npy')
            fig = make_subplots(1, 2, column_widths=[image_1.shape[1], image_2.shape[1]], shared_yaxes=True, y_title='Frequency (kHz)')
            fig.add_trace(go.Heatmap(x = [i for i in range(image_1.shape[1])], y= [i for i in range(image_1.shape[0])], z = image_1, showscale=False), 1, 1)

            fig.add_trace(go.Heatmap(x = [i for i in range(image_2.shape[1])], y= [i for i in range(image_2.shape[0])], z = image_2,showscale=False), 1, 2)
            fig.update_xaxes(title_text = 'Time (ms)', tickmode = 'array', tickvals = [i for i in np.arange(0,140,10)], ticktext=['{}'.format(i) for i in range(0,280,20)])

            segments=np.load('./dash/segments.npy')

            return True, fig

        return is_open, {}

    @app.callback(
        Output('retrain_batch', 'children'),
        [Input('b_y', 'n_clicks'),
        Input('b_n', 'n_clicks'),
        Input('b_stop', 'n_clicks'),
        ]
    )
    def retrain_batch(b_y, b_n, b_stop):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  

        if changed_id=='.':
            raise PreventUpdate
        pairwise_constraints=np.load('./dash/pwc.npy')

        help_=np.load('./dash/help.npy')

        [bn, cnt, x, y, dist, batch_size] = list(help_)
        stop = False

        bn, cnt, x, y, batch_size = int(bn), int(cnt), int(x), int(y), int(batch_size)
        if b_y and 'b_y' in changed_id:
            pairwise_constraints[bn*batch_size+x, bn*batch_size+y] = erf(25/dist)
            pairwise_constraints[bn*batch_size+y, bn*batch_size+x] = erf(25/dist)
        elif b_n and 'b_n' in changed_id:
            pairwise_constraints[bn*batch_size+x, bn*batch_size+y] = erf(-25/dist)
            pairwise_constraints[bn*batch_size+y, bn*batch_size+x] = erf(-25/dist)
        elif b_stop:
            stop =True

        np.save('./dash/pwc.npy', pairwise_constraints)
        
        if cnt==num_ann_per_batch or stop:
            
            while(1):
                kmeans_centers = np.load('./dash/kmeans_centers.npy')

                model = ConvAutoencoder()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                ckp_path = "./dash/checkpoint.pt"
                model, optimizer = load_ckp(ckp_path, model, optimizer)
                with open("./dash/batches.txt", "rb") as fp:
                    batches = pickle.load(fp)
                model, kmeans_centers, optimizer=tr_t.train_one_batch(model, optimizer, batches[bn],pairwise_constraints, kmeans_centers, bn)
                bn+=1
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, './dash/checkpoint.pt')
                np.save('./dash/kmeans_centers.npy', kmeans_centers)
                train_loader = torch.load('./dash/trainloader')

                cnt=0
                help_[0], help_[1]=bn, cnt
                np.save('./dash/help.npy', help_)
                if bn==len(train_loader):
                    model = tr_t.train_clust(model, train_loader)
                    with open("./dash/setup.txt", "rb") as fp:
                        setup = pickle.load(fp)
                    path = "./dash/model_{}_{}_{}_{}".format((args.input_file.split('/')[-1]).split('.')[0], setup[0], setup[1], setup[2])
                    torch.save(model, path)
                    return False

                if stop:
                    continue
                return True
        else:
            return False
    
    @app.callback(
        Output('update', 'children'),
        [Input('dropdown_cluster', 'value'),
         Input('dropdown_n_clusters', 'value'),
         Input('dropdown_feats_type', 'value'),
         Input('btn_s', 'n_clicks')]
    )
    def update_clustering(method, n_clusters, feats_type, n_clicks_s):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]        
        if n_clicks_s and n_clicks_s!='no' and 'btn_s' in changed_id:
            print("UPDATING...")
            path = "./dash/model_{}_{}_{}_{}".format((args.input_file.split('/')[-1]).split('.')[0], method, n_clusters, feats_type)
            seg_limits = np.load('./dash/seg_limits.npy')
            spectrogram = np.load('./dash/spectrogram.npy')
            sp_freq = np.load('./dash/sp_freq.npy')
            specs, images, f_points, f_points_init, \
            [feats_simple, feats_deep, outputs_deep], feat_names, [f1, f2], segments, seg_limits, selector, scaler, pca = ar.cluster_syllables(seg_limits, spectrogram,
                                                    sp_freq, f_low, f_high,  ST_STEP, model_name=path)
            reducer = umap.UMAP(random_state=1)
            # tsne = TSNE(n_components=2, perplexity = 50, n_iter = 5000, random_state = 1)
            feats_2d_s = reducer.fit_transform(feats_simple)
            # tsne = TSNE(n_components=2, perplexity = 50, n_iter = 5000, random_state = 1)
            feats_2d_d = reducer.fit_transform(feats_deep)
            list_contour = np.array(f_points, dtype=object)
            images = np.array(images, dtype=object) 

            np.save('./dash/list_contour.npy', list_contour)
            np.save('./dash/segments.npy', segments)
            np.save('./dash/images.npy', images)
            np.save('./dash/f1.npy', f1)
            np.save('./dash/f2.npy', f2)
            np.save('./dash/feats_simple.npy', feats_simple)
            np.save('./dash/feats_deep.npy', feats_deep)
            np.save('./dash/outputs_deep.npy', outputs_deep)
            np.save('./dash/feats_2d_s.npy', feats_2d_s)
            np.save('./dash/feats_2d_d.npy', feats_2d_d)
            np.save('./dash/seg_limits.npy', seg_limits)
            joblib.dump(selector, './dash/vt_selector.bin', compress=True)
            joblib.dump(scaler, './dash/std_scaler.bin', compress=True)
            joblib.dump(pca, './dash/pca.bin', compress=True)
            
            print("DONE")
            return True
        return False

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
    def update_cluster_table(method, n_clusters, feats_type, n_clicks_1,
                             n_clicks_2, n_clicks_3, click_data, table, total):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'intermediate_val_syllables.children' in changed_id:
            if click_data and  n_clicks_3!='{}':
                if not n_clicks_3[1]:
                    labels = np.load('./dash/labels.npy')
                    index=click_data['points'][0]['pointIndex']
                    table[int(labels[index])]['Annotated points'] +=1
                return table, total
            elif click_data:
                return table, total
        elif 'intermediate_val_clusters.children' in changed_id:
            if click_data and  n_clicks_2!='{}':
                labels = np.load('./dash/labels.npy')
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
         Input('btn_1', 'n_clicks')],
         State('clustering_info', 'data'))
         
    def point_annotation(click_data, val, info, n_clicks):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if click_data and (val=='approve' or val=='reject') and 'btn_1' in changed_id:
            labels = np.load('./dash/labels.npy')
            syllables = np.load('./dash/syllables.npy', allow_pickle=True)
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
                    if len(shared_items)==len(point_info)-1 or len(shared_items)==len(point_info):
                        data['point_annotations'][i]['annotation'] = val
                        ready = True
                        break
                if ready==False:
                    data['point_annotations'].append(point_info)
                x = json.dumps(data, indent=2)
                outfile.write(x)
            return val, ready

        return '{}'

    @app.callback(
        Output('intermediate_val_clusters', 'children'),
        [Input('cluster_graph', 'clickData'),
         Input('dropdown_cluster_annotation', 'value'),
         Input('clustering_info', 'data'),
         Input('btn_2', 'n_clicks')])
         
    def cluster_annotation(click_data, val, info, n_clicks):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if click_data and val and val!='no' and 'btn_2' in changed_id:

            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'r') as infile:
                data=json.load(infile)
            
            with open('annotations_eval_{}.json'.format((args.input_file.split('/')[-1]).split('.')[0]), 'w') as outfile:
                ready = False
                labels = np.load('./dash/labels.npy')
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
         Input('clustering_info', 'data'),
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
        segments=np.load('./dash/segments.npy')
        sp_time = np.load('./dash/sp_time.npy')
        sp_freq = np.load('./dash/sp_freq.npy')
        images = np.load('./dash/images.npy', allow_pickle=True)
        f1 = np.load('./dash/f1.npy')
        f2 = np.load('./dash/f2.npy')
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
        sp_freq = np.load('./dash/sp_freq.npy')
        list_contour = np.load('./dash/list_contour.npy', allow_pickle=True)
        fig = go.Figure(data = go.Scatter(x = list_contour[index][0], y=list_contour[index][1], mode='lines+markers'), 
                        layout = go.Layout(title = 'Points of max frequency per time window of syllable', margin=dict(l=0, r=0, b=40, t=40, pad=4), 
                                        xaxis=dict(visible=True, title = 'Time (Sec)'), yaxis=dict(visible=True, autorange=False, range=[sp_freq[0], sp_freq[-1]], title='Freq (Hz)')))

        return fig

    app.run_server(debug=True)
