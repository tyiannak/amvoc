# main_live.py:
# The real-time AMVOC

import argparse
import numpy as np
import pyaudio
import struct
import sys
import datetime
import signal
import time
import scipy.io.wavfile as wavfile
import audio_process as ap
from pyAudioAnalysis import audioBasicIO as io
import utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sounddevice as sd
import csv
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
import math 

global fs
global all_data
global outstr
global wav_signal


buff_size = 0.01  # recording buffer size in seconds
mid_buffer_size = 0.75 # processing buffer size in seconds

config_data = utils.load_config("config.json")
ST_WIN = config_data['params']['ST_WIN']
ST_STEP = config_data['params']['ST_STEP']
MIN_VOC_DUR = config_data['params']['MIN_VOC_DUR']
F1 = config_data['params']['F1']
F2 = config_data['params']['F2']
thres = config_data['params']['thres']
factor = config_data['params']['factor']
model_name = config_data['params']['model']


wav_signal = None

class ConvAutoencoder(nn.Module):
    def __init__(self, n_clusters = 5, kmeans_centers=None):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 64), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 64, 3, padding =1)  
        # conv layer (depth from 64 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        # conv layer (depth from 32 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 2, 3, padding=1)

        self.pool = nn.MaxPool2d((2,2), 2)

        self.flatten = nn.Flatten()
        ## decoder layers ##
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(2, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def forward(self, x, decode = True, clustering = False, kmeans_centers=None):
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
       

        if self.training and decode:
            ## decode ##
            # add transpose conv layers, with relu activation function
            y = F.relu(self.t_conv1(x))        
            y = F.relu(self.t_conv2(y))
            # output layer (with sigmoid for scaling from 0 to 1)
            y = F.sigmoid(self.t_conv3(y)) 
            if not clustering:
              return x, y

        if self.training and clustering:
             x = self.flatten(x)
             dist =torch.cdist(torch.reshape(x,(1,x.shape[0],x.shape[1])), torch.reshape(kmeans_centers,(1,kmeans_centers.shape[0],kmeans_centers.shape[1])))
             q = torch.div((1+dist.pow(2)).pow(-1),torch.sum((1+dist.pow(2)).pow(-1),dim=1))
             if decode:
              return x, y, q
             else:
               return x, q
        return x



def signal_handler(signal, frame):
    """
    This function is called when Ctr + C is pressed
    """
    sys.exit(0)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", nargs=None,
                        help="Input audio file (soundcard if none is provided")
    parser.add_argument("-c", "--classifier", nargs=None,
                        help="Pretrained classifier-optional")
    return parser.parse_args()


def feats_extr(image, encoder, selector, scaler, pca):
    
    time_limit=64
    if len(image)>time_limit:
        image = image[int((len(image)-time_limit)/2):int((len(image)-time_limit)/2)+time_limit,:]/np.amax(image)
    elif len(image)<time_limit:
        image = np.pad(image/np.amax(image), ((int((time_limit-image.shape[0])/2), (time_limit-image.shape[0]) - int((time_limit-image.shape[0])/2)),(0,0)))
    else:
        image = image/np.amax(image)
    
    spec = np.array(image)
    spec = spec.reshape(1, 1, spec.shape[0], spec.shape[1])
    dataset = TensorDataset(torch.tensor(spec, dtype = torch.float))
    test_loader = torch.utils.data.DataLoader(dataset, shuffle = False)
    outputs = []
    encoder.eval()
    with torch.no_grad():
        for data in test_loader:
            outputs += encoder(data[0], False)
    for i in range(len(outputs)):
        outputs[i] = outputs[i].detach().numpy().flatten()
    outputs=np.array(outputs)
    features = outputs
    
    features = selector.transform(features)
    features = scaler.transform(features)
    features = pca.transform(features)
    return features

def print_and_write(to_print, start_time, end_time, i_s, len_seg_limits):
    to_print[0] = start_time
    to_print[1] = end_time
    print(to_print)
    if i_s!=len_seg_limits-1:
        with open(voc_file, "a") as fp:
            writer=csv.writer(fp)
            writer.writerow(to_print)

def write_last(to_print_last):
    with open(voc_file, "a") as fp:
        writer=csv.writer(fp)
        writer.writerow(to_print_last)

if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    clf = args.classifier
    global fs, voc_file
    outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")
    if input_file:
        fs, wav_signal = io.read_audio_file(input_file)
        voc_file = 'realtime_{}.csv'.format((args.input_file.split('/')[-1]).split('.')[0])
    else:
        input_Fs = input("Input desired recording frequency (in Hz): ")
        fs = int(input_Fs)
        voc_file = 'realtime_{}.csv'.format(outstr)
    signal.signal(signal.SIGINT, signal_handler)
    if clf:
        loaded_model = pickle.load(open(clf, 'rb'))
        model = torch.load(model_name, map_location=torch.device('cpu'))
        # centers = np.load(clf)
        selector = joblib.load('vt_selector_' + clf[4:-4]+'.bin')
        scaler = joblib.load('std_scaler_' + clf[4:-4]+'.bin')
        pca = joblib.load('pca_' + clf[4:-4]+'.bin')
    all_data = []
    mid_buffer = []
    
    if wav_signal is None:
        print("Microphone options on this device:")
        print(sd.query_devices())
        input_mic_index = input("\nIndex of microphone to use "
                                "in this recording: ")
       
        # initialize soundcard for recording:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                         input_device_index=int(input_mic_index),
                         input=True, frames_per_buffer=int(fs * buff_size))

    means = []
    count_bufs = 0
    count_mid_bufs = 0
    check = False # is true when the first USV is detected

    # get spectral sequences:
    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0
    with open(voc_file, "w") as fp:
        writer=csv.writer(fp)
        if clf:
            writer.writerow(["Start time", "End time", "Class"])
        else:
            writer.writerow(["Start time", "End time"])
    while 1:  # for each recorded window (until ctr+c) is pressed
        if wav_signal is None:
            # get current block and convert to list of short ints,
            block = stream.read(int(fs * buff_size), exception_on_overflow=False)
            format = "%dh" % (len(block) / 2)
            shorts = struct.unpack(format, block)
            shorts_list = list(shorts)
            len_wav_signal = math.inf
        else:
            shorts_list = wav_signal[int(count_bufs * buff_size * fs):
                                 int((count_bufs + 1) * buff_size * fs)].tolist()
            len_wav_signal = len(wav_signal)
        # then normalize and convert to numpy array:
        x = np.double(shorts_list) / (2**15)
        seg_len = len(x)
        all_data += shorts_list
        mid_buffer += shorts_list
        if len(mid_buffer) >= int(mid_buffer_size * fs) or \
                (int((count_bufs + 1) * buff_size * fs) > len_wav_signal and
                 len(mid_buffer)>0):
            
            # get spectrogram:
            if count_mid_bufs>0:
                # calculate the spectrogram of the signal in the
                # mid_buffer and 100 msec before
                spectrogram, sp_time, sp_freq, _  = \
                    ap.get_spectrogram_buffer(all_data[-len(mid_buffer)-
                                                       int(0.1*fs):], fs,
                                              ST_WIN, ST_STEP)
            else:
                spectrogram, sp_time, sp_freq, _  = \
                    ap.get_spectrogram_buffer(mid_buffer, fs, ST_WIN, ST_STEP)

            # define feature sequence for vocalization detection
            f1 = np.argmin(np.abs(sp_freq - f_low))
            f2 = np.argmin(np.abs(sp_freq - f_high))
            spectral_energy, mean_values, max_values = \
                ap.prepare_features(spectrogram[:,f1:f2])
            
            means.append(spectral_energy.mean())

            time_sec = 100
            seg_limits, thres_sm = ap.get_syllables(spectral_energy,
                                                    mean_values,
                                                    max_values,
                                                    ST_STEP,
                                                    threshold_per=thres * 100,
                                                    factor=factor, 
                                                    min_duration=MIN_VOC_DUR,
                                                    threshold_buf = means,
                                                    )
            win = ST_STEP
            # the following lines save the detected
            # vocalizations in a .csv file and correct the split ones
            for i_s, s in enumerate(seg_limits):
                start = int(s[0]/win)
                end = int(s[1]/win)
                if clf:
                    # label calculation
                    cur_image = spectrogram[start:end, f1:f2]
                    feature_vector = feats_extr(cur_image, model, selector, scaler, pca)
                    label = loaded_model.predict(feature_vector)[0]
                    to_print = [0,0,label]
                else:
                    to_print = [0,0]
                # plt.imshow(cur_image.T)
                # plt.show()
                if check:
                    # actual times of occurrence
                    real_start = round(count_mid_bufs * mid_buffer_size-0.1 + s[0],3)
                    real_end = round(count_mid_bufs * mid_buffer_size-0.1 + s[1],3)
                    
                    if s[0]<0.1 and s[1] >=0.1:
                        # vocalization interrupted, correct only if the newly
                        # detected vocalization has an overlap
                        # with the old one with a 10 ms tolerance
                        if real_start - last[1]<0.01:
                            print("correction")
                            print_and_write(to_print, min(last[0], real_start), real_end, i_s, len(seg_limits))

                        # otherwise, keep both 
                        else:
                            write_last(to_print_last)
                            print_and_write(to_print, real_start, real_end, i_s, len(seg_limits))

                        last[2]=1 # mark last vocalization of previous buffer as written
                    elif s[0]<0.1 and s[1]<0.1:
                        # no need to add or change an entry
                        continue
                    else:            
                        if last[1] <real_start and last[2]==0:
                            write_last(to_print_last)
                            last[2]=1
                        print_and_write(to_print, real_start, real_end, i_s, len(seg_limits))
                                
                else:
                    if count_mid_bufs > 0:
                        real_start = round(count_mid_bufs * mid_buffer_size -0.1 + s[0],3)
                        real_end = round(count_mid_bufs * mid_buffer_size -0.1 + s[1],3)
                    else:
                        real_start = round(count_mid_bufs * mid_buffer_size + s[0],3)
                        real_end = round(count_mid_bufs * mid_buffer_size + s[1],3)
                    print_and_write(to_print, real_start, real_end, i_s, len(seg_limits))
                
                if i_s==len(seg_limits)-1:
                    # keep last USV in case it needs correction 
                    # last[2] is a marker of whether the last USV 
                    # from previous buffer is written in the file
                    if clf:
                        last=[real_start, real_end, 0, label] 
                        
                        to_print_last = [last[0], last[1], last[3]]
                    else:
                        last=[real_start, real_end, 0,-1]
                        to_print_last=[last[0], last[1]]

            if seg_limits!=[]:
                #first USVs detected
                check=True
            
            all_data = all_data[-len(mid_buffer):]
            mid_buffer = []
            count_mid_bufs += 1
        if int((count_bufs + 1) * buff_size * fs) > len_wav_signal:
            if last[2]==0: # if last USV from previous buffer isn't written in the file
                write_last(to_print_last)
            break
        count_bufs += 1