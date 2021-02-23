# main_lite.py:
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

global fs
global all_data
global outstr
global wav_signal


buff_size = 0.01  # recording buffer size in seconds
mid_buffer_size = 0.5 # processing buffer size in seconds

config_data = utils.load_config("config.json")
ST_WIN = config_data['params']['ST_WIN']
ST_STEP = config_data['params']['ST_STEP']
MIN_VOC_DUR = config_data['params']['MIN_VOC_DUR']
F1 = config_data['params']['F1']
F2 = config_data['params']['F2']
thres = 0.7


wav_signal = None

def signal_handler(signal, frame):
    """
    This function is called when Ctr + C is pressed and is used to output the
    final buffer into a WAV file
    """
    # write final buffer to wav file
    global fs
    if len(all_data) > 1:
        wavfile.write(outstr + ".wav", fs, np.int16(all_data))


    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(outstr+".wav",
                                                           ST_WIN, ST_STEP)

    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

    # define feature sequence for vocalization detection
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    spectral_energy_1 = spectrogram.sum(axis=1)
    spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)
    crit =  (np.argmax(spectrogram[:,f1:f2], axis =1), np.argmax(spectrogram, axis =0), spectrogram[:,f1:f2])
    seg_limits, thres_sm, _ = ap.get_syllables(spectral_energy_2,
                                               spectral_energy_1,
                                               ST_STEP,
                                               crit,
                                               threshold_per=thres * 100,
                                               min_duration=MIN_VOC_DUR)
    for s in seg_limits:
        with open("debug_offline.csv", "a") as fp:
            fp.write(f'{count_mid_bufs * mid_buffer_size + s[0]},'
                     f'{count_mid_bufs * mid_buffer_size + s[1]}\n')

    sys.exit(0)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", nargs=None,
                        help="Input audio file (soundcard if none is provided")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    global fs
    if input_file:
        fs, wav_signal = io.read_audio_file(input_file)
    else:
        fs = 44100
    signal.signal(signal.SIGINT, signal_handler)

    all_data = []
    mid_buffer = []
    time_start = time.time()
    outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")
    # wav_signal = wav_signal[8*fs:-4*fs]
    # plt.plot(wav_signal)
    # plt.show()
    if wav_signal is None:
        # initialize soundcard for recording:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                         input=True, frames_per_buffer=int(fs * buff_size))
    means = []
    count_bufs = 0
    count_mid_bufs = 0

    # get spectral sequences:
    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

    with open("debug_realtime.csv", "w") as fp:
        pass
    # print(min(wav_signal))
    # print(max(wav_signal))
    # wav_signal = StandardScaler().fit_transform(wav_signal.reshape(-1,1))
    # wav_signal = wav_signal[:,0]
    # wav_signal = wav_signal/(max(wav_signal))
    # wav_signal = list(wav_signal)
    while 1:  # for each recorded window (until ctr+c) is pressed
        if wav_signal is None:
            # get current block and convert to list of short ints,
            block = stream.read(int(fs * buff_size))
            format = "%dh" % (len(block) / 2)
            shorts = struct.unpack(format, block)
            shorts_list = list(shorts)
        else:
            shorts_list = wav_signal[int(count_bufs * buff_size * fs):
                                 int((count_bufs + 1) * buff_size * fs)].tolist()
            if int((count_bufs + 1) * buff_size * fs) > len(wav_signal):
                break
        # then normalize and convert to numpy array:
        x = np.double(shorts_list) / (2**15)
        seg_len = len(x)
        # shorts_list = MinMaxScaler().fit_transform(np.array(shorts_list).reshape(-1,1))
        # shorts_list = list(shorts_list[:,0])
        # shorts_list = np.array(shorts_list)/(max(shorts_list))
        # shorts_list = list(shorts_list)
        # print(shorts_list)
        all_data += shorts_list

        mid_buffer += shorts_list
        if len(mid_buffer) >= int(mid_buffer_size * fs):
            # get spectrogram:
            # print(len(mid_buffer))
            spectrogram, sp_time, sp_freq, _  = ap.get_spectrogram_buffer(mid_buffer,
                                                                          fs,
                                                                          ST_WIN,
                                                                          ST_STEP)
            # print((np.where(spectrogram>0.1))[0].shape)
            # print(np.amin(spectrogram))
            # print(spectrogram.T[10,:])
            # for i in range(spectrogram.shape[0]):
            #     for j in range(50):
            #         if spectrogram[i,j]>0.005:
            #             spectrogram[i,j] = 0.001
            
            # spectrogram  = spectrogram/np.amax(spectrogram)
            # print(spectrogram.shape)
            # spectrogram = MinMaxScaler().fit_transform(spectrogram)
            # define feature sequence for vocalization detection
            f1 = np.argmin(np.abs(sp_freq - f_low))
            f2 = np.argmin(np.abs(sp_freq - f_high))
            # spectrogram[:,:f1] = 0.001
            # spectrogram[:,f2:] = 0.001
            # spectrogram = spectrogram[:,:250]
            # spectrogram[:, :50] = 0
            # plt.imshow(spectrogram[:,f1:f2].T)
            # plt.show()
            spectral_energy_1 = spectrogram.sum(axis=1)
            # print(spectral_energy_1)
            spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)
            # crit = (np.amax(spectrogram, axis=1)-np.mean(spectrogram,axis=1)-np.var(spectrogram,axis=1))
            var = np.var(spectrogram[:,f1:f2], axis=1)
            mean = np.mean(spectrogram[:,f1:f2], axis=1)
            crit =  (np.argmax(spectrogram[:,f1:f2], axis =1), np.argmax(spectrogram, axis =0), spectrogram[:,f1:f2])
            # var = np.var(spectrogram/np.amax(spectrogram), axis = 1)
            means.append(spectral_energy_2.mean())
            time_sec = 100
            seg_limits, thres_sm, _ = ap.get_syllables(spectral_energy_2,
                                                       spectral_energy_1,
                                                       ST_STEP,
                                                       crit,
                                                       threshold_per=thres * 100,
                                                       min_duration=MIN_VOC_DUR,
                                                       threshold_buf = means,
                                                    #    prev_time_frames=int(time_sec/mid_buffer_size)
                                                       )
            kmeans_centers = np.load('kmeans_centers.npy')
            # print(kmeans_centers)
            win = ST_STEP
            for s in seg_limits:

                # # for each detected syllable (vocalization)

                # A. get the spectrogram area in the defined frequency range
                start = int(s[0] / win)
                end = int(s[1] / win)
                # # np.save('segments.npy', segments)
                cur_image = spectrogram[start:end, f1:f2]
                # print(cur_image)
                # plt.imshow(cur_image.T)
                # plt.show()
                # var = np.var(cur_image/np.amax(cur_image), axis = 1)
                if cur_image.shape[0]==0 or cur_image.shape[1]==0:
                    continue
                temp_image = cur_image/np.amax(cur_image)

                vec = [np.mean(temp_image),np.var(temp_image), np.mean(cur_image-np.amax(cur_image)), np.var(cur_image-np.amax(cur_image))]
                if np.linalg.norm(vec-kmeans_centers[1]) < np.linalg.norm(vec-kmeans_centers[0]):
                    # print(mentemp_image)
                    # print([start, end])
                    # plt.imshow(temp_image.T)
                    # plt.show()
                    continue
                if np.linalg.norm(vec[1:]-kmeans_centers[1][1:]) < np.linalg.norm(vec[1:]-kmeans_centers[0][1:]):
                    # print(mentemp_image)
                    # print([start, end])
                    # plt.imshow(temp_image.T)
                    # plt.show()
                    continue
                # stop = False
                # for i in np.arange(0,160,40):
                #     vec = [np.mean(temp_image[:,i:i+40]),np.var(temp_image[:,i:i+40]), np.mean(cur_image[:,i:i+40]-np.amax(cur_image[:,i:i+40])), np.var(cur_image[:,i:i+40]-np.amax(cur_image[:,i:i+40]))]
                #     if np.linalg.norm(vec-kmeans_centers[1]) < np.linalg.norm(vec-kmeans_centers[0]):
                #         stop = True
                #         break
                # if stop==True:
                    # continue
                # print(kmeans_centers)
                # print(vec)
                # if vec[2] > -0.05:
                #     continue
                
                # plt.imshow(cur_image.T)
                # plt.show()
                print([count_mid_bufs * mid_buffer_size + s[0], count_mid_bufs * mid_buffer_size + s[1]])
                with open("debug_realtime.csv", "a") as fp:
                    fp.write(f'{count_mid_bufs * mid_buffer_size + s[0]},'
                             f'{count_mid_bufs * mid_buffer_size + s[1]}\n')

            mid_buffer = []
            count_mid_bufs += 1

        count_bufs += 1
    # print(means)

