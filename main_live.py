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
import sounddevice as sd
import csv

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

    spectral_energy, means, max_values = ap.prepare_features(spectrogram[:, f1:f2])

    time_sec = 100
    seg_limits, thres_sm = ap.get_syllables(spectral_energy,
                                                means,
                                                max_values,
                                                ST_STEP,
                                                threshold_per=thres * 100,
                                                min_duration=MIN_VOC_DUR,
                                                threshold_buf = means,
                                                )

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
        input_Fs = input("Input desired recording frequency (in Hz): ")
        fs = int(input_Fs)
    signal.signal(signal.SIGINT, signal_handler)

    all_data = []
    mid_buffer = []
    time_start = time.time()
    outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")
    
    if wav_signal is None:
        print("Microphone options on this device:")
        print(sd.query_devices())
        input_mic_index = input("\nIndex of microphone to use in this recording: ")
       
        # initialize soundcard for recording:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                         input_device_index=int(input_mic_index),
                         input=True, frames_per_buffer=int(fs * buff_size))

    means = []
    count_bufs = 0
    count_mid_bufs = 0
    cnt = 0

    # get spectral sequences:
    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0

    with open("realtime_vocalizations.csv", "w") as fp:
        pass
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
        # then normalize and convert to numpy array:
        x = np.double(shorts_list) / (2**15)
        seg_len = len(x)
        all_data += shorts_list

        mid_buffer += shorts_list
        if len(mid_buffer) >= int(mid_buffer_size * fs) or (int((count_bufs + 1) * buff_size * fs) > len(wav_signal) and len(mid_buffer)>0):    
            # get spectrogram:
            if cnt>0:
                # calculate the spectrogram of the signal in the mid_buffer and 100 msec before 
                spectrogram, sp_time, sp_freq, _  = ap.get_spectrogram_buffer(all_data[-len(mid_buffer)-int(0.1*fs):],
                                                                          fs,
                                                                          ST_WIN,
                                                                          ST_STEP)
            else:
                spectrogram, sp_time, sp_freq, _  = ap.get_spectrogram_buffer(mid_buffer,
                                                                          fs,
                                                                          ST_WIN,
                                                                          ST_STEP)

            # define feature sequence for vocalization detection
            f1 = np.argmin(np.abs(sp_freq - f_low))
            f2 = np.argmin(np.abs(sp_freq - f_high))

            spectral_energy, mean_values, max_values = ap.prepare_features(spectrogram[:,f1:f2])
            
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
            # the following lines save the detected vocalizations in a .csv file and correct the split ones 
            for s in seg_limits:
                if cnt>0:
                    real_start = count_mid_bufs * mid_buffer_size-0.1 + s[0]
                    real_end = count_mid_bufs * mid_buffer_size-0.1 + s[1]
                    if s[0]<=0.1 and s[1] >=0.1:
                        # last vocalization should be changed
                        syllables_csv1 = []
                        print("correction")
                        # load the written vocalizations up to now
                        with open("realtime_vocalizations.csv", "r") as realtime:
                            reader = csv.reader(realtime)
                            for row in reader:
                                syllables_csv1.append(row)
                        with open("realtime_vocalizations.csv", "w") as fp:
                            for iS, syl in enumerate(syllables_csv1):
                                # rewrite the correct vocalizations
                                if iS<len(syllables_csv1)-1:
                                    fp.write(f'{syl[0]},'
                                            f'{syl[1]}\n')
                                else:
                                    # change the entry only if the newly detected vocalization has an overlap with the old one
                                    if float(syl[1]) > float(real_start):
                                        print([min(float(syl[0]), float(real_start)), real_end])
                                        fp.write(f'{min(float(syl[0]), float(real_start))},'
                                            f'{real_end}\n')
                                    # otherwise, keep both 
                                    else:
                                        fp.write(f'{syl[0]},'
                                            f'{syl[1]}\n')
                                        print([real_start, real_end])
                                        fp.write(f'{real_start},'
                                            f'{real_end}\n')
                    elif s[0]<0.1 and s[1]<0.1:
                        # no need to add or change an entry
                        continue
                    else:
                        print([real_start, real_end])
                        with open("realtime_vocalizations.csv", "a") as fp:
                            fp.write(f'{real_start},'
                                    f'{real_end}\n')
                else:
                    print([count_mid_bufs * mid_buffer_size + s[0], count_mid_bufs * mid_buffer_size + s[1]])
                    with open("realtime_vocalizations.csv", "a") as fp:
                            fp.write(f'{count_mid_bufs * mid_buffer_size+ s[0]},'
                                    f'{count_mid_bufs * mid_buffer_size+ s[1]}\n')
            cnt+=1

            mid_buffer = []
            count_mid_bufs += 1
        if int((count_bufs + 1) * buff_size * fs) > len(wav_signal):
            break
        count_bufs += 1