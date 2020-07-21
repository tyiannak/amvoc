# main_lite.py:
# The real-time AMVOC

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

global fs
global all_data
global outstr

fs = 96000                # sampling frequency in Hz
global wav_signal
wav_signal = None
# comment this line if you want realtime mode
fs, wav_signal = io.read_audio_file("data/B148_test_small.wav")

def signal_handler(signal, frame):
    """
    This function is called when Ctr + C is pressed and is used to output the
    final buffer into a WAV file
    """
    # write final buffer to wav file
    if len(all_data) > 1:
        wavfile.write(outstr + ".wav", fs, np.int16(all_data))
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
buff_size = 0.01             # recording buffer size in seconds
mid_buffer_size = 1.0        # processing buffer size in seconds
st_win = 0.002               # short-term window size in seconds
F1 = 30000                   # lower spectral energy sequence
F2 = 110000                  # higher spectral energy sequence
MIN_VOC_DUR = 0.005
thres = 1.3


all_data = []
mid_buffer = []
time_start = time.time()
outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")

if wav_signal is None:
    # initialize soundcard for recording:
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                     input=True, frames_per_buffer=int(fs * buff_size))

means = []
count_bufs = 0
count_mid_bufs = 0

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

    all_data += shorts_list

    mid_buffer += shorts_list
    if len(mid_buffer) >= int(mid_buffer_size * fs):
        # get spectrogram:
        spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram_buffer(mid_buffer,
                                                                      fs,
                                                                      st_win,
                                                                      st_win)

        # get spectral sequences:
        f_low = F1 if F1 < fs / 2.0 else fs / 2.0
        f_high = F2 if F2 < fs / 2.0 else fs / 2.0

        # define feature sequence for vocalization detection
        f1 = np.argmin(np.abs(sp_freq - f_low))
        f2 = np.argmin(np.abs(sp_freq - f_high))

        spectral_energy_1 = spectrogram.sum(axis=1)
        spectral_energy_2 = spectrogram[:, f1:f2].sum(axis=1)

        means.append(spectral_energy_2.mean())
        seg_limits, thres_sm, _ = ap.get_syllables(spectral_energy_2,
                                                   spectral_energy_1,
                                                   st_win,
                                                   threshold_per=thres * 100,
                                                   min_duration=MIN_VOC_DUR,
                                                   threshold_buf = means)
        for s in seg_limits:
            print(count_mid_bufs * mid_buffer_size + s[0],
                  count_mid_bufs * mid_buffer_size + s[1])

        mid_buffer = []
        count_mid_bufs += 1

    count_bufs += 1


