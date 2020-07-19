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

global fs
global all_data
global outstr

fs = 96000                # sampling frequency in Hz


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
mid_buffer_size = 0.5        # processing buffer size in seconds
st_win = 0.002               # short-term window size in seconds

all_data = []
mid_buffer = []
time_start = time.time()
outstr = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M%p")

# initialize soundcard for recording:
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                 input=True, frames_per_buffer=int(fs * buff_size))

while 1:  # for each recorded window (until ctr+c) is pressed
    # get current block and convert to list of short ints,
    block = stream.read(int(fs * buff_size))
    format = "%dh" % (len(block) / 2)
    shorts = struct.unpack(format, block)
    shorts_list = list(shorts)

    # then normalize and convert to numpy array:
    x = np.double(shorts_list) / (2**15)
    seg_len = len(x)

    all_data += shorts_list

    mid_buffer += shorts_list
    if len(mid_buffer) >= int(mid_buffer_size * fs):
        spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram_buffer(mid_buffer,
                                                                      fs,
                                                                      st_win,
                                                                      st_win)
        mid_buffer = []


