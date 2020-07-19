# main_lite.py:
# The real-time AMVOC

import numpy as np
import pyaudio
import struct

buff_size = 0.01          # window size in seconds
fs = 96000                # sampling frequency in Hz

# initialize soundcard for recording:
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs,
                 input=True, frames_per_buffer=int(fs * buff_size))

while 1:  # for each recorded window (until ctr+c) is pressed
    # get current block and convert to list of short ints,
    block = stream.read(int(fs * buff_size))
    format = "%dh" % (len(block) / 2)
    shorts = struct.unpack(format, block)

    # then normalize and convert to numpy array:
    x = np.double(list(shorts)) / (2**15)
    seg_len = len(x)
    print(len(x))
