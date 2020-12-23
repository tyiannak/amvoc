import os, glob
from scipy.io import wavfile
import audio_process as ap
import audio_recognize as ar
import numpy as np
from random import randrange
from scipy.io.wavfile import write


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--data_folder", required=True, nargs=None,
                        help="Folder")
    return parser.parse_args()

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

def load_syllables (filename):
  spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(filename,
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
  seg_limits, thres_sm, _ = ap.get_syllables(spectral_energy_2,
                                        spectral_energy_1,
                                        ST_STEP,
                                        threshold_per=thres * 100,
                                        min_duration=MIN_VOC_DUR)
  return seg_limits

if __name__ == '__main__':

    args = parse_arguments()

    path = args.data_folder
    samplerate=250000

    # path = "/content/drive/My Drive/amvoc/data/MaleMice_WildType_Songs"
    threshold = 50
    new_wav_dir_1 = np.zeros((72, 20*samplerate), dtype='int16')
    j=0
    for folder in os.walk(path):
          dir = folder[0]
          print(dir)
          for wavfile_ in folder[2]:
            print(wavfile_)
            if wavfile_[-3:]=='WAV' or wavfile_[-3:]=='wav':
              filename = dir+"/"+wavfile_
              samplerate, data = wavfile.read(filename)
              # find all syllables
              syllables = load_syllables(filename)
              cnt = 0
              while (1):
                # random interval
                start = randrange(0, 280)
                interval = [start, start+20]
                print(interval)
                # count num of syllables in the interval
                for syllable in syllables:
                  if syllable[0]> interval[0] and syllable[1]<interval[1]:
                    cnt+=1
                print(cnt)
                # choose interval where num_of_syllables > threshold
                if cnt > threshold:
                  new_wav_dir_1[j, :] = data[interval[0]*samplerate:interval[1]*samplerate]
                  j += 1
                  break

    write('/content/drive/My Drive/dir_ur_comp_stim.wav', samplerate, new_wav_dir_1[:36].flatten())
    write('/content/drive/My Drive/dir_fe_comp_stim.wav', samplerate, new_wav_dir_1[36:].flatten())