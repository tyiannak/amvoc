import matplotlib.pyplot as plt
import audio_recognize as ar
import audio_process as ap
import numpy as np
import utils
import argparse
import os.path
from os import path

config_data = utils.load_config("config.json")
ST_WIN = config_data['params']['ST_WIN']
ST_STEP = config_data['params']['ST_STEP']
MIN_VOC_DUR = config_data['params']['MIN_VOC_DUR']
F1 = config_data['params']['F1']
F2 = config_data['params']['F2']
thres = config_data['params']['thres']
factor = config_data['params']['factor']

def load_spectrograms (filename):
  train_data = []
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

  spectral_energy, mean_values, max_values = ap.prepare_features(spectrogram[:,f1:f2])

  time_sec = 100
  seg_limits, thres_sm = ap.get_syllables(spectral_energy,
                                          mean_values,
                                          max_values,
                                          ST_STEP,
                                          threshold_per=thres * 100,
                                          factor=factor, 
                                          min_duration=MIN_VOC_DUR,
                                          )
  
  
  train_data += (ar.cluster_syllables(seg_limits, spectrogram,
                                          sp_freq, f_low, f_high,  ST_STEP, train = True))
  print(len(train_data))
  return train_data

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    spectrogram = load_spectrograms(args.input_file)
    # pairwise_constraints = np.zeros((len(spectrogram), len(spectrogram)))
    if path.exists('pw_{}.npy'.format((args.input_file.split('/')[-1]).split('.')[0])):
        pairwise_constraints = np.load('pw_{}.npy'.format((args.input_file.split('/')[-1]).split('.')[0]))
    else:
        pairwise_constraints = np.zeros((len(spectrogram), len(spectrogram)))
    cnt=0
    answer = 'start'
    plt.ion()
    plt.show()
    while answer!='stop':
        while (1):
            x = np.random.randint(pairwise_constraints.shape[0])
            y = np.random.randint(pairwise_constraints.shape[0])
            if pairwise_constraints[x, y]==0 and pairwise_constraints[y, x]==0 and x!=y:
                cnt += 1
                print()
                print("No {}".format(cnt))
                break
        print(x,y)
        if answer!='stop':
            plt.figure(figsize=(5,5))
            plt.subplot(1,2,1)
            plt.imshow(np.flip(spectrogram[x], axis=1).T)
            plt.subplot(1,2,2)
            plt.imshow(np.flip(spectrogram[y], axis=1).T)
            plt.pause(0.001)
            plt.draw()
            answer = input("Should the two vocalizations belong to the same cluster? (y/n). If you want to stop, type 'stop'. \n")
            plt.close()

        while (answer!='stop'):
            if answer=='y':
                pairwise_constraints[x, y] = 1
                # pairwise_constraints[y, x] = 1
                break
            elif answer=='n':
                pairwise_constraints[x, y] = -1
                # pairwise_constraints[y, x] = -1
                break
            else:
                answer = input("Please provide a new answer (y/n) \n")
    np.save('pw_{}.npy'.format((args.input_file.split('/')[-1]).split('.')[0]), pairwise_constraints)

