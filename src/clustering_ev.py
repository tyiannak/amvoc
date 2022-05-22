import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import audio_recognize as ar
import audio_process as ap
import utils
import csv

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

def evaluate(pairs, clusters):
    rows_pos, cols_pos = np.where(pairs==1)
    y_true = []
    y_pred = []
    for i in range(len(rows_pos)):
        y_true.append(1)
        if clusters[rows_pos[i]]==clusters[cols_pos[i]]:
            y_pred.append(1)
        else:
            y_pred.append(0)
    rows_neg, cols_neg = np.where(pairs==-1)

    for i in range(len(rows_neg)):
        y_true.append(0)
        if clusters[rows_neg[i]]==clusters[cols_neg[i]]:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["different cluster","same cluster"]))


if __name__ == "__main__":
    args = parse_arguments()
    pairs = np.load('pw_{}.npy'.format((args.input_file.split('/')[-1]).split('.')[0]))

    clusters = np.load('labels_{}.npy'.format((args.input_file.split('/')[-1]).split('.')[0]))
    # syllables_csv2=[]
    # with open("realtime_vocalizations_2.csv", "r") as realtime:
    #     reader = csv.reader(realtime)
    #     for row in reader:
    #         syllables_csv2.append(row)
    # clusters = []
    # # print(syllables_csv2)
    # for i in range(len(syllables_csv2)):
    #     clusters.append(syllables_csv2[i][2])
    # print(clusters)
    evaluate(pairs, clusters)