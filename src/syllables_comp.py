import csv
import audio_process as ap
import matplotlib.pyplot as plt
import utils
import argparse
import numpy as np
import evaluation as eval

config_data = utils.load_config("config.json")
ST_WIN = config_data['params']['ST_WIN']
ST_STEP = config_data['params']['ST_STEP']
MIN_VOC_DUR = config_data['params']['MIN_VOC_DUR']
F1 = config_data['params']['F1']
F2 = config_data['params']['F2']
thres = config_data['params']['thres']


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Amvoc")
    parser.add_argument("-i", "--input_file", required=True, nargs=None,
                        help="File")
    parser.add_argument("-csv1", "--input_csv1", required=True, nargs=None,
                        help="File")
    parser.add_argument("-csv2", "--input_csv2", required=True, nargs=None,
                        help="File")


    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    csv1 = args.input_csv1
    csv2 = args.input_csv2

    syllables_csv1, syllables_csv2 = [], []

    with open(csv1, 'r') as file1:
        reader = csv.reader(file1)
        for row in reader:
            syllables_csv1.append(row)

    with open(csv2, 'r') as file2:
        reader = csv.reader(file2)
        for row in reader:
            syllables_csv2.append(row)
    
    syllables_csv1 = syllables_csv1[2:]
    syllables_csv2 = syllables_csv2[1:]

    print('Syllables of 1st csv file: {}'.format(len(syllables_csv1)))
    print('Syllables of 2nd csv file: {}'.format(len(syllables_csv2)))

    for i in range(len(syllables_csv2)):
        syllables_csv2[i] = [float(syllables_csv2[i][0]), float(syllables_csv2[i][1])]
    
    for i in range(len(syllables_csv1)):
        syllables_csv1[i] = [float(syllables_csv1[i][0]), float(syllables_csv1[i][1])]

    
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                            ST_WIN, ST_STEP)
    win = ST_STEP
    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    duration = spectrogram.shape[0] * ST_STEP
    t_precision, t_recall, f1_temporal = eval.temporal_evaluation(syllables_csv2, syllables_csv1, duration)
    f1_event = eval.event_evaluation(syllables_csv2, syllables_csv1)
    print("Temporal Precision: {}".format(t_precision))
    print("Temporal Recall: {}".format(t_recall))
    print("Temporal F1: {}".format(f1_temporal))
    print("Event F1: {}".format(f1_event))