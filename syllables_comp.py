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

    syllables_online, syllables_offline = [], []

    with open(csv1, 'r') as realtime:
        reader = csv.reader(realtime)
        for row in reader:
            syllables_online.append(row)

    with open(csv2, 'r') as offline:
        reader = csv.reader(offline)
        for row in reader:
            syllables_offline.append(row)

    print('Syllables offline: {}'.format(len(syllables_offline)))
    print('Syllables online: {}'.format(len(syllables_online)))

    max_len = max(len(syllables_offline), len(syllables_online))
    for i in range(len(syllables_offline)):
        syllables_offline[i] = [float(syllables_offline[i][0]), float(syllables_offline[i][1])]
    
    for i in range(len(syllables_online)):
        syllables_online[i] = [float(syllables_online[i][0]), float(syllables_online[i][1])]

    
    spectrogram, sp_time, sp_freq, fs = ap.get_spectrogram(args.input_file,
                                                            ST_WIN, ST_STEP)
    win = ST_STEP
    f_low = F1 if F1 < fs / 2.0 else fs / 2.0
    f_high = F2 if F2 < fs / 2.0 else fs / 2.0
    f1 = np.argmin(np.abs(sp_freq - f_low))
    f2 = np.argmin(np.abs(sp_freq - f_high))

    duration = spectrogram.shape[0] * ST_STEP
    precision, recall, accuracy_temporal = eval.temporal_evaluation(syllables_offline, syllables_online, duration)
    accuracy_event = eval.event_evaluation(syllables_offline, syllables_online)
    print(precision)
    print(recall)
    print(accuracy_temporal)
    print(accuracy_event)
    with open ('exp_ur_1.txt', 'a') as exp: 
        exp.write('Precision: {:.4f}\n'.format(precision))
        exp.write('Recall: {:.4f}\n'.format(recall))
        exp.write('Temporal accuracy: {:.4f}\n'.format(accuracy_temporal))
        exp.write('Event accuracy: {:.4f}\n'.format(accuracy_event))