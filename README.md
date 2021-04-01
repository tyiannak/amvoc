# amvoc: Python Tool for Analysis of Mouse Vocal Communication

## Intro
amvoc is a tool used to analyze mouse vocal communication through analyzing
 audio recordings from mouse communication signals.
 
## Setup
```
pip3 install -r requirements.txt
``` 

## Execution
The main GUI for analyzing mouse sounds is in the `main.py`. 
It takes the WAV filename in which the recording to analyze is stored.
Also, the user has to declare whether or not they want to proceed with the annotation and clustering of each vocalization or just get the vocalizations detected for the given recording (option -c y if they want to continue or -c n in the opposite case). In case they want to proceed with the clustering, they can also define whether spectrogram of the whole input signal 
should be displayed in the app (option -s). If the spectrogram is to be displayed
(not recommended for large wavfiles),
the user should choose one of the following options: -s 1 or -s True or -s true.

```
python3 main.py -i data/B148_test_small.wav -c y -s 1
```

The basic GUI is build among dash, so after running the script, a 
basic computation of the audio spectrogram is completed (or the spectrogram 
is loaded from file if it has been already cached), and then one can use the 
GUI through the dash local address `http://127.0.0.1:8050/`

![execution example](screenshot.png "execution example")

## Vocalization detection
If the user runs the `main.py`, the vocalizations are saved in a csv file named offline_vocalizations.csv.
By running the `main_live.py`, the user can get the detected vocalizations in online mode (every 750msec). It just takes the WAV filename of the recording.

```
python3 main_live.py -i data/B148_test_small.wav
```

If the user runs the `main_live.py`, the vocalizations are saved in a csv file named realtime_vocalizations.csv.  
Then, the detected vocalizations of a recording with our method can be compared to the detected vocalizations of the same recording using some other method, or to annotated vocalizations (ground truth). 
This comparison can be done using the `syllables_comp.py`, which takes the WAV filename of the recording, and the names of the two csv files to be compared.

```
python3 syllables_comp.py -i data/B148_test_small.wav -csv1 first_file.csv -csv2 second_file.csv
```
We have annotated some intervals of multiple recordings in order to evaluate the vocalization detection method and saved them in .csv files. These can be found in the google drive link https://drive.google.com/drive/folders/1gfRecT_0EYatHhvZMmmK8V_C0m562NCS?usp=sharing, along with the corresponding WAV recordings and a README file that defines which time interval of each recording we have annotated. Ground truth files have the name gt_{num}_fc.csv, for example gt_1_fc.csv. We have also included the detected vocalizations from 5 other methods: 2 versions of MSA, MUPET, VocalMat and DeepSqueak. Their csv files, e.g. deepsqueak_1_fc.csv can be used for comparison with our results or the ground truth annotations.
The .csv files without the _fc suffix include the same vocalizations, but start and end time correspond to the original and not the cut recording. 

The evaluation results are presented in the same google drive link mentioned above. 

AMVOC results are presented in columns offline and online. If the user wants to reproduce the results, they can download the whole folder from google drive and run for example the `main_live.py`:

```
python3 main_live.py -i path/to/folder_1/1/rec1.wav

```

In order to compare the detected vocalizations with the actual ones, the user should run the syllables_comp.py:

```
python3 syllables_comp.py -i path/to/folder_1/rec_1.wav -csv1 path/to/first_csv/realtime_vocalizations.csv -csv2 path/to/folder_1/1/gt_1_fc.csv
```

The evaluation metrics are displayed on terminal. 
