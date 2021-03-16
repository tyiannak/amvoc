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
Also, the user has to declare whether or not the spectrogram of the whole input signal 
should be created and displayed in the app (option -s). If the spectrogram is to be displayed
(not recommended for large wavfiles),
the user should choose one of the following options: -s 1 or -s True or -s true.
Otherwise, available options are: -s 0 or -s False or -s false.

```
python3 main.py -i data/B148_test_small.wav -s 0
```

The basic GUI is build among dash, so after running the script, a 
basic computation of the audio spectrogram is completed (or the spectrogram 
is loaded from file if it has been already cached), and then one can use the 
GUI through the dash local address `http://127.0.0.1:8050/`

![execution example](screenshot.png "execution example")

## Vocalization detection
The user can also use this tool for just extracting the detected vocalizations. If the user runs the `main.py`, the vocalizations are saved in a csv file named debug_offline.csv.
By running the `main_live.py`, the user can get the detected vocalizations in online mode (every 750msec). It just takes the WAV filename of the recording.

```
python3 main_live.py -i data/B148_test_small.wav
```

If the user runs the `main_live.py`, the vocalizations are saved in a csv file named debug_realtime.csv.  
Then, the detected vocalizations of a recording with our method can be compared to the detected vocalizations of the same recording using some other method, or to annotated vocalizations (ground truth). 
This comparison can be done using the `syllables_comp.py`, which takes the WAV filename of the recording, and the names of the two csv files to be compared.

```
python3 syllables_comp.py -i data/B148_test_small.wav -csv1 first_file.csv -csv2 second_file.csv
```
We have annotated some intervals of multiple recordings in order to evaluate the vocalization detection method and saved them in .csv files. These can be found in the google drive link https://drive.google.com/drive/folders/1gfRecT_0EYatHhvZMmmK8V_C0m562NCS?usp=sharing, along with the corresponding WAV recordings and a README file that defines which time interval of each recording we have annotated. 

The evaluation results are presented in https://docs.google.com/spreadsheets/d/1Kv2thpGLPzoflB7GHtn0pt0cJJ-utLC2u1QLK98DKP0/edit?usp=sharing. 

The method we chose correspond to column online_1 and offline_1. If the user wants to reproduce the results, they can download the whole folder from google drive and run for example the `main_live.py`:

```
python3 main_live.py -i path/to/recording/dir_fe_comp_stim_1.wav

```

Then, the user should open the debug_realtime.csv produced and keep only the entries that correspond to the same time interval of the ground truth file (found in README in google drive). The results are produced by the syllables_comp.py:

```
python3 syllables_comp.py -i path/to/recording/dir_fe_comp_stim_1.wav -csv1 path/to/first_csv/comp_stim_fe_1.csv -csv2 path/to/second_csv/debug_realtime.csv
```

The evaluation metrics are displayed on terminal. 
