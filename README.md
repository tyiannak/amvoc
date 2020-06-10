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
It takes the WAV filename in which the recording to analyze is stored:

```
python3 main.py -i data/B148_test_small.wav
```

The basic GUI is build among dash, so after running the script, a 
basic computation of the audio spectrogram is completed (or the spactrogram 
is loaded from file if it has been already cached), and then one can use the 
GUI through the dash local address `http://127.0.0.1:8050/`

![execution example](screenshot.png "execution example")


