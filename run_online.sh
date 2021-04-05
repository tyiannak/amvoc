#!/bin/bash

recordings=(1 2 3 4 5 6 7 8 9)

for rec in "${recordings[@]}"; do
     echo "--------------------------------------------"
     echo "Recording $rec"
     python3 main_live.py -i data/vocalizations_evaluation/"$rec"/rec_"$rec".wav
     python3 syllables_comp.py -i data/vocalizations_evaluation/"$rec"/rec_"$rec".wav -csv1 realtime_vocalizations.csv -csv2 data/vocalizations_evaluation/"$rec"/gt_"$rec".csv
done 

