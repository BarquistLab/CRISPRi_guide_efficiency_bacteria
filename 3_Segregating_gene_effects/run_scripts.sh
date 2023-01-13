#!/bin/bash

declare -a trainingsets=('0' '1' '2' '0,1,2')
declare -a NamesArray=('R75' 'C18' 'W' '3sets')
for i in {0..3}; do 
echo $i
training_set=${trainingsets[$i]}
foldername=${NamesArray[$i]}
choice=pasteur
echo $training_set $foldername $choice
#python datafusion.py -c $choice -o $foldername -training $training_set -r training/$foldername/regressor.pkl
python median_subtracting_model.py -c $choice -o $foldername -training $training_set -F pasteur
done


#nohup python datafusion.py -o R75 -training 0 -r training/R75/regressor.pkl  &
#nohup python datafusion.py -o C18 -training 1 -r training/C18/regressor.pkl &
#nohup python datafusion.py -o W -training 2  -r training/W/regressor.pkl &
#nohup python datafusion.py -o R75C18 -training 0,1  -r training/R75C18/regressor.pkl &
#nohup python datafusion.py -o R75W -training 0,2 -r training/R75W/regressor.pkl &
#nohup python datafusion.py -o C18W -training 1,2 -r training/C18W/regressor.pkl &
#nohup python datafusion.py -o 3sets -training 0,1,2 -r training/3sets/regressor.pkl &
#nohup python datafusion.py -o 3sets_rf -training 0,1,2 -inest random_forest -r training/3sets_rf/regressor.pkl &
