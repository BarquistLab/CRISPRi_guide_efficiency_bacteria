#!/bin/bash

declare -a trainingsets=('0' '1' '2' '0,1' '0,2' '1,2' '0,1,2' '0,1,2')
declare -a NamesArray=('R75' 'C18' 'W' 'R75C18' 'R75W' 'C18W' '3sets' '3sets_rf')
for i in {0..6}; do 
echo $i
training_set=${trainingsets[$i]}
foldername=${NamesArray[$i]}
choice=histgb
echo $training_set $foldername $choice
#python datafusion.py -c $choice -o $foldername -training $training_set -r training/$foldername/regressor.pkl
python datafusion.py -c $choice -o $foldername -training $training_set
done


#nohup python datafusion.py -o R75 -training 0 -r training/R75/regressor.pkl  &
#nohup python datafusion.py -o C18 -training 1 -r training/C18/regressor.pkl &
#nohup python datafusion.py -o W -training 2  -r training/W/regressor.pkl &
#nohup python datafusion.py -o R75C18 -training 0,1  -r training/R75C18/regressor.pkl &
#nohup python datafusion.py -o R75W -training 0,2 -r training/R75W/regressor.pkl &
#nohup python datafusion.py -o C18W -training 1,2 -r training/C18W/regressor.pkl &
#nohup python datafusion.py -o 3sets -training 0,1,2 -r training/3sets/regressor.pkl &
#nohup python datafusion.py -o 3sets_rf -training 0,1,2 -inest random_forest -r training/3sets_rf/regressor.pkl &
