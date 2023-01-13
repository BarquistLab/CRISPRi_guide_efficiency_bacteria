#!/bin/bash

nohup python autosklearn_feature_engineering.py -o only_seq -c only_seq &
nohup python autosklearn_feature_engineering.py -o add_distance  -c add_distance &
nohup python autosklearn_feature_engineering.py -o add_MFE -c add_MFE &
nohup python autosklearn_feature_engineering.py -o add_deltaGB -c add_deltaGB &
nohup python autosklearn_feature_engineering.py -o only_guide -c only_guide &
nohup python autosklearn_feature_engineering.py -o gene_seq -c gene_seq &

