#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:25:46 2021

@author: yanying
"""
import os
import random
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

import pytorch_lightning as pl
from sklearn.model_selection import GroupKFold
from src.utils import get_data, create_model_features, write_performance, reload_performance, write_predictions
import src.deep_models as dm 

from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

seed = 5555
random.seed(seed)
np.random.seed(seed)
pl.seed_everything(seed)

## get time stamp
date = datetime.now()
date = "{}-{}-{}".format(date.year, date.month, date.day)

## files and directories
output_models = "/storage/groups/haicu/workspace/crispri/models/" + date 
output_performance = "../reports/performance_training/" + date + "/deep_models" 
output_plots = "../reports/plots_training/" + date 

file_data_wang = '../datasets/data_wang.pickle'
file_one_hot_encoding_wang = '../datasets/one_hot_encoding_DL_wang.pickle'
file_kmer_wang = '../datasets/kmer_wang.pickle'

file_data_rousset_E18 = '../datasets/data_rousset_E18.pickle'
file_one_hot_encoding_rousset_E18 = '../datasets/one_hot_encoding_DL_rousset_E18.pickle'
file_kmer_rousset_E18 = '../datasets/kmer_rousset_E18.pickle'

file_data_rousset_E75 = '../datasets/data_rousset_E75.pickle'
file_one_hot_encoding_rousset_E75 = '../datasets/one_hot_encoding_DL_rousset_E75.pickle'
file_kmer_rousset_E75 = '../datasets/kmer_rousset_E75.pickle'


n_folds_outer = 5
n_folds_inner = 5

max_epochs = 500
batch_size = 64
patience = 30


features_gene_guide = ["gene_length", "gene_GC_content", "distance_operon", "operon_downstream_genes", "ess_gene_operon", "gene_expression_min", "gene_expression_max",
               "guide_GC_content", "distance_start_codon", "homopolymers", "MFE_hybrid_full", "MFE_hybrid_seed", "MFE_homodimer_guide", "MFE_monomer_guide", 
               "off_target_90_100", "off_target_80_90", "off_target_70_80", "off_target_60_70"]

features_guide = ["guide_GC_content", "distance_start_codon", "homopolymers", "MFE_hybrid_full", "MFE_hybrid_seed", "MFE_homodimer_guide", "MFE_monomer_guide", 
               "off_target_90_100", "off_target_80_90", "off_target_70_80", "off_target_60_70"]

datasets = ["wang_orig_guide-genes","wang_orig_guide","wang_median-sub_guide-genes","wang_median-sub_guide","wang_rank_guide-genes","wang_rank_guide",
            "rousset_E18_orig_guide-genes","rousset_E18_orig_guide","rousset_E18_median-sub_guide-genes","rousset_E18_median-sub_guide","rousset_E18_rank_guide-genes","rousset_E18_rank_guide",
            "rousset_E75_orig_guide-genes","rousset_E75_orig_guide","rousset_E75_median-sub_guide-genes","rousset_E75_median-sub_guide","rousset_E75_rank_guide-genes","rousset_E75_rank_guide",
            "wang_rousset_E18_orig_guide-genes","wang_rousset_E18_orig_guide","wang_rousset_E18_median-sub_guide-genes","wang_rousset_E18_median-sub_guide","wang_rousset_E18_rank_guide-genes","wang_rousset_E18_rank_guide",
            "wang_rousset_E75_orig_guide-genes","wang_rousset_E75_orig_guide","wang_rousset_E75_median-sub_guide-genes","wang_rousset_E75_median-sub_guide","wang_rousset_E75_rank_guide-genes","wang_rousset_E75_rank_guide",
            "wang_rousset_E18_rousset_E75_orig_guide-genes","wang_rousset_E18_rousset_E75_orig_guide","wang_rousset_E18_rousset_E75_median-sub_guide-genes","wang_rousset_E18_rousset_E75_median-sub_guide","wang_rousset_E18_rousset_E75_rank_guide-genes","wang_rousset_E18_rousset_E75_rank_guide"]

#datasets = ["wang_median-sub_guide", "rousset_E18_median-sub_guide", "rousset_E75_median-sub_guide", "wang_rousset_E18_median-sub_guide", "wang_rousset_E75_median-sub_guide", "wang_rousset_E18_rousset_E75_median-sub_guide"]

models = ["1DCNN", "GRU"]

## setup performance tables
perf = pd.DataFrame(columns = models, index = datasets)

# mse_train, mse_test, spearmanR_train, spearmanR_test
performance_table = [perf,perf,perf,perf]
#performance_table = reload_performance(output_performance)

## load data
data_wang = get_data(file_data_wang)
one_hot_encoding_wang = get_data(file_one_hot_encoding_wang)
kmer_wang = get_data(file_kmer_wang)

data_rousset_E18 = get_data(file_data_rousset_E18)
one_hot_encoding_rousset_E18 = get_data(file_one_hot_encoding_rousset_E18)
kmer_rousset_E18 = get_data(file_kmer_rousset_E18)

data_rousset_E75 = get_data(file_data_rousset_E75)
one_hot_encoding_rousset_E75 = get_data(file_one_hot_encoding_rousset_E75)
kmer_rousset_E75 = get_data(file_kmer_rousset_E75)

## Create X and Y variables
model_features_wang = create_model_features(data_wang,one_hot_encoding_wang,kmer_wang,features_gene_guide,features_guide)
model_features_rousset_E18 = create_model_features(data_rousset_E18,one_hot_encoding_rousset_E18,kmer_rousset_E18,features_gene_guide,features_guide)
model_features_rousset_E75 = create_model_features(data_rousset_E75,one_hot_encoding_rousset_E75,kmer_rousset_E75,features_gene_guide,features_guide)


# run all combinations of models and datasets
for model in models:
    
        for dataset in datasets:
            print(dataset)

            # get correct Y
            if "orig" in dataset:
                key_Y = "Y_orig"
            elif "median-sub" in dataset:
                key_Y = "Y_median_sub"
            elif "rank" in dataset:
                key_Y = "Y_rank"

            # get correct X
            if "guide-genes" in dataset:
                key_X = "X_gene_guide"
                features = features_gene_guide.copy()
            else:
                key_X = "X_guide"
                features = features_guide.copy()

            # get correct dataset
            if "wang_rousset_E18_rousset_E75" in dataset:
                model_features = model_features_wang
                model_features_add1 = model_features_rousset_E18
                model_features_add2 = model_features_rousset_E75
                X_add = pd.concat([model_features_add1[key_X].copy(), model_features_add2[key_X].copy()],axis=0)
                X_add.reset_index(inplace=True,drop=True)
                Y_add = pd.concat([model_features_add1[key_Y].copy(), model_features_add2[key_Y].copy()],axis=0)
                Y_add.reset_index(inplace=True,drop=True)
                group_ids_add = model_features_add1["geneid"] + model_features_add2["geneid"]

            elif "wang_rousset_E18" in dataset:
                model_features = model_features_wang
                model_features_add = model_features_rousset_E18
                X_add = model_features_add[key_X].copy()
                Y_add = model_features_add[key_Y].copy()
                group_ids_add = model_features_add["geneid"]

            elif "wang_rousset_E75" in dataset:
                model_features = model_features_wang
                model_features_add = model_features_rousset_E75
                X_add = model_features_add[key_X].copy()
                Y_add = model_features_add[key_Y].copy()
                group_ids_add = model_features_add["geneid"]

            elif "wang" in dataset:
                model_features = model_features_wang
                X_add = None
                Y_add = None
                group_ids_add = None

            elif "rousset_E18" in dataset:
                model_features = model_features_rousset_E18
                X_add = None
                Y_add = None
                group_ids_add = None

            elif "rousset_E75" in dataset:
                model_features = model_features_rousset_E75
                X_add = None
                Y_add = None
                group_ids_add = None


            #create directories and output plots
            output_dir_model = output_models + "/" + model + "/" + dataset

            filename_predictions = output_models + "/" + model + "/predictions_" + dataset + ".csv"
            os.makedirs(os.path.dirname(filename_predictions), exist_ok=True)

            #filename_plots = output_plots + "/" + model + "/" + dataset + ".pdf"
            #os.makedirs(os.path.dirname(filename_plots), exist_ok=True)
            #pp = PdfPages(filename_plots)

            #train model and write performance and predictions
            performance, predictions = dm.run_deep_model(model, model_features[key_X].copy(), model_features[key_Y].copy(), features, model_features["geneid"], GroupKFold(n_splits=n_folds_outer), 
                                                         n_folds_outer, n_folds_inner, batch_size, max_epochs, patience, output_dir_model,
                                                         X_add = X_add, Y_add = Y_add, group_ids_add = group_ids_add)


            performance_table = write_performance(performance_table, performance, model, dataset, outfiles = output_performance)
            write_predictions(predictions, model_features["Y_orig"].copy(), filename_predictions)
            #pp.close()