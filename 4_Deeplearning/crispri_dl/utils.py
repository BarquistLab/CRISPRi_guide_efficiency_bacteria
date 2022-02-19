############################################
# imports
############################################

import os
import pickle
import numpy as np
import pandas as pd
import itertools


############################################
# utils functions
############################################

def get_data(file):
    data = pickle.load(open(file, 'rb'))
    return data


############################################

def create_model_features(data, one_hot_encoding, kmer, features_gene_guide, features_guide):
    X_gene_guide_seq = pd.concat([data[features_gene_guide],one_hot_encoding],axis=1)
    X_guide_seq = pd.concat([data[features_guide],one_hot_encoding],axis=1)
    X_gene_guide_kmer = pd.concat([data[features_gene_guide],kmer],axis=1)
    X_guide_kmer = pd.concat([data[features_guide],kmer],axis=1)
    Y_orig = data['log2FC']
    Y_median_sub = data['log2FC'] - data['log2FC_gene_median']
    Y_rank = data["log2FC_normalized_rank"]
    geneid = data["geneid"].tolist()
    model_features = {'X_gene_guide':X_gene_guide_seq,'X_guide':X_guide_seq,'X_gene_guide_kmer':X_gene_guide_kmer,'X_guide_kmer':X_guide_kmer,'Y_orig':Y_orig,'Y_median_sub':Y_median_sub,'Y_rank':Y_rank,'geneid':geneid}
    return model_features


############################################

def scaling(SCALE, X, feature_to_normalize):
  X_scale = pd.DataFrame(SCALE.transform(X[feature_to_normalize]))
  X_scale.columns = X[feature_to_normalize].columns
  X_scale = pd.concat([X_scale,X.drop(feature_to_normalize, axis=1).reset_index(drop=True)],axis=1)
  X_scale.reset_index(inplace=True,drop=True)
  return X_scale


############################################

def write_performance(performance_table, performance, model, data_set, outfiles):
    performance_mean = performance.mean(axis=0).round(3)
    
    for i in range(len(performance_mean)):
        perf = performance_table[i].copy()
        perf.loc[[data_set],[model]] = performance_mean[i]
        performance_table[i] = perf
        
    # write performance to csv output
    os.makedirs(os.path.dirname(outfiles), exist_ok=True)
    performance_table[0].to_csv(outfiles + "_mse_train.csv", index=True)
    performance_table[1].to_csv(outfiles + "_mse_test.csv", index=True)
    performance_table[2].to_csv(outfiles + "_spearmanR_train.csv", index=True)
    performance_table[3].to_csv(outfiles + "_spearmanR_test.csv", index=True)
    
    return performance_table


############################################

def reload_performance(output_performance):
    
    # load performance from csv output
    mse_train = pd.read_csv(output_performance + "_mse_train.csv", index_col = 0)
    mse_test = pd.read_csv(output_performance + "_mse_test.csv", index_col = 0)
    spearmanR_train = pd.read_csv(output_performance + "_spearmanR_train.csv", index_col = 0)
    spearmanR_test = pd.read_csv(output_performance + "_spearmanR_test.csv", index_col = 0)

    performance_table = [mse_train,mse_test,spearmanR_train,spearmanR_test]
    
    return performance_table


############################################

def write_predictions(predictions, Y_orig, filename_predictions):
    predictions['log2FC_original'] = Y_orig
    predictions.to_csv(filename_predictions,index=False)

    
############################################

def get_colnames(item1, item2):
    colnames = []
    combinations = list(itertools.product(item1, item2))
    
    for combination in combinations:
        colnames.append(combination[0] + "_" + combination[1])
        
    return colnames


    
    
