#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:34:42 2019

@author: yanying
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import itertools
import os
import time 
import seaborn as sns
import pandas
import sklearn.model_selection
import sklearn.metrics
import autosklearn
import autosklearn.metrics
import autosklearn.regression
from scipy.stats import spearmanr,pearsonr
import sys
import warnings
warnings.filterwarnings('ignore')
start_time=time.time()
nts=['A','T','C','G']
items=list(itertools.product(nts,repeat=2))
dinucleotides=list(map(lambda x: x[0]+x[1],items))
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to optimize models with individual or fused datasets using auto-sklearn (tested version 0.10.0).

ensemble_size, folds, per_run_time_limit, time_left_for_this_task, include_estimators, and include_preprocessors are parameters for auto-sklearn. More description please check the API of auto-sklearn (https://automl.github.io/auto-sklearn/master/api.html)

Example: python autosklearn_datafusion.py -training 0,1
                  """)
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.2")
parser.add_argument("-training", type=str, default='0,1,2', 
                    help="""
Which datasets to use: 
    0: E75 Rousset
    1: E18 Cui
    2: Wang
    0,1: E75 Rousset & E18 Cui
    0,2: E75 Rousset & Wang
    1,2: E18 Cui & Wang
    0,1,2: all 3 datasets
default: 0,1,2""")
parser.add_argument("-e","--ensemble_size", type=int, default=1, help="Ensemble size, default: 1")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-prt","--per_run_time_limit", type=int, default=360, help="per_run_time_limit (in second), default: 360")
parser.add_argument("-ptt","--time_left_for_this_task", type=int, default=3600, help="time_left_for_this_task (in second), default: 3600")
parser.add_argument("-inest","--include_estimators", type=str, default=None, help="estimators to be included in auto-sklearn. Multiple input separated by ','. If None, then include all. Default: None")
parser.add_argument("-inprepro","--include_preprocessors", type=str, default=None, help="preprocessors to be included in auto-sklearn. Multiple input separated by ','. If None, then include all. Default: None")


args = parser.parse_args()
training_sets=args.training
ensemble_size=args.ensemble_size
folds=args.folds
per_run_time_limit=args.per_run_time_limit
time_left_for_this_task=args.time_left_for_this_task
include_estimators=args.include_estimators
include_preprocessors=args.include_preprocessors
datasets=['../0_Datasets/E75_Rousset.csv','../0_Datasets/E18_Cui.csv','../0_Datasets/Wang_dataset.csv']
training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}

### esitmator and preprocessor setting for auto sklearn
if include_estimators != None:
    include_estimators=include_estimators.split(',')
if include_preprocessors != None:
    include_preprocessors=include_preprocessors.split(',')
if training_sets != None:
    if ',' in training_sets:
        training_sets=[int(i) for i in training_sets.split(",")]
    else:
        training_sets=[int(training_sets)]
output_file_name = args.output
folds=args.folds
test_size=args.test_size
try:
    os.mkdir(output_file_name)
except:
    overwrite=input("File exists, do you want to overwrite? (y/n)")
    if overwrite == "y":
        os.system("rm -r %s"%output_file_name)
        os.mkdir(output_file_name)
    elif overwrite =="n":
        output_file_name=input("Please give a new output file name:")
        os.mkdir(output_file_name)
    else:
        print("Please input valid choice..\nAbort.")
        sys.exit()
        
def self_encode(sequence):#one-hot encoding for single nucleotide features
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

def dinucleotide(sequence):#encoding for dinucleotide features
    nts=['A','T','C','G']
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    encoded=np.zeros([(len(nts)**2)*(len(sequence)-1)],dtype=np.float64)
    for nt in range(len(sequence)-1):
        if sequence[nt] == 'N' or sequence[nt+1] =='N':
            print(sequence)
            continue
        encoded[nt*len(nts)**2+dinucleotides.index(sequence[nt]+sequence[nt+1])]=1
    return encoded

def DataFrame_input(df,coding_strand=1):
    ###keep guides for essential genes
    logging_file= open(output_file_name + '/log.txt','a')
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==coding_strand)]
    df=df.dropna()
    for i in list(set(list(df['geneid']))):
        df_gene=df[df['geneid']==i]
        for j in df_gene.index:
            df.at[j,'Nr_guide']=df_gene.shape[0]
    logging_file.write("Number of guides for essential genes: %s \n" % df.shape[0])
    df=df[df['Nr_guide']>=5]#keep only genes with more than 5 guides from all 3 datasets
    
    sequences=list(dict.fromkeys(df['sequence']))
    y=np.array(df['log2FC'],dtype=float)
    ### one hot encoded sequence features
    PAM_encoded=[]
    sequence_encoded=[]
    dinucleotide_encoded=[]
    for i in df.index:
        PAM_encoded.append(self_encode(df['PAM'][i]))
        sequence_encoded.append(self_encode(df['sequence'][i]))
        dinucleotide_encoded.append(dinucleotide(df['sequence_30nt'][i]))
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        df.at[i,'guideid']=sequences.index(df['sequence'][i])
    #check if the length of gRNA and PAM from all samples is the same
    if len(list(set(map(len,list(df['PAM'])))))==1:
        PAM_len=int(list(set(map(len,list(df['PAM']))))[0])
    else:
        print("error: PAM len")
    if len(list(set(map(len,list(df['sequence'])))))==1:   
        sequence_len=int(list(set(map(len,list(df['sequence']))))[0])
    else:
        print("error: sequence len")
    if len(list(set(map(len,list(df['sequence_30nt'])))))==1:   
        dinucleotide_len=int(list(set(map(len,list(df['sequence_30nt']))))[0])
    else:
        print("error: sequence len")
    
    guideids=np.array(list(df['guideid']))
    # remove columns that are not used in training
    drop_features=['Nr_guide','coding_strand','guideid',"intergenic","No.","genename","gene_biotype","gene_strand","gene_5","gene_3",
                   "genome_pos_5_end","genome_pos_3_end","guide_strand",'sequence','PAM','sequence_30nt','gene_essentiality',
                   'off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70']
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    X=df.drop(['log2FC'],1)
    dataset_col=np.array(X['dataset'],dtype=int)  
    headers=list(X.columns.values)

    X=X[headers]
    ### feat_type for auto sklearn
    feat_type=[]
    categorical_indicator=['intergenic','coding_strand','geneid','dataset',"gene_strand","gene"]
    feat_type=['Categorical' if headers[i] in categorical_indicator else 'Numerical' for i in range(len(headers)) ] 
    ### add one-hot encoded sequence features columns
    PAM_encoded=np.array(PAM_encoded)
    sequence_encoded=np.array(sequence_encoded)
    dinucleotide_encoded=np.array(dinucleotide_encoded)
    X=np.c_[X,sequence_encoded,PAM_encoded,dinucleotide_encoded]
    ###add one-hot encoded sequence features to headers
    for i in range(sequence_len):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
    for i in range(PAM_len):
        for j in range(len(nts)):
            headers.append('PAM_%s_%s'%(i+1,nts[j]))
    for i in range(dinucleotide_len-1):
        for dint in dinucleotides:
            headers.append(dint+str(i+1)+str(i+2))
    ###add one-hot encoded sequence features to feat_type
    for i in range(PAM_len*4+sequence_len*4+(dinucleotide_len-1)*4*4):
        feat_type.append('Categorical')
    
    X=pandas.DataFrame(data=X,columns=headers)
    logging_file.write("Number of features: %s\n" % len(headers))
    logging_file.write("Features: "+",".join(headers)+"\n\n")
    return X, y,feat_type, headers, guideids,sequences,dataset_col


def Evaluation(output_file_name,y,predictions,name):
    #scores
    output=open(output_file_name+"/result.txt","a")
    spearman_rho,spearman_p_value=spearmanr(y, predictions)
    pearson_rho,pearson_p_value=pearsonr(y, predictions)
    output.write(name+"\n")
    output.write("spearman correlation rho: "+str(spearman_rho)+"\n")
    output.write("spearman correlation p value: "+str(spearman_p_value)+"\n")
    output.write("pearson correlation rho: "+str(pearson_rho)+"\n")
    output.write("pearson correlation p value: "+str(pearson_p_value)+"\n")
    output.write("r2: "+str(sklearn.metrics.r2_score(y,predictions))+"\n")
    output.write("explained_variance_score score of "+name+" :"+str(sklearn.metrics.explained_variance_score(y, predictions))+"\n")
    output.write("Mean absolute error regression loss score of "+name+" :"+str(sklearn.metrics.mean_absolute_error(y, predictions))+"\n")
    y=np.array(y)
    
    # scatter plot
    plt.figure() 
    sns.set_palette("PuBu",2)
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    ax_main.scatter(y,predictions,edgecolors='white',alpha=0.8)
    ax_main.set(xlabel='Experimental log2FC',ylabel='Predicted log2FC')
    ax_xDist.hist(y,bins=70,align='mid',alpha=0.7)
    ax_xDist.set(ylabel='count')
    ax_xDist.tick_params(labelsize=6,pad=2)
    ax_yDist.hist(predictions,bins=70,orientation='horizontal',align='mid',alpha=0.7)
    ax_yDist.set(xlabel='count')
    ax_yDist.tick_params(labelsize=6,pad=2)
    ax_main.text(0.55,0.03,"Spearman R: {0}".format(round(spearman_rho,2)),transform=ax_main.transAxes,fontsize=10)
    ax_main.text(0.55,0.10,"Pearson R: {0}".format(round(pearson_rho,2)),transform=ax_main.transAxes,fontsize=10)
    plt.savefig(output_file_name+'/'+name+'_scatterplot.png',dpi=300)
    plt.close()
    


def main():
    open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
    open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
    df1=pandas.read_csv(datasets[0],sep="\t")
    df1 = df1.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    df1['dataset']=[0]*df1.shape[0]
    open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n"% (datasets[0],df1.shape[0]))
    df2=pandas.read_csv(datasets[1],sep="\t")
    df2 = df2.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    df2['dataset']=[1]*df2.shape[0]
    open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[1],df2.shape[0]))
    df3=pandas.read_csv(datasets[2],sep="\t")
    df3 = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    df3['dataset']=[2]*df3.shape[0]
    df2=df2.append(df3,ignore_index=True)  
    open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[2],df3.shape[0]))
    training_df=df1.append(df2,ignore_index=True)  
    training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    open(output_file_name + '/log.txt','a').write("Training dataset: %s\n"%training_set_list[tuple(training_sets)])
    #dropping unnecessary features and encode sequence features
    X,y,feat_type,headers,guideids, guide_sequence_set,dataset_col=DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    #adding guideid and dataset column for covenient train-test split
    X_df=pandas.DataFrame(data=np.c_[X,y,guideids,dataset_col],columns=headers+['log2FC','guideid','dataset_col'])
        
    guideid_set=list(set(guideids))
    logging_file= open(output_file_name + '/log.txt','a')
    ##split the combined training set into train and test
    guide_train, guide_test = sklearn.model_selection.train_test_split(guideid_set, test_size=test_size,random_state=np.random.seed(111))  
    X_train = X_df[X_df['guideid'].isin(guide_train)]
    X_train=X_train[X_train['dataset_col'].isin(training_sets)]
    y_train=np.array(X_train['log2FC'],dtype=float)
    X_train = X_train[headers]
    X_train=np.array(X_train,dtype=float)
    
    
    test = X_df[X_df['guideid'].isin(guide_test)]
    y_test=np.array(test['log2FC'],dtype=float)
    X_test = test[headers]
    X_test=np.array(X_test,dtype=float)
    
    estimator = autosklearn.regression.AutoSklearnRegressor(
            ensemble_size=ensemble_size,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            include_estimators=include_estimators,
            include_preprocessors=include_preprocessors,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': folds},
            tmp_folder=output_file_name+'/autosklearn_regression_example_tmp',
            output_folder=output_file_name+'/autosklearn_regression_example_out',
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
            disable_evaluator_output=False,
            ensemble_memory_limit=1024, ml_memory_limit= 3072,
            ensemble_nbest=50, seed = 1,
            exclude_estimators=None,exclude_preprocessors=None,get_smac_object_callback=None,
            initial_configurations_via_metalearning=25,
            logging_config=None, metadata_directory = None,
            n_jobs= None, smac_scenario_args= None,metric=autosklearn.metrics.mean_squared_error) #max_models_on_disc not included in version 0.5.2
    
    estimator.fit(X_train.copy(), y_train.copy(),feat_type=feat_type)
    estimator.refit(X_train.copy(), y_train.copy())

    logging_file.write("Get parameters:\n"+str(estimator.get_params())+"\n\n")
    logging_file.write("Show models: \n"+str(estimator.show_models())+"\n\n")
    logging_file.write("sprint statistics: %s \n\n"% (estimator.sprint_statistics()))
    
    predictions = estimator.predict(np.array(X_test,dtype=float))
    Evaluation(output_file_name,y_test,predictions,"X_test")
    

if __name__ == '__main__':
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
#%%
