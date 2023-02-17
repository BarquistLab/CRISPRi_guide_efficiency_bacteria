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
from scipy.stats import spearmanr,pearsonr
from collections import defaultdict
import shap
import sys
import pickle
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
This is used to train optimized models from auto-sklearn with diffferent feature sets and evaluate with 10-fold cross-validation. 

Example: python feature_sets.py -o test -c add_distance -r regressor.pkl
                  """)
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.2")
parser.add_argument("-r","--regressor", type=str, default=None, help="Saved regressor from autosklearn, default: None")
parser.add_argument("-c","--choice", type=str, default='all', 
                    help="""
Which feature sets to use: 
    all: all 137 features
    only_seq: only sequence features
    add_distance: sequence and distance features
    add_MFE:sequence, distance, and MFE features
    only_guide:all 128 guide features
    guide_geneid: all 128 guide features and geneID
    gene_seq:sequence features and gene features
    add_deltaGB: sequence, distance, and CRISPRoff score features
    all_deltaGB: replacing 4 MFE features with CRISPRoff score
default: all""")
args = parser.parse_args()
choice=args.choice
output_file_name = args.output
folds=args.folds
test_size=args.test_size
regressor=args.regressor
training_sets=[0]  ###For the results in the paper, only tested on E75 Rousset data.
datasets=['../0_Datasets/E75_Rousset.csv','../0_Datasets/E18_Cui.csv','../0_Datasets/Wang_dataset.csv']
training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}

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


def DataFrame_input(df,coding_strand=1):
    ###keep guides for essential genes
    logging_file= open(output_file_name + '/log.txt','a')
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==coding_strand)]
    df=df.dropna()
    # for i in list(set(list(df['geneid']))):
    #     df_gene=df[df['geneid']==i]
    #     for j in df_gene.index:
    #         df.at[j,'Nr_guide']=df_gene.shape[0]
    # print(len(set(df[df['dataset']==1]['geneid'])))
    for dataset in range(len(set(df['dataset']))):
        dataset_df=df[df['dataset']==dataset]
        for i in list(set(dataset_df['geneid'])):
            gene_df=dataset_df[dataset_df['geneid']==i]
            for j in gene_df.index:
                df.at[j,'Nr_guide']=gene_df.shape[0]
    logging_file.write("Number of guides for essential genes: %s \n" % df.shape[0])
    df=df[df['Nr_guide']>=5] #keep only genes with more than 5 guides from all 3 datasets
    logging_file.write("Number of guides after filtering: %s \n" % df.shape[0])
    # print(len(set(df[df['dataset']==1]['geneid'])))
    sequences=list(dict.fromkeys(df['sequence']))
    
    y=np.array(df['log2FC'],dtype=float)
    ### one hot encoded sequence features
    sequence_encoded=[]
    for i in df.index:
        sequence_encoded.append(self_encode(df['sequence_30nt'][i]))
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        df.at[i,'guideid']=sequences.index(df['sequence'][i])
    
    guideids=np.array(list(df['guideid']))
    # remove columns that are not used in training
    drop_features=['geneid','std','Nr_guide','coding_strand','guideid',"intergenic","No.","genename","gene_biotype","gene_strand","gene_5","gene_3",
                   "genome_pos_5_end","genome_pos_3_end","guide_strand",'sequence','PAM','sequence_30nt','gene_essentiality',
                   'off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70','spacer_self_fold','RNA_DNA_eng','DNA_DNA_opening']
    if choice=='all' or choice=='only_guide' or choice=='guide_geneid':
        drop_features+=['CRISPRoff_score']
    elif choice=='all_deltaGB':
        drop_features+=['MFE_hybrid_seed','MFE_homodimer_guide','MFE_hybrid_full','MFE_monomer_guide']
    if choice=='guide_geneid':
        drop_features.remove('geneid')
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
        
    X=df.drop(['log2FC'],1)
    dataset_col=np.array(X['dataset'],dtype=int)  
    headers=list(X.columns.values)
    gene_features=['dataset','geneid',"gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
    #different opptions for feature sets
    if choice=='only_guide':
        headers=[item for item in headers if item not in gene_features]
    elif choice=='add_distance':
        headers=['distance_start_codon','distance_start_codon_perc']
    elif choice=='add_MFE':
        headers=['distance_start_codon','distance_start_codon_perc']+['MFE_hybrid_full','MFE_hybrid_seed','MFE_homodimer_guide','MFE_monomer_guide']
    elif choice=='add_deltaGB':
        headers=['distance_start_codon','distance_start_codon_perc']+['CRISPRoff_score']
    elif choice=="guide_geneid":
        headers=[item for item in headers if item not in gene_features]+['geneid']
    elif choice=='gene_seq':
        headers=[item for item in headers if item in gene_features]
    X=X[headers]
    ### add one-hot encoded sequence features columns
    sequence_encoded=np.array(sequence_encoded)
    X=np.c_[X,sequence_encoded]
    ###add one-hot encoded sequence features to headers
    sequence_headers=list()
    for i in range(30):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
            sequence_headers.append('sequence_%s_%s'%(i+1,nts[j]))
    if choice=='only_seq':
        X=pandas.DataFrame(X,columns=headers)
        X=X[sequence_headers]
        headers=sequence_headers
    
    X=pandas.DataFrame(data=X,columns=headers)
    logging_file.write("Number of features: %s\n" % len(headers))
    logging_file.write("Features: "+",".join(headers)+"\n\n")
    return X, y, headers, guideids,sequences,dataset_col


def Evaluation(output_file_name,y,predictions,name):
    # scatter plot
    spearman_rho,spearman_p_value=spearmanr(y, predictions)
    pearson_rho,pearson_p_value=pearsonr(y, predictions)
    y=np.array(y)
    
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
    

def SHAP(estimator,X,y,headers):
    X=pandas.DataFrame(X,columns=headers)
    X=X.astype(float)
    explainer=shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X,check_additivity=False)
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')
    
    shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.svg",dpi=400)
    plt.close()
    
    for i in [10,15,30]:
        shap.summary_plot(shap_values, X,show=False,max_display=i,alpha=0.5)
        plt.subplots_adjust(left=0.45, top=0.95,bottom=0.2)
        plt.yticks(fontsize='small')
        plt.xticks(fontsize='small')
        plt.savefig(output_file_name+"/shap_value_top%s.svg"%(i),dpi=400)
        plt.close()    

def main():
    open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
    open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
    # load 3 datesets
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
    X,y,headers,guideids, guide_sequence_set,dataset_col=DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    
    
    numerical_indicator=["gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max",\
                          'distance_start_codon','distance_start_codon_perc','guide_GC_content','MFE_hybrid_seed','MFE_homodimer_guide','MFE_hybrid_full','MFE_monomer_guide','homopolymers']
    dtypes=dict()
    for feature in headers:
        if feature not in numerical_indicator:
            dtypes.update({feature:int})
    X=pandas.DataFrame(data=X,columns=headers)
    X=X.astype(dtypes)

    #optimized models from auto-sklearn
    if regressor !=None:
        estimator=pickle.load(open(regressor,'rb'))
        print(estimator.get_params())
        
        params=estimator.get_params()
        params.update({"random_state":np.random.seed(111)})
        if 'max_iter' in params.keys():
            params.update({'max_iter':512})
        if 'early_stop' in params.keys():
            params.pop('early_stop', None)
        if params['max_depth']=='None':
            params['max_depth']=None
        if params['max_leaf_nodes']=='None':
            params['max_leaf_nodes']=None
        
        if 'Gradient Boosting' in str(estimator):
            from sklearn.experimental import enable_hist_gradient_boosting
            from sklearn.ensemble import HistGradientBoostingRegressor
            estimator=HistGradientBoostingRegressor(**params)
        elif 'Extra Trees' in str(estimator):
            estimator=sklearn.ensemble.ExtraTreesRegressor(**params)
        elif 'Random Forest' in str(estimator):
            from sklearn.ensemble import RandomForestRegressor
            estimator=RandomForestRegressor(**params)
        else:
            print(str(estimator))
            estimator=estimator.set_params(**params)
    else:
        print("Please include the saved regressor from auto-sklearn with option -r. ")
        print("Abort.")
        sys.exit()
    
    open(output_file_name + '/log.txt','a').write("Estimator:"+str(estimator)+"\n")
    X_df=pandas.DataFrame(data=np.c_[X,y,guideids,dataset_col],columns=headers+['log2FC','guideid','dataset_col'])
    
    #k-fold cross validation
    evaluations=defaultdict(list)
    kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
    fold=1
    # plt.figure()
    guideid_set=list(set(guideids))
    for train_index, test_index in kf.split(guideid_set):
        ##split the combined training set into train and test based on guideid
        train_index=np.array(guideid_set)[train_index]
        test_index=np.array(guideid_set)[test_index]
        X_train = X_df[X_df['guideid'].isin(train_index)]
        X_train=X_train[X_train['dataset_col'].isin(training_sets)]
        y_train=X_train['log2FC']
        X_train=X_train[headers]
        X_train=X_train.astype(dtypes)
        
        test = X_df[X_df['guideid'].isin(test_index)]
        X_test=test[(test['dataset_col'].isin(training_sets))]
        y_test=X_test['log2FC']
        X_test=X_test[headers]
        X_test=X_test.astype(dtypes)
        estimator = estimator.fit(np.array(X_train,dtype=float),np.array(y_train,dtype=float))
        
        predictions = estimator.predict(np.array(X_train,dtype=float))
        evaluations['Rs_train'].append(spearmanr(y_train, predictions)[0])
        predictions = estimator.predict(np.array(X_test,dtype=float))
        evaluations['Rs'].append(spearmanr(y_test, predictions)[0])
        
        # sns.distplot(predictions,label='Fold %s'%fold)
        # Evaluation(output_file_name,y_test,predictions,"X_test_fold_%s"%fold)
        fold+=1
        if len(datasets)>1: #evaluation in mixed and each individual dataset(s)
            X_test_1=test[headers]
            y_test_1=np.array(test['log2FC'],dtype=float)
            predictions=estimator.predict(np.array(X_test_1,dtype=float))
            spearman_rho,_=spearmanr(y_test_1, predictions)
            evaluations['Rs_test_mixed'].append(spearman_rho)
            # Evaluation(output_file_name,y_test_1,predictions,"X_test_mixed")
            for dataset in range(len(datasets)):
                dataset1=test[test['dataset_col']==dataset]
                X_test_1=dataset1[headers]
                y_test_1=np.array(dataset1['log2FC'],dtype=float)
                predictions=estimator.predict(np.array(X_test_1,dtype=float))
                spearman_rho,_=spearmanr(y_test_1, predictions)
                evaluations['Rs_test%s'%(dataset+1)].append(spearman_rho)
                # Evaluation(output_file_name,y_test_1,predictions,"X_test_%s"%(dataset+1))
    # plt.legend()
    # plt.xlabel('Predicted values',fontsize=14)
    # plt.ylabel('Density',fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.savefig(output_file_name+"/predicted_dist.svg")
    # plt.close()
    evaluations=pandas.DataFrame.from_dict(evaluations)
    evaluations.to_csv(output_file_name+'/iteration_scores.csv',sep='\t',index=True)
    
    ##split the combined training set into train and test
    guide_train, guide_test = sklearn.model_selection.train_test_split(list(set(guideids)), test_size=test_size,random_state=np.random.seed(111))  
    X_train = X_df[X_df['guideid'].isin(guide_train)]
    X_train=X_train[X_train['dataset_col'].isin(training_sets)]
    y_train=X_train['log2FC']
    X_train = X_train[headers]
    X_train=X_train.astype(dtypes)
    estimator.fit(np.array(X_train,dtype=float),np.array(y_train,dtype=float))
    SHAP(estimator,X_train,y,headers) #model interpretation using SHAP
    

if __name__ == '__main__':
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
