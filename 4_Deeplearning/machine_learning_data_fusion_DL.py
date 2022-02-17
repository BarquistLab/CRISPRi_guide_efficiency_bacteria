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
import logging
import pandas
import sklearn.model_selection
import sklearn.metrics
import autosklearn.metrics
from scipy.stats import spearmanr,pearsonr
from sklearn.cluster import KMeans,FeatureAgglomeration
from sklearn.feature_selection import VarianceThreshold,GenericUnivariateSelect,f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,Normalizer,RobustScaler,MinMaxScaler
from collections import defaultdict
import shap
import sys
import textwrap
import pickle
import autosklearn
from scipy import sparse
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from autosklearn.pipeline.implementations.SparseOneHotEncoder import SparseOneHotEncoder
from sklearn.preprocessing import OneHotEncoder
import autosklearn.pipeline.implementations.CategoryShift
from autosklearn.pipeline.implementations.MinorityCoalescer import MinorityCoalescer
from autosklearn.pipeline.components.data_preprocessing.minority_coalescense.no_coalescense import NoCoalescence
start_time=time.time()
nts=['A','T','C','G']
items=list(itertools.product(nts,repeat=2))
dinucleotides=list(map(lambda x: x[0]+x[1],items))
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawDescriptionHelpFormatter,description=textwrap.dedent('''\
                  This is used to train regression models using autosklearn. 
                  
                  Example: python machine_learning_autosklearn_regressor.py Rousset_dataset.csv,Wang_dataset.csv
                  '''))
parser.add_argument("datasets", help="data csv file(s),multiple files separated by ','")
parser.add_argument("-training", type=str, default=None, help="Dataset for training. Count starts from 0. If None,then all input datasets")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.3")
parser.add_argument("-c","--choice", type=str, default='CNN', help="CNN/GRU, default: CNN")
parser.add_argument("-nc", default=False, help="Whether add non-coding strand targeting guides for training or not, default: False")
parser.add_argument("-s","--split", type=str, default='guide', help="Train-test split (gene/guide), default: guide")


args = parser.parse_args()
datasets=args.datasets
training_sets=args.training
choice=args.choice
split=args.split
if ',' in datasets:
    datasets=datasets.split(",")
else:
    datasets=[datasets]
if training_sets != None:
    if ',' in training_sets:
        training_sets=[int(i) for i in training_sets.split(",")]
    else:
        training_sets=[int(training_sets)]
    
else:
    training_sets=range(len(datasets))
output_file_name = args.output
folds=args.folds
test_size=args.test_size
add_nc=args.nc
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
def self_encode(sequence):
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

def dinucleotide(sequence):
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

def ROC_plot(fpr,tpr,roc_auc_score,name):
    plt.figure()
    plt.plot(fpr, tpr, color='#1f77b4',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(output_file_name+'/'+name+ '_ROC.png',dpi=300)
    plt.close()
def encode_sequence(sequence):
   
    alphabet = 'AGCT'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    #encoded_sequence_old = tf.keras.utils.to_categorical(integer_encoded, num_classes=4)
    encoded_sequence = np.eye(4)[integer_encoded]
    return encoded_sequence
def DataFrame_input(df,coding_strand=1):
    ###keep guides for essential genes
    logging_file= open(output_file_name + '/Output.txt','a')
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==coding_strand)]
    df=df.dropna()
    print(df.shape)
    for i in list(set(list(df['geneid']))):
        df_gene=df[df['geneid']==i]
        for j in df_gene.index:
            df.at[j,'Nr_guide']=df_gene.shape[0]
    df=df[df['Nr_guide']>=5]
    print(df.shape)
    logging_file.write("Number of guides for essential genes: %s \n" % df.shape[0])
    sequences=list(dict.fromkeys(df['sequence']))
    print(len(sequences))
    
    # dinucleotide_encoded=[]
    y=np.array(df['log2FC'],dtype=float)
    
    for i in df.index:
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        df.at[i,'guideid']=sequences.index(df['sequence'][i])
    
    if split=='guide':
        guideids=np.array(list(df['guideid']))
    elif split=='gene':
        guideids=np.array(list(df['geneid']))
    drop_features=['Nr_guide','coding_strand','guideid',"intergenic","No.","genename","gene_biotype","gene_strand","gene_5","gene_3",
                   "genome_pos_5_end","genome_pos_3_end","guide_strand",'sequence','PAM','gene_essentiality',
                   'off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70']
    if split=='gene':
        drop_features.append("geneid")
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    if add_nc == 'True':
        df['coding_strand']=[coding_strand]*df.shape[0]
    # if len(datasets)==1:
    #     df=df.drop('dataset',1)
    X=df.drop(['log2FC'],1)#activity_score
    dataset_col=np.array(X['dataset'],dtype=int)  
    X=X.rename(columns={'sequence_30nt':'sequence_40nt'})
    X['sequence_40nt'] = X.apply(lambda row : encode_sequence(row['sequence_40nt']), axis = 1)
    headers=list(X.columns.values)
    #use Kmeans clustering to cluster log2FC into two groups
    kmeans = KMeans(n_clusters=2, random_state=0).fit(y.reshape(-1,1))
    logging_file.write("kmeans cluster centers: %s, %s\n" % (round(kmeans.cluster_centers_[0][0],2),round(kmeans.cluster_centers_[1][0],2)))
    
    # features=['dataset','geneid',"gene_5","gene_strand","gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
    # guide_features=[item for item in headers if item not in features]
    # headers=guide_features.copy()
    # guide_features.append("dataset")
    # X=X[guide_features]
    ### feat_type for auto sklearn
    feat_type=[]
    categorical_indicator=['intergenic','coding_strand','geneid','dataset',"gene_strand","gene"]
    feat_type=['Categorical' if headers[i] in categorical_indicator else 'Numerical' for i in range(len(headers)) ] 
    ### add one-hot encoded sequence features columns
    # PAM_encoded=np.array(PAM_encoded)
    # sequence_encoded=np.array(sequence_encoded)
    # dinucleotide_encoded=np.array(dinucleotide_encoded)
    # X=np.c_[X,sequence_encoded,PAM_encoded,dinucleotide_encoded]
    
    ###add one-hot encoded sequence features to headers
    # sequence_headers=list()
    # sequence_headers=['base_A','base_T','base_C','base_G']
    # for dint in dinucleotides:
        # sequence_headers.append(dint)
    # feat_type=['Numerical']*20
    # for i in range(sequence_len):
    #     for j in range(len(nts)):
    #         headers.append('sequence_%s_%s'%(i+1,nts[j]))
    #         sequence_headers.append('sequence_%s_%s'%(i+1,nts[j]))
    # for i in range(PAM_len):
    #     for j in range(len(nts)):
    #         headers.append('PAM_%s_%s'%(i+1,nts[j]))
    #         sequence_headers.append('PAM_%s_%s'%(i+1,nts[j]))
    # for i in range(dinucleotide_len-1):
    #     for dint in dinucleotides:
    #         headers.append(dint+str(i+1)+str(i+2))
    #         sequence_headers.append(dint+str(i+1)+str(i+2))
    ###add one-hot encoded sequence features to feat_type
    # for i in range(PAM_len*4+sequence_len*4+(dinucleotide_len-1)*4*4):
        # feat_type.append('Categorical')
    
    X=pandas.DataFrame(data=X,columns=headers)
    print(X.shape)
    print(headers)
    logging_file.write("Number of features: %s\n" % len(headers))
    return X, y, kmeans,feat_type, headers, guideids,sequences,dataset_col


def Evaluation(output_file_name,y,predictions,kmeans,kmeans_train,name):
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
#    output.write("Max error of "+name+" :"+str(sklearn.metrics.max_error(y, predictions))+"\n") #Only with scikit-learn version >=0.20
    y=np.array(y)
    
    # transfer to a classification problem
    y_test_clustered=kmeans.predict(y.reshape(-1,1)) ## kmeans was trained with all samples, extract the first column (second one is useless)
    predictions_clustered=kmeans_train.predict(predictions.reshape(-1, 1))
    if list(kmeans.cluster_centers_).index(min(kmeans.cluster_centers_))==0: ## define cluster with smaller value as class 1 
        y_test_clustered=y_test_clustered
    else:
        y_test_clustered=[1 if x==0 else 0 for x in y_test_clustered] 
    fpr,tpr,_=sklearn.metrics.roc_curve(y_test_clustered,predictions)
    y_test_clustered=[1 if x==0 else 0 for x in y_test_clustered] 
    if list(kmeans_train.cluster_centers_).index(min(kmeans_train.cluster_centers_))==1: ## define cluster with smaller value as class 1 ('good' guides)
        predictions_clustered=predictions_clustered
    else:
        predictions_clustered=[1 if x==0 else 0 for x in predictions_clustered] 

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
    plt.savefig(output_file_name+'/'+name+'_classes_scatterplot.png',dpi=300)
    plt.close()
    
    # evaluations for classicfication
    with open(output_file_name+"/result.txt","a") as output:        
        output.write("Accuracy score of "+name+" :"+str(sklearn.metrics.accuracy_score(y_test_clustered, predictions_clustered))+"\n")
        output.write("precision_score of "+name+" :"+str(sklearn.metrics.precision_score(y_test_clustered, predictions_clustered))+"\n")
        output.write("recall_score of "+name+ ":"+str(sklearn.metrics.recall_score(y_test_clustered, predictions_clustered))+"\n")
        output.write("f1 of "+name+" :"+str(sklearn.metrics.f1_score(y_test_clustered, predictions_clustered))+"\n")
        output.write("roc_auc_score of "+name+" :"+str(sklearn.metrics.auc(fpr,tpr))+"\n\n")
        
        roc_auc_score=sklearn.metrics.auc(fpr,tpr)
        ROC_plot(fpr,tpr,roc_auc_score,name)
        output.write("The MCC score: %s\n\n" %sklearn.metrics.matthews_corrcoef(y_test_clustered, predictions_clustered))
    return fpr,tpr,roc_auc_score

def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, SparseOneHotEncoder) or isinstance(estimator, OneHotEncoder) :
            # handling all vectorizers
            return [feature_in[int(f.split("_")[0][1:])]+"_"+f.split("_")[1] for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, VarianceThreshold):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct,features):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        # print(name, estimator, features)
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
                
    return output_features

def SHAP(estimator,X_train,y,headers):
    X_train=pandas.DataFrame(X_train,columns=headers)
    shap_values = shap.TreeExplainer(estimator).shap_values(X_train,check_additivity=False)
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')
    
    shap.summary_plot(shap_values, X_train, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.png",dpi=400)
    plt.close()
    
    for i in [10,15,30]:
        shap.summary_plot(shap_values, X_train,show=False,max_display=i,alpha=0.05)
        plt.subplots_adjust(left=0.45, top=0.95,bottom=0.2)
        plt.yticks(fontsize='small')
        plt.xticks(fontsize='small')
        plt.savefig(output_file_name+"/shap_value_top%s.png"%(i),dpi=400)
        plt.close()    
def encode(seq):
    return np.array([[int(b==p) for b in seq] for p in ["A","T","G","C"]])
def find_target(df,before=20,after=20):
    from Bio import SeqIO
    fasta_sequences = SeqIO.parse(open("/home/yanying/projects/crispri/seq/NC_000913.3.fasta"),'fasta')    
    for fasta in fasta_sequences:  # input reference genome
        reference_fasta=fasta.seq 
    extended_seq=[]
    guides_index=list()
    for i in df.index.values:
        if len(df['sequence'][i])!=20 or df["genome_pos_5_end"][i]<20 or df["genome_pos_3_end"][i]<20 :
            continue
        guides_index.append(i)
        if df["genome_pos_5_end"][i] > df["genome_pos_3_end"][i]:
            extended_seq.append(str(reference_fasta[df["genome_pos_3_end"][i]-1-after:df["genome_pos_5_end"][i]+before].reverse_complement()))
        else:
            extended_seq.append(str(reference_fasta[df["genome_pos_5_end"][i]-1-before:df["genome_pos_3_end"][i]+after]))
    return extended_seq,guides_index
def encode_seqarr(seq,r):
    '''One hot encoding of the sequence. r specifies the position range.'''
    X = np.array(
            [encode(''.join([s[i] for i in r])) for s in seq]
        )
    X = X.reshape(X.shape[0], -1)
    return X

def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def convert_int_params(names, params):
    for int_type in names:
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params

def convert_float_params(names, params):
    for float_type in names:
        raw_val = params[float_type]
        if is_number(raw_val):
            params[float_type] = '{:.3f}'.format(raw_val)
    return params
def scorer(reg,X,y):
    return(-1*spearmanr(reg.predict(X),y)[0])
def gene_split_metrics(df):
    metrics={'Rs_per_gene':list(),'std':list(),'top3_log2FC':list(),'top20_pred':list(),'bottom20_pred':list()}
    for i in list(set(df['guideid'])):
        df_gene=df[df['guideid']==i]
        metrics['Rs_per_gene'].append(spearmanr(df_gene['log2FC'],df_gene['pred'])[0])
        metrics['std'].append(np.std(df_gene['log2FC']))
        if np.median(df_gene['log2FC'])<=-2 and df_gene.shape[0]>=3:
            df_gene=df_gene.sort_values(by='pred',ascending=False)
            metrics['top3_log2FC']+=list(df_gene['log2FC'])[:3]
        df_top=df_gene[df_gene['log2FC']<=np.quantile(df_gene['log2FC'],0.2)]
        metrics['top20_pred']+=list(df_top['pred'])
        df_bottom=df_gene[df_gene['log2FC']>=np.quantile(df_gene['log2FC'],0.8)]
        metrics['bottom20_pred']+=list(df_bottom['pred'])
    return metrics
def main():
    df1=pandas.read_csv(datasets[0],sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
    df1 = df1.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    df1['dataset']=[0]*df1.shape[0]
    open(output_file_name + '/Output.txt','a').write("Total number of guides in dataset %s: %s\n"% (datasets[0],df1.shape[0]))
    if len(datasets)>1:       
        df2=pandas.read_csv(datasets[1],sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
        df2 = df2.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
        df2['dataset']=[1]*df2.shape[0]
        open(output_file_name + '/Output.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[1],df2.shape[0]))
        if len(datasets)==3:
            df3=pandas.read_csv(datasets[2],sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
            df3 = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
            df3['dataset']=[2]*df3.shape[0]
            df2=df2.append(df3,ignore_index=True)  
            open(output_file_name + '/Output.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[2],df3.shape[0]))
            # split into training and validation
        training_df=df1.append(df2,ignore_index=True)  
        training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
        
        
    else:
        training_df=df1
    open(output_file_name + '/Output.txt','a').write("Combined training set:\n")
    X,y,kmeans_train,feat_type,headers,guideids, guide_sequence_set,dataset_col=DataFrame_input(training_df)
    print(len(headers))
    open(output_file_name + '/Output.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    open(output_file_name + '/Output.txt','a').write("Features: "+",".join(headers)+"\n\n")
    
    processed_headers=headers
    header=[i for i in headers if i !='sequence_40nt']
    X_rescaled=X
    numerical_indicator=["gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max",\
                          'distance_start_codon','distance_start_codon_perc','guide_GC_content','MFE_hybrid_seed','MFE_homodimer_guide','MFE_hybrid_full','MFE_monomer_guide',\
                        'off_target_60_70','off_target_70_80','off_target_80_90','off_target_90_100','homopolymers']
    dtypes=dict()
    for feature in processed_headers:
        if feature not in numerical_indicator:
            dtypes.update({feature:int})
    X_rescaled=pandas.DataFrame(data=X_rescaled,columns=processed_headers)

    max_epochs = 500
    batch_size = 64
    patience = 30
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    from src.dataloader import CrisprDatasetTrain
    

    # open(output_file_name + '/Output.txt','a').write("Estimator:"+str(estimator)+"\n")
    X_df=pandas.DataFrame(data=np.c_[X_rescaled,y,guideids,dataset_col],columns=processed_headers+['log2FC','guideid','dataset_col'])
    #k-fold cross validation
    evaluations=defaultdict(list)
    kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
    if len(datasets)>1:
        pasteur_test = training_df[(training_df['gene_essentiality']==1)&(training_df['coding_strand']==1)&(training_df['intergenic']==0)]
        pasteur_test=pasteur_test.dropna().reset_index(drop=True)
        for i in list(set(pasteur_test['geneid'])):
            p_gene=pasteur_test[pasteur_test['geneid']==i]
            for j in p_gene.index:
                pasteur_test.at[j,'Nr_guide']=p_gene.shape[0]
        pasteur_test=pasteur_test[pasteur_test['Nr_guide']>=5]
        import statistics
        for dataset in range(len(datasets)):
            test_data=pasteur_test[pasteur_test['dataset']==dataset]
            for i in list(set(test_data['geneid'])):
                gene_df=test_data[test_data['geneid']==i]
                for j in gene_df.index:
                    pasteur_test.at[j,'median']=statistics.median(gene_df['log2FC'])
                    if split=='guide':
                        pasteur_test.at[j,'guideid']=guide_sequence_set.index(pasteur_test['sequence'][j])
                    elif split=='gene':
                        pasteur_test.at[j,'guideid']=int(pasteur_test['geneid'][j][1:])
        
    guideid_set=list(set(guideids))
    fold_inner=0
    for train_index, test_index in kf.split(guideid_set):
        # X_rescaled=np.array(X_rescaled)
        # X_train, X_test = X_rescaled[train_index], X_rescaled[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        # guideid_train, guideid_test = guideids[train_index], guideids[test_index]
        train_index=np.array(guideid_set)[train_index]
        test_index=np.array(guideid_set)[test_index]
        X_train = X_df[X_df['guideid'].isin(train_index)]
        # X_train=pandas.DataFrame(data=np.c_[X_train,y_train,dataset_col[train_index]],columns=processed_headers+['log2FC','dataset_col'])
        X_train=X_train[X_train['dataset_col'].isin(training_sets)]
        
        X_test = X_df[X_df['guideid'].isin(test_index)]
        # X_test=pandas.DataFrame(data=np.c_[X_test,y_test,dataset_col[test_index],guideid_test],columns=processed_headers+['log2FC','dataset_col','guideid'])
        X_test=X_test[(X_test['dataset_col'].isin(training_sets))]
        y_test=X_test['log2FC']
        X_test=X_test[processed_headers]
        
        # train val split
        index_train, index_val = sklearn.model_selection.train_test_split(train_index, test_size=0.2,random_state=np.random.seed(111))
        X_train = X_df[X_df['guideid'].isin(index_train)]
        X_val = X_df[X_df['guideid'].isin(index_val)]
        y_train=X_train['log2FC']
        X_train=X_train[processed_headers]
        y_val=X_val['log2FC']
        X_val=X_val[processed_headers]
        print(X_train.shape,X_val.shape)
        
        #loader
        loader_train = CrisprDatasetTrain(X_train, y_train, header)
        loader_train = DataLoader(loader_train, batch_size=batch_size, num_workers = 6, shuffle = True, drop_last=True)
        dataset_val  = CrisprDatasetTrain(X_val, y_val, header)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers = 6, drop_last=True)
        loader_test = CrisprDatasetTrain(X_test, y_test, header)
        loader_test = DataLoader(loader_test, batch_size=X_test.shape[0], num_workers = 6, shuffle = False)
        #train
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            min_delta=0.0, 
            patience=patience, 
            verbose=False, 
            mode="min")
        checkpoint_callback = ModelCheckpoint(
                    monitor = 'val_loss',
                    dirpath = output_file_name,
                    filename = "model_"+str(fold_inner),
                    verbose = False,
                    save_top_k = 1,
                    mode = 'min',)
        estimator = pl.Trainer( callbacks=[early_stop_callback,checkpoint_callback], max_epochs=max_epochs, check_val_every_n_epoch=1, logger=True,progress_bar_refresh_rate = 0, weights_summary=None)
        from src.architectures import Crispr1DCNN, CrisprGRU
        if choice=='CNN':
            estimator.fit(Crispr1DCNN(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
            #load trained model
            filename_model = output_file_name + '/model_'+str(fold_inner) + ".ckpt"
            trained_model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        elif choice=='GRU':
            estimator.fit(CrisprGRU(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
            #load trained model
            filename_model = output_file_name + '/model_'+str(fold_inner) + ".ckpt"
            trained_model = CrisprGRU.load_from_checkpoint(filename_model, num_features = len(header))
        #test
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        trained_model = trained_model.to(device)
        trained_model.eval()
        trained_model.freeze()
        predictions=list()
        for x_sequence_40nt, x_features, _ in loader_test:
            with torch.no_grad():
                predictions_test = trained_model(x_sequence_40nt.to(device), x_features.to(device)).detach()
        predictions.extend(predictions_test.cpu().numpy())
        predictions=np.array(predictions).flatten()
        fold_inner+=1
        print('mean',spearmanr(y_test, predictions)[0])
        # estimator = estimator.fit(np.array(X_train,dtype=float),np.array(y_train,dtype=float))
        # predictions = estimator.predict(np.array(X_test,dtype=float))
        evaluations['Rs'].append(spearmanr(y_test, predictions)[0])
        evaluations['Rp'].append(pearsonr(y_test, predictions)[0])
        evaluations['MSE'].append(sklearn.metrics.mean_absolute_error(y_test, predictions))
        if split=='gene':
            metrics=gene_split_metrics(pandas.DataFrame(np.c_[X_df[X_df['guideid'].isin(test_index)],predictions],columns=headers+['log2FC','guideid','dataset_col','pred']))
            for key in metrics.keys():
                evaluations[key].append(metrics[key])
        if len(datasets)>1:
            if 'dataset' not in X.columns.values.tolist():
                X_combined=pandas.DataFrame(data=np.c_[X,y,guideids,dataset_col],columns=headers+['log2FC','guideid','dataset'])
            else:
                X_combined=pandas.DataFrame(data=np.c_[X,y,guideids],columns=headers+['log2FC','guideid'])
            X_combined=X_combined[X_combined['guideid'].isin(test_index)]
            #mixed datasets
            X_test_1=X_combined[processed_headers]
            y_test_1=X_combined['log2FC']
            
            loader_test = CrisprDatasetTrain(X_test_1, y_test_1, header)
            loader_test = DataLoader(loader_test, batch_size=X_test_1.shape[0], num_workers = 6, shuffle = False)
            
            
            predictions=list()
            for x_sequence_40nt, x_features, _ in loader_test:
                with torch.no_grad():
                    predictions_test = trained_model(x_sequence_40nt.to(device), x_features.to(device)).detach()
            predictions.extend(predictions_test.cpu().numpy())
            predictions=np.array(predictions).flatten()
            # predictions=estimator.predict(np.array(X_test_1,dtype=float))
            spearman_rho,_=spearmanr(y_test_1, predictions)
            evaluations['Rs_test_mixed'].append(spearman_rho)
            pearson_rho,_=pearsonr(y_test_1, predictions)
            evaluations['Rp_test_mixed'].append(pearson_rho)
            evaluations['MSE_test_mixed'].append(sklearn.metrics.mean_absolute_error(y_test_1, predictions))
            if split=='gene':
                metrics=gene_split_metrics(pandas.DataFrame(np.c_[X_combined,predictions],columns=headers+['log2FC','guideid','pred']))
                for key in metrics.keys():
                    evaluations[key+"_test_mixed"].append(metrics[key])
            for dataset in range(len(datasets)):
                dataset1=X_combined[X_combined['dataset']==dataset]
                X_test_1=dataset1[processed_headers]
                y_test_1=dataset1['log2FC']
                # X_test_1=column_transformer.transform(X_test_1)  
                # X_test_1=preprocessor.transform(X_test_1)
                
                loader_test = CrisprDatasetTrain(X_test_1, y_test_1, header)
                loader_test = DataLoader(loader_test, batch_size=X_test_1.shape[0], num_workers = 6, shuffle = False)
                predictions=list()
                for x_sequence_40nt, x_features, _  in loader_test:
                    with torch.no_grad():
                        predictions_test = trained_model(x_sequence_40nt.to(device), x_features.to(device)).detach()
                predictions.extend(predictions_test.cpu().numpy())
                predictions=np.array(predictions).flatten()
                spearman_rho,_=spearmanr(y_test_1, predictions)
                evaluations['Rs_test%s'%(dataset+1)].append(spearman_rho)
                pearson_rho,_=pearsonr(y_test_1, predictions)
                evaluations['Rp_test%s'%(dataset+1)].append(pearson_rho)
                evaluations['MSE_test%s'%(dataset+1)].append(sklearn.metrics.mean_absolute_error(y_test_1, predictions))
                if split=='gene':
                    metrics=gene_split_metrics(pandas.DataFrame(np.c_[dataset1,predictions],columns=headers+['log2FC','guideid','pred']))
                    for key in metrics.keys():
                        evaluations[key+"_test%s"%(dataset+1)].append(metrics[key])
            
            #Pasteur method test on our test set
            # pasteur_test_index =  np.array(pasteur_test.index)[test_index]
            # pasteur = pasteur_test.iloc[pasteur_test_index]
            # pasteur_test = pandas.DataFrame(data=pasteur_test,columns=training_df.columns.values.tolist())
            test_data=pasteur_test[(pasteur_test['guideid'].isin(test_index))]
            # print('pasteur',test_data.shape)
            training_seq,guides_index=find_target(test_data)
            training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
            training_seq=training_seq.reshape(training_seq.shape[0],-1)
            test_data=test_data.loc[guides_index]
            reg=pickle.load(open('/home/yanying/projects/crispri/CRISPRi_ml/results_new/comparison/Bikard/NAR2020_paper/reg.pkl','rb'))
            test_data['pasteur_score']=reg.predict(training_seq).reshape(-1, 1).ravel()
            
            for i in test_data.index:
                test_data.at[i,'predicted_score']=test_data['median'][i]-test_data['pasteur_score'][i]
            evaluations['Rs_pasteur_data_mixed'].append(spearmanr(test_data['log2FC'], test_data['predicted_score'])[0])
            evaluations['Rp_pasteur_data_mixed'].append(pearsonr(test_data['log2FC'], test_data['predicted_score'])[0])
            evaluations['MSE_pasteur_data_mixed'].append(sklearn.metrics.mean_absolute_error(test_data['log2FC'], test_data['predicted_score']))
            if split=='gene':
                metrics=gene_split_metrics(test_data.rename(columns={'pasteur_score':'pred'}))
                for key in metrics.keys():
                    evaluations[key+"_pasteur_data_mixed"].append(metrics[key])
            for dataset in range(len(datasets)):
                test_data=pasteur_test[(pasteur_test['dataset']==dataset)&(pasteur_test['guideid'].isin(test_index))]
                # print('pasteur',dataset,test_data.shape)
                training_seq,guides_index=find_target(test_data)
                training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
                training_seq=training_seq.reshape(training_seq.shape[0],-1)
                test_data=test_data.loc[guides_index]
                test_data['pasteur_score']=reg.predict(training_seq).reshape(-1, 1).ravel()
                
                for i in test_data.index:
                    test_data.at[i,'predicted_score']=test_data['median'][i]-test_data['pasteur_score'][i]
                evaluations['Rs_pasteur_data%s'%(dataset+1)].append(spearmanr(test_data['log2FC'], test_data['predicted_score'])[0])
                evaluations['Rp_pasteur_data%s'%(dataset+1)].append(pearsonr(test_data['log2FC'], test_data['predicted_score'])[0])
                evaluations['MSE_pasteur_data%s'%(dataset+1)].append(sklearn.metrics.mean_absolute_error(test_data['log2FC'], test_data['predicted_score']))
                if split=='gene':
                    metrics=gene_split_metrics(test_data.rename(columns={'pasteur_score':'pred'}))
                    for key in metrics.keys():
                        evaluations[key+"_pasteur_data%s"%(dataset+1)].append(metrics[key])
     
    evaluations=pandas.DataFrame.from_dict(evaluations)
    evaluations.to_csv(output_file_name+'/iteration_scores.csv',sep='\t',index=True)
    
    logging_file= open(output_file_name + '/Output.txt','a')
    '''
    ##split the combined training set into train and test
    guide_train, guide_test = sklearn.model_selection.train_test_split(list(set(guideids)), test_size=test_size,random_state=np.random.seed(111))  
    X_train = X_df[X_df['guideid'].isin(guide_train)]
    X_train=X_train[X_train['dataset_col'].isin(training_sets)]
    y_train=X_train['log2FC']
    X_train = X_train[processed_headers]
    X_train=X_train.astype(dtypes)
    if add_nc=='True':
        #add non-coding strand guides
        X_train=np.concatenate((X_train, X_ncs),axis=0)
        y_train=np.concatenate((y_train, y_ncs),axis=0)
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_rescaled, y, test_size=test_size,random_state=np.random.seed(111))  
    # X_train=pandas.DataFrame(data=X_train,columns=processed_headers)
    
    
    X_test = X_df[X_df['guideid'].isin(guide_test)]
    X_test=X_test[(X_test['dataset_col'].isin(training_sets))]
    y_test=X_test['log2FC']
    X_test = X_test[processed_headers]
    # X_test=pandas.DataFrame(data=X_test,columns=processed_headers)
    X_test=X_test.astype(dtypes)
    estimator.fit(np.array(X_train,dtype=float),np.array(y_train,dtype=float))
    predictions = estimator.predict(np.array(X_test,dtype=float))
    fpr,tpr,roc_auc_score=Evaluation(output_file_name,y_test,predictions,kmeans_train,kmeans_train,"X_test")
    plt.figure()
    sns.set_palette("PuBu",len(datasets)+1)
    plt.subplots_adjust(bottom=0.15, top=0.96,right=0.96,left=0.15)
    plt.plot(fpr, tpr,lw=2, label='ROC curve (area = %0.2f) of test' % roc_auc_score)
    ### model validation with validation dataset
    if len(datasets)>1:
        
        if 'dataset' not in X.columns.values.tolist():
            X_combined=pandas.DataFrame(data=np.c_[X,y,guideids,dataset_col],columns=headers+['log2FC','guideid','dataset'])
        else:
            X_combined=pandas.DataFrame(data=np.c_[X,y,guideids],columns=headers+['log2FC','guideid'])
        X_combined=X_combined[X_combined['guideid'].isin(guide_test)]
        
        #mixed datasets
        X_test_1=X_combined[headers]
        y_test_1=np.array(X_combined['log2FC'],dtype=float)
        kmeans_test1=KMeans(n_clusters=2, random_state=0).fit(np.array(y_test_1).reshape(-1,1))
        # X_test_1=column_transformer.transform(X_test_1)  
        # X_test_1=preprocessor.transform(X_test_1)
    
        X_test_1=pandas.DataFrame(data=X_test_1,columns=processed_headers)
        X_test_1=X_test_1.astype(dtypes)
        # print('mixed',X_test_1.shape)
        predictions=estimator.predict(np.array(X_test_1,dtype=float))
        fpr_test1,tpr_test1,roc_auc_score_test1=Evaluation(output_file_name,y_test_1,predictions,kmeans_test1,kmeans_train,"dataset_mixed")
        plt.plot(fpr_test1, tpr_test1,
                 lw=2, label='ROC curve (area = %0.2f) of mixed datasets' % roc_auc_score_test1)
        for dataset in range(len(datasets)):
            dataset1=X_combined[X_combined['dataset']==dataset]
            X_test_1=dataset1[headers]
            y_test_1=np.array(dataset1['log2FC'],dtype=float)
            kmeans_test1=KMeans(n_clusters=2, random_state=0).fit(np.array(y_test_1).reshape(-1,1))
        
            # X_test_1=column_transformer.transform(X_test_1)  
            # X_test_1=preprocessor.transform(X_test_1)
        
            X_test_1=pandas.DataFrame(data=X_test_1,columns=processed_headers)
            X_test_1=X_test_1.astype(dtypes)
            # print(dataset+1,X_test_1.shape)
            predictions=estimator.predict(np.array(X_test_1,dtype=float))
            fpr_test1,tpr_test1,roc_auc_score_test1=Evaluation(output_file_name,y_test_1,predictions,kmeans_test1,kmeans_train,"dataset_%s"%(dataset+1))
            plt.plot(fpr_test1, tpr_test1,
                 lw=2, label='ROC curve (area = %0.2f) of dataset%s' % (roc_auc_score_test1,dataset+1))
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(output_file_name+'/'+ 'all_ROC.png',dpi=400)
    plt.close()
        
    coef=pandas.DataFrame(data={'coef':estimator.feature_importances_,'feature':processed_headers})
    coef=coef.sort_values(by='coef',ascending=False)
    coef=coef[:15]
    print(coef)
    pal = sns.color_palette('pastel')
    sns.barplot(data=coef,x='feature',y='coef',color=pal.as_hex()[0])
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(output_file_name+"/coef.svg",dpi=400)
    plt.close()
    # SHAP(estimator,X_train,y,processed_headers)
    '''
    logging_file.write("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))    
    logging_file.close()
    

if __name__ == '__main__':
#    logging_file= open(output_file_name + '/Output.txt','a')
#    logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)
    open(output_file_name + '/Output.txt','a').write('Input data file:\n%s,\nTraining sets:\n%s,\nFolds of cross-validation: %s test size: %s \n\n' \
                 %('\n'.join(datasets),training_sets,folds,test_size))
    main()
    open(output_file_name + '/Output.txt','a').write("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))    
#%%
