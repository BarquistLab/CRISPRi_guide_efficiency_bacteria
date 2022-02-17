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
# import logging
import pandas
import sklearn.model_selection
import sklearn.metrics
# import autosklearn.metrics
from scipy.stats import spearmanr,pearsonr
from sklearn import linear_model

# from sklearn.cluster import KMeans
# from sklearn.feature_selection import VarianceThreshold #,GenericUnivariateSelect,f_regression
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler,Normalizer,RobustScaler,MinMaxScaler
from collections import defaultdict
import shap
import sys
import textwrap
import pickle
# import autosklearn
# from scipy import sparse
# from sklearn.compose import make_column_selector, ColumnTransformer
# from sklearn.pipeline import Pipeline
# from autosklearn.pipeline.implementations.SparseOneHotEncoder import SparseOneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
# import autosklearn.pipeline.implementations.CategoryShift
# from autosklearn.pipeline.implementations.MinorityCoalescer import MinorityCoalescer
# from autosklearn.pipeline.components.data_preprocessing.minority_coalescense.no_coalescense import NoCoalescence
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
                  This is used to train regression models. 
                  
                  Example: python machine_learning_data_fusion.py Rousset_dataset.csv,Wang_dataset.csv
                  '''))
parser.add_argument("datasets", help="data csv file(s),multiple files separated by ','")
parser.add_argument("-training", type=str, default=None, help="Dataset for training. Count starts from 0. If None,then all input datasets")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.3")
parser.add_argument("-c","--choice", type=str, default='histgb', help="Model type: \nlr: simple linear regression\nlasso\nelasticnet\nsvr\nhistgb: Histogram-based gradient boosting\nrf: random forest\n'lasso_hyperopt','elasticnet_hyperopt','svr_hyperopt','histgb_hyperopt','rf_hyperopt','histgb_autosklearn','rf_autosklearn','autosklearn'. default: histgb")
# parser.add_argument("-nc", default=False, help="Whether add non-coding strand targeting guides for training or not, default: False")
parser.add_argument("-s","--split", type=str, default='guide', help="Train-test split (gene/guide), default: guide")

args = parser.parse_args()
datasets=args.datasets
training_sets=args.training
choice=args.choice
if choice not in ['lr','lasso','elasticnet','svr','rf','histgb','lasso_hyperopt','elasticnet_hyperopt','svr_hyperopt','rf_hyperopt','histgb_hyperopt','histgb_autosklearn','rf_autosklearn','autosklearn']:
    print("The option for model type is not included. Please check --help for more info...\nAbort.")
    sys.exit()

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
# add_nc=args.nc
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
    # df['gene_expression_max']=np.log10(df['gene_expression_max']+0.01)
    # df['gene_expression_min']=np.log10(df['gene_expression_min']+0.01)
    ### one hot encoded sequence features
    PAM_encoded=[]
    sequence_encoded=[]
    dinucleotide_encoded=[]
    y=np.array(df['log2FC'],dtype=float)
    for i in df.index:
        PAM_encoded.append(self_encode(df['PAM'][i]))
        sequence_encoded.append(self_encode(df['sequence'][i]))
        dinucleotide_encoded.append(dinucleotide(df['sequence_30nt'][i]))
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        df.at[i,'guideid']=sequences.index(df['sequence'][i])
        # df.at[i,'base_A']=df['sequence'][i].count("A")
        # df.at[i,'base_T']=df['sequence'][i].count("T")
        # df.at[i,'base_C']=df['sequence'][i].count("C")
        # df.at[i,'base_G']=df['sequence'][i].count("G")
        # dinu_freq=dict()
        # for dinu in dinucleotides:
        #     dinu_freq[dinu]=0
        # for s in range(len(df['sequence'][i])-1):
        #     dinu_freq[df['sequence'][i][s:s+2]]+=1
        # for dinu in dinucleotides:
        #     df.at[i,dinu]=dinu_freq[dinu]
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
    
    if split=='guide':
        guideids=np.array(list(df['guideid']))
    elif split=='gene':
        guideids=np.array(list(df['geneid']))
    drop_features=['std','Nr_guide','coding_strand','guideid',"intergenic","No.","genename","gene_biotype","gene_strand","gene_5","gene_3",
                   "genome_pos_5_end","genome_pos_3_end","guide_strand",'sequence','PAM','sequence_30nt','gene_essentiality',
                   'off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70']
    if split=='gene':
        drop_features.append("geneid")
    
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    # if add_nc == 'True':
    #     df['coding_strand']=[coding_strand]*df.shape[0]
        
    # if len(datasets)==1:
    #     df=df.drop('dataset',1)
    X=df.drop(['log2FC'],1)#activity_score
    dataset_col=np.array(X['dataset'],dtype=int)  
    headers=list(X.columns.values)
    #use Kmeans clustering to cluster log2FC into two groups
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(y.reshape(-1,1))
    # logging_file.write("kmeans cluster centers: %s, %s\n" % (round(kmeans.cluster_centers_[0][0],2),round(kmeans.cluster_centers_[1][0],2)))
    gene_features=['dataset','geneid',"gene_5","gene_strand","gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
        
    if choice=='only_guide':
        headers=[item for item in headers if item not in gene_features]
        # headers=guide_features.copy()
        # guide_features.append("dataset")
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
    sequence_headers=list()
    # sequence_headers=['base_A','base_T','base_C','base_G']
    # for dint in dinucleotides:
        # sequence_headers.append(dint)
    # feat_type=['Numerical']*20
    for i in range(sequence_len):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
            sequence_headers.append('sequence_%s_%s'%(i+1,nts[j]))
    for i in range(PAM_len):
        for j in range(len(nts)):
            headers.append('PAM_%s_%s'%(i+1,nts[j]))
            sequence_headers.append('PAM_%s_%s'%(i+1,nts[j]))
    for i in range(dinucleotide_len-1):
        for dint in dinucleotides:
            headers.append(dint+str(i+1)+str(i+2))
            sequence_headers.append(dint+str(i+1)+str(i+2))
    if choice=='only_seq':
        X=pandas.DataFrame(X,columns=headers)
        feat_type=list()
        X=X[sequence_headers]
        headers=sequence_headers
    ###add one-hot encoded sequence features to feat_type
    for i in range(PAM_len*4+sequence_len*4+(dinucleotide_len-1)*4*4):
        feat_type.append('Categorical')
    
    X=pandas.DataFrame(data=X,columns=headers)
    # print(X.shape)
    logging_file.write("Number of features: %s\n" % len(headers))
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
    output.write("Max error of "+name+" :"+str(sklearn.metrics.max_error(y, predictions))+"\n") #Only with scikit-learn version >=0.20
    y=np.array(y)
    
    # transfer to a classification problem
    # y_test_clustered=kmeans.predict(y.reshape(-1,1)) ## kmeans was trained with all samples, extract the first column (second one is useless)
    # predictions_clustered=kmeans_train.predict(predictions.reshape(-1, 1))
    # if list(kmeans.cluster_centers_).index(min(kmeans.cluster_centers_))==0: ## define cluster with smaller value as class 1 
    #     y_test_clustered=y_test_clustered
    # else:
    #     y_test_clustered=[1 if x==0 else 0 for x in y_test_clustered] 
    # fpr,tpr,_=sklearn.metrics.roc_curve(y_test_clustered,predictions)
    # y_test_clustered=[1 if x==0 else 0 for x in y_test_clustered] 
    # if list(kmeans_train.cluster_centers_).index(min(kmeans_train.cluster_centers_))==1: ## define cluster with smaller value as class 1 ('good' guides)
    #     predictions_clustered=predictions_clustered
    # else:
    #     predictions_clustered=[1 if x==0 else 0 for x in predictions_clustered] 

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
    # with open(output_file_name+"/result.txt","a") as output:        
    #     output.write("Accuracy score of "+name+" :"+str(sklearn.metrics.accuracy_score(y_test_clustered, predictions_clustered))+"\n")
    #     output.write("precision_score of "+name+" :"+str(sklearn.metrics.precision_score(y_test_clustered, predictions_clustered))+"\n")
    #     output.write("recall_score of "+name+ ":"+str(sklearn.metrics.recall_score(y_test_clustered, predictions_clustered))+"\n")
    #     output.write("f1 of "+name+" :"+str(sklearn.metrics.f1_score(y_test_clustered, predictions_clustered))+"\n")
    #     output.write("roc_auc_score of "+name+" :"+str(sklearn.metrics.auc(fpr,tpr))+"\n\n")
        
    #     roc_auc_score=sklearn.metrics.auc(fpr,tpr)
    #     # ROC_plot(fpr,tpr,roc_auc_score,name)
    #     output.write("The MCC score: %s\n\n" %sklearn.metrics.matthews_corrcoef(y_test_clustered, predictions_clustered))
    # return fpr,tpr,roc_auc_score


def SHAP(estimator,X_train,headers):
    # X_train=pandas.DataFrame(X_train,columns=headers)
    X_train=X_train.astype(float)
    # gene_features=['dataset','geneid',"gene_GC_content","distance_operon","distance_operon_perc",
    #                "operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
    # others=[i for i in headers if i not in gene_features]
    # X_train=X_train[gene_features+others]
    print(X_train.columns.values[:10])
    # feature_index=[headers.index(i) for i in gene_features]
    print('shap')
    if os.path.isfile(output_file_name+"/explainer.pkl")==True:
        explainer=pickle.load(open(output_file_name+"/explainer.pkl",'rb'))
    else:
        explainer=shap.TreeExplainer(estimator)
        pickle.dump(explainer, open(output_file_name+"/explainer.pkl", 'wb'))
    print(explainer)
    
    # if os.path.isfile(output_file_name+"/shap_interaction_values.pkl")==True:
    #     shap_interaction_values=pickle.load(open(output_file_name+"/shap_interaction_values.pkl",'rb'))
    # else:
    #     shap_interaction_values=shap.TreeExplainer(estimator).shap_interaction_values(X_train.iloc[:1000,:])
    #     print(shap_interaction_values.shape)
    #     shap_interaction_values=shap_interaction_values[:,:10,:10]
    #     print(shap_interaction_values.shape)
    #     pickle.dump(shap_interaction_values, open(output_file_name+"/shap_interaction_values.pkl", 'wb'))
    # print(shap_interaction_values.shape)
    # for feature in gene_features:
    #     shap.dependence_plot(("ess_gene_operon", feature),    shap_interaction_values, X_train.iloc[:1000,:10],alpha=0.3,show=False)
    #     plt.subplots_adjust(left=0.15,right=0.95, top=0.95)
    #     plt.savefig(output_file_name+"/shap_interaction_plot_ess_gene_operon_%s.png"%feature,dpi=400)
    #     plt.close()
        
        
    
    # shap.summary_plot(shap_interaction_values, X_train.iloc[:1000,:10],alpha=0.3,show=False)
    # plt.savefig(output_file_name+"/shap_interaction_summary_plot.svg",dpi=400)
    # plt.close()
    
    # tmp = np.abs(shap_interaction_values).sum(0)
    # print(tmp.shape)
    # for i in range(tmp.shape[0]):
    #     tmp[i,i] = 0
    # inds = np.argsort(-tmp.sum(0))[:10]
    # tmp2 = tmp[inds,:][:,inds]
    # import matplotlib.pylab as pl
    # pl.figure(figsize=(12,12))
    # pl.imshow(tmp2)
    # pl.yticks(range(tmp2.shape[0]), X_train.columns[inds], rotation=50.4, horizontalalignment="right")
    # pl.xticks(range(tmp2.shape[0]), X_train.columns[inds], rotation=50.4, horizontalalignment="left")
    # pl.gca().xaxis.tick_top()
    # pl.colorbar()
    # pl.savefig(output_file_name+"/shap_interaction_heatmap.svg",dpi=400)
    # pl.close()
    if os.path.isfile(output_file_name+"/shap_values.pkl")==True:
        shap_values=pickle.load(open(output_file_name+"/shap_values.pkl",'rb'))
    else:
        shap_values = explainer.shap_values(X_train)
        # pickle.dump(shap_values, open(output_file_name+"/shap_values.pkl", 'wb'))
    # print(shap_values.shape)
    # values=pandas.DataFrame(shap_values,columns=headers)
    # X_train=X_train.reset_index(drop=True)
    # values=pandas.concat([X_train,values['gene_expression_max']],names=headers+['gene_expression_max_shap'],axis=1)
    # values.to_csv(output_file_name+"/shap_value.csv",index=False,sep='\t')
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')
    
    shap.summary_plot(shap_values, X_train, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.svg",dpi=400)
    plt.close()
    print(shap_values)
    for i in [10]:
        shap.summary_plot(shap_values,X_train,show=False,max_display=i,alpha=0.05)
        plt.subplots_adjust(left=0.45, top=0.95,bottom=0.2)
        # plt.yticks(range(i),shap_values.columns.values[:i],fontsize='small')
        plt.xticks(fontsize='small')
        plt.savefig(output_file_name+"/shap_value_top%s.svg"%(i),dpi=400)
        plt.close()  
    # for feature in gene_features:
    #     shap.dependence_plot(feature , shap_values,X_train,show=False,alpha=0.3)
    #     plt.subplots_adjust(left=0.15,right=0.95, top=0.95)
    #     plt.savefig(output_file_name+"/shap_dependence_plot_%s.png"%feature,dpi=400)
    #     plt.close()
        
        
def encode(seq):
    return np.array([[int(b==p) for b in seq] for p in ["A","T","G","C"]])
def find_target(df,before=20,after=20):
    from Bio import SeqIO
    fasta_sequences = SeqIO.parse(open("/home/yan/Projects/CRISPRi_related/seq/NC_000913.3.fasta"),'fasta')    
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
    X,y,feat_type,headers,guideids, guide_sequence_set,dataset_col=DataFrame_input(training_df)
    print(len(headers))
    open(output_file_name + '/Output.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    open(output_file_name + '/Output.txt','a').write("Features: "+",".join(headers)+"\n\n")
    
 
    
    processed_headers=headers
    X_rescaled=X
    numerical_indicator=["gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max",\
                          'distance_start_codon','distance_start_codon_perc','guide_GC_content','MFE_hybrid_seed','MFE_homodimer_guide','MFE_hybrid_full','MFE_monomer_guide',\
                        'off_target_60_70','off_target_70_80','off_target_80_90','off_target_90_100','homopolymers']
    dtypes=dict()
    for feature in processed_headers:
        if feature not in numerical_indicator:
            dtypes.update({feature:int})
    X_rescaled=pandas.DataFrame(data=X_rescaled,columns=processed_headers)
    X_rescaled=X_rescaled.astype(dtypes)
    # if add_nc=='True':
    #     #add non-coding strand guides
    #     X_ncs=column_transformer.transform(X_ncs)
    #     X_ncs=preprocessor.transform(X_ncs)
    
    
        
    #Linear regression
    if choice =='lr':
        estimator= linear_model.LinearRegression()
    if choice =='lasso':
        estimator = linear_model.Lasso(random_state = np.random.seed(111))
    if choice=='elasticnet':
        estimator = linear_model.ElasticNet(random_state = np.random.seed(111))
    #svr
    if choice=='svr':
        from sklearn.svm import SVR
        estimator=SVR()
    if choice in ['histgb','histgb_hyperopt','histgb_autosklearn']:
        # from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor
        if choice == 'histgb':
            estimator=HistGradientBoostingRegressor(random_state=np.random.seed(111))
        elif choice == 'histgb_autosklearn':
        # auto-sklearn include only gradient boosting
            estimator=HistGradientBoostingRegressor(loss='least_squares',learning_rate=0.11742891344336259,
                        max_iter=512,
                        min_samples_leaf=12,
                        max_depth=None,
                        max_leaf_nodes=14,
                        max_bins=255,
                        l2_regularization=3.0777178597597097e-10,
                        tol=1e-07,scoring='loss',
                        n_iter_no_change=0,
                        validation_fraction=None,verbose=0,warm_start=False,random_state=np.random.seed(111))
    if choice in ['rf','rf_hyperopt','rf_autosklearn']:
        from sklearn.ensemble import RandomForestRegressor
        if choice=='rf':
            estimator=RandomForestRegressor(random_state=np.random.seed(111))
        elif choice=='rf_autosklearn':
            # auto-sklearn include only rf
            estimator=RandomForestRegressor(bootstrap=False,criterion='friedman_mse',
                                            n_estimators=512,
                                            min_samples_leaf=16,
                                            min_samples_split=6,
                                            max_depth=None,
                                            max_leaf_nodes=None,
                                            max_features=0.3723573926944208,
                                            min_impurity_decrease=0.0,
                                            min_weight_fraction_leaf=0,
                                            random_state=np.random.seed(111))
    if choice == 'autosklearn':
        # auto-sklearn include all 
        estimator=RandomForestRegressor(bootstrap=False,criterion='friedman_mse',
                        n_estimators=512,
                        min_samples_leaf=18,
                        min_samples_split=16,
                        max_depth=None,
                        max_leaf_nodes=None,
                        max_features=0.22442857329791677,
                        min_impurity_decrease=0.0,
                        min_weight_fraction_leaf=0,
                        random_state=np.random.seed(111))
    
    ### hyperparameter tuning with hyperopt
    if choice in ['lasso_hyperopt','elasticnet_hyperopt','svr_hyperopt','rf_hyperopt','histgb_hyperopt']:
        from sklearn.model_selection import  cross_val_score,GroupKFold
        from hyperopt import hp, tpe, Trials
        import hyperopt
        from hyperopt.fmin import fmin
    
        def objective_sklearn(params):
            if choice=='rf_hyperopt': 
                int_types=['max_depth','n_estimators','min_samples_leaf']
            elif choice=='histgb_hyperopt': 
                int_types=['max_depth','max_iter','min_samples_leaf'] 
            else:
                int_types=[]
            params = convert_int_params(int_types, params)
            # Extract the boosting type
            if choice=='lasso_hyperopt':
                estimator = linear_model.Lasso(random_state = np.random.seed(111),**params)
            elif choice=='elasticnet_hyperopt':
                estimator = linear_model.ElasticNet(random_state = np.random.seed(111),**params)
            elif choice=='svr_hyperopt': 
                from sklearn.svm import SVR
                estimator=SVR(cache_size=2000,max_iter=2048,**params)
            elif choice=='rf_hyperopt': 
                estimator=RandomForestRegressor(random_state=np.random.seed(111),**params)
            elif choice=='histgb_hyperopt': 
                estimator=HistGradientBoostingRegressor(random_state = np.random.seed(111),**params)
            # estimator = LGBMRegressor(random_state = np.random.seed(111),**params)
            # estimator = XGBRegressor(random_state = np.random.seed(111),**params)
            # estimator = CatBoostRegressor( logging_level='Silent',random_state = np.random.seed(111),**params)
            #
            score = cross_val_score(estimator, X_rescaled, y, groups=guideids,scoring=scorer, cv=GroupKFold(n_splits=5)).mean()
            #using logloss here for the loss but uncommenting line below calculates it from average accuracy
            result = {"loss": score, "params": params, 'status': hyperopt.STATUS_OK}
            return result
    
        #hyperparameter space

        #Lasso
        if choice=='lasso_hyperopt':
            space={'alpha': hp.uniform('alpha', 0.0, 1.0)}
        elif choice=='elasticnet_hyperopt':
            space={'alpha': hp.uniform('alpha', 0.0, 1.0),
                'l1_ratio':hp.uniform('l1_ratio', 0.0, 1.0)} #elastic net
        #SVR
        elif choice=='svr_hyperopt': 
            space={'kernel':hp.choice('kernel',['linear', 'rbf', 'sigmoid']),
                'C': hp.uniform('C', 0.1, 1.0),
                'epsilon':hp.choice('epsilon',[0.1,0.01,0.001])}
        elif choice=='rf_hyperopt': 
        # #RF
            space = {'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(50, 500, 50)),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 100, 1)}
        elif choice=='histgb_hyperopt': 
        # histogram gb
            space = {
            #this is just piling on most of the possible parameter values for LGBM
            'loss': hp.choice('loss',['least_squares', 'least_absolute_deviation']),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'max_iter': hyperopt.hp.choice('max_iter', np.arange(50, 250, 25)),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 100, 1),
            'l2_regularization': hp.uniform('l2_regularization', 0.0, 1.0),
            }
        n_trials = 25
        trials = Trials()
        best = fmin(fn=objective_sklearn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=n_trials,
                    trials=trials)
  

        # # # # # # find the trial with lowest loss value. this is what we consider the best one
        idx = np.argmin(trials.losses())
        # # these should be the training parameters to use to achieve the best score in best trial
        params = trials.trials[idx]["result"]["params"]
        open(output_file_name + '/Output.txt','a').write("Hyperopt estimated optimum {}".format(params)+"\n\n")
        if choice=='lasso_hyperopt':
            estimator=linear_model.Lasso(random_state = np.random.seed(111),**params)
        elif choice=='elasticnet_hyperopt':
            estimator=linear_model.ElasticNet(random_state = np.random.seed(111),**params)
        elif choice=='svr_hyperopt': 
            estimator=SVR(random_state = np.random.seed(111),**params)
        elif choice=='rf_hyperopt': 
            estimator=RandomForestRegressor(random_state=np.random.seed(111),**params)
        elif choice=='histgb_hyperopt': 
            estimator=HistGradientBoostingRegressor(random_state=np.random.seed(111),**params)
    
    
    # estimator=RandomForestRegressor(random_state=np.random.seed(111),bootstrap=True, ccp_alpha=0.0, criterion='mse',
    #                                    max_depth=10, max_features='auto', max_leaf_nodes=None,
    #                                    max_samples=None, min_impurity_decrease=0.0,
    #                                    min_impurity_split=None, min_samples_leaf=8,
    #                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
    #                                    n_estimators=300, n_jobs=None, oob_score=False,
    #                                     verbose=0, warm_start=False)
    # estimator = linear_model.ElasticNet(random_state = np.random.seed(111),alpha=0.04380363432080181, copy_X=True, fit_intercept=True,l1_ratio=0.35437758856242874, max_iter=1000, normalize=False,positive=False, precompute=False, selection='cyclic', tol=0.0001, warm_start=False)
    # estimator = linear_model.Lasso(random_state = np.random.seed(111),alpha=0.015821043616692503, copy_X=True, fit_intercept=True,max_iter=1000, normalize=False, positive=False, precompute=False,selection='cyclic', tol=0.0001, warm_start=False)
    
    open(output_file_name + '/Output.txt','a').write("Estimator:"+str(estimator)+"\n")
    X_df=pandas.DataFrame(data=np.c_[X_rescaled,y,guideids,dataset_col],columns=processed_headers+['log2FC','guideid','dataset_col'])
    print(X_df.shape)
    
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
        y_train=X_train['log2FC']
        X_train=X_train[processed_headers]
        X_train=X_train.astype(dtypes)
        # if add_nc=='True':
        #     #add non-coding strand guides
        #     X_train=np.concatenate((X_train, X_ncs),axis=0)
        #     y_train=np.concatenate((y_train, y_ncs),axis=0)
        X_test = X_df[X_df['guideid'].isin(test_index)]
        # X_test=pandas.DataFrame(data=np.c_[X_test,y_test,dataset_col[test_index],guideid_test],columns=processed_headers+['log2FC','dataset_col','guideid'])
        X_test=X_test[(X_test['dataset_col'].isin(training_sets))]
        y_test=X_test['log2FC']
        X_test=X_test[processed_headers]
        X_test=X_test.astype(dtypes)
           
        # print('test',X_test.shape)
        # X_train=X_train.iloc[:int(X_train.shape[0]/2),:]
        # y_train=y_train[:int(y_train.shape[0]/2)]
        estimator = estimator.fit(np.array(X_train,dtype=float),np.array(y_train,dtype=float))
        # print(spearmanr(y_train,estimator.predict(X_train)))
        predictions = estimator.predict(np.array(X_test,dtype=float))
        # evaluations['Rs_train'].append(spearmanr(y_train,estimator.predict(X_train))[0])
        evaluations['Rs'].append(spearmanr(y_test, predictions)[0])
        evaluations['Rp'].append(pearsonr(y_test, predictions)[0])
        Evaluation(output_file_name,y_test,predictions,"X_test_kfold")
        # evaluations['AUC'].append(roc_auc_score)
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
            X_test_1=X_combined[headers]
            y_test_1=np.array(X_combined['log2FC'],dtype=float)
            # kmeans_test1=KMeans(n_clusters=2, random_state=0).fit(np.array(y_test_1).reshape(-1,1))
            # X_test_1=column_transformer.transform(X_test_1)  
            # X_test_1=preprocessor.transform(X_test_1)
            X_test_1=pandas.DataFrame(data=X_test_1,columns=processed_headers)
            X_test_1=X_test_1.astype(dtypes)
            # print('mixed',X_test_1.shape)
            predictions=estimator.predict(np.array(X_test_1,dtype=float))
            spearman_rho,_=spearmanr(y_test_1, predictions)
            evaluations['Rs_test_mixed'].append(spearman_rho)
            pearson_rho,_=pearsonr(y_test_1, predictions)
            evaluations['Rp_test_mixed'].append(pearson_rho)
            Evaluation(output_file_name,y_test_1,predictions,"X_test_mixed")
            # evaluations['AUC_test_mixed'].append(roc_auc_score)
            evaluations['MSE_test_mixed'].append(sklearn.metrics.mean_absolute_error(y_test_1, predictions))
            if split=='gene':
                metrics=gene_split_metrics(pandas.DataFrame(np.c_[X_combined,predictions],columns=headers+['log2FC','guideid','pred']))
                for key in metrics.keys():
                    evaluations[key+"_test_mixed"].append(metrics[key])
            for dataset in range(len(datasets)):
                dataset1=X_combined[X_combined['dataset']==dataset]
                X_test_1=dataset1[headers]
                y_test_1=np.array(dataset1['log2FC'],dtype=float)
                # kmeans_test1=KMeans(n_clusters=2, random_state=0).fit(np.array(y_test_1).reshape(-1,1))
            
                X_test_1=pandas.DataFrame(data=X_test_1,columns=processed_headers)
                X_test_1=X_test_1.astype(dtypes)
                # print(dataset+1,X_test_1.shape)
                predictions=estimator.predict(np.array(X_test_1,dtype=float))
                spearman_rho,_=spearmanr(y_test_1, predictions)
                # print(spearman_rho)
                evaluations['Rs_test%s'%(dataset+1)].append(spearman_rho)
                pearson_rho,_=pearsonr(y_test_1, predictions)
                evaluations['Rp_test%s'%(dataset+1)].append(pearson_rho)
                Evaluation(output_file_name,y_test_1,predictions,"X_test_%s"%(dataset+1))
                # evaluations['AUC_test%s'%(dataset+1)].append(roc_auc_score)
                evaluations['MSE_test%s'%(dataset+1)].append(sklearn.metrics.mean_absolute_error(y_test_1, predictions))
                if split=='gene':
                    metrics=gene_split_metrics(pandas.DataFrame(np.c_[dataset1,predictions],columns=headers+['log2FC','guideid','pred']))
                    for key in metrics.keys():
                        evaluations[key+"_test%s"%(dataset+1)].append(metrics[key])
            
            #Pasteur method test on our test set
            # test_data=pasteur_test[(pasteur_test['guideid'].isin(test_index))]
            # # print('pasteur',test_data.shape)
            # training_seq,guides_index=find_target(test_data)
            # training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
            # training_seq=training_seq.reshape(training_seq.shape[0],-1)
            # test_data=test_data.loc[guides_index]
            # reg=pickle.load(open('/home/yanying/projects/crispri/CRISPRi_ml/results_new/comparison/Bikard/NAR2020_paper/reg.pkl','rb'))
            # test_data['pasteur_score']=reg.predict(training_seq).reshape(-1, 1).ravel()
            
            # for i in test_data.index:
            #     test_data.at[i,'predicted_score']=test_data['median'][i]-test_data['pasteur_score'][i]
            # evaluations['Rs_pasteur_data_mixed'].append(spearmanr(test_data['log2FC'], test_data['predicted_score'])[0])
            # evaluations['Rp_pasteur_data_mixed'].append(pearsonr(test_data['log2FC'], test_data['predicted_score'])[0])
            # Evaluation(output_file_name,np.array(test_data['log2FC']),np.array(test_data['predicted_score']),"X_pasteur_test_mixed")
            # # evaluations['AUC_pasteur_data_mixed'].append(roc_auc_score)
            # evaluations['MSE_pasteur_data_mixed'].append(sklearn.metrics.mean_absolute_error(test_data['log2FC'], test_data['predicted_score']))
            # if split=='gene':
            #     metrics=gene_split_metrics(test_data.rename(columns={'pasteur_score':'pred'}))
            #     for key in metrics.keys():
            #         evaluations[key+"_pasteur_data_mixed"].append(metrics[key])
            # for dataset in range(len(datasets)):
            #     test_data=pasteur_test[(pasteur_test['dataset']==dataset)&(pasteur_test['guideid'].isin(test_index))]
            #     # print('pasteur',dataset,test_data.shape)
            #     training_seq,guides_index=find_target(test_data)
            #     training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
            #     training_seq=training_seq.reshape(training_seq.shape[0],-1)
            #     test_data=test_data.loc[guides_index]
            #     test_data['pasteur_score']=reg.predict(training_seq).reshape(-1, 1).ravel()
                
            #     for i in test_data.index:
            #         test_data.at[i,'predicted_score']=test_data['median'][i]-test_data['pasteur_score'][i]
            #     evaluations['Rs_pasteur_data%s'%(dataset+1)].append(spearmanr(test_data['log2FC'], test_data['predicted_score'])[0])
            #     evaluations['Rp_pasteur_data%s'%(dataset+1)].append(pearsonr(test_data['log2FC'], test_data['predicted_score'])[0])
            #     Evaluation(output_file_name,np.array(test_data['log2FC']),np.array(test_data['predicted_score']),"X_pasteur_test%s"%(dataset+1))
            #     # evaluations['AUC_pasteur_data%s'%(dataset+1)].append(roc_auc_score)
            #     evaluations['MSE_pasteur_data%s'%(dataset+1)].append(sklearn.metrics.mean_absolute_error(test_data['log2FC'], test_data['predicted_score']))
            #     if split=='gene':
            #         metrics=gene_split_metrics(test_data.rename(columns={'pasteur_score':'pred'}))
            #         for key in metrics.keys():
            #             evaluations[key+"_pasteur_data%s"%(dataset+1)].append(metrics[key])
     
    evaluations=pandas.DataFrame.from_dict(evaluations)
    evaluations.to_csv(output_file_name+'/iteration_scores.csv',sep='\t',index=True)
    
    ### Inplemented functions
    logging_file= open(output_file_name + '/Output.txt','a')
    ##split the combined training set into train and test
    
    # gene_features=['dataset','geneid',"gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
    # others=[i for i in processed_headers if i not in gene_features]
    # processed_headers=gene_features+others
    # X_df['gene_expression_max']=np.log10(np.array(X_df['gene_expression_max'],dtype=float))
    # X_df['gene_expression_min']=np.log10(np.array(X_df['gene_expression_min'],dtype=float))
    guide_train, guide_test = sklearn.model_selection.train_test_split(list(set(guideids)), test_size=test_size,random_state=np.random.seed(111))  
    X_train = X_df[X_df['guideid'].isin(guide_train)]
    X_train=X_train[X_train['dataset_col'].isin(training_sets)]
    y_train=X_train['log2FC']
    X_train = X_train[processed_headers]
    X_train=X_train.astype(dtypes)
    # if add_nc=='True':
    #     #add non-coding strand guides
    #     X_train=np.concatenate((X_train, X_ncs),axis=0)
    #     y_train=np.concatenate((y_train, y_ncs),axis=0)
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
    Evaluation(output_file_name,y_test,predictions,"X_test")
    # plt.figure()
    # sns.set_palette("PuBu",len(datasets)+1)
    # plt.subplots_adjust(bottom=0.15, top=0.96,right=0.96,left=0.15)
    # plt.plot(fpr, tpr,lw=2, label='ROC curve (area = %0.2f) of test' % roc_auc_score)
    # ### model validation with validation dataset
    # if len(datasets)>1:
        
    #     if 'dataset' not in X.columns.values.tolist():
    #         X_combined=pandas.DataFrame(data=np.c_[X,y,guideids,dataset_col],columns=headers+['log2FC','guideid','dataset'])
    #     else:
    #         X_combined=pandas.DataFrame(data=np.c_[X,y,guideids],columns=headers+['log2FC','guideid'])
    #     X_combined=X_combined[X_combined['guideid'].isin(guide_test)]
        
    #     #mixed datasets
    #     X_test_1=X_combined[headers]
    #     y_test_1=np.array(X_combined['log2FC'],dtype=float)
    #     kmeans_test1=KMeans(n_clusters=2, random_state=0).fit(np.array(y_test_1).reshape(-1,1))
    #     # X_test_1=column_transformer.transform(X_test_1)  
    #     # X_test_1=preprocessor.transform(X_test_1)
    
    #     X_test_1=pandas.DataFrame(data=X_test_1,columns=processed_headers)
    #     X_test_1=X_test_1.astype(dtypes)
    #     # print('mixed',X_test_1.shape)
    #     predictions=estimator.predict(np.array(X_test_1,dtype=float))
    #     fpr_test1,tpr_test1,roc_auc_score_test1=Evaluation(output_file_name,y_test_1,predictions,kmeans_test1,kmeans_train,"dataset_mixed")
    #     plt.plot(fpr_test1, tpr_test1,
    #              lw=2, label='ROC curve (area = %0.2f) of mixed datasets' % roc_auc_score_test1)
    #     for dataset in range(len(datasets)):
    #         dataset1=X_combined[X_combined['dataset']==dataset]
    #         X_test_1=dataset1[headers]
    #         y_test_1=np.array(dataset1['log2FC'],dtype=float)
    #         kmeans_test1=KMeans(n_clusters=2, random_state=0).fit(np.array(y_test_1).reshape(-1,1))
        
    #         # X_test_1=column_transformer.transform(X_test_1)  
    #         # X_test_1=preprocessor.transform(X_test_1)
        
    #         X_test_1=pandas.DataFrame(data=X_test_1,columns=processed_headers)
    #         X_test_1=X_test_1.astype(dtypes)
    #         # print(dataset+1,X_test_1.shape)
    #         predictions=estimator.predict(np.array(X_test_1,dtype=float))
    #         fpr_test1,tpr_test1,roc_auc_score_test1=Evaluation(output_file_name,y_test_1,predictions,kmeans_test1,kmeans_train,"dataset_%s"%(dataset+1))
    #         plt.plot(fpr_test1, tpr_test1,
    #              lw=2, label='ROC curve (area = %0.2f) of dataset%s' % (roc_auc_score_test1,dataset+1))
        
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right")
    # plt.savefig(output_file_name+'/'+ 'all_ROC.png',dpi=400)
    # plt.close()
    if choice in ['lr','lasso','elasticnet','lasso_hyperopt','elasticnet_hyperopt']:
        coef=pandas.DataFrame(data={'coef':estimator.coef_,'feature':processed_headers})
        coef=coef.sort_values(by='coef',ascending=False)
        coef=coef[:15]
        pal = sns.color_palette('pastel')
        sns.barplot(data=coef,x='feature',y='coef',color=pal.as_hex()[0])
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.35)
        plt.savefig(output_file_name+"/coef.svg",dpi=400)
        plt.close()
    elif choice in ['histgb','rf','histgb_hyperopt','rf_hyperopt','histgb_autosklearn','rf_autosklearn','autosklearn']:
        print(processed_headers)
        SHAP(estimator,X_train,processed_headers)
    logging_file.write("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))    
    logging_file.close()
    

if __name__ == '__main__':
    # logging_file= open(output_file_name + '/Output.txt','w')
    # logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)
    open(output_file_name + '/Output.txt','a').write('Input data file:\n%s,\nTraining sets:\n%s,\nFolds of cross-validation: %s test size: %s model type: %s, split based on:%s\n\n' \
                 %('\n'.join(datasets),training_sets,folds,test_size,choice,split))
    main()
    open(output_file_name + '/Output.txt','a').write("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))    
#%%
  # LightGB
    # space = {
    #     #this is just piling on most of the possible parameter values for LGBM
    #     'boosting_type': hp.choice('boosting_type',
    #                                 ['gbdt','dart', 'goss']),
    #     'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(50, 250, 25)),
    #     'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    #     'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    #     'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    #     'verbose': -1,
    #     'subsample': hp.uniform('subsample', 0.5, 1.0), 
    #     'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    #     'min_child_weight': hp.uniform('min_child_weight', 0.01, 0.75), 
    #     'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    #     'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    #     'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    #     'objective':'regression'
    # }
    # #XGB
    # space = {
    #     'eta': hp.loguniform('eta', np.log(0.01), np.log(0.2)),
    #     'min_child_weight': hp.uniform('min_child_weight', 0.01, 0.75), 
    #     'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(50, 250, 25)),
    #     'max_depth': hp.quniform('max_depth', 2, 10, 1),
    #     'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    #     'gamma': hp.uniform('gamma', 0.0, 0.5),
    #     'subsample': hp.uniform('subsample', 0.5, 1.0),
    #     'alpha': hp.uniform('alpha', 0.0, 1.0),
    #     'lambda': hp.uniform('lambda', 0.0, 1.0),
    # }
    #cat
    # space = {
    #     'eta': hp.loguniform('eta', np.log(0.01), np.log(0.2)),
    #     'iterations': hyperopt.hp.choice('iterations', np.arange(50, 250, 25)),
    #     'max_depth': hp.quniform('max_depth', 2, 10, 1),
    #     'subsample': hp.uniform('subsample', 0.5, 1.0),
    #     'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 1.0),
    #     'cat_features':[processed_headers[i] for i in range(len(processed_headers)) if processed_headers[i] not in numerical_indicator],
    #     'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    #     'border_count': hyperopt.hp.choice('border_count', np.arange(5, 200, 5))
    # }
 # estimator = XGBRegressor(random_state = np.random.seed(111))#,  **params)# 
    #                         eta= 0.12969164540999284,
    #                         min_child_weight= 0.525042613775923,
    #                         n_estimators= 200,
    #                         max_depth= 10,
    #                         colsample_bytree = 0.7571524225503363,
    #                         subsample= 0.8853134869585851,
    #                         gamma= 0.04481008880965251,
    #                         alpha= 0.27053623983841407,
    #                         reg_lambda= 0.8814836048738078)
    # estimator = LGBMRegressor(random_state = np.random.seed(111))#, **params)# 
    #                           boosting_type ='gbdt',
    #                           n_estimators= 200,
    #                           num_leaves= 109,
    #                           learning_rate=0.07104765124530392,
    #                           subsample_for_bin= 180000,
    #                           verbose= -1,
    #                           subsample= 0.7570299894343023, 
    #                           min_child_samples= 20,
    #                           min_child_weight=0.29125776146124227, 
    #                           reg_alpha= 0.6003257400529137,
    #                           reg_lambda= 0.37481885215866073,
    #                           colsample_bytree= 0.8939030801137798,
    #                           objective='regression')
    # estimator = CatBoostRegressor(random_state = np.random.seed(111),logging_level='Silent',**params) #
                                    # cat_features=[processed_headers[i] for i in range(len(processed_headers)) if processed_headers[i] not in numerical_indicator],
                                    # border_count = 180, eta = 0.12028081547829111, 
                                    # iterations = 225, l2_leaf_reg = 0.19393899495075212, 
                                    # max_depth = 10, min_child_samples = 165.0, subsample = 0.9220382895249094)

#%%

   # #data preprocessing
    # category_coalescence=MinorityCoalescer(minimum_fraction=0.005788594071307701)# train on Rousset/Wang/Mixed
    # if sparse.issparse(X):
    #     encoder= SparseOneHotEncoder()
    # else:
    #     encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    # categorical_transformer=Pipeline(steps=[
    #         ["categoryshift",autosklearn.pipeline.implementations.CategoryShift.CategoryShift()],
    #         ["imputation", SimpleImputer(strategy='constant', fill_value=2, copy=False)],
    #         ["category_coalescence", category_coalescence]
    #         # ["categorical_encoding", encoder]
    #         ]) #
    # # train on Rousset/Wang/Mixed
    # numerical_imputer=SimpleImputer(add_indicator=False, copy= False, fill_value= None, missing_values= np.nan, strategy='median', verbose= 0)
    # numerical_scaler= MinMaxScaler()
    
    # numerical_transformer=Pipeline(steps=[["imputation", numerical_imputer],
    #     ["variance_threshold", VarianceThreshold(threshold=0.0)],
    #     ["rescaling", numerical_scaler]])
    # sklearn_transf_spec = [
    #         ["categorical_transformer", categorical_transformer, [headers[i] for i in range(len(feat_type)) if feat_type[i]=='Categorical']],
    #         ["numerical_transformer", numerical_transformer, [headers[i] for i in range(len(feat_type)) if feat_type[i]=='Numerical']]
    #         ]
    # if choice=='only_seq':
    #     sklearn_transf_spec = [
    #         ["categorical_transformer", categorical_transformer, [headers[i] for i in range(len(feat_type)) if feat_type[i]=='Categorical']]]
    # column_transformer = ColumnTransformer(transformers=sklearn_transf_spec,sparse_threshold=float(sparse.issparse(X)))
    # preprocessor=GenericUnivariateSelect(score_func=f_regression, param=0.41365987272753285, mode='fdr')# train on Rousset/Wang/Mixed
    # if add_nc=='True':
    #     nc_df=training_df[training_df['dataset'].isin(training_sets)]
    #     X_ncs,y_ncs,kmeans_ncs,_,_,guideids_ncs,guide_sequence_set_ncs=DataFrame_input(nc_df,coding_strand=0)
    #     X_total=np.concatenate((X, X_ncs),axis=0)
    #     y_datasets=np.concatenate((y, y_ncs),axis=0)
    #     X_datasets=pandas.DataFrame(data=X_total,columns=headers)
    #     # dataset_col=np.array(X_total['dataset'],dtype=int)  
    #     # X_datasets=column_transformer.fit_transform(X_total,y_total)
    #     # X_rescaled=column_transformer.transform(X)
    #     # X_total=preprocessor.fit_transform(X_total,y_total)
    #     # X_rescaled=preprocessor.transform(X_rescaled)
    # else:
    #     #only use training data for fitting
    #     if 'dataset' not in X.columns.values.tolist():
    #         X_datasets=pandas.DataFrame(data=np.c_[X,y,dataset_col],columns=headers+['log2FC','dataset'])
    #     else:
    #         X_datasets=pandas.DataFrame(data=np.c_[X,y],columns=headers+['log2FC'])
    #     X_datasets=X_datasets[X_datasets['dataset'].isin(training_sets)]
    #     print('datasets',X_datasets.shape)
    #     y_datasets=np.array(X_datasets['log2FC'],dtype=float)
    #     X_datasets=X_datasets[headers]
    # X_datasets=column_transformer.fit_transform(X_datasets)
    # preprocessor=preprocessor.fit(X_datasets,y_datasets)
    
    # X_rescaled=column_transformer.transform(X)
    # X_rescaled=preprocessor.transform(X_rescaled)
    # processed_headers=get_ct_feature_names(column_transformer,headers)
    # mask=preprocessor.get_support()
    # if False not in mask:
    #     preprocessed_headers=processed_headers
    # else:
    #     preprocessed_headers=[]
    #     for i in range(len(mask)):
    #         if mask[i]==True:
    #             preprocessed_headers.append(processed_headers[i])
    # processed_headers=preprocessed_headers