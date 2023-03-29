#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:21:14 2021

@author: yanying
"""
'''
Thank you for reading this!

Here are the codes for the figures in the study.

The resulting figures are indicated in each block of code.

'''

from collections import defaultdict
import itertools
import pandas
import matplotlib.pyplot as plt
import numpy as np
# import regex as re
from scipy.stats import spearmanr,pearsonr,kendalltau
import argparse
import subprocess
import seaborn as sns
from Bio import SeqIO
import math
import statistics
import random
import pickle
import matplotlib as mpl
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler
import scipy
# print((mpl.rcParams))
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['font.sans-serif']='Arial'
mpl.rcParams['font.size']=14
mpl.rcParams['legend.title_fontsize']=10
mpl.rcParams['legend.fontsize']=10
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12
# print((mpl.rcParams))
import sklearn
import warnings
warnings.filterwarnings('ignore')
sns.set_palette("Set2")
pal=sns.color_palette('Set2')
sns.set_style("white")
#%%
'''
Figure 1B
'''
###collect results for figure 1B into a table
'''
The results are assumed to saved in folders with the names in the list ("only_seq","add_distance","add_MFE","only_guide",'guide_geneid','gene_seq','all')
Please change the path or folder names or the code accordingly
'''
D=np.zeros((10,1))
for test in ["only_seq","add_distance","add_MFE","only_guide",'guide_geneid','gene_seq','all']:#add_deltaGB['3sets_gene','3sets_hyperopt_gene','3sets_automl_CV_gene']:
    try:
        df=pandas.read_csv("%s/iteration_scores.csv"%(test),sep='\t')
    except FileNotFoundError:
        continue
    print(test)
    d=defaultdict(list)
    iteration=1
    metrics=['Rs']
    groups=['','_test_mixed','_test1','_test2','_test3']
    for i in range(iteration-1,df.shape[0],iteration):
      for metric in metrics:
          for group in groups:
              d[metric+group].append(df[metric+group][i])
    d=pandas.DataFrame.from_dict(d)
    D=np.c_[D,d]
D=pandas.DataFrame(D)
D.to_csv("figure1B.csv",sep='\t',index=False)
#%%
#
'''
Figure 1B 
I saved the results to the summplementary tables and plotted from the excel sheets
'''
df=pandas.read_excel("Yu_CRISPRi_supplementary_tables_YYu_v3.5.xlsx",sheet_name="TableS3_feature_engineering")

for f in ['1B']:
    headers=df.columns.values.tolist()
    headers.remove('Fold')
    plot=defaultdict(list)
    print(headers)
    for i in range(0,len(headers),5):
        plot['value']+=list(df.loc[1:10,headers[i]]) #16:25
        plot['group']+=[df.loc[0,headers[i]]]*10
        
    ### Change the list according to the results included in the excel sheet
    for i in ['Sequence features', '+ distance features', '+ thermodynamic features', '128 guide features',
              '128 guide features + target gene',
              'Sequence + gene features','137 guide + gene features']:
        plot['test']+=[i]*10
    for key in plot.keys():
        print(len(plot[key]))
    sns.set_style("whitegrid")
    plot=pandas.DataFrame.from_dict(plot)
    pal=sns.color_palette('Set2')
    ax=sns.boxplot(data=plot,x='test',y='value',color=pal.as_hex()[1])#
    plt.xticks(fontsize=12, rotation=30)
    plt.xlabel("")
    plt.ylabel("Spearman correlation")
    plt.savefig("Figure%s.svg"%f)
    plt.show()
    plt.close()
#%%
'''
Figure S2
'''
def overlapping_guides(training_df,validation_df,set1,set2):
    pal=sns.color_palette("pastel")
    training_df=training_df[(training_df['gene_essentiality']==1)&(training_df['intergenic']==0)&(training_df['coding_strand']==1)]
    training_df=training_df.dropna()
    validation_df=validation_df[(validation_df['gene_essentiality']==1)&(validation_df['intergenic']==0)&(validation_df['coding_strand']==1)]
    validation_df=validation_df.dropna()
    print(validation_df.shape)
    for i in list(set(training_df['geneid'])):
        gene_df=training_df[training_df['geneid']==i]
        for j in gene_df.index:
            training_df.at[j,'nr']=gene_df.shape[0]
    for i in list(set(validation_df['geneid'])):
        gene_df=validation_df[validation_df['geneid']==i]
        for j in gene_df.index:
            validation_df.at[j,'nr']=gene_df.shape[0]
    training_ess=list(set(training_df['geneid']))
    validation_ess=list(set(validation_df['geneid']))
    overlap_ess=[ess for ess in training_ess if ess in validation_ess]
    print(len(overlap_ess))
    t_overlap_log2fc=[]
    v_overlap_log2fc=[]
    overlapping_guides=0
    for ess in overlap_ess:
        t=training_df[training_df['geneid']==ess]
        v=validation_df[validation_df['geneid']==ess]
        overlap_pos=[pos for pos in list(t['distance_start_codon']) if pos in list(v['distance_start_codon'])]
        overlapping_guides+=len(overlap_pos)
        if len(overlap_pos)==0:
            continue
        for pos in overlap_pos:
            t_pos=t[t['distance_start_codon']==pos]
            v_pos=v[v['distance_start_codon']==pos]
            t_overlap_log2fc.append(sum(t_pos['log2FC']))
            v_overlap_log2fc.append(sum(v_pos['log2FC']))
    r,_=pearsonr(t_overlap_log2fc,v_overlap_log2fc)
    print(r,overlapping_guides)
    sns.set_style('white')
    plt.figure()
    ax=sns.scatterplot(t_overlap_log2fc,v_overlap_log2fc,alpha=0.5,edgecolors='w',color=pal.as_hex()[0])
    plt.text(0.01,0.85,"N = {0}".format(len(t_overlap_log2fc)),fontsize=14,transform=ax.transAxes)
    plt.text(0.01,0.8,"Pearson R: {0}".format(round(r,2)),fontsize=14,transform=ax.transAxes)
    plt.xlabel("logFC in %s"%set1)
    plt.ylabel("logFC in %s"%set2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.savefig("%s_%s.svg"%(set1,set2))
    plt.show()
    plt.close()
path="~/Projects/CRISPRi_related/doc/CRISPRi_manuscript/github_code"
folds=10
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
rousset=pandas.read_csv(datasets[0],sep="\t")
rousset = rousset.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
rousset['dataset']=[0]*rousset.shape[0]
rousset18=pandas.read_csv(datasets[1],sep="\t")
rousset18 = rousset18.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
rousset18['dataset']=[1]*rousset18.shape[0]
wang=pandas.read_csv(datasets[2],sep="\t")
wang = wang.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
wang['dataset']=[2]*wang.shape[0]
overlapping_guides(rousset,wang,'E75 Roussset','Wang')    
overlapping_guides(rousset18,wang,'E18 Cui','Wang')    
overlapping_guides(rousset,rousset18,'E75 Roussset','E18 Cui')   

#%%
'''
Figure 1D
'''
###distribution of logFC
def DataFrame_input(df):
    ###keep guides for essential genes
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==1)]
    df=df.dropna()
    for i in list(set(list(df['geneid']))):
        df_gene=df[df['geneid']==i]
        for j in df_gene.index:
            df.at[j,'Nr_guide']=df_gene.shape[0]
    df=df[df['Nr_guide']>=5]#keep only genes with more than 5 guides from each dataset
    plt.figure(figsize=(6,4))
    sns.set_style("whitegrid")
    sns.set_palette("Set2",4)
    essential1=df[df['dataset']==0]
    essential2=df[df['dataset']==1]
    essential3=df[df['dataset']==2]
    sns.distplot(a=essential1['log2FC'],hist=False,kde=True,rug=False,kde_kws={"shade":False, "bw":0.2},label ='E75 Rousset (%s gRNAs)'%len(essential1))
    sns.distplot(a=essential2['log2FC'],hist=False,kde=True,rug=False,kde_kws={"shade":False, "bw":0.2},label ='E18 Cui (%s gRNAs)'%len(essential2))
    sns.distplot(a=essential3['log2FC'],hist=False,kde=True,rug=False,kde_kws={"shade":False, "bw":0.2},label ='Wang (%s gRNAs)'%len(essential3))
    plt.legend(loc='upper right',fontsize=11)
    plt.xlabel("logFC",fontsize=14)
    plt.ylabel("Proportion",fontsize=14)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    plt.savefig("logFC_distrbution.svg")
    plt.show()
    plt.close()
    
'''
Please change the path accordingly to direct the code to the dataset files
'''

path="github_code"
folds=10
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
df1=pandas.read_csv(datasets[0],sep="\t")
df1_shuffled = df1.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df1_shuffled['dataset']=[0]*df1_shuffled.shape[0]
df2=pandas.read_csv(datasets[1],sep="\t")
df2_shuffled = df2.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df2_shuffled['dataset']=[1]*df2_shuffled.shape[0]
df3=pandas.read_csv(datasets[2],sep="\t")
df3_shuffled = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df3_shuffled['dataset']=[2]*df3_shuffled.shape[0]
df2_shuffled=df2_shuffled.append(df3_shuffled,ignore_index=True)  
training_df=df1_shuffled.append(df2_shuffled,ignore_index=True)  
training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
DataFrame_input(training_df)
#%%
'''
Figure 1E
Similar to Figure 1B
'''
###collect results for figure 2 into a table
for alg in ['LR','LASSO','ElasticNet','SVR','RF','HistGB','autosklearn']:
    D=np.zeros((10,1))    
    for test in ['R75','C18','W','R75C18','R75W','C18W','3sets']:
        try:
            df=pandas.read_csv("%s/%s/iteration_scores.csv"%(alg,test),sep='\t')
        except FileNotFoundError:
            continue
        print(alg,test)
        d=defaultdict(list)
        iteration=1
        metrics=['Rs']
        groups=['','_test_mixed','_test1','_test2','_test3']
        for i in range(iteration-1,df.shape[0],iteration):
          for metric in metrics:
              for group in groups:
                  d[metric+group].append(df[metric+group][i])
        d=pandas.DataFrame.from_dict(d)
        D=np.c_[D,d]
    D=pandas.DataFrame(D)
    D.to_csv("figureS3_%s.csv"%alg,sep='\t',index=False)


#%%
'''
#Figure 1E & S3
'''
groug_num=5
if fig!='1E':
    algs= ['Linear Regression','LASSO','Elastic Net','SVR','RF','HistGB']
    df=pandas.read_excel("Yu_CRISPRi_supplementary_tables_YYu_v3.5.xlsx",sheet_name="TableS6_datafusion_modeltype")
    headers=df.columns.values.tolist()
    headers.remove('Model type')
    headers.remove('train')
    rotation=30
else:
    algs=['autosklearn']
    df=pandas.read_excel("Yu_CRISPRi_supplementary_tables_YYu_v3.5.xlsx",sheet_name="TableS5_data_fusion")
    headers=df.columns.values.tolist()
    headers.remove('Fold')
    rotation=0


sns.set_style('whitegrid')

for alg in algs:
    plot=defaultdict(list)
    for i in range(0,len(headers),groug_num):
        for j in range(groug_num):
            plot['value']+=list(df.loc[1+algs.index(alg)*15:10+algs.index(alg)*15,headers[i+j]]) #16:25
            plot['group']+=[df.loc[0,headers[i+j]]]*10
    for i in ['E75 Rousset','E18 Cui', 'Wang', 
             'E75 Rousset & E18 Cui', 'E75 Rousset & Wang', 'E18 Cui & Wang', 
              '3 datasets']: 
        plot['test']+=[i]*10*groug_num
    try:
        plot=pandas.DataFrame.from_dict(plot)
    except:
        for key in plot.keys():
            print(len(plot[key]))
    
    plot=plot[plot['group']!='Test']
    if fig=='1E':
        plot=plot[plot['test'].isin(['E75 Rousset','E18 Cui', 'Wang', '3 datasets'])]
        plt.figure(figsize=(5,3))
    else:
        plt.figure(figsize=(6,3))
    sns.set_palette("Set2")
    ax=sns.boxplot(data=plot,x='test',y='value',hue='group')#
    legend=plt.legend(loc='lower right',fontsize=8,title="")
    if fig !='1E':
        ax.legend().remove()
        plt.title(alg,fontsize=14)
    plt.xticks(fontsize=10,  rotation=0)
    plt.xlabel("")
    plt.ylabel("Spearman correlation",fontsize=14)
    plt.savefig("%s.svg"%alg.replace(" ","_"))
    plt.show()
    plt.close()


#%%
'''
Figure 2B
Please change the training_datasets list to include the results with the identical names in the output table
'''
###
### split on genes
from collections import defaultdict
from sklearn.metrics import ndcg_score
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
algs=['MERF','Pasteur']
xticks={'MERF':"MERF",
        'CNN':'CNN','CRISPRon_deltaGB':'CGx_CRISPRi (deltaGB)','CRISPRon':'CGx_CRISPRi',
        'pasteur':'Pasteur','Pasteur':'Pasteur (retrained)'}
dataset_labels= ['E75 Rousset','E18 Cui','Wang']
training_labels={'R75':'E75 Rousset','C18':'E18 Cui','W': "Wang",
                    'R75W':'E75 Rousset & Wang','R75C18':'E75 Rousset & E18 Cui', 'C18W':'E18 Cui & Wang', '3sets': "3 datasets"}
plot=defaultdict(list)
output_dataset=defaultdict(list)
for alg in algs:
    if alg=='MERF':
        training_datasets=['R75','C18','W','3sets','3sets_deltaGB','3sets_dropdistance','3sets_CAI']
    elif alg in ['Pasteur']:
        training_datasets=['R75','C18','W','3sets']
    else:
        training_datasets=['3sets']
    for training_dataset in training_datasets:
        print(alg)
        output_dataset[alg].append(training_dataset)
        if alg=='pasteur':
            df=pandas.read_csv("RF/3sets/iteration_predictions.csv",sep='\t')
        elif alg=='CNN' or alg=='CRISPRon' or alg=='CRISPRon_deltaGB':
            df=pandas.read_csv("%s/iteration_predictions.csv"%(alg),sep='\t')

        elif alg =='Pasteur':
            df=pandas.read_csv("%s/%s/iteration_predictions.csv"%(alg,training_dataset),sep='\t')
        else:
            df=pandas.read_csv("%s/%s/iteration_predictions.csv"%(alg,training_dataset),sep='\t')

        for i in list(df.index):
            d=defaultdict(list)
            log2FC=list(map(float,df['log2FC'][i].replace("[","").replace("]","").split(",")))
            dataset=list(map(float,df['dataset'][i].replace("[","").replace("]","").split(",")))
            if alg=='pasteur':
                pred=list(map(float,df['pasteur_score'][i].replace("[","").replace("]","").split(",")))
            else:
                pred=list(map(float,df['pred'][i].replace("[","").replace("]","").split(",")))
            if alg=='MERF' or alg=='MERF_noness' or 'GPB' in alg or 'hyper' in alg:
                geneid=[float(i) for i in df['clusters'][i].replace("[","").replace("]","").split(",")]
            else:
                geneid=list(map(float,df['geneid'][i].replace("[","").replace("]","").split(",")))
            d['log2FC']+=log2FC
            d['pred']+=pred
            d['geneid']+=geneid
            d['dataset']+=dataset
            D=pandas.DataFrame.from_dict(d)
            for k in range(3):
                D_dataset=D[D['dataset']==k]
                for j in list(set(D_dataset['geneid'])):
                    D_gene=D_dataset[D_dataset['geneid']==j]
                    if D_gene.shape[0]<5:
                        continue
                    if 'MERF' in alg:
                        sr,_=spearmanr(D_gene['log2FC'],D_gene['pred']) 
                    else:
                        sr,_=spearmanr(D_gene['log2FC'],-D_gene['pred']) 
                    plot['sr'].append(sr)
                    plot['gene'].append(j)
                    plot['alg'].append(xticks[alg])
                    plot['dataset'].append(dataset_labels[k])
                    if training_dataset not in training_labels.keys():
                        training_labels.update({training_dataset:training_dataset})
                    plot['training'].append(training_labels[training_dataset])
plot_all=pandas.DataFrame.from_dict(plot)

metrics=defaultdict(list)
for alg in algs:
    training_datasets=output_dataset[alg]
    for training_dataset in training_datasets:
        
        p=plot_all[(plot_all['training']==training_labels[training_dataset])&(plot_all['alg']==xticks[alg])]
        p=p.dropna(subset=['sr'])
        metrics['Model'].append(xticks[alg])
        metrics['Training datasets'].append(training_labels[training_dataset])
        for dataset in dataset_labels:
            p_d=p[p['dataset']==dataset]
            p_d=p_d.dropna(subset=['sr'])
            metrics[dataset].append((np.median(p_d['sr'])))
        metrics['Mixed'].append(np.median(p['sr']))
        
metrics=pandas.DataFrame.from_dict(metrics)
metrics=metrics.dropna()
print(metrics)
metrics=metrics.sort_values(by='Training datasets')
metrics.to_csv("sr_heldout_median.csv",sep='\t',index=False)
#%%
'''
Latex code for the table
https://texviewer.herokuapp.com/
https://tableconvert.com/excel-to-latex

\pdfminorversion=4
\documentclass[]{article}


%%%%%%%%%%%%%%%%%%%
% Packages/Macros %
%%%%%%%%%%%%%%%%%%%
\usepackage{amssymb,latexsym,amsmath}     % Standard packages
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{multicol}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
%%%%%%%%%%%
% Margins %
%%%%%%%%%%%
\addtolength{\textwidth}{1.5in}
\addtolength{\textheight}{1.50in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\oddsidemargin}{-0.65in}
\addtolength{\topmargin}{-.50in}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theorem/Proof Environments %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newtheorem{theorem}{Theorem}
\newenvironment{proof}{\noindent{\bf Proof:}}{$\hfill \Box$ \vspace{10pt}}  

%%%%%%%%%%%%
% Document %
%%%%%%%%%%%%
\begin{document}

\renewcommand{\arraystretch}{1.3}
\begin{table}[!ht]
    \centering
    \resizebox{\textwidth}{!}{\begin{tabular}{ccccccc} \hline
        \multirow{3}{*}{\bfseries Model} & \multirow{3}{*}{\bfseries Training data} & \multicolumn{4}{c}{\bfseries Median Spearman Correlation} \\
        &  & \multicolumn{4}{c}{\bfseries Across held-out genes} \\ \cline{3-6}
         &  & \bfseries E75 Rousset & \bfseries E18 Cui & \bfseries Wang & \bfseries Mixed \\ \hline
        \multirow{4}{*}{\bfseries MERF} & E75 Rousset & 0.327 & 0.287 & 0.279 & 0.296 \\ 
         & E18 Cui & 0.391 & \bfseries 0.400 & 0.302 & 0.333 \\ 
         & Wang & 0.373 & 0.367 & 0.327 & 0.344 \\ 
         & 3 datasets & \bfseries 0.409 & \bfseries 0.400 & \bfseries0.373 & \bfseries0.396 \\ \hline
        \multirow{4}{*}{\bfseries Pasteur (retrained)} & E75 Rousset & 0.333 & 0.333 & 0.256 & 0.310 \\ 
        & E18 Cui & 0.363 & 0.352 & 0.286 & 0.322 \\ 
         & Wang & 0.367 & 0.377 & 0.307 & 0.339 \\ 
         & 3 datasets & 0.394 & \bfseries 0.400 & 0.327 & 0.366 \\ \hline
    \end{tabular}}
    
\end{table}
\end{document}

'''

#%%
'''
##Figure 3 feature interpretation
## 3B
'''
shap_values=pandas.read_csv("MERF/3sets/shap_value_mean.csv",sep='\t')
shap_values=shap_values.sort_values(by='shap_values',ascending=False)
feature_list=list(shap_values['features'])
to_plot=[i for i in feature_list[:30] if i not in ['distance_start_codon','distance_start_codon_perc','homopolymers'] and 'MFE' not in i]
plt.figure()
plt.bar(range(len(to_plot)),shap_values[shap_values['features'].isin(to_plot)]['shap_values'])
plt.xticks(range(len(to_plot)),to_plot,rotation=90)
plt.savefig("3B.svg")
plt.show()
plt.close()
#%%
'''
##Figure 3 feature interpretation
## interaction SHAP plots
'''
import shap
import time
start=time.time()
path="github_code"
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}
training_sets=[0,1,2]
feature_set='all'

### Load saved files from running MERF
guide_features=pickle.load(open('MERF/3sets/saved_model/CRISPRi_headers.sav','rb'))
shap_values=pickle.load(open('MERF/3sets/shap_values_1000.pkl','rb'))
shap_values=np.array(shap_values,dtype=float)
X_index=pickle.load(open('MERF/3sets/X_index.pkl','rb'))
print("Done cal SHAP values: %s s"%round(time.time()-start,3))
marker=['-','+']
pairs=[['sequence_23_G','sequence_24_C'],['sequence_23_C','sequence_25_C'],
       ['sequence_24_G','sequence_28_C'],['sequence_24_A','sequence_25_A']]

feature_labels={'sequence_28_C':'+1 C','sequence_24_G':'20 G','sequence_24_C':'20 C','sequence_28_G':'+1 G','sequence_28_A':'+1 G','sequence_25_T':'P1 T',
        'sequence_24_A':'20 A','sequence_25_C':'P1 C','sequence_23_C':'19 C','sequence_23_G':'19 G','sequence_25_A':'P1 A'}
sns.set_style('whitegrid')
coms=[[0,0],[1,0],[0,1],[1,1]]
for pair in pairs:
    p=defaultdict(list)
    for i in coms:
        sample_df=X_index[(X_index[pair[0]]==i[0])&(X_index[pair[1]]==i[1])] 
        if coms.index(i)==1:
            f1=np.median(shap_values[sample_df.index,guide_features.index(pair[0])])
        if coms.index(i)==2:
            f2=np.median(shap_values[sample_df.index,guide_features.index(pair[1])])
        sample_df=X_index[(X_index[pair[0]]==i[0])&(X_index[pair[1]]==i[1])] 
        # print(np.median(shap_values[sample_df.index,guide_features.index(pair[0])]+shap_values[sample_df.index,guide_features.index(pair[1])]))
        p['pattern']+=[marker[i[0]]+" / "+marker[i[1]]]*sample_df.shape[0]
        p['value']+=list(shap_values[sample_df.index,guide_features.index(pair[0])]+shap_values[sample_df.index,guide_features.index(pair[1])])
    # print(f1+f2)
    p=pandas.DataFrame.from_dict(p)
    plot=p.dropna()
    plot=p[p['pattern']!='- / -']
    plt.figure(figsize=(5,4))
    ax=sns.boxplot(data=plot,x='pattern',y='value',order=['+ / -','- / +','+ / +'],color='lightgrey')
    ax.axhline(f1+f2,color='r',xmin=0.7,xmax=0.965)
    plt.xticks(rotation=0,fontsize='large')
    plt.xlabel("")
    plt.title(feature_labels[pair[0]]+' / '+feature_labels[pair[1]],fontsize='large')
    plt.ylabel("sum SHAP value",fontsize='large')
    plt.subplots_adjust(left=0.2)
    plt.savefig("S6_%s.svg"%'_'.join(pair))
    plt.show()
    plt.close()    
#%%
'''
Functions for plotting the figures for validation experiments
'''
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
    # encoded=list()
    for nt in range(len(sequence)-1):
        if sequence[nt]=='N' or sequence[nt+1] =='N':
            continue
        encoded[nt*len(nts)**2+dinucleotides.index(sequence[nt]+sequence[nt+1])]=1
        # encoded.append(sequence[nt:nt+2])
    return encoded
def encode_sequence(sequence):
    alphabet = 'AGCT'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    encoded_sequence = np.eye(4)[integer_encoded]
    return encoded_sequence
def encode(seq):
    return np.array([[int(b==p) for b in seq] for p in ["A","T","G","C"]])
def encode_seqarr(seq,r):
    '''One hot encoding of the sequence. r specifies the position range.'''
    X = np.array(
            [encode(''.join([s[i] for i in r])) for s in seq]
        )
    X = X.reshape(X.shape[0], -1)
    return X
def DataFrame_input(df):
    PAM_encoded=[]
    sequence_encoded=[]
    dinucleotide_encoded=[]
    df['coding_strand']=[1]*df.shape[0]
    for i in df.index:
        sequence_encoded.append(self_encode(df['sequence_30nt'][i]))
    df['sequence_40nt'] = df.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1)
    headers=df.columns.values.tolist()
    nts=['A','T','C','G']
    for i in range(30):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
        # headers.append('sequence_%s'%(i+1))
        # headers.append("pos"+str(i+1)+str(i+2))
    
    df=pandas.DataFrame(data=np.c_[df,sequence_encoded],columns=headers)
    return df
labels={'MERF':"MERF",
        'CRISPRon':'CGx_CRISPRi','CNN':'CNN', 'CRISPRon_deltaGB': 'CGx_CRISPRi (deltaGB)',
        'pasteur_score':'Pasteur','Pasteur':'Pasteur (retrained)',
        'Doench_score(with aa)':"gRNA Designer",'Doench_score(without aa)':"Doench (w/o aa)",
        'DeepSpCas9':"DeepSpCas9",'TUSCAN':"TUSCAN","SSC":"SSC" }
#%%
'''
###Performance of deGFP
'''
df=pandas.read_csv("deGFP_gRNAs.tsv",sep='\t')
algs=['Pasteur','MERF']
ecoli=list()
salmonella=list()
tests=list()
metric_scores=defaultdict(list)
df=DataFrame_input(df)
for alg in algs:
    if 'Pasteur' not in alg and 'pasteur' not in alg and alg not in ['CRISPRon','CNN']:
        if alg=='MERF':
            datasets=['3sets']
        elif alg in  ['RF','LASSO']:
            datasets=['3sets']
        for dataset in datasets:#,'R75','C18','W']:#'R75W','R75C18','C18W',,'3sets_dataset'
            estimator=pickle.load(open("%s/%s/saved_model/CRISPRi_model.sav"%(alg,dataset),'rb'))
            headers=pickle.load(open("%s/%s/saved_model/CRISPRi_headers.sav"%(alg,dataset),'rb'))
            df_sub=df[headers]
            predictions=estimator.predict(df_sub)
            if dataset != '3sets':
                df[alg+"("+dataset+")"]=predictions
                labels.update({alg+"("+dataset+")":labels[alg]+" ("+dataset+")"})
                tests.append(alg+"("+dataset+")")
            else:
                df[alg]=predictions
                tests.append(alg)
    elif alg in ['CRISPRon','CNN']:
        filename_model = "%s/model_10.ckpt"%(alg)
        header=pickle.load(open("%s/CRISPRi_headers.sav"%(alg+""),'rb'))
        SCALE=pickle.load(open("%s/SCALEr.sav"%(alg+""),'rb'))
        df_sub=df[header]
        df_sub['sequence_30nt'] = df_sub.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1)
        header=[i for i in header if i !='sequence_30nt']
        df_sub[header] = SCALE.transform(df_sub[header])
        from crispri_dl.architectures import Crispr1DCNN, CrisprGRU, CrisprOn1DCNN
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        from pytorch_lightning.callbacks import ModelCheckpoint
        from crispri_dl.dataloader import CrisprDatasetTrain
        loader_test = CrisprDatasetTrain(df_sub, df['pasteur_score'], header)
        loader_test = DataLoader(loader_test, batch_size=df_sub.shape[0])
        if alg=='CNN' :
            trained_model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        elif alg=='CRISPRon' :
            trained_model = CrisprOn1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        #test
        max_epochs = 500
        batch_size = 32
        patience = 5
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            min_delta=0.0, 
            patience=patience, 
            verbose=False, 
            mode="min")
        checkpoint_callback = ModelCheckpoint(
                    monitor = 'val_loss',
                    # dirpath = output_file_name,
                    # filename = "model_"+str(fold_inner),
                    verbose = True,
                    save_top_k = 1,
                    mode = 'min',)
        estimator = pl.Trainer(gpus=0,  max_epochs=max_epochs, check_val_every_n_epoch=1, logger=True,progress_bar_refresh_rate = 0, weights_summary=None)

        predictions_test  = estimator.predict(
                                    model=trained_model,
                                    dataloaders=loader_test,
                                    return_predictions=True,
                                    ckpt_path=filename_model)
        predictions = predictions_test[0].cpu().numpy().flatten()
        df[alg]=predictions
        tests.append(alg)
    else:
        if alg=='Pasteur':
            datasets=['3sets']
        for dataset in datasets:
            if alg=='Pasteur':
                estimator=pickle.load(open("%s/%s/saved_model/CRISPRi_model.sav"%(alg,dataset),'rb'))
            training_seq=list(df['seq_60nt'])
            training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
            training_seq=training_seq.reshape(training_seq.shape[0],-1)
            predictions=estimator.predict(training_seq).reshape(-1, 1).ravel()
            if dataset != '3sets':
                df[alg+"("+dataset+")"]=predictions
                labels.update({alg+"("+dataset+")":labels[alg]+" ("+dataset+")"})
                tests.append(alg+"("+dataset+")")
            else:
                df[alg]=predictions
                tests.append(alg)
tests=['Doench_score(with aa)','TUSCAN','DeepSpCas9','pasteur_score']+tests
for i in df.index:
    ecoli.append(np.nanmean([df['E.coli_rep1'][i],df['E.coli_rep2'][i],df['E.coli_rep3'][i]]))
    salmonella.append(np.nanmean([df['Salmonella_rep1'][i],df['Salmonella_rep2'][i],df['Salmonella_rep3'][i],df['Salmonella_rep4'][i]]))
for exp in ['ecoli','salmonella']:
    metric_scores['metric'].append('Spearman correlation')
    plt.figure(figsize=(5,5))
    sns.set_palette("PuBu")
    sns.set_style('white')
    if exp=='ecoli':
        df['experimental']=ecoli.copy()
        metric_scores['validation'].append("deGFP in Ecoli")
    else:
        df['experimental']=salmonella.copy()
        metric_scores['validation'].append("deGFP in Salmonella")
        # metric_scores['validation'].append("deGFP in Salmonella")
    df_drop=df.dropna(subset=['experimental'])
    df_drop=df_drop.sort_values(by='experimental',ascending=True)
    experimental=list(df_drop['experimental'])
    bars=list()
    for test in tests:
        if test=='MERF':
            scatter=True
        else:
            scatter=False
        predicted=list(df_drop[test])
        if 'MERF' in test:
            pred=[2**(-1 * i) for i in predicted]
        else:
            pred=[2**(1 * i) for i in predicted]
            predicted=MinMaxScaler().fit_transform(-1*np.array(df_drop[test]).reshape(-1,1))
            predicted=np.hstack(predicted)
        if test=='MERF':
            ax= sns.regplot(predicted,experimental,
                        label=labels[test],scatter=scatter,color="#1f77b4")
        true=[2**(-1 * i) for i in experimental]
        ndcg=ndcg_score(np.asarray([true]),np.asarray([pred])) 
        bars.append(spearmanr(predicted,experimental,nan_policy='omit')[0])
        metric_scores[labels[test]].append(spearmanr(predicted,experimental)[0])
    
    if exp =='salmonella':
        plt.title("GFP silencing efficiency in Salmonella typhimurium SL1344")
    else:
        plt.title("GFP silencing efficiency in E. coli")#.  
    plt.xlabel("Predicted Score from MERF",fontsize=14)
    plt.ylabel("Experimental logFC",fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("deGFP_%s.svg"%exp,dpi=400)
    # plt.show()
    plt.close()


    plt.figure(figsize=(3,1))
    sns.set_palette("PuBu",len(bars))
    sns.set_style('white')
    ax=sns.barplot([i for i in range(len(bars))],bars,color=sns.color_palette('PuBu').as_hex()[2])
    for i, v in enumerate(bars):
        if v<0.3:
            ax.text(i-0.28,0 , str(round(v,2)), color='k',rotation=90,fontsize=12)
        else:
            ax.text(i-0.28,v-0.3 , str(round(v,2)), color='k',rotation=90,fontsize=12)
    plt.xticks([i for i in range(len(bars))],[labels[i] for i in tests],rotation=30,fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("")
    plt.title("Spearman correlation",fontsize=14)
    plt.savefig("deGFP_%s_bar_CVscale.svg"%exp)
    # plt.show()
    plt.close() 
# metric_scores=pandas.DataFrame.from_dict(metric_scores)
# metric_scores.to_csv("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/metric_scores.csv",sep='\t',index=False)
#%%
'''
### lacZ
'''
df=pandas.read_csv("lacZ_gRNAs.tsv",sep='\t')
algs=['Pasteur','MERF']
tests=list()
metric_scores=defaultdict(list)
df=DataFrame_input(df)
for alg in algs:
    if 'Pasteur' not in alg and 'pasteur' not in alg and alg not in ['CRISPRon','CNN']:
        if alg=='MERF':
            datasets=['3sets']
        for dataset in datasets:
            estimator=pickle.load(open("%s/%s/saved_model/CRISPRi_model.sav"%(alg,dataset),'rb'))
            headers=pickle.load(open("%s/%s/saved_model/CRISPRi_headers.sav"%(alg,dataset),'rb'))
            df_sub=df[headers]
            predictions=estimator.predict(df_sub)
            if dataset != '3sets':
                df[alg+"("+dataset+")"]=predictions
                labels.update({alg+"("+dataset+")":labels[alg]+" ("+dataset+")"})
                tests.append(alg+"("+dataset+")")
            else:
                df[alg]=predictions
                tests.append(alg)
    elif alg in ['CRISPRon','CNN']:
        filename_model = "%s/model_10.ckpt"%(alg)
        header=pickle.load(open("%s/CRISPRi_headers.sav"%(alg+""),'rb'))
        SCALE=pickle.load(open("%s/SCALEr.sav"%(alg+""),'rb'))
        df_sub=df[header]
        df_sub['sequence_30nt'] = df_sub.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1)
        header=[i for i in header if i !='sequence_30nt']
        df_sub[header] = SCALE.transform(df_sub[header])
        from crispri_dl.architectures import Crispr1DCNN, CrisprGRU, CrisprOn1DCNN
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        from pytorch_lightning.callbacks import ModelCheckpoint
        from crispri_dl.dataloader import CrisprDatasetTrain
        loader_test = CrisprDatasetTrain(df_sub, df['pasteur_score'], header)
        loader_test = DataLoader(loader_test, batch_size=df_sub.shape[0])
        if alg=='CNN' :
            trained_model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        elif alg=='CRISPRon' :
            trained_model = CrisprOn1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        #test
        max_epochs = 500
        batch_size = 32
        patience = 5
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            min_delta=0.0, 
            patience=patience, 
            verbose=False, 
            mode="min")
        checkpoint_callback = ModelCheckpoint(
                    monitor = 'val_loss',
                    # dirpath = output_file_name,
                    # filename = "model_"+str(fold_inner),
                    verbose = True,
                    save_top_k = 1,
                    mode = 'min',)
        estimator = pl.Trainer(gpus=0,  max_epochs=max_epochs, check_val_every_n_epoch=1, logger=True,progress_bar_refresh_rate = 0, weights_summary=None)

        predictions_test  = estimator.predict(
                                    model=trained_model,
                                    dataloaders=loader_test,
                                    return_predictions=True,
                                    ckpt_path=filename_model)
        predictions = predictions_test[0].cpu().numpy().flatten()
        df[alg]=predictions
        tests.append(alg)
    else:
        if alg=='Pasteur':
            datasets=['3sets']
        elif alg=='MERF_pasteur':
            datasets=['3sets']
        for dataset in datasets:#,'R75','C18','W']:#,'R75','C18','W']:#'R75W','R75C18','C18W',,'3sets_dataset'
            if alg=='Pasteur':
                estimator=pickle.load(open("%s/%s/saved_model/CRISPRi_model.sav"%(alg,dataset),'rb'))
            elif alg=='MERF_pasteur':
                estimator=pickle.load(open("%s/saved_model/CRISPRi_model.sav"%(alg),'rb'))
            training_seq=list(df['seq_60nt'])
            training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
            training_seq=training_seq.reshape(training_seq.shape[0],-1)
            predictions=estimator.predict(training_seq).reshape(-1, 1).ravel()
            if dataset != '3sets':
                df[alg+"("+dataset+")"]=predictions
                labels.update({alg+"("+dataset+")":labels[alg]+" ("+dataset+")"})
                tests.append(alg+"("+dataset+")")
            else:
                df[alg]=predictions
                tests.append(alg)

tests=['Doench_score(with aa)','TUSCAN','DeepSpCas9','pasteur_score']+tests
measured=list(df['measured_activity'])
bars=list()
metric_scores['metric'].append('Spearman correlation')
metric_scores['validation'].append("lacZ")
plt.figure(figsize=(5,5))
sns.set_palette("PuBu")
sns.set_style('white')
for test in tests:
    predicted=df[test]
    if test=='MERF':
        scatter=True
    else:
        scatter=False
    
    if 'MERF' in test:
        pred=[2**(-1 * i) for i in predicted]
    else:
        pred=[2**(1 * i) for i in predicted]
        predicted=MinMaxScaler().fit_transform(-1*np.array(df[test]).reshape(-1,1))
        predicted=np.hstack(predicted)
    true=[2**(-1 * i) for i in measured]
    if test=='MERF':
        ax= sns.regplot(predicted,measured,
                    label=labels[test],scatter=scatter,color="#1f77b4")
    ndcg=ndcg_score(np.asarray([true]),np.asarray([pred])) 
    metric_scores[labels[test]].append(spearmanr(predicted,measured)[0])
    bars.append(spearmanr(predicted,measured,nan_policy='omit')[0])
plt.title("Miller assay")  #.E. coli K12 MG1655
plt.xlabel("Predicted Score from MERF",fontsize=14)
plt.ylabel("Experimental logFC",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("lacZ.svg",dpi=400)
# plt.show()
plt.close()

plt.figure(figsize=(3,1))
sns.set_style('white')
ax=sns.barplot([i for i in range(len(bars))],bars,color=sns.color_palette('PuBu').as_hex()[2])
for i, v in enumerate(bars):
    if v<0:
        ax.text(i-0.2,0 , str(round(v,2)), color='k',rotation=90,fontsize=12)
    else:
        ax.text(i-0.2,v-0.4 , str(round(v,2)), color='k',rotation=90,fontsize=12)
plt.xticks([i for i in range(len(bars))],[labels[i] for i in tests],fontsize=10,rotation=30)
plt.ylabel("")
plt.title("Spearman correlation",fontsize=14)
plt.savefig("lacZ_bar_CVscale.svg")
# plt.show()
plt.close() 
# metric_scores=pandas.DataFrame.from_dict(metric_scores)
# metric_scores.to_csv("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/metric_scores.csv",sep='\t',index=False)

#%%
'''
###Figure 3A demonstration of gRNA design for purine genes
'''
df=pandas.read_csv("pur_gRNAs.csv",sep='\t')
length=df.groupby('gene_name').mean()
pal=sns.color_palette("PuBu")
plt.figure()
fig, ax = plt.subplots(figsize=(12,5))
sns.set_style('white')
for l in length.index:
    gene_df=df[df['gene_name']==l]
    plt.hlines(length.shape[0]-list(length.index).index(l),xmin=0,xmax=length['gene_length'][l],alpha=0.8)
    plt.text(length['gene_length'][l]+50,length.shape[0]-list(length.index).index(l)+0.2,l,weight='bold',fontsize='large')
    plt.text(length['gene_length'][l]+260,length.shape[0]-list(length.index).index(l)+0.2," ("+str(int(length['gene_length'][l]))+"bp/ "+str(gene_df.shape[0])+"gRNAs)",fontsize='large')
    for j in gene_df.index:
        plt.vlines(gene_df['distance_start_codon'][j],ymin=length.shape[0]-list(length.index).index(l),ymax=length.shape[0]-list(length.index).index(l)+0.6,alpha=0.3,color=pal.as_hex()[5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()
#%%
'''
Purine screen
'''
df=pandas.read_csv("pur_gRNAs.tsv",sep='\t')
###
'''
S6A
'''
df=pandas.melt(df,id_vars=['gene_name'],value_vars=["OD02_edgeR.batch","OD06_edgeR.batch","OD1_edgeR.batch"])
print(df)
df=df.replace({"OD02_edgeR.batch":"OD 0.2","OD06_edgeR.batch":"OD 0.6","OD1_edgeR.batch":"OD 1"})
plt.figure(figsize=(6,4))
sns.set_palette('PuBu',3)
sns.boxplot(data=df,x='gene_name',y='value',hue='variable')
plt.legend(title="",loc='lower left')
plt.xlabel("")
plt.ylabel("logFC")
plt.savefig("S6A.svg",dpi=400)
plt.show()
plt.close()
###
'''
Figure 4B-D
'''
algs=['MERF','Pasteur']
sr_plot='4B'
ppv_plot='4D'
perc_plot='4C'
ppv=5
pasteur_ori=True
tests=list()
metric_scores=defaultdict(list)
df_sub=DataFrame_input(df)
for alg in algs:
    if 'Pasteur' not in alg and 'pasteur' not in alg and alg not in ['CRISPRon','CNN','CRISPRon_deltaGB']:
        if alg=='MERF':
            datasets=['3sets']
        for dataset in datasets:
            estimator=pickle.load(open("%s/%s/saved_model/CRISPRi_model.sav"%(alg,dataset),'rb'))
            headers=pickle.load(open("%s/%s/saved_model/CRISPRi_headers.sav"%(alg,dataset),'rb'))
            df_sub=DataFrame_input(df)
            df_sub=df_sub[headers]
            predictions=estimator.predict(df_sub)
            if dataset != '3sets':
                if 'dropdistance' in dataset:
                    dataset="Drop distance"
                elif 'CAI' in dataset:
                    dataset='CAI'
                elif 'deltaGB' in dataset:
                    dataset='deltaGB'
                df[alg+"("+dataset+")"]=predictions
                labels.update({alg+"("+dataset+")":labels[alg]+" ("+dataset+")"})
                tests.append(alg+"("+dataset+")")
            else:
                df[alg]=predictions
                tests.append(alg)
        
    elif alg in ['CRISPRon','CNN','CRISPRon_deltaGB']:
        filename_model = "%s/model_10.ckpt"%(alg)
        header=pickle.load(open("%s/CRISPRi_headers.sav"%(alg+""),'rb'))
        SCALE=pickle.load(open("%s/SCALEr.sav"%(alg+""),'rb'))
        df_sub=df[header]
        df_sub['sequence_30nt'] = df_sub.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1)
        header=[i for i in header if i !='sequence_30nt']
        df_sub[header] = SCALE.transform(df_sub[header])
        from crispri_dl.architectures import Crispr1DCNN, CrisprGRU, CrisprOn1DCNN
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        from pytorch_lightning.callbacks import ModelCheckpoint
        from crispri_dl.dataloader import CrisprDatasetTrain
        loader_test = CrisprDatasetTrain(df_sub, df['pasteur_score'], header)
        loader_test = DataLoader(loader_test, batch_size=df_sub.shape[0])
        if alg=='CNN' :
            trained_model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        elif alg=='CRISPRon' or alg=='CRISPRon_deltaGB':
            trained_model = CrisprOn1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        #test
        max_epochs = 500
        batch_size = 32
        patience = 5
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            min_delta=0.0, 
            patience=patience, 
            verbose=False, 
            mode="min")
        checkpoint_callback = ModelCheckpoint(
                    monitor = 'val_loss',
                    verbose = True,
                    save_top_k = 1,
                    mode = 'min',)
        estimator = pl.Trainer(gpus=0,  max_epochs=max_epochs, check_val_every_n_epoch=1, logger=True,progress_bar_refresh_rate = 0, weights_summary=None)

        predictions_test  = estimator.predict(
                                    model=trained_model,
                                    dataloaders=loader_test,
                                    return_predictions=True,
                                    ckpt_path=filename_model)
        predictions = predictions_test[0].cpu().numpy().flatten()
        df[alg]=predictions
        tests.append(alg)
    
    
    else:
        datasets=['3sets']
        for dataset in datasets:
            if alg=='Pasteur':
                estimator=pickle.load(open("%s/%s/saved_model/CRISPRi_model.sav"%(alg,dataset),'rb'))
            training_seq=list(df['seq_60nt'])
            training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
            training_seq=training_seq.reshape(training_seq.shape[0],-1)
            predictions=estimator.predict(training_seq).reshape(-1, 1).ravel()
            if dataset != '3sets':
                df[alg+"("+dataset+")"]=predictions
                labels.update({alg+"("+dataset+")": alg+"("+dataset+")"})
                tests.append(alg+"("+dataset+")")
            else:
                df[alg]=predictions
                tests.append(alg)
print(tests)
if pasteur_ori:
    tests=tests+['pasteur_score']
# tests=['Doench_score(with aa)','TUSCAN','DeepSpCas9','SSC']+tests    
genes=list(set(df['gene_name']))
genes.sort()
correlations=defaultdict(list)

###different metrics per gene per model
for prediction in tests:
    for OD in ['OD02','OD06','OD1']:#
        test=OD+"_edgeR.batch"
        for gene in genes:
            if gene in ['purE','purK']:
                continue
            df_gene=df[df['gene_name']==gene]
            df_gene=df_gene.dropna(subset=[test])
            df_gene=df_gene.sort_values(by=test,ascending=True)
            true=[np.exp2(-1 * i) for i in df_gene[test]]
            predicted=np.array(df_gene[prediction])
            if 'MERF' in prediction or 'GPB' in prediction  or ('hyper' in prediction and 'RF' not in prediction):
                pred=[np.exp2(-1 * i) for i in predicted]
                Rs,ps=spearmanr(df_gene[test],df_gene[prediction])
            else:
                pred=[np.exp2(1 * i) for i in predicted]
                Rs,ps=spearmanr(df_gene[test],-df_gene[prediction])
            correlations['spearmanr'].append(Rs)
            correlations['gene'].append(gene+"("+str(df_gene.shape[0])+")")
            correlations['timepoint'].append(OD)
            correlations['method'].append(prediction)
            
            metric_scores['metric'].append('Spearman correlation')
            metric_scores['validation'].append(gene)
            metric_scores['timepoint'].append(OD)
            metric_scores['method'].append(labels[prediction])
            metric_scores['value'].append(Rs)
correlations=pandas.DataFrame.from_dict(correlations)
# metric_scores=pandas.DataFrame.from_dict(metric_scores)
# metric_scores.to_csv("metric_scores.csv",sep='\t',index=False)
methods=defaultdict(list)
included_methods=list(set(correlations['method']))

for test in ['spearmanr']:
    sns.set_palette('Set2',len(tests))
    sns.set_style("whitegrid")
    plt.figure(figsize=(4,5))
    ax=sns.boxplot(data=correlations[correlations['method'].isin(included_methods)],x='timepoint',y=test,hue='method')
    sns.swarmplot(data=correlations[correlations['method'].isin(included_methods)],x='timepoint',y=test,hue='method',dodge=True,s=2,color='k')
    plt.xlabel("")
    plt.ylabel("Spearman correlation",fontsize=14)
    plt.xticks([0,1,2],["OD 0.2","OD 0.6","OD 1"],fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right',fontsize='x-small')
    plt.subplots_adjust(right=0.8)
    handles, predictions = ax.get_legend_handles_labels()
    plt.legend(handles[:len(included_methods)], [labels[prediction] for prediction in predictions][:len(included_methods)],
                fontsize=12, loc='center',
                ncol=2,bbox_to_anchor=(0,-0.25, 1, .102))
    plt.savefig("%s.svg"%sr_plot,dpi=400)
    plt.show()
    plt.close()  
'''
###PPV
'''
gene_wise_corr=defaultdict(list)
###different metrics per gene per model
for fold in [1.5,2,2.5,3,3.5,4,4.5,5]:
    for prediction in tests:
        mccs=list()
        PPVs=list()
        for OD in ['OD02','OD06','OD1']:#
            test=OD+"_edgeR.batch"
            true_genes=list()
            pred_genes=list()
            for gene in genes:
                if gene in ['purE','purK']:
                    continue
                df_gene=df[df['gene_name']==gene]
                df_gene=df_gene.dropna(subset=[test])
                df_gene=df_gene.sort_values(by=test,ascending=True)
                true=[np.exp2(-1 * i) for i in df_gene[test]]
                predicted=np.array(df_gene[prediction])
                if 'MERF' in prediction or 'GPB' in prediction  or ('hyper' in prediction and 'RF' not in prediction):
                    pred=[np.exp2(-1 * i) for i in predicted]
                else:
                    pred=[np.exp2(1 * i) for i in predicted]
                
                top=np.max(true)
                true=[1  if i >=(top/fold) else 0 for i in true]
                good_pred=list()
                for i in range(ppv):
                    if i >true.count(1):
                        break
                    good_pred.append(pred.index(sorted(pred,reverse=True)[i]))
                pred=[1 if i in good_pred else 0 for i in range(len(pred))]
                true_genes+=true
                pred_genes+=pred
            mcc=sklearn.metrics.matthews_corrcoef(true_genes,pred_genes)
            TP = np.sum([1 if true_genes[i]==1 and pred_genes[i]==1 else 0 for i in range(len(pred_genes))])
            FP = np.sum([1 if true_genes[i]==0 and pred_genes[i]==1 else 0 for i in range(len(pred_genes))])
            PPV=TP/(TP+FP)
            mccs.append(mcc)
            PPVs.append(PPV)
            gene_wise_corr['fold'].append(fold)
            gene_wise_corr['OD'].append(OD)
            gene_wise_corr['method'].append(labels[prediction])
            gene_wise_corr['PPV'].append(PPV)
gene_wise_corr=pandas.DataFrame.from_dict(gene_wise_corr)
gene_wise_corr.to_csv("purine_PPV.csv",sep='\t',index=False)

for test in ['PPV']:
    sns.set_style("white")
    plt.figure(figsize=(4,5))
    ax=sns.lineplot(data=gene_wise_corr,y=test,x='fold',hue='method',style='method',markers=True)
    plt.xlabel("Fold")
    if test=='mcc':
        plt.ylabel("Matthews correlation coefficient",fontsize=14)
    elif test=='PPV':
        plt.ylabel("Positive predictive value",fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Fold',fontsize=14)
    plt.legend(fontsize=12,title="",loc='lower right')#,bbox_to_anchor=(1,1))#
    plt.subplots_adjust(right=0.8)
    plt.savefig("%s.svg"%ppv_plot,dpi=400)
    plt.show()
    plt.close()  

'''
###enrichement of different quantile guides in each predicted catergory, efficient: parameter perc 
'''

perc=0.2
for gene in genes:
    for OD in ['OD02','OD06','OD1']:#
        test=OD+"_edgeR.batch"
    
        df_gene=df[df['gene_name']==gene]
        df_gene=df_gene.dropna(subset=[test])
        df_gene=df_gene.sort_values(by=test,ascending=True)
        
        for i in df_gene.index:
            df.at[i,'percent_rank_'+OD]=(list(df_gene.index).index(i)+1)/df_gene.shape[0]
for prediction in tests:
    for gene in genes:
        df_gene=df[df['gene_name']==gene]
        if 'MERF' in prediction:
            df_gene=df_gene.sort_values(by=prediction,ascending=True)
        else:
            df_gene=df_gene.sort_values(by=prediction,ascending=False)
        for i in df_gene.index:
            df.at[i,'pred_'+prediction]=(list(df_gene.index).index(i)+1)/df_gene.shape[0]
        
  
    
'''
###enrichement of efficient guides in each predicted efficient guides, efficient: parameter perc 
'''
# csv_save=defaultdict(list) # saving the scores
for perc in [0.2]:
    correlations=defaultdict(list)
    for prediction in tests:
        for OD in ['OD02','OD06','OD1']:#
            test=OD+"_edgeR.batch"
            true_genes=list()
            pred_genes=list()
            for gene in genes:
                if gene in ['purE','purK']:
                    continue
                df_gene=df[df['gene_name']==gene]
                df_gene=df_gene.dropna(subset=[test])
                df_gene=df_gene.sort_values(by=test,ascending=True)
                true_high=list(df_gene[df_gene['percent_rank_'+OD]<perc].index)
                predicted_high=list(df_gene[df_gene['pred_'+prediction]<perc].index)
                enriched_perc=len([i for i in predicted_high if i in true_high])/len(predicted_high)*100
                
                correlations['true_good_in_pred_good'].append(enriched_perc)             
                correlations['timepoint'].append(OD)
                correlations['method'].append(prediction)
                # csv_save['method'].append(prediction)
                # csv_save['timepoint'].append(OD)
                # csv_save['gene'].append(gene)
                # csv_save['perc'].append(perc)
                # csv_save['true_good_in_pred_good'].append(enriched_perc)             
                
                
    correlations=pandas.DataFrame.from_dict(correlations)
    sns.set_palette('Set2',len(tests))
    sns.set_style("whitegrid")
    plt.figure(figsize=(4,5))
    ax=sns.boxplot(data=correlations,x='timepoint',y='true_good_in_pred_good',hue='method')#
    sns.swarmplot(data=correlations,x='timepoint',y='true_good_in_pred_good',hue='method',dodge=True,s=2,color='k')
    plt.xlabel("")
    plt.ylabel("Percentage of efficient gRNAs\n in predicted efficient gRNAs ("+str(perc*100)+"%)",fontsize=14)
    plt.xticks([0,1,2],["OD 0.2","OD 0.6","OD 1"],fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right',fontsize='x-small')
    plt.subplots_adjust(right=0.8)
    handles, predictions = ax.get_legend_handles_labels()
    plt.legend(handles[:len(included_methods)], [labels[prediction] for prediction in predictions][:len(included_methods)],
                fontsize=12, loc='center',
                ncol=2,bbox_to_anchor=(0,-0.25, 1, .102))
    plt.savefig("%s.svg"%perc_plot,dpi=400)
    plt.show()
    plt.close()  
#%%
'''
Figure 4E & S6B
'''
#log2FC vs distance start codon
df=pandas.read_csv("pur_gRNAs.csv",sep='\t')
sns.set_palette('PuBu',3)
sns.set_style("white")
fig, axes=plt.subplots(3,3)
for gene in genes:
    gene_df=df[(df['gene_name']==gene) ]
    for OD in ['OD02','OD06','OD1']:#
        test=OD+"_edgeR.batch"
        ax=sns.regplot(data=gene_df,x='distance_start_codon',y=test,ax=axes[genes.index(gene)//3,genes.index(gene)%3],label=OD,line_kws={"linewidth":1},scatter_kws={"alpha":0.7,'s':1})
    axes[genes.index(gene)//3,genes.index(gene)%3].set_title(gene,fontsize='small',pad=2.5)
    axes[genes.index(gene)//3,genes.index(gene)%3].tick_params(labelsize=6,pad=-2.5)
    axes[genes.index(gene)//3,genes.index(gene)%3].set_xlabel("")
    axes[genes.index(gene)//3,genes.index(gene)%3].set_ylabel("")
plt.subplots_adjust(wspace=0.15,hspace=0.3)
fig.text(0.5, 0.01, 'Distance to start codon (bp)', ha='center',fontsize='small')
fig.text(0.06, 0.5, 'logFC', va='center', rotation='vertical',fontsize='small')
handles, labels = ax.get_legend_handles_labels()
lgnd=fig.legend(handles[0:3], ['OD 0.2', 'OD 0.6', 'OD 1'], fontsize='xx-small',bbox_to_anchor=(0.22, 0.01, 0.5, 0.17),loc='center',
       ncol=3)
lgnd.legendHandles[0]._sizes=[40]
lgnd.legendHandles[1]._sizes=[40]
lgnd.legendHandles[2]._sizes=[40]
plt.savefig("distance_logFC.svg")
plt.show()
plt.close()    

    
#%%
'''
###Calculating hamming distance between gRNAs
'''
from numpy import savetxt,loadtxt
from scipy.spatial.distance import hamming
from tqdm import tqdm

def DataFrame_input(df,split):
    ###keep guides for essential genes
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==1)]
    df=df.dropna()
    if split=='guide':
        for i in list(set(list(df['geneid']))):
            df_gene=df[df['geneid']==i]
            for j in df_gene.index:
                df.at[j,'Nr_guide']=df_gene.shape[0]
    elif split=='gene':
        for dataset in range(len(set(df['dataset']))):
            dataset_df=df[df['dataset']==dataset]
            for i in list(set(dataset_df['geneid'])):
                gene_df=dataset_df[dataset_df['geneid']==i]
                for j in gene_df.index:
                    df.at[j,'Nr_guide']=gene_df.shape[0]
    df=df[df['Nr_guide']>=5]#keep only genes with more than 5 guides from all 3 datasets
    print(df.shape)
    sequences=list(dict.fromkeys(df['sequence']))
    ### one hot encoded sequence features
    for i in df.index:
        if split=='guide':
            df.at[i,'guideid']=sequences.index(df['sequence'][i])
        elif split=='gene':
            df.at[i,'guideid']=int(df['geneid'][i][1:])
    X=df[['sequence','sequence_30nt','guideid','dataset','geneid']]
    guideids=np.array(list(df['guideid']))
    sequences=list(dict.fromkeys(df['sequence_30nt']))
    return  X,guideids,sequences

path="github_code"
folds=10
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
df1=pandas.read_csv(datasets[0],sep="\t")
df1 = df1.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df1['dataset']=[0]*df1.shape[0]
df2=pandas.read_csv(datasets[1],sep="\t")
df2 = df2.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df2['dataset']=[1]*df2.shape[0]
df3=pandas.read_csv(datasets[2],sep="\t")
df3 = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df3['dataset']=[2]*df3.shape[0]
df2=df2.append(df3,ignore_index=True)  
training_df=df1.append(df2,ignore_index=True)  
training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
#dropping unnecessary features and encode sequence features
X_df,guideids, guide_sequence_set=DataFrame_input(training_df,'guide')
'''
###save unique sequences 
'''
guide_seq_file=open("unique_30nt_guides.txt","w")
for i in guide_sequence_set:
    guide_seq_file.writelines(i+'\n')
guide_seq_file.close()

'''
### calculate the pair-wise hamming distance across all gRNAs
'''
dist=np.zeros((len(guide_sequence_set),len(guide_sequence_set)),dtype=np.int0)
# dist[dist==0]=np.nan
for i in tqdm(range(len(guide_sequence_set))):
    for j in range(len(guide_sequence_set)):
        distance=hamming(list(guide_sequence_set[i]), list(guide_sequence_set[j]))
        # print(list(guide_sequence_set[i]), list(guide_sequence_set[j]))
        dist[i][j]=distance*30
dist=dist.astype(np.int0)
savetxt("gRNA_hamming_distance.csv",dist,delimiter='\t')
dist = loadtxt("gRNA_hamming_distance.csv", delimiter='\t')
'''
### plot the distribution of hamming distance
'''
lower=dist[np.tril_indices_from(dist,k=-1)]
print(np.median(lower.flatten()),np.min(lower.flatten()))
import sklearn.model_selection
'''
Calculate and plot the hamming distance between train and test samples for guide-wise split
'''
X_df,guideids, guide_sequence_set=DataFrame_input(training_df,'guide')
guide_sequence_set=open("unique_30nt_guides.txt","r")
guide_sequence_set=guide_sequence_set.readlines()
guide_sequence_set=[i.strip() for i in guide_sequence_set]
#k-fold cross validation
training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}
guideid_set=list(set(guideids))
plot=defaultdict(list)
fold=0
size=defaultdict(list)

for training_sets in training_set_list.keys():
    training_sets=list(training_sets)
    print(training_set_list[tuple(training_sets)])
    kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
    plot=defaultdict(list)
    fold=0
    for train_index, test_index in kf.split(guideid_set):##split the combined training set into train and test based on guideid
        train_index=np.array(guideid_set)[train_index]
        test_index=np.array(guideid_set)[test_index]
        X_train = X_df[X_df['guideid'].isin(train_index)]
        X_train=X_train[X_train['dataset'].isin(training_sets)]
        
        test = X_df[X_df['guideid'].isin(test_index)]
        X_test=test
        
        X_train_seq_inds=[guide_sequence_set.index(i) for i in list(X_train['sequence_30nt'])]
        X_test_seq_inds=[guide_sequence_set.index(i) for i in list(X_test['sequence_30nt'])]
        print(X_train.shape,len(set(X_train['geneid'])),len(train_index),len(X_train_seq_inds))
        print(X_test.shape,len(set(X_test['geneid'])),len(test_index),len(X_test_seq_inds))
        fold+=1
        dist_train_test=dist[np.ix_(X_train_seq_inds,X_test_seq_inds)]
        print(fold,np.min(dist_train_test),np.max(dist_train_test),np.median(dist_train_test))
        size['split'].append('guide-wise')
        size['training_data'].append(training_set_list[tuple(training_sets)])
        size['fold'].append(fold)
        size['train_guide'].append(X_train.shape[0])
        size['test_guide'].append(X_test.shape[0])
        size['train_gene'].append(len(set(X_train['geneid'])))
        size['test_gene'].append(len(set(X_test['geneid'])))
        size['minimum_hamming'].append(np.min(dist_train_test))
        size['average_hamming'].append(np.mean(dist_train_test))
        
# plot=pandas.DataFrame.from_dict(plot)
# sns.boxplot(x='fold',y='value',color='steelblue')
# plt.xlabel('CV fold')
# plt.ylabel("Hamming distance between train and test")
# plt.show()
# plt.close()

'''
Calculate and plot the hamming distance between train and test samples for gene-wise split
'''
training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}
X_df,guideids, guide_sequence_set=DataFrame_input(training_df,'gene')
# print(X_df)
guide_sequence_set=open("unique_30nt_guides.txt","r")
guide_sequence_set=guide_sequence_set.readlines()
guide_sequence_set=[i.strip() for i in guide_sequence_set]
print(len(guide_sequence_set))
training_sets=[0,1,2]
guideid_set=list(set(guideids))
print(len(guideid_set))
plot=defaultdict(list)
for training_sets in training_set_list.keys():
    training_sets=list(training_sets)
    print(training_set_list[tuple(training_sets)])
    guideid_set=list(set(guideids))
    kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
    fold=0
    for train_index, test_index in kf.split(guideid_set):##split the combined training set into train and test based on guideid
        train_index=np.array(guideid_set)[train_index]
        test_index=np.array(guideid_set)[test_index]
        X_train = X_df[X_df['guideid'].isin(train_index)]
        X_train=X_train[X_train['dataset'].isin(training_sets)]
        
        test = X_df[X_df['guideid'].isin(test_index)]
        X_test=test
        
        X_train_seq_inds=[guide_sequence_set.index(i) for i in list(X_train['sequence_30nt'])]
        X_test_seq_inds=[guide_sequence_set.index(i) for i in list(X_test['sequence_30nt'])]
        # print([i for i in list(set(X_train['sequence'])) if i in list(set(X_test['sequence']))])
        print(X_train.shape,len(set(X_train['sequence'])),len(train_index))
        print(X_test.shape,len(set(X_test['sequence'])),len(test_index))
        fold+=1
        dist_train_test=dist[np.ix_(X_train_seq_inds,X_test_seq_inds)]
        print(fold,np.min(dist_train_test),np.max(dist_train_test),np.median(dist_train_test))
        size['split'].append('gene-wise')
        size['training_data'].append(training_set_list[tuple(training_sets)])
        size['fold'].append(fold)
        size['train_guide'].append(X_train.shape[0])
        size['test_guide'].append(X_test.shape[0])
        size['train_gene'].append(len(set(X_train['geneid'])))
        size['test_gene'].append(len(set(X_test['geneid'])))
        size['minimum_hamming'].append(np.min(dist_train_test))
        size['average_hamming'].append(np.mean(dist_train_test))
size=pandas.DataFrame.from_dict(size)
size.to_csv("between_train_test.tsv",sep='\t',index=False)
#%%
'''
###Hamming distance across test sets in 10 iterations of gene-split cross-validation
'''
X_df,guideids, guide_sequence_set=DataFrame_input(training_df,'guide')
guide_sequence_set=open("unique_30nt_guides.txt","r")
guide_sequence_set=guide_sequence_set.readlines()
guide_sequence_set=[i.strip() for i in guide_sequence_set]
training_sets=[0,1,2]
guideid_set=list(set(guideids))
plot=defaultdict(list)
guideid_set=list(set(guideids))
kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
fold=0
test_inds=[tuple(i[1]) for i in kf.split(guideid_set)]
combs=list(itertools.combinations(test_inds,2))
size=defaultdict(list)
for c in combs:
    test_1=np.array(guideid_set)[list(c[0])]
    test_2=np.array(guideid_set)[list(c[1])]
    test_1 = X_df[X_df['guideid'].isin(test_1)]
    test_2 = X_df[X_df['guideid'].isin(test_2)]
    
    test_1=[guide_sequence_set.index(i) for i in list(test_1['sequence_30nt'])]
    test_2=[guide_sequence_set.index(i) for i in list(test_2['sequence_30nt'])]
    dist_test_test=dist[np.ix_(test_1,test_2)]
    print(len(test_1),len(test_2),np.median(dist_test_test))
    size['fold_test1'].append(test_inds.index(c[0])+1)
    size['fold_test2'].append(test_inds.index(c[1])+1)
    size['minimum_hamming'].append(np.min(dist_test_test))
    size['average_hamming'].append(np.mean(dist_test_test))
size=pandas.DataFrame.from_dict(size)
print(min(list(size['minimum_hamming'])))
size.to_csv("between_test_test.tsv",sep='\t',index=False)
#%%
'''
### hamming distance  between training data and validation data
'''
path="github_code"
folds=10
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
df1=pandas.read_csv(datasets[0],sep="\t")
df1 = df1.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df1['dataset']=[0]*df1.shape[0]
df2=pandas.read_csv(datasets[1],sep="\t")
df2 = df2.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df2['dataset']=[1]*df2.shape[0]
df3=pandas.read_csv(datasets[2],sep="\t")
df3 = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df3['dataset']=[2]*df3.shape[0]
df2=df2.append(df3,ignore_index=True)  
training_df=df1.append(df2,ignore_index=True)  
training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
#dropping unnecessary features and encode sequence features
X_df,guideids, guide_sequence_set=DataFrame_input(training_df,'guide')
print(len(guide_sequence_set))
val=['deGFP','lacZ','pur'] #,
size=defaultdict(list)
for v in val:
    df=pandas.read_csv("%s_gRNAs.csv"%v,sep='\t')
    vali_seq=list(df['sequence_30nt'])
    print(v,len(vali_seq))
    '''
    ###calculate the hamming distance
    # dist=np.zeros((len(guide_sequence_set),len(vali_seq)),dtype=np.int0)
    # # dist[dist==0]=np.nan
    # for i in tqdm(range(len(guide_sequence_set))):
    #     for j in range(len(vali_seq)):
    #         distance=hamming(list(guide_sequence_set[i]), list(vali_seq[j]))
    #         # print(list(guide_sequence_set[i]), list(guide_sequence_set[j]))
    #         dist[i][j]=distance*30
    # dist=dist.astype(np.int0)
    # savetxt("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/figure1/hamming_distance_%s.csv"%v,dist,delimiter='\t')
   '''
    from numpy import savetxt,loadtxt
    dist = loadtxt("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/figure1_2/hamming_distance/hamming_distance_%s.csv"%v, delimiter='\t')
    print(np.median(dist.flatten()),np.min(dist.flatten()))
 
    ###plot the distribution
    # plt.figure(figsize=(3,3))
    # sns.displot(   dist.flatten(),bins=23,color='steelblue')
    # plt.ylabel("Count",fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.xlabel("Hamming distance of 30nt extended sequences",fontsize=12)
    # plt.title("between training data and "+v,fontsize=14)
    # plt.show()
    # plt.close()
    '''
    size['val'].append(v)
    size['minimum_hamming'].append(np.min(dist.flatten()))
    size['average_hamming'].append(np.mean(dist.flatten()))
size=pandas.DataFrame.from_dict(size)
size.to_csv("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/figure1/hamming_distance/between_train_vali.tsv",sep='\t',index=False)
    '''
#%%
'''
##Table S2
### output the table to specify which guides for training and which for test in guide and gene-wise split
'''
from tqdm import tqdm
###Calculating hamming distance between gRNAs
def DataFrame_input(df,split):
    ###keep guides for essential genes
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==1)]
    df=df.dropna()
    for dataset in range(len(set(df['dataset']))):
        dataset_df=df[df['dataset']==dataset]
        for i in list(set(dataset_df['geneid'])):
            gene_df=dataset_df[dataset_df['geneid']==i]
            for j in gene_df.index:
                df.at[j,'Nr_guide']=gene_df.shape[0]
    df=df[df['Nr_guide']>=5]#keep only genes with more than 5 guides from all 3 datasets
    print(df.shape)
    sequences=list(dict.fromkeys(df['sequence']))
    ### one hot encoded sequence features
    for i in df.index:
        if split=='guide':
            df.at[i,'guideid']=sequences.index(df['sequence'][i])
        elif split=='gene':
            df.at[i,'guideid']=int(df['geneid'][i][1:])
    X=df[['No.','sequence','sequence_30nt','guideid','dataset','geneid']]
    guideids=np.array(list(df['guideid']))
    sequences=list(dict.fromkeys(df['sequence_30nt']))
    return  X,guideids,sequences

path="github_code"
folds=10
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
df1=pandas.read_csv(datasets[0],sep="\t")
df1_shuffled = df1.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df1_shuffled['dataset']=[0]*df1_shuffled.shape[0]
df2=pandas.read_csv(datasets[1],sep="\t")
df2_shuffled = df2.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df2_shuffled['dataset']=[1]*df2_shuffled.shape[0]
df3=pandas.read_csv(datasets[2],sep="\t")
df3_shuffled = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
df3_shuffled['dataset']=[2]*df3_shuffled.shape[0]
df2_shuffled=df2_shuffled.append(df3_shuffled,ignore_index=True)  
training_df=df1_shuffled.append(df2_shuffled,ignore_index=True)  
training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
#dropping unnecessary features and encode sequence features
for split in ['guide','gene']:
    X_df,guideids, guide_sequence_set=DataFrame_input(training_df,split)
    guideid_set=list(set(guideids))
    kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
    fold=1
    for train_index, test_index in kf.split(guideid_set):##split the combined training set into train and test based on guideid
        train_index=np.array(guideid_set)[train_index]
        test_index=np.array(guideid_set)[test_index]
        train = X_df[X_df['guideid'].isin(train_index)]
        test = X_df[X_df['guideid'].isin(test_index)]
        for i in    train.index:
            if train['dataset'][i]==0:
                df1_guide=df1[df1['No.']==train['No.'][i]]
                if df1_guide.shape[0]!=1:
                    print('df1',df1_guide)
                    sys.exit()
                df1.at[list(df1_guide.index)[0],split+'-wise_fold_%s'%fold]='train'
            elif train['dataset'][i]==1:
                df2_guide=df2[df2['No.']==train['No.'][i]]
                if df2_guide.shape[0]!=1:
                    print('df2',df2_guide)
                    sys.exit()
                df2.at[list(df2_guide.index)[0],split+'-wise_fold_%s'%fold]='train'
            elif train['dataset'][i]==2:
                df3_guide=df3[df3['No.']==train['No.'][i]]
                if df3_guide.shape[0]!=1:
                    print('df3',df3_guide)
                    sys.exit()
                df3.at[list(df3_guide.index)[0],split+'-wise_fold_%s'%fold]='train'
        for i in    test.index:
            if test['dataset'][i]==0:
                df1_guide=df1[df1['No.']==test['No.'][i]]
                if df1_guide.shape[0]!=1:
                    print('df1',df1_guide)
                    sys.exit()
                df1.at[list(df1_guide.index)[0],split+'-wise_fold_%s'%fold]='test'
            elif test['dataset'][i]==1:
                df2_guide=df2[df2['No.']==test['No.'][i]]
                if df2_guide.shape[0]!=1:
                    print('df2',df2_guide)
                    sys.exit()
                df2.at[list(df2_guide.index)[0],split+'-wise_fold_%s'%fold]='test'
            elif test['dataset'][i]==2:
                df3_guide=df3[df3['No.']==test['No.'][i]]
                if df3_guide.shape[0]!=1:
                    print('df3',df3_guide)
                    sys.exit()
                df3.at[list(df3_guide.index)[0],split+'-wise_fold_%s'%fold]='test'
        fold+=1
df1=df1.dropna(subset=['guide-wise_fold_1'])
df2=df2.dropna(subset=['guide-wise_fold_1'])
df3=df3.dropna(subset=['guide-wise_fold_1'])
cols=df1.columns.values.tolist()
cols=[i for i in cols if 'wise_fold' in i]
cols=['No.','geneid','genename','sequence','dataset']+cols
df1=df1[cols]
df2=df2[cols]
df3=df3[cols]

df1.to_csv("E75_Rousset_train_test_split.csv",sep='\t',index=False)
df2.to_csv("E18_Cui_train_test_split.csv",sep='\t',index=False)
df3.to_csv("Wang_dataset_train_test_split.csv",sep='\t',index=False)
    