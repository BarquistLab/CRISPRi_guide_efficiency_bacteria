#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:35:49 2020

@author: yanying
"""
#%%

import h2o
from h2o.automl import H2OAutoML
import pandas
import sklearn.model_selection
import numpy as np
from scipy.stats import spearmanr,pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
# from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from collections import defaultdict
import os
import itertools
import logging
from Bio import SeqIO
import pickle
import statistics
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['figure.dpi'] = 300
# h2o.init()
# h2o.remove_all()
def position_specific(seqs):
    encoded=defaultdict(list)
    for seq in seqs:
        for i in range(1,5):
            for j in range(1,len(seq)-i+2):
                index=str(j)
                for k in range(i):
                    if k != 0:
                        index+="_"+str(j+k)
                encoded['pos_%s'%index].append(seq[j-1:j+i-1])
    encoded=pandas.DataFrame.from_dict(encoded)
    return encoded

def position_independent(seqs):
    nts=['A','T','C','G']
    encoded=defaultdict(list)
    for i in range(1,5):
        motifs=list(itertools.product(nts,repeat=i))
        for motif in motifs:
            motif="".join(motif)
            for seq in seqs:
                pos=0
                count=0
                while pos < len(seq):
                    motif_pos=seq[pos:].find(motif)
                    if motif_pos >=0:
                        count+=1
                        pos+=motif_pos+1
                    else:
                        break
                encoded[motif].append(count)
    encoded=pandas.DataFrame.from_dict(encoded)
    return encoded
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
#%%

# h2o.shutdown()
h2o.init()
h2o.remove_all()
h2o.no_progress()
name="H2O"
path="/home/yanying/projects/crispri/doc/CRISPRi_manuscript/result/figure1/dataset_fusion/%s"%name
if os.path.isdir(path)==False:
    os.mkdir(path)

logging_file= path+"/Log.txt"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)

#data fusion
rousset=pandas.read_csv("/home/yanying/projects/crispri/CRISPRi_ml/datasets/Rousset_dataset_new.csv",sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
rousset['dataset']=[0]*rousset.shape[0]
rousset = rousset.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
rousset18=pandas.read_csv("/home/yanying/projects/crispri/CRISPRi_ml/datasets/Rousset_dataset_fit18.csv",sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
rousset18['dataset']=[1]*rousset18.shape[0]
rousset18 = rousset18.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
wang=pandas.read_csv("/home/yanying/projects/crispri/CRISPRi_ml/datasets/Wang_dataset_new.csv",sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
wang['dataset']=[2]*wang.shape[0]
wang = wang.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
combined = rousset.append(rousset18,ignore_index=True)
combined = combined.append(wang,ignore_index=True)
combined = combined.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
combined = combined[(combined['gene_essentiality']==1)&(combined['intergenic']==0)&(combined['coding_strand']==1)] #
combined = combined.dropna()
for i in list(set(list(combined['geneid']))):
    df_gene=combined[combined['geneid']==i]
    for j in df_gene.index:
        combined.at[j,'Nr_guide']=df_gene.shape[0]
combined=combined[combined['Nr_guide']>=5]
logging.info("Number of guides for essential genes: %s \n" % combined.shape[0])
guide_sequence_set=list(dict.fromkeys(combined['sequence']))
print(len(guide_sequence_set))
PAM_encoded=[]
sequence_encoded=[]
dinucleotide_encoded=[]
for i in combined.index:
    PAM_encoded.append(self_encode(combined['PAM'][i]))
    sequence_encoded.append(self_encode(combined['sequence'][i]))   
    dinucleotide_encoded.append(dinucleotide(combined['sequence_30nt'][i]))
    combined.at[i,'geneid']=int(combined['geneid'][i][1:])
    combined.at[i,'guideid']=guide_sequence_set.index(combined['sequence'][i])
if len(list(set(map(len,list(combined['PAM'])))))==1:
    PAM_len=int(list(set(map(len,list(combined['PAM']))))[0])
else:
    print("error: PAM len")
if len(list(set(map(len,list(combined['sequence'])))))==1:   
    sequence_len=int(list(set(map(len,list(combined['sequence']))))[0])
else:
    print("error: sequence len")
if len(list(set(map(len,list(combined['sequence_30nt'])))))==1:   
    dinucleotide_len=int(list(set(map(len,list(combined['sequence_30nt']))))[0])
else:
    print("error: sequence len")
guideids=np.array(list(combined['guideid']))
#drop features
X=combined.drop(['guideid',"No.","genename","gene_strand","gene_5","gene_biotype","gene_3","genome_pos_5_end","genome_pos_3_end",\
                 'gene_essentiality','intergenic','guide_strand','coding_strand','PAM','sequence','sequence_30nt','Nr_guide','off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70'],1)
features=X.columns.values.tolist()
X=np.c_[X,sequence_encoded,PAM_encoded,dinucleotide_encoded]
    
###add one-hot encoded sequence features to headers
nts=['A','T','C','G']
for i in range(sequence_len):
    for j in range(len(nts)):
        features.append('sequence_%s_%s'%(i+1,nts[j]))
for i in range(PAM_len):
    for j in range(len(nts)):
        features.append('PAM_%s_%s'%(i+1,nts[j]))
items=list(itertools.product(nts,repeat=2))
dinucleotides=list(map(lambda x: x[0]+x[1],items))
for i in range(dinucleotide_len-1):
    for dint in dinucleotides:
        features.append(dint+str(i+1)+str(i+2))
print(len(features))
print(features)
X=pandas.DataFrame(data=X,columns=features)
logging.info("Number of features: %s\n" % len(features))
kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X['log2FC']).reshape(-1,1))
# X=h2o.H2OFrame(X)
y="log2FC"

# splits = X.split_frame(ratios = [0.8], seed = 111)
# train = splits[0]
# test = splits[1]
# print(train.shape,test.shape)
guide_train, guide_test = sklearn.model_selection.train_test_split(range(len(guide_sequence_set)), test_size=0.2,random_state=np.random.seed(111))  
X_df=pandas.DataFrame(data=np.c_[X,guideids],columns=features+['guideid'])
train = X_df[X_df['guideid'].isin(guide_train)]
train=train.drop('guideid',1)
test = X_df[X_df['guideid'].isin(guide_test)]
test=test.drop('guideid',1)
features.remove("log2FC")
print(train.shape,test.shape)
# train, test = X.iloc[train_index], X.iloc[test_index]
train=h2o.H2OFrame(train)
test=h2o.H2OFrame(test)
#set params for h2o automl
aml = H2OAutoML(max_runtime_secs = 0, seed = 1, project_name = "comparison", 
                        # include_algos= ['DRF'], 
                      exclude_algos= ['StackedEnsemble'],
                       sort_metric='MSE',
                      keep_cross_validation_predictions=True, keep_cross_validation_fold_assignment=True)
aml.train(x = features, y = y, training_frame = train)

#save resulting params
params=defaultdict(list)
for key in aml.leader.params.keys():
    if key in ['training_frame','validation_frame','model_id']:
        continue
    params['parameter'].append(key)
    params['default'].append(aml.leader.params[key]['default'])
    params['actual'].append(aml.leader.params[key]['actual'])
    params['input'].append(aml.leader.params[key]['input'])
params=pandas.DataFrame.from_dict(params)
params.to_csv("%s/gene_params.csv"%path,sep='\t',index=False)
print(aml.leader.summary)

#save variable importance
varimp=aml.leader.varimp(use_pandas=True)
varimp.to_csv("%s/gene_varimp.csv"%path,sep='\t',index=False)
aml.leader.varimp_plot()
plt.savefig("%s/varimp_plot.png"%path,dpi=400)
# plt.show()
plt.close()

#save performance evaluation
perf = aml.leader.model_performance(test)
logging.info("Performance of gene model:\n %s" %str(perf))
print(perf)

#save model
model=aml.leader
model_path=h2o.save_model(model, path=path)
# model=h2o.load_model(path+"/XGBoost_grid__1_AutoML_20210130_203407_model_12")
# X= h2o.as_list(X, use_pandas=True)
evaluations=defaultdict(list)
kf=sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=np.random.seed(111))

# X_df=pandas.DataFrame(data=np.c_[X,guideids],columns=X.columns.values.tolist()+['guideid'])
# features.remove("log2FC")
for train_index, test_index in kf.split(range(len(guide_sequence_set))):
    train = X_df[X_df['guideid'].isin(train_index)]
    train=train.drop('guideid',1)
    test = X_df[X_df['guideid'].isin(test_index)]
    test=test.drop('guideid',1)
    print(train.shape,test.shape)
    # train, test = X.iloc[train_index], X.iloc[test_index]
    train=h2o.H2OFrame(train)
    test=h2o.H2OFrame(test)
    model.train(x = features, y = y, training_frame = train)
    
    #satterplot
    predictions = model.predict(test)
    test= h2o.as_list(test, use_pandas=True)
    print(test.shape)
    predictions= h2o.as_list(predictions, use_pandas=True)
    print(predictions.shape)
    spearman_rho,_=spearmanr(np.array(test[y]), np.array(predictions['predict']))
    pearsonr_rho,_=pearsonr(np.array(test[y]), np.array(predictions['predict']))
    print(spearman_rho,pearsonr_rho)
    evaluations['Rs'].append(spearman_rho)
    evaluations['Rp'].append(pearsonr_rho)
    test_clustered=kmeans.predict(np.array(test['log2FC']).reshape(-1,1)) ## kmeans was trained with all samples
    if list(kmeans.cluster_centers_).index(min(kmeans.cluster_centers_))==0: ## define cluster with smaller value as class 1 
        test_clustered=test_clustered
    else:
        test_clustered=[1 if x==0 else 0 for x in test_clustered] 
    fpr,tpr,thresholds=sklearn.metrics.roc_curve(test_clustered,predictions)  
    roc_auc_score=sklearn.metrics.auc(fpr,tpr)
    evaluations['AUC'].append(roc_auc_score)
    evaluations['MSE'].append(sklearn.metrics.mean_absolute_error(np.array(test[y]), np.array(predictions['predict'])))
    
    for dataset in range(3):
        test_1 = test[test['dataset']==dataset]
        print(test_1.shape)
        test_1=h2o.H2OFrame(test_1)
        predictions_1 = model.predict(test_1)
        test_1= h2o.as_list(test_1, use_pandas=True)
        predictions_1= h2o.as_list(predictions_1, use_pandas=True)
        spearman_rho,spearman_p_value=spearmanr(np.array(test_1[y]), np.array(predictions_1['predict']))
        print(spearman_rho,pearsonr_rho)
        evaluations['Rs_test%s'%(dataset+1)].append(spearman_rho)
        pearsonr_rho,_=pearsonr(np.array(test_1[y]), np.array(predictions_1['predict']))
        evaluations['Rp_test%s'%(dataset+1)].append(pearsonr_rho)
        test_clustered=kmeans.predict(np.array(test_1['log2FC']).reshape(-1,1)) ## kmeans was trained with all samples
        if list(kmeans.cluster_centers_).index(min(kmeans.cluster_centers_))==0: ## define cluster with smaller value as class 1 
            test_clustered=test_clustered
        else:
            test_clustered=[1 if x==0 else 0 for x in test_clustered] 
        fpr,tpr,thresholds=sklearn.metrics.roc_curve(test_clustered,predictions_1['predict'])  
        roc_auc_score=sklearn.metrics.auc(fpr,tpr)
        evaluations['AUC_test%s'%(dataset+1)].append(roc_auc_score)
        evaluations['MSE_test%s'%(dataset+1)].append(sklearn.metrics.mean_absolute_error(np.array(test_1[y]), np.array(predictions_1['predict'])))
            
    
evaluations=pandas.DataFrame.from_dict(evaluations)
evaluations.to_csv(path+'/iteration_scores.csv',sep='\t',index=True)

def Plotting(y,predictions,kmeans,kmeans_train,name):
    # transfer to a classification problem
    y_test_clustered=kmeans.predict(y.reshape(-1,1)) ## kmeans was trained with all samples, extract the first column (second one is useless)
    predictions_clustered=kmeans_train.predict(predictions.reshape(-1, 1))
    if list(kmeans.cluster_centers_).index(min(kmeans.cluster_centers_))==0: ## define cluster with smaller value as class 1 
        y_test_clustered=y_test_clustered
    else:
        y_test_clustered=[1 if x==0 else 0 for x in y_test_clustered] 
    fpr,tpr,_=sklearn.metrics.roc_curve(y_test_clustered,predictions)
    roc_auc_score=sklearn.metrics.auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
              lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(path+'/'+name+ '_ROC.png',dpi=300)
    plt.close()
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
    plt.savefig(path+'/'+name+'_classes_scatterplot.png',dpi=300)
    plt.close()
    return fpr,tpr,roc_auc_score
kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X['log2FC']).reshape(-1,1))
guide_train, guide_test = sklearn.model_selection.train_test_split(range(len(guide_sequence_set)), test_size=0.2,random_state=np.random.seed(111))  
X_train = X_df[X_df['guideid'].isin(guide_train)]
train = X_train.drop('guideid',1)

X_test = X_df[X_df['guideid'].isin(guide_test)]
test = X_test.drop('guideid',1)

train=h2o.H2OFrame(train)
test=h2o.H2OFrame(test)
# X=h2o.H2OFrame(X)
# splits = X.split_frame(ratios = [0.8], seed = np.random.seed(111))
# train = splits[0]
# test = splits[1]

model.train(x = features, y = y, training_frame = train)
predictions = model.predict(test)
test= h2o.as_list(test, use_pandas=True)
predictions= h2o.as_list(predictions, use_pandas=True)
spearman_rho,spearman_p_value=spearmanr(np.array(test[y]), np.array(predictions['predict']))
logging.info("Spearman corelation of combined test: {0}".format(spearman_rho))
pearson_rho,_=pearsonr(np.array(test[y]), np.array(predictions['predict']))
logging.info("Pearson corelation of combined test: {0}".format(pearson_rho))
print(spearman_rho,pearson_rho)

fpr,tpr,roc_auc_score=Plotting(np.array(test[y]),np.array(predictions['predict']),kmeans,kmeans,'combined_test')
plt.figure()
plt.plot(fpr, tpr,lw=2, label='ROC curve (area = %0.2f) of combined test' % roc_auc_score)
for dataset in range(3):
    test_1 = test[test['dataset']==dataset]
    kmeans_test=KMeans(n_clusters=2, random_state=0).fit(np.array(test_1['log2FC']).reshape(-1,1))
    test_1=h2o.H2OFrame(test_1)
    predictions_1 = model.predict(test_1)
    test_1= h2o.as_list(test_1, use_pandas=True)
    predictions_1= h2o.as_list(predictions_1, use_pandas=True)
    logging.info("Spearman corelation of Test dataset {0}: {1}".format(dataset+1,spearmanr(np.array(test_1[y]), np.array(predictions_1['predict']))[0]))
    logging.info("Pearson corelation of Test dataset {0}: {1}".format(dataset+1,pearsonr(np.array(test_1[y]), np.array(predictions_1['predict']))[0]))
    fpr,tpr,roc_auc_score=Plotting(np.array(test_1[y]),np.array(predictions_1['predict']),kmeans_test,kmeans,'test_dataset%s'%(dataset+1))
    plt.plot(fpr, tpr,lw=2, label='ROC curve (area = {0}) of dataset {1} test'.format(round(roc_auc_score,2),dataset+1))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
# plt.show()
plt.savefig(path+"/ROC.png",dpi=400)
plt.close()



#SHAP values
X=h2o.H2OFrame(X)
contributions = model.predict_contributions(X)
contributions = h2o.as_list(contributions, use_pandas=True)
cols=contributions.columns.values.tolist()
shap_values = np.array(contributions[cols[:-1]])
X= h2o.as_list(X, use_pandas=True)
X=X[cols[:-1]]
import shap
shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
plt.subplots_adjust(left=0.4, top=0.95,bottom=0.2)
plt.xticks(fontsize='medium')
# plt.show()
plt.savefig(path+"/model_shap_value_bar.png",dpi=400)
plt.close()
for i in [10,15,30]:
    shap.summary_plot(shap_values, X,show=False,max_display=i,alpha=0.05)
    plt.subplots_adjust(left=0.4, top=0.95,bottom=0.2)
    plt.yticks(fontsize='small')
    plt.xticks(fontsize='small')
    plt.savefig(path+"/_model_shap_value_top%s.png"%(i),dpi=400)
    plt.close()
#%%
