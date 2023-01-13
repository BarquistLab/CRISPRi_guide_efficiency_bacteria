#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:35:49 2020

@author: yanying
"""
import time
start_time=time.time()

import pandas
import sklearn.model_selection
import numpy as np
from scipy.stats import spearmanr,pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from collections import defaultdict
import os
import itertools
import logging
import argparse
import sys
import warnings
import shap
warnings.filterwarnings('ignore')
mpl.rcParams['figure.dpi'] = 300

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to optimize models with fused datasets using H2O and evaluate the best performed model with 10 fold cross-validation.
                  
Example: python h2o_crispri.py -o test -training 0,1,2
                  """)
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
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

args = parser.parse_args()
training_sets=args.training
if ',' in training_sets:
    training_sets=[int(i) for i in training_sets.split(",")]
else:
    training_sets=[int(training_sets)]
folds=args.folds
test_size=args.test_size
output_file_name = args.output
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
import h2o
from h2o.automl import H2OAutoML
h2o.init()
h2o.remove_all()
def self_encode(sequence):#one-hot encoding for single nucleotide features
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

# h2o.shutdown()
h2o.no_progress()
logging_file= output_file_name+"/log.txt"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)
open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
#data fusion
datasets=['../0_Datasets/E75_Rousset.csv','../0_Datasets/E18_Cui.csv','../0_Datasets/Wang_dataset.csv']
training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}
# load 3 datesets
rousset=pandas.read_csv(datasets[0],sep="\t")
rousset['dataset']=[0]*rousset.shape[0]
rousset = rousset.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
rousset18=pandas.read_csv(datasets[1],sep="\t")
rousset18['dataset']=[1]*rousset18.shape[0]
rousset18 = rousset18.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
wang=pandas.read_csv(datasets[2],sep="\t")
wang['dataset']=[2]*wang.shape[0]
wang = wang.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
combined = rousset.append(rousset18,ignore_index=True)
combined = combined.append(wang,ignore_index=True)
combined = combined.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
combined = combined[(combined['gene_essentiality']==1)&(combined['intergenic']==0)&(combined['coding_strand']==1)] #
combined = combined.dropna()
open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n"% (datasets[0],rousset.shape[0]))
open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[1],rousset18.shape[0]))
open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[2],wang.shape[0]))
open(output_file_name + '/log.txt','a').write("Training dataset: %s\n"%training_set_list[tuple(training_sets)])

        
for dataset in range(len(set(combined['dataset']))):
    dataset_df=combined[combined['dataset']==dataset]
    for i in list(set(dataset_df['geneid'])):
        gene_df=dataset_df[dataset_df['geneid']==i]
        for j in gene_df.index:
            combined.at[j,'Nr_guide']=gene_df.shape[0]
open(output_file_name + '/log.txt','a').write("Number of guides for essential genes: %s \n" % combined.shape[0])
combined=combined[combined['Nr_guide']>=5]#keep only genes with more than 5 guides from all 3 datasets
open(output_file_name + '/log.txt','a').write("Number of guides after filtering: %s \n" % combined.shape[0])
guide_sequence_set=list(dict.fromkeys(combined['sequence']))
### one hot encoded sequence features
sequence_encoded=[]
for i in combined.index:
    sequence_encoded.append(self_encode(combined['sequence_30nt'][i]))   
    combined.at[i,'guideid']=guide_sequence_set.index(combined['sequence'][i])
guideids=np.array(list(combined['guideid']))
#drop features
X=combined.drop(['geneid','guideid',"No.","genename","gene_strand","gene_5","gene_biotype","gene_3","genome_pos_5_end","genome_pos_3_end",\
                 'gene_essentiality','intergenic','guide_strand','coding_strand','PAM','sequence','sequence_30nt','Nr_guide',\
                 'off_target_90_100','off_target_80_90','off_target_70_80','off_target_60_70','spacer_self_fold','RNA_DNA_eng','DNA_DNA_opening','CRISPRoff_score'],1)
features=X.columns.values.tolist()
X=np.c_[X,sequence_encoded]
    
###add one-hot encoded sequence features to headers
nts=['A','T','C','G']
for i in range(30):
    for j in range(len(nts)):
        features.append('sequence_%s_%s'%(i+1,nts[j]))
X=pandas.DataFrame(data=X,columns=features)
logging.info("Number of features: %s" % len(features))
logging.info("Features: "+",".join(features)+"\n")
open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
# X=h2o.H2OFrame(X)
y="log2FC"
##split the combined training set into train and test
guide_train, guide_test = sklearn.model_selection.train_test_split(range(len(guide_sequence_set)), test_size=test_size,random_state=np.random.seed(111))  
X_df=pandas.DataFrame(data=np.c_[X,guideids],columns=features+['guideid'])
train = X_df[X_df['guideid'].isin(guide_train)]
train=train[train['dataset'].isin(training_sets)]
train=train.drop('guideid',1)
test = X_df[X_df['guideid'].isin(guide_test)]
test=test.drop('guideid',1)
features.remove("log2FC")
train=h2o.H2OFrame(train)
test=h2o.H2OFrame(test)
#set params for h2o automl
aml = H2OAutoML(max_runtime_secs = 0, seed = 1, project_name = "comparison", 
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
params.to_csv("%s/gene_params.csv"%output_file_name,sep='\t',index=False)

#save variable importance
varimp=aml.leader.varimp(use_pandas=True)
varimp.to_csv("%s/gene_varimp.csv"%output_file_name,sep='\t',index=False)
# aml.leader.varimp_plot()
# plt.savefig("%s/varimp_plot.png"%output_file_name,dpi=400)
# plt.show()
# plt.close()

#save performance evaluation
perf = aml.leader.model_performance(test)
logging.info("Performance of gene model:\n %s" %str(perf))

#save model
model=aml.leader
model_path=h2o.save_model(model, path=output_file_name)
open(output_file_name + '/log.txt','a').write("Estimator:"+str(model)+"\n")
#k-fold cross validation
evaluations=defaultdict(list)
kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
guideid_set=list(set(guideids))
for train_index, test_index in kf.split(guideid_set):##split the combined training set into train and test based on guideid
    train = X_df[X_df['guideid'].isin(train_index)]
    train=train[train['dataset'].isin(training_sets)]
    train=train.drop('guideid',1)
    test = X_df[X_df['guideid'].isin(test_index)]
    test=test.drop('guideid',1)
    train=h2o.H2OFrame(train)
    test=h2o.H2OFrame(test)
    model.train(x = features, y = y, training_frame = train)
    
    #satterplot
    predictions = model.predict(test)
    test= h2o.as_list(test, use_pandas=True)
    predictions= h2o.as_list(predictions, use_pandas=True)
    spearman_rho,_=spearmanr(np.array(test[y]), np.array(predictions['predict']))
    evaluations['Rs'].append(spearman_rho)
    for dataset in range(3):
        test_1 = test[test['dataset']==dataset]
        test_1=h2o.H2OFrame(test_1)
        predictions_1 = model.predict(test_1)
        test_1= h2o.as_list(test_1, use_pandas=True)
        predictions_1= h2o.as_list(predictions_1, use_pandas=True)
        spearman_rho,spearman_p_value=spearmanr(np.array(test_1[y]), np.array(predictions_1['predict']))
        evaluations['Rs_test%s'%(dataset+1)].append(spearman_rho)
    
evaluations=pandas.DataFrame.from_dict(evaluations)
evaluations.to_csv(output_file_name+'/iteration_scores.csv',sep='\t',index=True)


guide_train, guide_test = sklearn.model_selection.train_test_split(guideid_set, test_size=test_size,random_state=np.random.seed(111))  
train = X_df[X_df['guideid'].isin(guide_train)]
train=train[train['dataset'].isin(training_sets)]
train = train.drop('guideid',1)
X_test = X_df[X_df['guideid'].isin(guide_test)]
test = X_test.drop('guideid',1)

train=h2o.H2OFrame(train)
test=h2o.H2OFrame(test)
model.train(x = features, y = y, training_frame = train)
predictions = model.predict(test)
test= h2o.as_list(test, use_pandas=True)
predictions= h2o.as_list(predictions, use_pandas=True)
spearman_rho,spearman_p_value=spearmanr(np.array(test[y]), np.array(predictions['predict']))
logging.info("Spearman corelation of combined test: {0}".format(spearman_rho))

plt.figure() 
sns.set_palette("PuBu",2)
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
ax_main.scatter(np.array(test[y]),np.array(predictions['predict']),edgecolors='white',alpha=0.8)
ax_main.set(xlabel='Experimental log2FC',ylabel='Predicted log2FC')
ax_xDist.hist(np.array(test[y]),bins=70,align='mid',alpha=0.7)
ax_xDist.set(ylabel='count')
ax_xDist.tick_params(labelsize=6,pad=2)
ax_yDist.hist(np.array(predictions['predict']),bins=70,orientation='horizontal',align='mid',alpha=0.7)
ax_yDist.set(xlabel='count')
ax_yDist.tick_params(labelsize=6,pad=2)
ax_main.text(0.55,0.03,"Spearman R: {0}".format(round(spearman_rho,2)),transform=ax_main.transAxes,fontsize=10)
plt.savefig(output_file_name+'/scatterplot.svg')
plt.savefig(output_file_name+'/scatterplot.png',dpi=300)
plt.close()

#SHAP values
X=h2o.H2OFrame(X)
contributions = model.predict_contributions(X)
contributions = h2o.as_list(contributions, use_pandas=True)
cols=contributions.columns.values.tolist()
shap_values = np.array(contributions[cols[:-1]])
X= h2o.as_list(X, use_pandas=True)
X=X[cols[:-1]]

shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
plt.subplots_adjust(left=0.4, top=0.95,bottom=0.2)
plt.xticks(fontsize='medium')
plt.savefig(output_file_name+"/shap_value_bar.svg")
plt.savefig(output_file_name+"/shap_value_bar.png",dpi=400)
plt.close()
for i in [10,15,30]:
    shap.summary_plot(shap_values, X,show=False,max_display=i,alpha=0.05)
    plt.subplots_adjust(left=0.4, top=0.95,bottom=0.2)
    plt.yticks(fontsize='small')
    plt.xticks(fontsize='small')
    plt.savefig(output_file_name+"/shap_value_top%s.svg"%(i))
    plt.savefig(output_file_name+"/shap_value_top%s.png"%(i),dpi=400)
    plt.close()

