#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:35:49 2020

@author: yanying
"""
#%%
import sys
sys.path.append("/vol/projects/yyu/CRISPRi-ge/lib/python3.7/site-packages/numpy")
import pandas
from merf import MERF
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import spearmanr,pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import os
import itertools
import logging
import pickle
import statistics
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['figure.dpi'] = 300
import time
start=time.time()

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
def DataFrame_input(df):
    ###keep guides for essential genes
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==1)] #
    df=df.dropna()
    for dataset in range(len(set(df['dataset']))):
        dataset_df=df[df['dataset']==dataset]
        for i in list(set(dataset_df['geneid'])):
            gene_df=dataset_df[dataset_df['geneid']==i]
            median=statistics.median(gene_df['log2FC'])
            for j in gene_df.index:
                df.at[j,'median']=median
                df.at[j,'std']=np.std(gene_df['log2FC'])
    if 'CAI' in choice:
        cai=pandas.read_csv('/home/yanying/projects/crispri/seq/NC_000913.3_CAI_values.csv',sep='\t',index_col=0)
    for i in list(set(list(df['geneid']))):
        df_gene=df[df['geneid']==i]
        for j in df_gene.index:
            df.at[j,'nr_guides']=df_gene.shape[0]
    df=df[df['nr_guides']>=5]
    ### one hot encoded sequence features
    PAM_encoded=[]
    sequence_encoded=[]
    dinucleotide_encoded=[]
    guide_sequence_set=list(dict.fromkeys(df['sequence']))
    df['guideid']=[0]*df.shape[0]
    clusters=[str(i)+"_"+str(j) for i,j in zip(list(df['geneid']),list(df['dataset']))] #
    for i in df.index:
        PAM_encoded.append(self_encode(df['PAM'][i]))
        sequence_encoded.append(self_encode(df['sequence'][i]))   
        dinucleotide_encoded.append(dinucleotide(df['sequence_30nt'][i]))
        if 'CAI' in choice:
            try:
                df.at[i,'CAI']=float(cai['CAI'][df['geneid'][i]])
            except KeyError:
                df.at[i,'CAI']=0
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        df.at[i,'guideid']=guide_sequence_set.index(df['sequence'][i])
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
    if split=='guide' or split=='guide_dropdistance':
        guideids=np.array(list(df['guideid']))
    elif split=='gene':
        guideids=np.array(list(df['geneid']))
    else:
        print('Unexpected split method...')
        sys.exit()
    medians=np.array(df['median'])
    cols=np.array(df['dataset'])
    nr_guides=np.array(df['nr_guides'])
    stds=np.array(df['std'])
    #drop features
    y=np.array(df['log2FC'],dtype=float)
    drop_features=['std','nr_guides','median','guideid','log2FC',"intergenic","No.","genename","coding_strand",
                   "gene_biotype","gene_strand","gene_5","gene_3","genome_pos_5_end","genome_pos_3_end","guide_strand",
                   'sequence','PAM','sequence_30nt','gene_essentiality']
    if 'CAI' in choice:
        drop_features=drop_features+['dataset','geneid',"distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_expression_min","gene_expression_max"]
    if split=='gene':
        drop_features.append("geneid")
    elif split=='guide_dropdistance':
        drop_features+=["distance_start_codon","distance_start_codon_perc"]
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    
    X=df.copy()
    gene_fea=['overlapping','dataset','geneid',"gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max",'CAI']#
    headers=list(X.columns.values)
    gene_features=[item for item in gene_fea if item in headers]
    if ( split=='guide' or split=='guide_dropdistance') and  "CAI" not in choice:
        gene_features.remove("geneid")
    if  "CAI" not in choice:
        gene_features.remove("dataset")
    X_gene=X[gene_features]
    # #scale
    
    guide_features=[item for item in headers if item not in gene_fea]
    X_guide=np.c_[X[guide_features],sequence_encoded,PAM_encoded,dinucleotide_encoded] #
    ###add one-hot encoded sequence features to headers
    nts=['A','T','C','G']
    sequence_features=list()
    for i in range(sequence_len):
        for j in range(len(nts)):
            guide_features.append('sequence_%s_%s'%(i+1,nts[j])) #,nts[j]
            sequence_features.append('sequence_%s_%s'%(i+1,nts[j]))
    for i in range(PAM_len):
        for j in range(len(nts)):
            guide_features.append('PAM_%s_%s'%(i+1,nts[j])) #
            sequence_features.append('PAM_%s_%s'%(i+1,nts[j]))
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    for i in range(dinucleotide_len-1):
        for dint in dinucleotides:
            guide_features.append(dint+str(i+1)+str(i+2))
            sequence_features.append(dint+str(i+1)+str(i+2))
    X_guide=pandas.DataFrame(data=X_guide,columns=guide_features)
    logging.info('Guide features: %s'%",".join(guide_features))
    return X_gene,X_guide, y, gene_features,guide_features,guide_sequence_set,guideids,clusters,medians,cols,sequence_features,stds,nr_guides


import argparse
import sys
import textwrap
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s [options]',formatter_class=argparse.RawDescriptionHelpFormatter,description=textwrap.dedent('''\
                  Example: python machine_learning_MERF.py 
                  '''))
parser.add_argument("-training", type=str, default=None, help="Dataset for training. Count starts from 0. If None,then all input datasets")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-c", "--choice", default="", help="If train on simplified random-effect model with CAI values, -c CAI. default: None")
parser.add_argument("-s", "--split", default='guide', help="default: guide")
args = parser.parse_args()
training_sets=args.training
split=args.split
if training_sets != None:
    if ',' in training_sets:
        training_sets=[int(i) for i in training_sets.split(",")]
    else:
        training_sets=[int(training_sets)]
    
else:
    training_sets=list(range(3))
path=args.output
choice=args.choice
    
if os.path.isdir(path)==False:
    os.mkdir(path)
logging_file= path+"/Log.txt"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)

#data fusion
datasets=['../0_Datasets/E75_Rousset.csv','../0_Datasets/E18_Cui.csv','../0_Datasets/Wang_dataset.csv']
labels= ['E75 Rousset','E18 Cui','Wang'] #
rousset=pandas.read_csv(datasets[0],sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
rousset['dataset']=[0]*rousset.shape[0]
rousset = rousset.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
rousset18=pandas.read_csv(datasets[1],sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
rousset18['dataset']=[1]*rousset18.shape[0]
rousset18 = rousset18.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
wang=pandas.read_csv(datasets[2],sep="\t",dtype={'log2FC':float,'gene_length':int, 'gene_GC_content':float,'gene_essentiality':int, 'guide_GC_content':float,'intergenic':int, 'distance_start_codon':int,'distance_start_codon_perc':float, 'distance_operon':int, 'operon_downstream_genes':int, 'coding_strand':int, 'homopolymers':int, 'MFE_hybrid_full':float, 'MFE_hybrid_seed':float, 'MFE_homodimer_guide':float, 'MFE_monomer_guide':float, 'off_target_90_100':int, 'off_target_80_90':int, 'off_target_70_80':int, 'off_target_60_70':int})
wang['dataset']=[2]*wang.shape[0]
wang = wang.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
combined = rousset.append(rousset18,ignore_index=True)
combined = combined.append(wang,ignore_index=True)
combined = combined.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
X_gene,X_guide, y, gene_features,guide_features,guide_sequence_set,guideids,clusters,medians,datasets,sequence_features,stds,nr_guides=DataFrame_input(combined)

open(path + '/Output.txt','a').write("Training sets: %s\n" % training_sets)
open(path + '/Output.txt','a').write("Number of guide features: %s\n" % len(guide_features))
open(path + '/Output.txt','a').write("Guide features: %s\n" % (guide_features))
open(path + '/Output.txt','a').write("Number of gene features: %s\n" % len(gene_features))
open(path + '/Output.txt','a').write("Gene features: %s\n" % (gene_features))
open(path + '/Output.txt','a').write("Number of clusters: %s\n" % len(set(clusters)))
open(path + '/Output.txt','a').write("Done processing input: %s s"%round(time.time()-start,3))
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
estimator=RandomForestRegressor(bootstrap=True, criterion='friedman_mse', max_depth=None, #origin & test
                        max_features=0.22442857329791677, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, 
                        min_samples_leaf=18, min_samples_split=16,
                        min_weight_fraction_leaf=0.0, n_estimators=512, n_jobs=1,
                        verbose=0, warm_start=False,random_state = np.random.seed(111))
guide_features=X_guide.columns.values.tolist()
X_df=pandas.DataFrame(data=np.c_[X_gene,X_guide,y,clusters,guideids,medians,datasets,stds,nr_guides],columns=gene_features+guide_features+['log2FC','geneid','guideid','median','dataset','std','nr_guides'])
X_df = X_df.loc[:,~X_df.columns.duplicated()]
guideid_set=list(set(guideids))
open(path + '/Output.txt','a').write("Estimator:"+str(estimator)+"\n")
dtypes=dict()
for feature in X_df.columns.values:
    if feature != 'geneid':
        dtypes.update({feature:float})
X_df=X_df.astype(dtypes)
'''
evaluations=defaultdict(list)
iteration_predictions=defaultdict(list)
kf=sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=np.random.seed(111))
iteration=0
for train_index, test_index in kf.split(guideid_set):
    guide_train = np.array(guideid_set)[train_index]
    test_index = np.array(guideid_set)[test_index]
    guide_train, guide_val = sklearn.model_selection.train_test_split(guide_train, test_size=0.2,random_state=np.random.seed(111))  
   
    train = X_df[X_df['guideid'].isin(guide_train)]
    train=train[train['dataset'].isin(training_sets)]
    y_train=train['log2FC']
    X_train=train[guide_features]
    Z_train=train[gene_features]
    clusters_train=train['geneid']
    
    val = X_df[X_df['guideid'].isin(guide_val)]
    val=val[val['dataset'].isin(training_sets)]
    y_val=val['log2FC']
    X_val=val[guide_features]
    Z_val=val[gene_features]
    clusters_val=val['geneid']
    
    ### keep the same test from 3 datasets
    test = X_df[X_df['guideid'].isin(test_index)]
    y_test=test['log2FC']
    X_test=test[guide_features]
    Z_test=test[gene_features]
    clusters_test=test['geneid']
    
    
    filename = path+'/saved_model/Merf_model_%s.sav'%iteration
    if os.path.isfile(filename)==True:
        mrf_lgbm=pickle.load(open(filename,'rb'))
    else:
        mrf_lgbm = MERF(estimator,max_iterations=15)
        mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)
        pickle.dump(mrf_lgbm, open(filename, 'wb'))
    iteration+=1
    test['pred'] = mrf_lgbm.predict(X_test,Z_test,clusters_test) 
    test['log2FC'] = y_test
    test['guide_pred']=mrf_lgbm.trained_fe_model.predict(X_test)
    iteration_predictions['log2FC'].append(list(y_test))
    iteration_predictions['pred'].append(list(mrf_lgbm.trained_fe_model.predict(X_test)))
    iteration_predictions['iteration'].append(iteration)
    iteration_predictions['dataset'].append(list(test['dataset']))
    iteration_predictions['geneid'].append(list(test['geneid']))
    # random effect coef
    cluster_counts = clusters_train.value_counts()
    train_re = train.groupby("geneid").mean()
    train_re=train_re.loc[np.array(mrf_lgbm.trained_b.index)]
    median_train_re=train_re['median']
    Z_train_re=np.array(train_re[gene_features])
    pred=np.sum(mrf_lgbm.trained_b * Z_train_re,axis=1)
    for i in pred.index:
        train_gene=train[train['geneid']==i]
        for j in train_gene.index:
            train.at[j,'gene_pred']=pred[i]
    if split=='guide' or split=='guide_dropdistance':        
        coef=pandas.DataFrame(mrf_lgbm.trained_b,index=mrf_lgbm.trained_b.index)
        cluster_counts = clusters_test.value_counts()
        test_re = test.groupby("geneid").mean()
        test_re=test_re.loc[np.array(cluster_counts.index)]
        median_test_re=test_re['median']
        coef=coef.loc[np.array(cluster_counts.index)]
        Z_test_re=np.array(test_re[gene_features])
        pred_test=np.sum(coef * Z_test_re,axis=1)
        for i in pred_test.index:
            test_gene=test[test['geneid']==i]
            for j in test_gene.index:
                test.at[j,'gene_pred']=pred_test[i]

    # test in 3 datasets
    predictions = mrf_lgbm.predict(X_test, Z_test, clusters_test)        
    spearman_rho,_=spearmanr(np.array(y_test), np.array(predictions))
    pearsonr_rho,_=pearsonr(np.array(y_test), np.array(predictions))
    evaluations['Rs'].append(spearman_rho)
    if split=='guide' or split=='guide_dropdistance':
        evaluations['Rs_guide'].append(spearmanr(np.array(y_test-test['gene_pred']), mrf_lgbm.trained_fe_model.predict(X_test))[0])
        
        evaluations['Rs_median'].append(spearmanr(median_test_re,pred_test)[0])
    for dataset in range(len(set(datasets))):
        test_1 = test[test['dataset']==dataset]
        y_test=test_1['log2FC']
        X_test=test_1[guide_features]
        Z_test=test_1[gene_features]
        clusters_test=test_1['geneid']
        predictions_1 = mrf_lgbm.predict(X_test, Z_test, clusters_test)        
        spearman_rho,spearman_p_value=spearmanr(np.array(y_test), np.array(predictions_1))
        evaluations['Rs_test%s'%(dataset+1)].append(spearman_rho)
        if split=='guide' or split=='guide_dropdistance':
            evaluations['Rs_guide_test%s'%(dataset+1)].append(spearmanr(np.array(y_test-test_1['gene_pred']), mrf_lgbm.trained_fe_model.predict(X_test))[0])
            test_1_group=test_1.groupby("geneid").mean()
            evaluations['Rs_median_test%s'%(dataset+1)].append(spearmanr(test_1_group['median'], test_1_group['gene_pred'])[0])
    
evaluations=pandas.DataFrame.from_dict(evaluations)
evaluations.to_csv(path+'/iteration_scores.csv',sep='\t',index=True)
iteration_predictions=pandas.DataFrame.from_dict(iteration_predictions)
iteration_predictions.to_csv(path+'/iteration_predictions.csv',sep='\t',index=False)
'''
if os.path.isdir(path+'/saved_model')==False:  
    os.mkdir(path+'/saved_model')
filename = path+'/saved_model/CRISPRi_headers.sav'
pickle.dump(guide_features, open(filename, 'wb'))
filename = path+'/saved_model/Merf_model.sav'
mrf_lgbm = MERF(estimator,max_iterations=15)
mrf_lgbm.fit(X_df[X_df['dataset'].isin(training_sets)][guide_features], X_df[X_df['dataset'].isin(training_sets)][gene_features], X_df[X_df['dataset'].isin(training_sets)]['geneid'], X_df[X_df['dataset'].isin(training_sets)]['log2FC'])
pickle.dump(mrf_lgbm, open(filename, 'wb'))
filename = path+'/saved_model/CRISPRi_model.sav'
pickle.dump(mrf_lgbm.trained_fe_model, open(filename, 'wb')) 
filename = path+'/saved_model/random_coef.sav'
pickle.dump(mrf_lgbm.trained_b, open(filename, 'wb'))
#SHAP values for fixed-effect model
import shap
treexplainer = shap.TreeExplainer(mrf_lgbm.trained_fe_model)
shap_values = treexplainer.shap_values(X_df[X_df['dataset'].isin(training_sets)][guide_features],check_additivity=False)
values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':guide_features})
values.to_csv(path+"/shap_value_mean.csv",index=False,sep='\t')
open(path + '/Output.txt','a').write("Done calculating SHAP values: %s s"%round(time.time()-start,3))
shap.summary_plot(shap_values, X_df[X_df['dataset'].isin(training_sets)][guide_features], plot_type="bar",show=False,color_bar=True,max_display=10)
plt.subplots_adjust(left=0.35, top=0.95)
plt.title("Test")
plt.savefig(path+"/shap_value_bar_test.svg",dpi=400)
plt.close()

for i in [10,15,30]:
    shap.summary_plot(shap_values, X_df[X_df['dataset'].isin(training_sets)][guide_features],show=False,max_display=i,alpha=0.05)
    plt.subplots_adjust(left=0.4, top=0.95,bottom=0.1)
    plt.yticks(fontsize='medium')
    plt.xticks(fontsize='medium')
    plt.savefig(path+"/shap_value_top%s.svg"%(i),dpi=400)
    plt.close()    

shap_interaction_values=treexplainer.shap_interaction_values(X_df[X_df['dataset'].isin(training_sets)][guide_features])
pickle.dump(shap_interaction_values, open(path+"/shap_interaction_values_all.pkl", 'wb'))
open(path + '/Output.txt','a').write("Done saving SHAP interaction values: %s s"%round(time.time()-start,3))

tmp = np.abs(shap_interaction_values).mean(0)
tmp_1d=defaultdict(list)
for i in range(len(guide_features)):
    for j in range(i+1,len(guide_features)):
        tmp_1d['feature1'].append(guide_features[i])
        tmp_1d['feature2'].append(guide_features[j])
        tmp_1d['mean_absolute_interaction_value'].append(tmp[i,j])
tmp_1d=pandas.DataFrame.from_dict(tmp_1d)
tmp_1d=tmp_1d.sort_values(by='mean_absolute_interaction_value',ascending=False).reset_index(drop=True)
tmp_1d=tmp_1d.astype({'mean_absolute_interaction_value':float})
numeric_features=['distance_start_codon','distance_start_codon_perc','homopolymers','guide_GC_content',
                   'MFE_hybrid_full', 'MFE_hybrid_seed', 'MFE_homodimer_guide', 'MFE_monomer_guide']
p=defaultdict(list)
X_index=X_df[X_df['dataset'].isin(training_sets)][guide_features].reset_index(drop=True)
coms=[[0,0],[1,0],[0,1],[1,1]]
marker=['-','+']
tmp_1d['global interaction rank']=tmp_1d.index+1
for rank in tmp_1d.index[:5000]:
# main_feature='GC2425'
    pair=[tmp_1d['feature1'][rank],tmp_1d['feature2'][rank]]
    for i in coms:
        if pair[0] not in numeric_features and  pair[1] not in numeric_features:
            sample_df=X_index[(X_index[pair[0]]==i[0])&(X_index[pair[1]]==i[1])]     
        elif pair[0] in numeric_features and pair[1] not in numeric_features:
            if i[0]==1:
                sample_df=X_index[(X_index[pair[0]]>=np.quantile(X_index[pair[0]],0.5))&(X_index[pair[1]]==i[1])]    
            elif i[0]==0:
                sample_df=X_index[(X_index[pair[0]]<np.quantile(X_index[pair[0]],0.5))&(X_index[pair[1]]==i[1])]    
        elif pair[0] not in numeric_features and pair[1] in numeric_features:
            if i[1]==1:
                sample_df=X_index[(X_index[pair[1]]>=np.quantile(X_index[pair[1]],0.5))&(X_index[pair[0]]==i[0])]    
            elif i[1]==0:
                sample_df=X_index[(X_index[pair[1]]<np.quantile(X_index[pair[1]],0.5))&(X_index[pair[0]]==i[0])]    
        elif pair[0] in numeric_features and pair[1] in numeric_features:
            if i[0]==1 and i[1]==1:
                sample_df=X_index[(X_index[pair[0]]>=np.quantile(X_index[pair[0]],0.5))&(X_index[pair[1]]>=np.quantile(X_index[pair[1]],0.5))]    
            elif i[0]==0 and i[1]==1:
                sample_df=X_index[(X_index[pair[0]]<np.quantile(X_index[pair[0]],0.5))&(X_index[pair[1]]>=np.quantile(X_index[pair[1]],0.5))]    
            elif i[0]==1 and i[1]==0:
                sample_df=X_index[(X_index[pair[0]]>=np.quantile(X_index[pair[0]],0.5))&(X_index[pair[1]]<np.quantile(X_index[pair[1]],0.5))]    
            elif i[0]==0 and i[1]==0:
                sample_df=X_index[(X_index[pair[0]]<np.quantile(X_index[pair[0]],0.5))&(X_index[pair[1]]<np.quantile(X_index[pair[1]],0.5))]    
        if coms.index(i)==1:
            f1=np.median(shap_values[sample_df.index,guide_features.index(pair[0])])
            f2_m=np.median(shap_values[sample_df.index,guide_features.index(pair[1])])
        if coms.index(i)==2:
            f2=np.median(shap_values[sample_df.index,guide_features.index(pair[1])])
            f1_m=np.median(shap_values[sample_df.index,guide_features.index(pair[0])])
   
        tmp_1d.at[rank,marker[i[0]]+" / "+marker[i[1]]]=np.median(shap_values[sample_df.index,guide_features.index(pair[0])]+shap_values[sample_df.index,guide_features.index(pair[1])])
    tmp_1d.at[rank,'expected_+/+']=f1+f2
    tmp_1d.at[rank,'expected_-/-']=f1_m+f2_m
tmp_1d.to_csv(path+"/interaction_pair_sumSHAPvalues.csv",index=False,sep='\t')

marker=['-','+']
pairs=[['sequence_20_C','GC2728'],['sequence_20_G','GC2728'],['sequence_20_A','GG2728'],['GG2728','TG2526']]
labels={'GC2728':'+1 C','sequence_20_G':'20 G','sequence_20_C':'20 C','GG2728':'+1 G','sequence_20_A':'20 A','TG2526':'P1 T'}
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
        p['pattern']+=[marker[i[0]]+" / "+marker[i[1]]]*sample_df.shape[0]
        p['value']+=list(shap_values[sample_df.index,guide_features.index(pair[0])]+shap_values[sample_df.index,guide_features.index(pair[1])])
    p=pandas.DataFrame.from_dict(p)
    plot=p.dropna()
    plot=p[p['pattern']!='- / -']
    plt.figure(figsize=(5,4))
    ax=sns.boxplot(data=plot,x='pattern',y='value',order=['+ / -','- / +','+ / +'],color='lightgrey')
    ax.axhline(f1+f2,color='r',xmin=0.7,xmax=0.98)
    plt.xticks(rotation=0,fontsize='large')
    plt.xlabel("")
    plt.title(labels[pair[0]]+' / '+labels[pair[1]],fontsize='large')
    plt.ylabel("sum SHAP value",fontsize='large')
    plt.subplots_adjust(left=0.2)
    plt.savefig(path+"/shap_dependence_plot_%s_%s.svg"%(pair[0],pair[1]),dpi=400)
    plt.close()    


###
guide_train, guide_test = sklearn.model_selection.train_test_split(guideid_set, test_size=0.2,random_state=np.random.seed(111))  
guide_train, guide_val = sklearn.model_selection.train_test_split(guide_train, test_size=0.2,random_state=np.random.seed(111))  
train = X_df[X_df['guideid'].isin(guide_train)]
train=train[train['dataset'].isin(training_sets)]
y_train=train['log2FC']
X_train=train[guide_features]
Z_train=train[gene_features]
clusters_train=train['geneid']

val = X_df[X_df['guideid'].isin(guide_val)]
val=val[val['dataset'].isin(training_sets)]
y_val=val['log2FC']
X_val=val[guide_features]
Z_val=val[gene_features]
clusters_val=val['geneid']

test = X_df[X_df['guideid'].isin(guide_test)]
y_test=test['log2FC']
X_test=test[guide_features]
Z_test=test[gene_features]
clusters_test=test['geneid']
print("Training model...")
mrf_lgbm = MERF(estimator,max_iterations=15)
mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)
pickle.dump(mrf_lgbm.trained_fe_model, open(path+"/trained_fe_model.pkl", 'wb'))
predictions = mrf_lgbm.predict(X_test, Z_test, clusters_test)  
spearman_rho,spearman_p_value=spearmanr(np.array(y_test), np.array(predictions))
open(path + '/Output.txt','a').write("Spearman corelation of combined test: {0}\n".format(spearman_rho))
pearson_rho,_=pearsonr(np.array(y_test), np.array(predictions))
open(path + '/Output.txt','a').write("Pearson corelation of combined test: {0}\n".format(pearson_rho))
print(spearman_rho,pearson_rho)
print("Done training: %s s"%round(time.time()-start,3))
#random effect model predictions
cluster_counts = clusters_train.value_counts()
coef=pandas.DataFrame(mrf_lgbm.trained_b,index=mrf_lgbm.trained_b.index)
train_re = train.groupby("geneid").mean()
train_re=train_re.loc[np.array(mrf_lgbm.trained_b.index)]
median_train_re=train_re['median']
Z_train_re=np.array(train_re[gene_features])
pred=np.sum(mrf_lgbm.trained_b * Z_train_re,axis=1)
for i in pred.index:
    train_gene=train[train['geneid']==i]
    for j in train_gene.index:
        train.at[j,'gene_pred']=pred[i]
train_re['gene_pred']=pred
train_re['random_med'] = abs(train_re['median'] - train_re['gene_pred'])
open(path + '/Output.txt','a').write("Spearman corelation between random effects and median  (train): {0}\n".format(spearmanr(median_train_re,pred)[0]))



###feature importance for random effect model        
coef.columns=gene_features
plot=defaultdict(list)
for i in coef.index:
    for feature in gene_features:
        plot['dataset'].append(['E75 Rousset', 'E18 Cui', 'Wang'][int(i.split("_")[1])])
        plot['value'].append(coef[feature][i])
        plot['feature'].append(feature)

sns.set_style("whitegrid")
sns.boxplot(data=plot,x='value',y='feature',hue='dataset',orient='h',showfliers=False,palette='Blues')
plt.xlabel("Coefficient")
plt.xticks(fontsize='small')
plt.subplots_adjust(left=0.35)
plt.savefig(path+'/Coef_random_effect_model.svg',dpi=400)
# plt.show()
plt.close()

effect=pandas.DataFrame(data=coef * Z_train_re,columns=gene_features)
plot=defaultdict(list)
for i in effect.index:
    for feature in gene_features:
        plot['dataset'].append(['E75 Rousset', 'E18 Cui', 'Wang'][int(i.split("_")[1])])
        plot['value'].append(effect[feature][i])
        plot['feature'].append(feature)

sns.boxplot(data=plot,x='value',y='feature',hue='dataset',orient='h',showfliers=False,palette='Blues')
plt.xlabel("Feature effects")
plt.xticks(fontsize='small')
plt.subplots_adjust(left=0.35)
plt.savefig(path+'/feature_effects_random_effect_model.svg',dpi=400)
# plt.show()
plt.close()

if split=='guide' or split=='guide_dropdistance':
    cluster_counts = clusters_test.value_counts()
    test_re = test.groupby("geneid").mean()
    test_re=test_re.loc[np.array(cluster_counts.index)]
    median_test_re=test_re['median']
    coef=coef.loc[np.array(cluster_counts.index)]
    Z_test_re=np.array(test_re[gene_features])
    pred_test=np.sum(coef * Z_test_re,axis=1)
    for i in pred_test.index:
        test_gene=test[test['geneid']==i]
        for j in test_gene.index:
            test.at[j,'gene_pred']=pred_test[i]
    test_re['gene_pred']=pred_test
    test_re['random_med'] = abs(test_re['median'] - test_re['gene_pred'])
    open(path + '/Output.txt','a').write("Spearman corelation between random effects and median (test): {0}\n".format(spearmanr(median_test_re,pred_test)[0]))



#scatter plot of whole MERF model
test['pred'] = mrf_lgbm.predict(X_test,Z_test,clusters_test) 
markers=['x','D','s']
sns.set_palette('Set2',len(set(training_sets)))
plt.figure()
for data in training_sets:
    median_dataset=test[test['dataset']==data]
    ax=sns.scatterplot(median_dataset['log2FC'],median_dataset['pred'],label=labels[data],marker=markers[data],alpha=0.5,edgecolors='white')
    plt.text(0.65,0.10-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(median_dataset['log2FC'],median_dataset['pred'])[0],3)),transform=ax.transAxes,fontsize='x-small')
plt.legend()
plt.xlabel("Measured log2FC")
plt.ylabel("Predictions")   
plt.savefig(path+"/merf.png",dpi=400)
# plt.show()
plt.close()


#scatter plots for random effect model and fixed-effect model (both test and train)
labels= ['E75 Rousset','E18 Cui','Wang']
sns.set_style("whitegrid")
plt.figure()
for data in training_sets:
    median_dataset=train_re[train_re['dataset']==data]
    ax=sns.scatterplot(median_dataset['median'],median_dataset['gene_pred'],label=labels[data],alpha=0.5,edgecolors='white')
    plt.text(0.55,0.15-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(median_dataset['median'],median_dataset['gene_pred'])[0],3)),transform=ax.transAxes,fontsize='small')
plt.legend()
plt.xlabel("Median")
plt.ylabel("Predicted Random effects")   
plt.title('Train')
plt.savefig(path+"/random_median_train.png",dpi=400)
# plt.show()
plt.close()

plt.figure()
for data in training_sets:
    train_dataset=train[train['dataset']==data]
    ax=sns.scatterplot(train_dataset['log2FC']-train_dataset['gene_pred'],mrf_lgbm.trained_fe_model.predict(train_dataset[guide_features]),label=labels[data],alpha=0.5)
    plt.text(0.55,0.15-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(train_dataset['log2FC']-train_dataset['gene_pred'],mrf_lgbm.trained_fe_model.predict(train_dataset[guide_features]))[0],3)),transform=ax.transAxes,fontsize='small')
plt.legend()
plt.xlabel("Residual of log2FC")
plt.ylabel("Predicted Fixed effects")
plt.title('Train')
plt.savefig(path+"/fixed_train.png",dpi=400)
# plt.show()
plt.close()

if split=='guide' or split=='guide_dropdistance':
    labels= ['E75 Rousset','E18 Cui','Wang']
    markers=['x','D','s']
    plt.figure()
    for data in training_sets:
        median_dataset=test_re[test_re['dataset']==data]
        ax=sns.distplot(median_dataset['gene_pred']-median_dataset['median'],label=labels[data])
        print(labels[data],np.mean(median_dataset['gene_pred']-median_dataset['median']),np.std(median_dataset['gene_pred']-median_dataset['median']))
    plt.legend()
    plt.xlabel("Predicted Random effects - Median log2FC of gRNAs for each gene")
    # plt.title('Test')
    plt.savefig(path+"/random_minus_median_test.png",dpi=400)
    # plt.show()
    plt.close()

    plt.figure()
    for data in training_sets:
        median_dataset=test_re[test_re['dataset']==data]
        ax=sns.scatterplot(median_dataset['median'],median_dataset['gene_pred'],label=labels[data],marker=markers[data],alpha=0.5,edgecolors='white')
        plt.text(0.65,0.1-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(median_dataset['median'],median_dataset['gene_pred'])[0],3)),transform=ax.transAxes,fontsize='x-small')
    plt.legend()
    plt.xlabel("Median log2FC of gRNAs for each gene")
    plt.ylabel("Predicted Random effects")
    # plt.title('Test')
    plt.savefig(path+"/random_median_test.png",dpi=400)
    # plt.show()
    plt.close()

    plt.figure()
    for data in training_sets:
        test_dataset=test[test['dataset']==data]
        ax=sns.scatterplot(test_dataset['log2FC']-test_dataset['gene_pred'],mrf_lgbm.trained_fe_model.predict(test_dataset[guide_features]),marker=markers[data],label=labels[data],alpha=0.5)
        plt.text(0.65,0.1-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(test_dataset['log2FC']-test_dataset['gene_pred'],mrf_lgbm.trained_fe_model.predict(test_dataset[guide_features]))[0],3)),transform=ax.transAxes,fontsize='x-small')
    plt.legend(loc='upper left')
    plt.xlabel("Residual of log2FC")
    plt.ylabel("Predicted Fixed effects")
    # plt.title('Test')
    plt.savefig(path+"/fixed_test.png",dpi=400)
    # plt.show()
    plt.close()
    
    
  
   








