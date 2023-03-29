# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:35:49 2020

@author: yanying
"""
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
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['font.sans-serif']='Arial'
mpl.rcParams['font.size']=14
mpl.rcParams['legend.title_fontsize']=10
mpl.rcParams['legend.fontsize']=10
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['figure.figsize']=[5,5]
import time
start=time.time()

import argparse
import sys
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to separate gene and guide effects using MERF (tested version 1.0).
Using this script, one can run the individual fold in 10-fold CV separately to save the total run-time with -i. 
When a value lower than 0 is passed to -i, the script will train a model with all guides instead of 10-fold CV.

Example: python MERF_crispri_run_individual_iteration.py -training 0,1,2 -i -1
                  """)
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
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-c", "--choice", default="", help="If train on simplified random-effect model with CAI values, -c CAI; or use feature set from the Pasteur model -c pasteur. default: None")
parser.add_argument("-s", "--split", default='gene', help="train-test split stratege. gene/gene_dropdistance. guide_dropdistance: To test the models without distance associated features. default: gene")
parser.add_argument("-F", "--feature_set", default='all',type=str, help="feature set for training. all/pasteur. Pasteur: sequence features used for Pasteur model. default: all")

parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.2")
parser.add_argument("-r","--random_seed", type=int, default=111, help="random seed for train-test split, default: 111")
parser.add_argument("-m","--model", type=str, default='hyperopt', help="""
                    tree-based model for fixed-effect model, 
                    autosklearn_rf: optimized model from auto-sklearn to predict depletion; 
                    hyperopt: hyperparameter tunning using hyperopt.;
                    default: hyperopt""")
parser.add_argument("-S","--saved_cv_models", type=str, default=None, help="The folder including saved models for 10-fold CV or trained with all samples, named after format of MERF_model_N.sav (for CV, N=0-9 iteration) or Merf_model.sav (for all samples), default: None")
parser.add_argument("-i", "--tuning_iteration", default=10,type=int, help="The iteration to be run for hyperparamter tunning. default: 0")

args = parser.parse_args()
training_sets=args.training
split=args.split
feature_set=args.feature_set
folds=args.folds
test_size=args.test_size
random_seed=args.random_seed
model=args.model
saved_cv_models=args.saved_cv_models
tuning_iteration=args.tuning_iteration
if tuning_iteration >= folds:
    print("The selected iteration for tuning is out of range of the folds. The value of tuning itaration should be < folds, starting from 0. Or use any value < 0 and -m hyperopt to tune the hyperparameters for all samples")
    sys.exit()
if training_sets != None:
    if ',' in training_sets:
        training_sets=[int(i) for i in training_sets.split(",")]
    else:
        training_sets=[int(training_sets)]
    
else:
    training_sets=list(range(3))
    
output_file_name=args.output
choice=args.choice
    
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
open(output_file_name + '/log.txt','a').write(time.asctime())
open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)    
datasets=['../0_Datasets/E75_Rousset.csv','../0_Datasets/E18_Cui.csv','../0_Datasets/Wang_dataset.csv']
##for cluster
# datasets=['/vol/projects/yyu/crispri/CRISPRi-guide-efficiency-in-bacteria/0_Datasets/E75_Rousset.csv','/vol/projects/yyu/crispri/CRISPRi-guide-efficiency-in-bacteria/0_Datasets/E18_Cui.csv','/vol/projects/yyu/crispri/CRISPRi-guide-efficiency-in-bacteria/0_Datasets/Wang_dataset.csv']

training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}

logging_file= output_file_name+"/log.txt"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)


def self_encode(sequence):#one-hot encoding for single nucleotide features
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

def encode(seq):
    return np.array([[int(b==p) for b in seq] for p in ["A","T","G","C"]])
def find_target(df,before=20,after=20):
    from Bio import SeqIO
    fasta_sequences = SeqIO.parse(open("../0_Datasets/NC_000913.3.fasta"),'fasta')    
    for fasta in fasta_sequences:  # input reference genome
        reference_fasta=fasta.seq 
    extended_seq=[]
    # guides_index=list()
    for i in df.index.values:
        # if len(df['sequence'][i])!=20 or df["genome_pos_5_end"][i]<20 or df["genome_pos_3_end"][i]<20 :
            # continue
        # guides_index.append(i)
        if df["genome_pos_5_end"][i] > df["genome_pos_3_end"][i]:
            extended_seq.append(str(reference_fasta[df["genome_pos_3_end"][i]-1-after:df["genome_pos_5_end"][i]+before].reverse_complement()))
        else:
            extended_seq.append(str(reference_fasta[df["genome_pos_5_end"][i]-1-before:df["genome_pos_3_end"][i]+after]))
    return extended_seq
def encode_seqarr(seq,r):
    '''One hot encoding of the sequence. r specifies the position range.'''
    X = np.array(
            [encode(''.join([s[i] for i in r])) for s in seq]
        )
    X = X.reshape(X.shape[0], -1)
    return X

def DataFrame_input(df):
    ###keep guides for essential genes
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==1)] #
    df=df.dropna()
    logging.info("Number of guides for essential genes: %s \n" % df.shape[0])
    for dataset in range(len(set(df['dataset']))):
        dataset_df=df[df['dataset']==dataset]
        for i in list(set(dataset_df['geneid'])):
            gene_df=dataset_df[dataset_df['geneid']==i]
            median=statistics.median(gene_df['log2FC'])
            for j in gene_df.index:
                df.at[j,'median']=median
                df.at[j,'nr_guides']=gene_df.shape[0]
    if 'CAI' in choice:
        cai=pandas.read_csv('NC_000913.3_CAI_values.csv',sep='\t',index_col=0)
    
    #keep only genes with more than 5 guides from each dataset
    df=df[df['nr_guides']>=5]
    logging.info("Number of guides after filtering: %s \n" % df.shape[0])
    ### one hot encoded sequence features
    sequence_encoded=[]
    numbers_dataset=len(set(df['dataset']))
    for i in df.index:
        if 'CAI' in choice:
            try:
                df.at[i,'CAI']=float(cai['CAI'][df['geneid'][i]])
            except KeyError:
                df.at[i,'CAI']=0
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        # df.at[i,'guideid']=guide_sequence_set.index(df['sequence'][i])
        sequence_encoded.append(self_encode(df['sequence_30nt'][i]))  
        df.at[i,'intercept']=1
        for dataset in range(1,numbers_dataset): #dummy encode the dataset feature
            if df['dataset'][i]==dataset:
                df.at[i,'dataset_%s'%dataset]=1
            else:
                df.at[i,'dataset_%s'%dataset]=0
    #define guideid based on chosen split method
    guideids=np.array(list(df['geneid']))
    clusters=list(df['geneid'])
    medians=np.array(df['median'])
    dataset_col=np.array(df['dataset'])
    #drop features
    y=np.array(df['log2FC'],dtype=float)
    
    if feature_set=='pasteur':
        nts=["A","T","G","C"]
        training_seq=find_target(df)
        training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
        training_seq=training_seq.reshape(training_seq.shape[0],-1)
        guide_features=list()
        for j in range(len(nts)):
            for i in range(14,20):
                guide_features.append('sequence_%s_%s'%(i+1,nts[j]))
            for i in range(1):
                guide_features.append('PAM_%s_%s'%(i+1,nts[j]))
            for i in range(0,16):
                guide_features.append('plus_%s_%s'%(i+1,nts[j]))
        X_guide=pandas.DataFrame(data=training_seq,columns=guide_features)
        
    drop_features=['dataset','std','nr_guides','median','guideid','log2FC',"intergenic","No.","genename","coding_strand",'geneid',
                   "gene_biotype","gene_strand","gene_5","gene_3","genome_pos_5_end","genome_pos_3_end","guide_strand",
                   'sequence','PAM','sequence_30nt','gene_essentiality','off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70',
                   'CRISPRoff_score','spacer_self_fold','RNA_DNA_eng','DNA_DNA_opening']
    if feature_set=='deltaGB':
        drop_features=['dataset','std','nr_guides','median','guideid','log2FC',"intergenic","No.","genename","coding_strand",'geneid',
                       "gene_biotype","gene_strand","gene_5","gene_3","genome_pos_5_end","genome_pos_3_end","guide_strand",
                       'sequence','PAM','sequence_30nt','gene_essentiality','off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70',
                       'spacer_self_fold','RNA_DNA_eng','DNA_DNA_opening','MFE_hybrid_seed','MFE_homodimer_guide','MFE_hybrid_full','MFE_monomer_guide']
    if 'CAI' in choice:
        drop_features=drop_features+["distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_expression_min","gene_expression_max"]
    if 'intercept' not in choice:
        drop_features.append('intercept')
    if 'droppromoter' in choice:
        drop_features.append('if_promoter')
    if split=='gene_dropdistance':
        drop_features+=["distance_start_codon","distance_start_codon_perc"]
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    
    X=df.copy()
    # dataframe for gene features (random-effect model)
    gene_fea=['intercept','dataset',"gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max",'CAI']#
    for j in range(1,numbers_dataset): #dummy encode the dataset feature
        gene_fea.append('dataset_%s'%j)
    headers=list(X.columns.values)
    gene_features=[item for item in gene_fea if item in headers]
    if choice =='only_dataset':
        gene_features=['gene_GC_content','gene_length']+[i for i in gene_features if 'dataset' in i]
    if choice =='only_intercept':
        gene_features=['intercept']
    if choice=='intercept_dataset':
        gene_features=['intercept']+[i for i in gene_features if 'dataset' in i]
    X_gene=X[gene_features] 
    if feature_set !='pasteur':
        guide_features=[item for item in headers if item not in gene_fea]
        X_guide=np.c_[X[guide_features],sequence_encoded] #
        ###add one-hot encoded sequence features to headers
        nts=['A','T','C','G']
        for i in range(30):
            for j in range(len(nts)):
                guide_features.append('sequence_%s_%s'%(i+1,nts[j]))
        X_guide=pandas.DataFrame(data=X_guide,columns=guide_features)
        
        
    logging.info('Number of Guide features: %s'%len(guide_features))
    logging.info('Number of Gene features: %s'%len(gene_features))
    logging.info('Guide features: %s'%",".join(guide_features))
    logging.info('Gene features: %s'%",".join(gene_features))
    
    return X_gene,X_guide, y, gene_features,guide_features,guideids,clusters,medians,dataset_col



#data fusion
labels= ['E75 Rousset','E18 Cui','Wang'] #
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
open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n"% (datasets[0],rousset.shape[0]))
open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[1],rousset18.shape[0]))
open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[2],wang.shape[0]))
open(output_file_name + '/log.txt','a').write("Training dataset: %s\n"%training_set_list[tuple(training_sets)])

X_gene,X_guide, y, gene_features,guide_features,guideids,clusters,medians,dataset_col=DataFrame_input(combined)
open(output_file_name + '/log.txt','a').write("Number of clusters: %s\n" % len(set(clusters)))
open(output_file_name + '/log.txt','a').write("Done processing input: %s s\n\n"%round(time.time()-start,3))
X_df=pandas.DataFrame(data=np.c_[X_gene,X_guide,y,clusters,guideids,medians,dataset_col],columns=gene_features+guide_features+['log2FC','clusters','guideid','median','dataset'])
X_df = X_df.loc[:,~X_df.columns.duplicated()]
guideid_set=list(set(guideids)) 
dtypes=dict()
for feature in X_df.columns.values:
    if feature != 'clusters':
        dtypes.update({feature:float})
    if feature in ['genome_pos_5_end','genome_pos_3_end']:
        dtypes.update({feature:int})
X_df=X_df.astype(dtypes)
numeric_gene_features=[i for i in gene_features if 'dataset' not in i and i!='intercept']
categorical_gene_feature=[i for i in gene_features if 'dataset' in i or i =='intercept']
print(numeric_gene_features,categorical_gene_feature)
open(output_file_name + '/log.txt','a').write("Numeric gene features: %s \n"%numeric_gene_features)
open(output_file_name + '/log.txt','a').write("Categorical gene features: %s \n\n"%categorical_gene_feature)
if model=='autosklearn_rf':
    #optimized RF model from auto-skelearn
    estimator=RandomForestRegressor(bootstrap=False, criterion='friedman_mse', max_depth=None, 
                    max_features=0.22442857329791677, max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=18, min_samples_split=16,
                    min_weight_fraction_leaf=0.0, n_estimators=512, n_jobs=1,
                    verbose=0, warm_start=False,random_state = np.random.seed(111))
#####################
# Tune hyperparameters on 5-CV of all samples 
# Or import the selected hyperparameters
# Train the model with all samples and save
#####################
if tuning_iteration <0:
    training=True
    if saved_cv_models != None:
        try:
            filename = saved_cv_models+'/CRISPRi_headers.sav'
            guide_features=pickle.load(open(filename,'rb'))
            filename = saved_cv_models+'/Merf_model.sav'
            mrf_lgbm=pickle.load(open(filename,'rb'))
            estimator=mrf_lgbm.trained_fe_model
            training=False
        except:
            training=True
    if model=='hyperopt' and training==True:
        from hyperopt import hp, tpe, Trials
        import hyperopt
        from hyperopt.fmin import fmin    
        from hyperopt.early_stop import no_progress_loss
        space = {'bootstrap':hp.choice('bootstrap', [True,False]),
            'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(50, 1000, 10)),
                 'max_features':hp.uniform('max_features', 0.0, 1.0),
                  'max_depth': hp.quniform('max_depth', 2, 30, 1),
                    'min_samples_leaf': hyperopt.hp.choice('min_samples_leaf', np.arange(1, 20, 1)),
                    'min_samples_split': hyperopt.hp.choice('min_samples_split', np.arange(2, 20, 1))}
        def objective_sklearn(params):
            int_types=['n_estimators','max_depth','min_samples_leaf','min_samples_split']
            params = convert_int_params(int_types, params)
            estimator=RandomForestRegressor(criterion='friedman_mse',
                                            random_state=np.random.seed(111),**params)
            #get the mean score of 5 folds
            kf=sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=np.random.seed(111))
            scores=list()
            for train_index, test_index in kf.split(guideid_set):##split the combined training set into train and test based on guideid
                guide_train = np.array(guideid_set)[train_index]
                test_index = np.array(guideid_set)[test_index]
               
                guide_train, guide_val = sklearn.model_selection.train_test_split(guide_train, test_size=test_size,random_state=np.random.seed(111))  
               
                train = X_df[X_df['guideid'].isin(guide_train)]
                train=train[train['dataset'].isin(training_sets)]
                y_train=train['log2FC']
                X_train=train[guide_features]
                Z_train=train[gene_features]
                clusters_train=train['clusters']
                
                val = X_df[X_df['guideid'].isin(guide_val)]
                val=val[val['dataset'].isin(training_sets)]
                y_val=val['log2FC']
                X_val=val[guide_features]
                Z_val=val[gene_features]
                clusters_val=val['clusters']
                
                if len(numeric_gene_features)>0:
                    scaler=StandardScaler()
                    Z_train=np.c_[scaler.fit_transform(Z_train[numeric_gene_features]),np.array(Z_train[categorical_gene_feature])]
                    Z_val=np.c_[scaler.transform(Z_val[numeric_gene_features]),np.array(Z_val[categorical_gene_feature])]
                
                ### keep the same test from 3 datasets
                test = X_df[X_df['guideid'].isin(test_index)]
                y_test=test['log2FC']
                X_test=test[guide_features]
                clusters_test=test['clusters']
                mrf_lgbm = MERF(estimator,max_iterations=10)
    
                mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)            
                # estimator.fit(X_train,y_train)
                d=defaultdict(list)
                d['log2FC']+=list(y_test)
                d['pred']+=list(mrf_lgbm.trained_fe_model.predict(X_test))
                # d['pred']+=list(estimator.predict(X_test))
                d['clusters']+=list(clusters_test)
                d['dataset']+=list(test['dataset'])
                D=pandas.DataFrame.from_dict(d)
                for k in range(3):
                    D_dataset=D[D['dataset']==k]
                    for j in list(set(D_dataset['clusters'])):
                        D_gene=D_dataset[D_dataset['clusters']==j]
                        sr,_=spearmanr(D_gene['log2FC'],D_gene['pred']) 
                        scores.append(-sr)
            score=np.median(scores)
            result = {"loss": score, "params": params, 'status': hyperopt.STATUS_OK}
            return result
    
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
    
        n_trials = 100
        trials = Trials()
        best = fmin(fn=objective_sklearn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=n_trials,
                    trials=trials,
                    early_stop_fn=no_progress_loss(25),
                    rstate=np.random.default_rng(111))
        idx = np.argmin(trials.losses())
        params = trials.trials[idx]["result"]["params"]
        with open(output_file_name+"/trials.hyperopt", "wb") as f:
            pickle.dump(trials, f)
        trail_results=defaultdict(list)
        for i in list(trials.trials):
            trail_results['loss'].append(i["result"]['loss'])
            for p in list(params.keys()):
                trail_results[p].append(i["result"]['params'][p])
        trail_results=pandas.DataFrame.from_dict(trail_results)
        trail_results.to_csv(output_file_name+'/trail_results.csv',sep='\t',index=False)
        estimator=RandomForestRegressor( criterion='friedman_mse',random_state=np.random.seed(111),**params)
        open(output_file_name + '/log.txt','a').write("Hyperopt estimated optimum {}".format(params)+"\n\n")
    open( output_file_name+ '/log.txt','a').write("Estimator:"+str(estimator)+"\n\n\n")
    
    if os.path.isdir(output_file_name+'/saved_model')==False:  
        os.mkdir(output_file_name+'/saved_model')
    
    print(time.asctime(),'Start saving model...')    
    #save model trained with all guides
    
    filename = output_file_name+'/saved_model/CRISPRi_headers.sav'
    pickle.dump(guide_features, open(filename, 'wb'))
    filename = output_file_name+'/saved_model/Merf_model.sav'
    X_all=X_df[X_df['dataset'].isin(training_sets)][guide_features]
    guide_train, guide_val = sklearn.model_selection.train_test_split(guideid_set, test_size=test_size,random_state=np.random.seed(random_seed))  
    
    train = X_df[X_df['guideid'].isin(guide_train)]
    train=train[train['dataset'].isin(training_sets)]
    y_train=train['log2FC']
    X_train=train[guide_features]
    Z_train=train[gene_features]
    clusters_train=train['clusters']
    
    val = X_df[X_df['guideid'].isin(guide_val)]
    val=val[val['dataset'].isin(training_sets)]
    y_val=val['log2FC']
    X_val=val[guide_features]
    Z_val=val[gene_features]
    clusters_val=val['clusters']
    if len(numeric_gene_features)>0:
        scaler=StandardScaler()
        Z_train=np.c_[scaler.fit_transform(Z_train[numeric_gene_features]),np.array(Z_train[categorical_gene_feature])]
        Z_val=np.c_[scaler.transform(Z_val[numeric_gene_features]),np.array(Z_val[categorical_gene_feature])]
    if choice=='pasteur':
        training_seq=find_target(train)
        training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
        X_train=training_seq.reshape(training_seq.shape[0],-1)
        training_seq=find_target(val)
        training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
        X_val=training_seq.reshape(training_seq.shape[0],-1)
    if training:
        mrf_lgbm = MERF(estimator,max_iterations=15)
        mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)
    pickle.dump(mrf_lgbm, open(filename, 'wb'))
    filename = output_file_name+'/saved_model/CRISPRi_model.sav'
    pickle.dump(mrf_lgbm.trained_fe_model, open(filename, 'wb')) 
    coef=pandas.DataFrame(mrf_lgbm.trained_b,index=mrf_lgbm.trained_b.index)
    coef.columns=gene_features
    coef.to_csv(output_file_name+"/saved_model/random_coef.csv",sep='\t',index=True)
    train_re = train.groupby("clusters").mean()
    train_re=train_re.loc[coef.index]
    if len(numeric_gene_features)>0:
        Z_train_re=np.c_[scaler.transform(train_re[numeric_gene_features]),np.array(train_re[categorical_gene_feature])]
    else:
        Z_train_re=np.array(train_re[categorical_gene_feature])
    effect=pandas.DataFrame(data=coef * Z_train_re,columns=gene_features)
    effect.to_csv(output_file_name+"/saved_model/random_effect.csv",sep='\t',index=True)
    open(output_file_name + '/log.txt','a').write("Done saving model: %s s\n"%round(time.time()-start,3))
    
    ## plotting 
    #random effect model predictions
    pred=np.sum(coef * Z_train_re,axis=1)
    for i in pred.index:
        train_gene=train[train['clusters']==i]
        for j in train_gene.index:
            train.at[j,'gene_pred']=pred[i]
    train_re['gene_pred']=pred
    
    ###feature importance for random effect model  
    plot=defaultdict(list)
    for i in coef.index:
        for feature in gene_features:
            plot['value'].append(coef[feature][i])
            plot['feature'].append(feature)
    
    sns.set_style("whitegrid")
    sns.boxplot(data=plot,x='value',y='feature',orient='h',showfliers=False,palette='Blues')
    plt.xlabel("Coefficient")
    plt.xticks(fontsize='small')
    plt.subplots_adjust(left=0.35)
    plt.savefig(output_file_name+'/Coef_random_effect_model.svg',dpi=400)
    # plt.show()
    plt.close()
    
    effect=pandas.DataFrame(data=coef * Z_train_re,columns=gene_features)
    plot=defaultdict(list)
    for i in effect.index:
        for feature in gene_features:
            # plot['dataset'].append(['E75 Rousset', 'E18 Cui', 'Wang'][int(i.split("_")[1])])
            plot['value'].append(effect[feature][i])
            plot['feature'].append(feature)
    
    sns.boxplot(data=plot,x='value',y='feature',orient='h',showfliers=False,palette='Blues')
    plt.xlabel("Feature effects")
    plt.xticks(fontsize='small')
    plt.subplots_adjust(left=0.35)
    plt.savefig(output_file_name+'/feature_effects_random_effect_model.svg',dpi=400)
    # plt.show()
    plt.close()
    
    
    #scatter plot of whole MERF model
    train['pred'] = mrf_lgbm.predict(X_train,Z_train,clusters_train) 
    markers=['x','D','s']
    sns.set_palette('Set2',len(set(training_sets)))
    plt.figure()
    for data in training_sets:
        train_dataset=train[train['dataset']==data]
        ax=sns.scatterplot(train_dataset['log2FC'],train_dataset['pred'],label=labels[data],marker=markers[data],alpha=0.5,edgecolors='white')
        plt.text(0.65,0.10-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(train_dataset['log2FC'],train_dataset['pred'])[0],3)),transform=ax.transAxes,fontsize='x-small')
    plt.legend()
    plt.xlabel("Measured logFC")
    plt.ylabel("Predictions")   
    plt.savefig(output_file_name+"/merf.svg",dpi=400)
    # plt.show()
    plt.close()
    
    
    #scatter plots for random effect model and fixed-effect model (train)
    labels= ['E75 Rousset','E18 Cui','Wang']
    sns.set_style("whitegrid")
    plt.figure()
    for data in training_sets:
        median_dataset=train[train['dataset']==data]
        median_dataset=median_dataset.groupby('clusters').mean()
        ax=sns.scatterplot(median_dataset['median'],median_dataset['gene_pred'],label=labels[data],alpha=0.5,edgecolors='white')
        plt.text(0.55,0.15-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(median_dataset['median'],median_dataset['gene_pred'])[0],3)),transform=ax.transAxes,fontsize='small')
    plt.legend()
    plt.xlabel("Median logFC of gRNAs for each gene",fontsize=14)
    plt.ylabel("Predicted Random effects",fontsize=14)   
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('Train')
    plt.savefig(output_file_name+"/random_median_train.svg",dpi=400)
    # plt.show()
    plt.close()
    
    plt.figure()
    for data in training_sets:
        train_dataset=train[train['dataset']==data]
        X_dataset=train_dataset[guide_features]
        ax=sns.scatterplot(train_dataset['log2FC']-train_dataset['gene_pred'],mrf_lgbm.trained_fe_model.predict(X_dataset),label=labels[data],alpha=0.5)
        plt.text(0.55,0.15-data*0.05,labels[data]+" Spearman R: {0}".format(round(spearmanr(train_dataset['log2FC']-train_dataset['gene_pred'],mrf_lgbm.trained_fe_model.predict(train_dataset[guide_features]))[0],3)),transform=ax.transAxes,fontsize='small')
    plt.legend()
    plt.xlabel("Residual of logFC")
    plt.ylabel("Predicted Fixed effects")
    plt.savefig(output_file_name+"/fixed_train.svg",dpi=400)
    # plt.show()
    plt.close()
    print(time.asctime(),'Start model interpretation...')    
    ##SHAP values for fixed-effect model
    import shap
    treexplainer = shap.TreeExplainer(mrf_lgbm.trained_fe_model)
    shap_values = treexplainer.shap_values(X_all,check_additivity=False)
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':guide_features})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')
    open(output_file_name + '/log.txt','a').write("Done calculating SHAP values: %s s\n"%round(time.time()-start,3))
    shap.summary_plot(shap_values, X_all, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.svg",dpi=400)
    plt.close()
    
    for i in [10,15,30]:
        shap.summary_plot(shap_values, X_all,show=False,max_display=i,alpha=0.05)
        plt.subplots_adjust(left=0.4, top=0.95,bottom=0.1)
        plt.yticks(fontsize='medium')
        plt.xticks(fontsize='medium')
        plt.savefig(output_file_name+"/shap_value_top%s.svg"%(i),dpi=400)
        plt.close()    
    '''
    ###The SHAP interaction values takes a fairly long time and more RAM to run
    '''
    print(time.asctime(),'Start calculating interaction values.') 
    #SHAP interaction values
    shap_values = treexplainer.shap_values(X_all.iloc[:1000,:],check_additivity=False)
    shap_interaction_values=treexplainer.shap_interaction_values(X_all.iloc[:1000,:])
    pickle.dump(shap_interaction_values, open(output_file_name+"/shap_interaction_values_1000.pkl", 'wb'))
    open(output_file_name + '/log.txt','a').write("Done calculating SHAP interaction values: %s s\n\n"%round(time.time()-start,3))
    #mean absolute values
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
    X_index=X_all.iloc[:1000,:].reset_index(drop=True) #use the same index as SHAP values
    coms=[[0,0],[1,0],[0,1],[1,1]] # 4 different types of feature combinations, 0 for -, 1 for +
    marker=['-','+']
    tmp_1d['global interaction rank']=tmp_1d.index+1
    for rank in tmp_1d.index[:5000]: #calculate the combination SHAP values for the top 5000 interaction pairs
        pair=[tmp_1d['feature1'][rank],tmp_1d['feature2'][rank]]
        for i in coms:
            # for sequence features, 0 and 1 means absent or present of the nucleotide of corresponding positions
            # for numeric features, 0 means values in the lower 0.5 quantile, and 1 means values in the upper 0.5 quantile.
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
    tmp_1d.to_csv(output_file_name+"/interaction_pair_sumSHAPvalues.csv",index=False,sep='\t')
    
    pickle.dump(shap_values, open(output_file_name+"/shap_values_1000.pkl", 'wb'))
    pickle.dump(X_index, open(output_file_name+"/X_index.pkl", 'wb'))



####
# Nested cross-validation to evaluate the models
# The model hyperparameters are tuned in the inner 5-CV
# And training on the outer train and tested on the outer test
###
if tuning_iteration < 0:
    print(time.asctime(),'Done.')     
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start))) 
    sys.exit()

if os.path.isdir(output_file_name+'/saved_model')==False:  
    os.mkdir(output_file_name+'/saved_model')    
print(time.asctime(),'Start 10-fold CV...')    
evaluations=defaultdict(list)
iteration_predictions=defaultdict(list)
kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(random_seed))
iteration=0

for train_index, test_index in kf.split(guideid_set):##split the combined training set into train and test based on guideid
    if iteration!=tuning_iteration:
        iteration+=1
        continue
    guide_train = np.array(guideid_set)[train_index]
    test_index = np.array(guideid_set)[test_index]
    training=True
    if saved_cv_models != None:  ### when saved optimized models are provided, the model is directly loaded for train-test
        filename = saved_cv_models+'/MERF_model_%s.sav'%iteration
        if os.path.isfile(filename)==True:
            mrf_lgbm=pickle.load(open(filename,'rb'))
            training=False
            estimator=mrf_lgbm.trained_fe_model
        else:
            print('Saved model not found. (Note, iteration number starts from 0)')
            training=True
    if training == True and model=='hyperopt':  ### If not, the the model needs to be optimized
        ### inner 5-CV fold for hyperopt tuning
        from hyperopt import hp, tpe, Trials
        import hyperopt
        from hyperopt.fmin import fmin    
        from hyperopt.early_stop import no_progress_loss
        space = {'bootstrap':hp.choice('bootstrap', [True,False]),
            'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(50, 1000, 10)),
                 'max_features':hp.uniform('max_features', 0.0, 1.0),
                  'max_depth': hp.quniform('max_depth', 2, 30, 1),
                    'min_samples_leaf': hyperopt.hp.choice('min_samples_leaf', np.arange(1, 20, 1)),
                    'min_samples_split': hyperopt.hp.choice('min_samples_split', np.arange(2, 20, 1))}
        def objective_sklearn(params):
            int_types=['n_estimators','max_depth','min_samples_leaf','min_samples_split']
            params = convert_int_params(int_types, params)
            estimator=RandomForestRegressor(criterion='friedman_mse',
                                            random_state=np.random.seed(111),**params)
            #get the mean score of 5 folds
            kf=sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=np.random.seed(111))
            scores=list()
            for train_inner, test_inner in kf.split(train_index):##split the  training set into train and test based on train_index
                train_inner = np.array(guideid_set)[train_inner]
                test_inner = np.array(guideid_set)[test_inner] # this is for hyperparameter tunning
                # print((train_inner),len(test_inner))
                ## this train and val is for MERF fitting
                train_inner, val_inner = sklearn.model_selection.train_test_split(train_inner, test_size=test_size,random_state=np.random.seed(111))  
               
                train = X_df[X_df['guideid'].isin(train_inner)]
                train=train[train['dataset'].isin(training_sets)]
                y_train=train['log2FC']
                X_train=train[guide_features]
                Z_train=train[gene_features]
                clusters_train=train['clusters']
                
                val = X_df[X_df['guideid'].isin(val_inner)]
                val=val[val['dataset'].isin(training_sets)]
                y_val=val['log2FC']
                X_val=val[guide_features]
                Z_val=val[gene_features]
                clusters_val=val['clusters']
                if len(numeric_gene_features)>0:
                    scaler=StandardScaler()
                    Z_train=np.c_[scaler.fit_transform(Z_train[numeric_gene_features]),np.array(Z_train[categorical_gene_feature])]
                    Z_val=np.c_[scaler.transform(Z_val[numeric_gene_features]),np.array(Z_val[categorical_gene_feature])]
                
                ### evaluate the hyperparameter on the inner test set
                test = X_df[X_df['guideid'].isin(test_inner)]
                y_test=test['log2FC']
                X_test=test[guide_features]
                clusters_test=test['clusters']
                mrf_lgbm = MERF(estimator,max_iterations=10)
                mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)            
                d=defaultdict(list)
                d['log2FC']+=list(y_test)
                d['pred']+=list(mrf_lgbm.trained_fe_model.predict(X_test))
                # d['pred']+=list(estimator.predict(X_test))
                d['clusters']+=list(clusters_test)
                d['dataset']+=list(test['dataset'])
                D=pandas.DataFrame.from_dict(d)
                for k in range(3):
                    D_dataset=D[D['dataset']==k]
                    for j in list(set(D_dataset['clusters'])):
                        D_gene=D_dataset[D_dataset['clusters']==j]
                        sr,_=spearmanr(D_gene['log2FC'],D_gene['pred']) 
                        scores.append(-sr)
            score=np.median(scores)
            result = {"loss": score, "params": params, 'status': hyperopt.STATUS_OK}
            return result

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

        n_trials = 100
        trials = Trials()
        best = fmin(fn=objective_sklearn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=n_trials,
                    trials=trials,
                    early_stop_fn=no_progress_loss(25),
                    rstate=np.random.default_rng(111))
        idx = np.argmin(trials.losses())
        params = trials.trials[idx]["result"]["params"]
        with open(output_file_name+"/trials.hyperopt", "wb") as f:
            pickle.dump(trials, f)
        trail_results=defaultdict(list)
        for i in list(trials.trials):
            trail_results['loss'].append(i["result"]['loss'])
            for p in list(params.keys()):
                trail_results[p].append(i["result"]['params'][p])
        trail_results=pandas.DataFrame.from_dict(trail_results)
        trail_results.to_csv(output_file_name+'/trail_results_%s.csv'%iteration,sep='\t',index=False)
        estimator=RandomForestRegressor( criterion='friedman_mse',random_state=np.random.seed(111),**params)
        mrf_lgbm = MERF(estimator,max_iterations=10)
        print(params)
        open(output_file_name + '/log.txt','a').write("Outer fold {0}: Hyperopt estimated optimum {1}".format(iteration,params)+"\n\n")
    elif training==False:
        open( output_file_name+ '/log.txt','a').write("Estimator:"+str(estimator)+"\n\n\n")
    else:
        # print('else??',model,saved_cv_models)     
        open( output_file_name+ '/log.txt','a').write("Estimator:"+str(estimator)+"\n\n\n") # if there is no saved trained models
    if training==True: 
        ## after tunning hyperparamters on the inner CV, with the tuned hyperparameter, train the model on the outer split
        guide_train, guide_val = sklearn.model_selection.train_test_split(guide_train, test_size=test_size,random_state=np.random.seed(random_seed))  
       
        train = X_df[X_df['guideid'].isin(guide_train)]
        train=train[train['dataset'].isin(training_sets)]
        y_train=train['log2FC']
        X_train=train[guide_features]
        Z_train=train[gene_features]
        clusters_train=train['clusters']
        
        val = X_df[X_df['guideid'].isin(guide_val)]
        val=val[val['dataset'].isin(training_sets)]
        y_val=val['log2FC']
        X_val=val[guide_features]
        Z_val=val[gene_features]
        clusters_val=val['clusters']
        if len(numeric_gene_features)>0:
            scaler=StandardScaler()
            Z_train=np.c_[scaler.fit_transform(Z_train[numeric_gene_features]),np.array(Z_train[categorical_gene_feature])]
            Z_val=np.c_[scaler.transform(Z_val[numeric_gene_features]),np.array(Z_val[categorical_gene_feature])]
        mrf_lgbm = MERF(estimator,max_iterations=15)
        mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)
        filename = output_file_name+'/saved_model/MERF_model_%s.sav'%iteration
        pickle.dump(mrf_lgbm, open(filename, 'wb')) ## save the model from each iteration
        
        
    ### keep the same test from 3 datasets
    test = X_df[X_df['guideid'].isin(test_index)]
    y_test=test['log2FC']
    X_test=test[guide_features]
    Z_test=test[gene_features]
    clusters_test=test['clusters']
    iteration+=1
    iteration_predictions['log2FC'].append(list(y_test))
    if feature_set=='pasteur':
        iteration_predictions['pred'].append(list(mrf_lgbm.trained_fe_model.predict(X_test).reshape(-1, 1).ravel()))
    else:   
        iteration_predictions['pred'].append(list(mrf_lgbm.trained_fe_model.predict(X_test)))
    iteration_predictions['iteration'].append(iteration)
    iteration_predictions['dataset'].append(list(test['dataset']))
    iteration_predictions['clusters'].append(list(test['clusters']))
    
iteration_predictions=pandas.DataFrame.from_dict(iteration_predictions)
iteration_predictions.to_csv(output_file_name+'/iteration_predictions.csv',sep='\t',index=False)
open(output_file_name + '/log.txt','a').write("\n\nDone 10-fold CV: %s s\n\n\n"%round(time.time()-start,3))

open(output_file_name + '/log.txt','a').write("Median Spearman correlation for all gRNAs of each gene: \n")
labels= ['E75 Rousset','E18 Cui','Wang']
df=iteration_predictions.copy()
plot=defaultdict(list)
for i in list(df.index):
    d=defaultdict(list)
    d['log2FC']+=list(df['log2FC'][i])
    d['pred']+=list(df['pred'][i])
    d['clusters']+=list(df['clusters'][i])
    d['dataset']+=list(df['dataset'][i])
    D=pandas.DataFrame.from_dict(d)
    for k in range(3):
        D_dataset=D[D['dataset']==k]
        for j in list(set(D_dataset['clusters'])):
            D_gene=D_dataset[D_dataset['clusters']==j]
            sr,_=spearmanr(D_gene['log2FC'],D_gene['pred']) 
            plot['sr'].append(sr)
            plot['dataset'].append(k)
plot=pandas.DataFrame.from_dict(plot)
for k in range(3):
    p=plot[plot['dataset']==k]
    open(output_file_name + '/log.txt','a').write("%s (median/mean): %s / %s \n" % (labels[k],np.nanmedian(p['sr']),np.nanmean(p['sr'])))
open(output_file_name + '/log.txt','a').write("Mixed 3 datasets (median/mean): %s / %s \n\n\n" % (np.nanmedian(plot['sr']),np.nanmean(plot['sr'])))


print(time.asctime(),'Done.')     
open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start)))    







