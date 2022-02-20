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
start_time=time.time()
import warnings
warnings.filterwarnings('ignore')
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to separate gene and guide effects using median subtracting method.

Example: python median_subtracting_model.py -training 0,1,2 -c rf -o test
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
parser.add_argument("-c", "--choice", default="rf", help="If train on random forest or LASSO model, rf/lasso. default: rf")
parser.add_argument("-s", "--split", default='guide', help="train-test split stratege. guide/gene/guide_dropdistance. guide_dropdistance: To test the models without distance associated features. default: guide")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.2")

args = parser.parse_args()
training_sets=args.training
if training_sets != None:
    if ',' in training_sets:
        training_sets=[int(i) for i in training_sets.split(",")]
    else:
        training_sets=[int(training_sets)]
else:
    training_sets=list(range(3))
split=args.split
folds=args.folds
test_size=args.test_size
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
datasets=['../0_Datasets/E75_Rousset.csv','../0_Datasets/E18_Cui.csv','../0_Datasets/Wang_dataset.csv']
training_set_list={tuple([0]): "E75 Rousset",tuple([1]): "E18 Cui",tuple([2]): "Wang", tuple([0,1]): "E75 Rousset & E18 Cui", tuple([0,2]): "E75 Rousset & Wang",  tuple([1,2]): "E18 Cui & Wang",tuple([0,1,2]): "all 3 datasets"}

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
    logging_file= open(output_file_name + '/log.txt','a')
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==1)]
    df=df.dropna()
    import statistics
    for dataset in range(len(datasets)):
        dataset_df=df[df['dataset']==dataset]
        for i in list(set(dataset_df['geneid'])):
            gene_df=dataset_df[dataset_df['geneid']==i]
            median=statistics.median(gene_df['log2FC'])
            for j in gene_df.index:
                df.at[j,'median']=median
                df.at[j,'activity_score']=median-df['log2FC'][j]
    logging_file.write("Number of guides for essential genes: %s \n" % df.shape[0])
    for i in list(set(list(df['geneid']))):
        df_gene=df[df['geneid']==i]
        for j in df_gene.index:
            df.at[j,'Nr_guide']=df_gene.shape[0]
    df=df[df['Nr_guide']>=5]
    sequences=list(dict.fromkeys(df['sequence']))
    log2FC=np.array(df['log2FC'],dtype=float)
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
    
    drop_features=['std','Nr_guide','coding_strand','guideid',"intergenic","No.","genename","gene_biotype","gene_strand","gene_5","gene_3",
                   "genome_pos_5_end","genome_pos_3_end","guide_strand",'sequence','PAM','sequence_30nt','gene_essentiality','off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70']
    if split=='gene':
        drop_features.append("geneid")
    elif split=='guide_dropdistance':
        drop_features+=["distance_start_codon","distance_start_codon_perc"]#,'guide_GC_content', 'homopolymers', 'MFE_hybrid_full', 'MFE_hybrid_seed', 'MFE_homodimer_guide', 'MFE_monomer_guide']
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    y=np.array(df['activity_score'],dtype=float)
    median=np.array(df['median'],dtype=float)
    dataset_col=np.array(df['dataset'],dtype=float)
    X=df.drop(['log2FC','activity_score','median'],1)
    headers=list(X.columns.values)
    features=['dataset','geneid',"gene_5","gene_strand","gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
    guide_features=[item for item in headers if item not in features]
    X=X[guide_features]
    headers=list(X.columns.values)
    
    ### add one-hot encoded sequence features columns
    PAM_encoded=np.array(PAM_encoded)
    sequence_encoded=np.array(sequence_encoded)
    dinucleotide_encoded=np.array(dinucleotide_encoded)
    X=np.c_[X,sequence_encoded,PAM_encoded,dinucleotide_encoded]
    
    ###add one-hot encoded sequence features to headers
    nts=['A','T','C','G']
    for i in range(sequence_len):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
    for i in range(PAM_len):
        for j in range(len(nts)):
            headers.append('PAM_%s_%s'%(i+1,nts[j]))
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    for i in range(dinucleotide_len-1):
        for dint in dinucleotides:
            headers.append(dint+str(i+1)+str(i+2))
    X=pandas.DataFrame(data=X,columns=headers)
    
    logging_file.write('Number of features: %s\n'%len(headers))
    logging_file.write('Features: %s\n'%",".join(headers))
    
    return X, y, headers,dataset_col,log2FC,median , guideids,sequences


def Evaluation(output_file_name,y,predictions,name):
    spearman_rho,spearman_p_value=spearmanr(y, predictions)
    pearson_rho,pearson_p_value=pearsonr(y, predictions)
    y=np.array(y)
    # scatter plot
    plt.figure() 
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    ax_main.scatter(y,predictions,edgecolors='white',alpha=0.8)
    ax_main.set(xlabel='Experimental log2FC',ylabel='Predicted log2FC')
    # ax_main.legend(fontsize='small')
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
    
def SHAP(estimator,X,headers):
    X_train=pandas.DataFrame(X,columns=headers)
    X_train=X_train.astype(float)
    shap_values = shap.TreeExplainer(estimator).shap_values(X,check_additivity=False)
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')
    
    shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.png",dpi=400)
    plt.close()
    
    for i in [10,15,30]:
        shap.summary_plot(shap_values, X,show=False,max_display=i,alpha=0.05)
        plt.subplots_adjust(left=0.45, top=0.95,bottom=0.2)
        plt.yticks(fontsize='small')
        plt.xticks(fontsize='small')
        plt.savefig(output_file_name+"/shap_value_top%s.png"%(i),dpi=400)
        plt.close()    
def encode(seq):
    return np.array([[int(b==p) for b in seq] for p in ["A","T","G","C"]])
def find_target(df,before=20,after=20):
    from Bio import SeqIO
    fasta_sequences = SeqIO.parse(open("../0_Datasets/NC_000913.3.fasta"),'fasta')    
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
    if len(datasets)==3:
        df3=pandas.read_csv(datasets[2],sep="\t")
        df3 = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
        df3['dataset']=[2]*df3.shape[0]
        df2=df2.append(df3,ignore_index=True)  
        open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[2],df3.shape[0]))
    training_df=df1.append(df2,ignore_index=True)  
    training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    open(output_file_name + '/log.txt','a').write("Training dataset: %s\n"%training_set_list[tuple(training_sets)])
    X,y,headers,dataset_col,log2FC,median,guideids, guide_sequence_set = DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    
    if choice=='rf':
        from sklearn.ensemble import RandomForestRegressor
        estimator=RandomForestRegressor(bootstrap=True, criterion='friedman_mse', max_depth=None, 
                        max_features=0.22442857329791677, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=18, min_samples_split=16,
                        min_weight_fraction_leaf=0.0, n_estimators=512, n_jobs=1,
                        verbose=0, warm_start=False,random_state = np.random.seed(111))
    if choice=='lasso':
        from sklearn import linear_model
        from sklearn.model_selection import  cross_val_score,GroupKFold
        from hyperopt import hp, tpe, Trials
        import hyperopt
        from hyperopt.fmin import fmin
    
        def objective_sklearn(params):
            int_types=[]
            params = convert_int_params(int_types, params)
            # Extract the boosting type
            estimator = linear_model.Lasso(random_state = np.random.seed(111),**params)
            score = cross_val_score(estimator, X, y, groups=guideids,scoring=scorer, cv=GroupKFold(n_splits=5)).mean()
            #using logloss here for the loss but uncommenting line below calculates it from average accuracy
            result = {"loss": score, "params": params, 'status': hyperopt.STATUS_OK}
            return result
        #hyperparameter space
        #Lasso
        space={'alpha': hp.uniform('alpha', 0.0, .1)}
     
        n_trials = 100
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
        open(output_file_name + '/log.txt','a').write("Hyperopt estimated optimum {}".format(params)+"\n\n")
        estimator=linear_model.Lasso(random_state = np.random.seed(111),**params)
        ### tested optimized lasso
        # estimator=linear_model.Lasso(random_state = np.random.seed(111),alpha=0.010780717789770466, copy_X=True, fit_intercept=True,max_iter=1000, normalize=False, positive=False, precompute=False,selection='cyclic', tol=0.0001, warm_start=False)
        
        
        
    open(output_file_name + '/log.txt','a').write("Estimator:"+str(estimator)+"\n")
    
    X_df=pandas.DataFrame(data=np.c_[X,y,log2FC,median,guideids,dataset_col],
                              columns=headers+['activity','log2FC','median','guideid','dataset_col'])
    dtypes=dict()
    for feature in X_df.columns.values:
        if feature != 'geneid':
            dtypes.update({feature:float})
    X_df=X_df.astype(dtypes)
    guideid_set=list(set(guideids))
    #k-fold cross validation
    evaluations=defaultdict(list)
    iteration_predictions=defaultdict(list)
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
                    if split=='guide' or split=='guide_dropdistance':
                        pasteur_test.at[j,'guideid']=guide_sequence_set.index(pasteur_test['sequence'][j])
                    elif split=='gene':
                        pasteur_test.at[j,'guideid']=int(pasteur_test['geneid'][j][1:])
                    pasteur_test.at[j,'activity_score']=statistics.median(gene_df['log2FC'])-pasteur_test['log2FC'][j]
    
    iteration=0
    for train_index, test_index in kf.split(guideid_set):
        train_index = np.array(guideid_set)[train_index]
        test_index = np.array(guideid_set)[test_index]
        X_train = X_df[X_df['guideid'].isin(train_index)]
        X_train=X_train[X_train['dataset_col'].isin(training_sets)]
        y_train=np.array(X_train['activity'])
        X_train=np.array(X_train[headers])
        test = X_df[X_df['guideid'].isin(test_index)]
        y_test=np.array(test['activity'])
        log2FC_test = np.array( test['log2FC'])
        median_test =np.array( test['median'])
        X_test=np.array(test[headers])
        estimator = estimator.fit(X_train,y_train)
        predictions = estimator.predict(X_test)
        iteration+=1
        iteration_predictions['log2FC'].append(list(log2FC_test))
        iteration_predictions['pred'].append(list(predictions))
        iteration_predictions['iteration'].append([iteration]*len(y_test))
        iteration_predictions['dataset'].append(list(test['dataset_col']))
        iteration_predictions['geneid'].append(list(test['guideid']))
        
        evaluations['Rs_activity'].append(spearmanr(y_test, predictions)[0])
        evaluations['Rs_depletion'].append(spearmanr(log2FC_test, median_test-predictions)[0])
        evaluations['Rs_median'].append(spearmanr(log2FC_test, median_test)[0])
        if len(datasets)>1:
            X_combined = np.c_[X,y,log2FC,median,dataset_col]
            # X_combined=pandas.DataFrame(data=X_combined,columns=headers+['activity_score','log2FC','median','dataset'])
            
            X_combined=pandas.DataFrame(data=np.c_[X_combined,guideids],columns=headers+['activity_score','log2FC','median','dataset','guideid'])
            X_combined=X_combined[X_combined['guideid'].isin(test_index)]
            for dataset in range(len(datasets)):
                dataset1=X_combined[X_combined['dataset']==dataset]
                X_test_1=dataset1[headers]
                y_test_1=np.array(dataset1['activity_score'],dtype=float)
                log2FC_test_1=np.array(dataset1['log2FC'],dtype=float)
                median_test_1=np.array(dataset1['median'],dtype=float)
                X_test_1=np.array(X_test_1)
                predictions=estimator.predict(X_test_1)
                spearman_rho,_=spearmanr(y_test_1, predictions)
                evaluations['Rs_activity_test%s'%(dataset+1)].append(spearman_rho)
                evaluations['Rs_depletion_test%s'%(dataset+1)].append(spearmanr(log2FC_test_1, median_test_1-predictions)[0])
                evaluations['Rs_median_test%s'%(dataset+1)].append(spearmanr(log2FC_test_1, median_test_1)[0])
            
            #Pasteur method test on our test set
            pasteur=pasteur_test[(pasteur_test['guideid'].isin(test_index))]
            test_data=pasteur
            training_seq,guides_index=find_target(test_data)
            training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
            training_seq=training_seq.reshape(training_seq.shape[0],-1)
            test_data=test_data.loc[guides_index]
            reg=pickle.load(open('Pasteur_model.pkl','rb'))
            test_data['pasteur_score']=reg.predict(training_seq).reshape(-1, 1).ravel()
            iteration_predictions['pasteur_score'].append(list(reg.predict(training_seq).reshape(-1, 1).ravel()))
            for i in test_data.index:
                test_data.at[i,'predicted_score']=test_data['median'][i]-test_data['pasteur_score'][i]
            evaluations['Rs_depletion_pasteur_data'].append(spearmanr(test_data['log2FC'], test_data['predicted_score'])[0])
            evaluations['Rs_activity_pasteur_data'].append(spearmanr(test_data['activity_score'], test_data['pasteur_score'])[0])
            for dataset in range(len(datasets)):
                test_data=pasteur[pasteur['dataset']==dataset]
                training_seq,guides_index=find_target(test_data)
                training_seq=encode_seqarr(training_seq,list(range(34,41))+list(range(43,59)))
                training_seq=training_seq.reshape(training_seq.shape[0],-1)
                test_data=test_data.loc[guides_index]
                test_data['pasteur_score']=reg.predict(training_seq).reshape(-1, 1).ravel()
                
                for i in test_data.index:
                    test_data.at[i,'predicted_score']=test_data['median'][i]-test_data['pasteur_score'][i]
                evaluations['Rs_depletion_pasteur_data%s'%(dataset+1)].append(spearmanr(test_data['log2FC'], test_data['predicted_score'])[0])
                # Evaluation(output_file_name,np.array(test_data['log2FC']),np.array(test_data['predicted_score']),"X_pasteur_test%s"%(dataset+1))
                evaluations['Rs_activity_pasteur_data%s'%(dataset+1)].append(spearmanr(test_data['activity_score'], test_data['pasteur_score'])[0])

    evaluations=pandas.DataFrame.from_dict(evaluations)
    evaluations.to_csv(output_file_name+'/iteration_scores.csv',sep='\t',index=True)
    iteration_predictions=pandas.DataFrame.from_dict(iteration_predictions)
    iteration_predictions.to_csv(output_file_name+'/iteration_predictions.csv',sep='\t',index=False)
    
    ### Inplemented functions
    logging_file= open(output_file_name + '/log.txt','a')
    
    #save models
    estimator.fit(np.array(X_df[X_df['dataset_col'].isin(training_sets)][headers]),np.array(X_df[X_df['dataset_col'].isin(training_sets)]['activity']))
    os.mkdir(output_file_name+'/saved_model')
    filename = output_file_name+'/saved_model/CRISPRi_model.sav'
    pickle.dump(estimator, open(filename, 'wb')) 
    filename = output_file_name+'/saved_model/CRISPRi_headers.sav'
    pickle.dump(headers, open(filename, 'wb'))
    
    ##split the combined training set into train and test
    guide_train, guide_test = sklearn.model_selection.train_test_split(guideid_set, test_size=test_size,random_state=np.random.seed(111))  
    
    X_train = X_df[X_df['guideid'].isin(guide_train)]
    X_train=X_train[X_train['dataset_col'].isin(training_sets)]
    y_train=np.array(X_train['activity'])
    X_train = np.array(X_train[headers])
    
    X_test = X_df[X_df['guideid'].isin(guide_test)]
    y_test=np.array(X_test['activity'])
    log2FC_test=np.array(X_test['log2FC'])
    median_test=np.array(X_test['median'])
    X_test = np.array(X_test[headers])
    estimator.fit(X_train,y_train)
    predictions = estimator.predict(X_test)
    Evaluation(output_file_name,y_test,predictions,"X_test")
    
    ### model validation with validation dataset
    X_combined=pandas.DataFrame(data=np.c_[X,y,log2FC,median,dataset_col,guideids],columns=headers+['activity_score','log2FC','median','dataset','guideid'])
    X_combined=X_combined[X_combined['guideid'].isin(guide_test)]
    if choice =='lasso':
        coef=pandas.DataFrame(data={'coef':estimator.coef_,'feature':headers})
        coef=coef.sort_values(by='coef',ascending=False)
        coef=coef[:15]
        pal = sns.color_palette('pastel')
        sns.barplot(data=coef,x='feature',y='coef',color=pal.as_hex()[0])
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.35)
        plt.savefig(output_file_name+"/coef.svg",dpi=400)
        plt.close()
    # elif choice=='rf':
        # SHAP(estimator,X_train,headers)
        
    if split=='gene':
        logging_file.write("Median Spearman correlation for all gRNAs of each gene: \n")
        labels= ['E75 Rousset','E18 Cui','Wang']
        df=iteration_predictions.copy()
        plot=defaultdict(list)
        for i in list(df.index): #each iteration/CV split
            d=defaultdict(list)
            d['log2FC']+=list(df['log2FC'][i])
            d['pred']+=list(df['pred'][i])
            d['geneid']+=list(df['geneid'][i])
            d['dataset']+=list(df['dataset'][i])
            D=pandas.DataFrame.from_dict(d)
            for k in training_sets:
                D_dataset=D[D['dataset']==k]
                for j in list(set(D_dataset['geneid'])):
                    D_gene=D_dataset[D_dataset['geneid']==j]
                    sr,_=spearmanr(D_gene['log2FC'],-D_gene['pred']) 
                    plot['sr'].append(sr)
                    plot['dataset'].append(k)
        plot=pandas.DataFrame.from_dict(plot)
        for k in training_sets:
            p=plot[plot['dataset']==k]
            logging_file.write("%s (median/mean): %s / %s \n" % (labels[k],np.nanmedian(p['sr']),np.nanmean(p['sr'])))
        logging_file.write("Mixed 3 datasets (median/mean): %s / %s \n" % (np.nanmedian(plot['sr']),np.nanmean(p['sr'])))
    logging_file.close()
    


if __name__ == '__main__':
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
#%%
