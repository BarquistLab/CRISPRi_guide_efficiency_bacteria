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
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from crispri_dl.dataloader import CrisprDatasetTrain
warnings.filterwarnings('ignore')
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to separate gene and guide effects using median subtracting method and simple deep learning model

Example: python median_subtracting_model_DL.py -training 0,1,2 -c cnn -o test
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
parser.add_argument("-c", "--choice", default="cnn", help="If train on CNN or GRU model, cnn/gru. default: cnn")
parser.add_argument("-s", "--split", default='guide', help="train-test split stratege. guide/gene. default: guide")
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

def encode_sequence(sequence):
    alphabet = 'AGCT'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    encoded_sequence = np.eye(4)[integer_encoded]
    return encoded_sequence
def DataFrame_input(df):
    ###keep guides for essential genes
    logging_file= open(output_file_name + '/log.txt','a')
    df=df[(df['gene_essentiality']==1)&(df['intergenic']==0)&(df['coding_strand']==1)]
    df=df.dropna()
    for i in list(set(list(df['geneid']))):
        df_gene=df[df['geneid']==i]
        for j in df_gene.index:
            df.at[j,'Nr_guide']=df_gene.shape[0]
    logging_file.write("Number of guides for essential genes: %s \n" % df.shape[0])       
    df=df[df['Nr_guide']>=5]
    log2FC=np.array(df['log2FC'],dtype=float)
    sequences=list(dict.fromkeys(df['sequence']))
    ### one hot encoded sequence features
    for i in df.index:
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        df.at[i,'guideid']=sequences.index(df['sequence'][i])
    if split=='guide':
        guideids=np.array(list(df['guideid']))
    elif split=='gene':
        guideids=np.array(list(df['geneid']))
    import statistics
    for dataset in range(len(datasets)):
        dataset_df=df[df['dataset']==dataset]
        for i in list(set(dataset_df['geneid'])):
            gene_df=dataset_df[dataset_df['geneid']==i]
            median=statistics.median(gene_df['log2FC'])
            for j in gene_df.index:
                df.at[j,'median']=median
                df.at[j,'activity_score']=median-df['log2FC'][j]
    drop_features=['std','Nr_guide','coding_strand','guideid',"intergenic","No.","genename","gene_biotype","gene_strand","gene_5","gene_3",
                   "genome_pos_5_end","genome_pos_3_end","guide_strand",'sequence','PAM','gene_essentiality']
    if split=='gene':
        drop_features.append("geneid")
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    y=np.array(df['activity_score'],dtype=float)
    median=np.array(df['median'],dtype=float)
    dataset_col=np.array(df['dataset'],dtype=float)
    X=df.drop(['log2FC','activity_score','median'],1)
    X['sequence_30nt'] = X.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1)
    headers=list(X.columns.values)
    
    features=['dataset','geneid',"gene_5","gene_strand","gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
    guide_features=[item for item in headers if item not in features]
    X=X[guide_features]
    headers=list(X.columns.values)
    
    logging_file.write("Number of features: %s\n" % len(headers))
    logging_file.write('Features: %s\n'%",".join(headers))
    X=pandas.DataFrame(data=X,columns=headers)
    return X, y, headers,dataset_col,log2FC,median, guideids,sequences


def Evaluation(output_file_name,y,predictions,kmeans,kmeans_train,name):
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
    

def SHAP(estimator,X,headers):
    X=pandas.DataFrame(X,columns=headers)
    shap_values = shap.TreeExplainer(estimator).shap_values(X,check_additivity=False)
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')
    
    shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.svg",dpi=400)
    plt.close()
    
    for i in [10,15,30]:
        shap.summary_plot(shap_values, X,show=False,max_display=i,alpha=0.05)
        plt.subplots_adjust(left=0.45, top=0.95,bottom=0.2)
        plt.yticks(fontsize='small')
        plt.xticks(fontsize='small')
        plt.savefig(output_file_name+"/shap_value_top%s.svg"%(i),dpi=400)
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
        # split into training and validation
    training_df=df1.append(df2,ignore_index=True)  
    training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    open(output_file_name + '/log.txt','a').write("Training dataset: %s\n"%training_set_list[tuple(training_sets)])
    X,y,headers,dataset_col,log2FC,median,guideids, guide_sequence_set = DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    
    header=[i for i in headers if i !='sequence_30nt']
    filename = output_file_name+'/CRISPRi_headers.sav'
    pickle.dump(headers, open(filename, 'wb'))
    max_epochs = 500
    batch_size = 64
    patience = 30
    
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
                pasteur_test.at[j,'std']=np.std(p_gene['log2FC'])
        pasteur_test=pasteur_test[pasteur_test['std']>=np.quantile(pasteur_test['std'],0.2)]
        pasteur_test=pasteur_test[pasteur_test['Nr_guide']>=5]
        import statistics
        for dataset in range(len(datasets)):
            test_data=pasteur_test[pasteur_test['dataset']==dataset]
            for i in list(set(test_data['geneid'])):
                gene_df=test_data[test_data['geneid']==i]
                for j in gene_df.index:
                    pasteur_test.at[j,'median']=statistics.median(gene_df['log2FC'])
                    pasteur_test.at[j,'guideid']=guide_sequence_set.index(pasteur_test['sequence'][j])
                    pasteur_test.at[j,'activity_score']=statistics.median(gene_df['log2FC'])-pasteur_test['log2FC'][j]
    # for train_index, test_index in kf.split(X_rescaled,y):
    X_df=pandas.DataFrame(data=np.c_[X,y,log2FC,median,guideids,dataset_col],
                              columns=headers+['activity','log2FC','median','guideid','dataset_col'])
    fold_inner=0
    guideid_set=list(set(guideids))
    for train_index, test_index in kf.split(guideid_set):
        train_index = np.array(guideid_set)[train_index]
        test_index = np.array(guideid_set)[test_index]
        test = X_df[X_df['guideid'].isin(test_index)]
        y_test=test['activity']
        log2FC_test = np.array( test['log2FC'])
        median_test =np.array( test['median'])
        X_test=test[headers]
        
        # train val split
        index_train, index_val = sklearn.model_selection.train_test_split(train_index, test_size=test_size,random_state=np.random.seed(111))
        X_train = X_df[X_df['guideid'].isin(index_train)]
        X_train=X_train[X_train['dataset_col'].isin(training_sets)]
        X_val = X_df[X_df['guideid'].isin(index_val)]
        X_val=X_val[X_val['dataset_col'].isin(training_sets)]
        y_train=X_train['activity']
        X_train=X_train[headers]
        y_val=X_val['activity']
        X_val=X_val[headers]
        
        #loader
        loader_train = CrisprDatasetTrain(X_train, y_train, header)
        print(str(loader_train))
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
        open(output_file_name + '/log.txt','a').write("Estimator:"+str(estimator)+"\n")
    
        from crispri_dl.architectures import Crispr1DCNN, CrisprGRU
        filename_model = output_file_name + '/model_'+str(fold_inner) + ".ckpt"
        
        #load trained model
        if choice=='cnn':
            estimator.fit(Crispr1DCNN(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
            trained_model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        elif choice=='gru':
            estimator.fit(CrisprGRU(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
            trained_model = CrisprGRU.load_from_checkpoint(filename_model, num_features = len(header))
    
        #test
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        trained_model = trained_model.to(device)
        trained_model.eval()
        trained_model.freeze()
        predictions=list()
        for x_sequence_30nt, x_features, _ in loader_test:
            with torch.no_grad():
                predictions_test = trained_model(x_sequence_30nt.to(device), x_features.to(device)).detach()
        predictions.extend(predictions_test.cpu().numpy())
        predictions=np.array(predictions).flatten()
        fold_inner+=1
        iteration_predictions['log2FC'].append(list(log2FC_test))
        iteration_predictions['pred'].append(list(predictions))
        iteration_predictions['iteration'].append([fold_inner]*len(y_test))
        iteration_predictions['dataset'].append(list(test['dataset_col']))
        iteration_predictions['geneid'].append(list(test['guideid']))
        evaluations['Rs_activity'].append(spearmanr(y_test, predictions)[0])
        evaluations['Rs_depletion'].append(spearmanr(log2FC_test, median_test-predictions)[0])
        
        if len(datasets)>1:
            X_combined = np.c_[X,y,log2FC,median,dataset_col]
            X_combined=pandas.DataFrame(data=np.c_[X_combined,guideids],columns=headers+['activity_score','log2FC','median','dataset','guideid'])
            X_combined=X_combined[X_combined['guideid'].isin(test_index)]
            for dataset in range(len(datasets)):
                dataset1=X_combined[X_combined['dataset']==dataset]
                X_test_1=dataset1[headers]
                y_test_1=dataset1['activity_score']
                log2FC_test_1=np.array(dataset1['log2FC'],dtype=float)
                median_test_1=np.array(dataset1['median'],dtype=float)
                loader_test = CrisprDatasetTrain(X_test_1, y_test_1, header)
                loader_test = DataLoader(loader_test, batch_size=X_test_1.shape[0], num_workers = 6, shuffle = False)
                predictions=list()
                for x_sequence_30nt, x_features, _  in loader_test:
                    with torch.no_grad():
                        predictions_test = trained_model(x_sequence_30nt.to(device), x_features.to(device)).detach()
                predictions.extend(predictions_test.cpu().numpy())
                predictions=np.array(predictions).flatten()
                spearman_rho,_=spearmanr(y_test_1, predictions)
                evaluations['Rs_activity_test%s'%(dataset+1)].append(spearman_rho)
                evaluations['Rs_depletion_test%s'%(dataset+1)].append(spearmanr(log2FC_test_1, median_test_1-predictions)[0])
            

    evaluations=pandas.DataFrame.from_dict(evaluations)
    evaluations.to_csv(output_file_name+'/iteration_scores_test.csv',sep='\t',index=True)
    iteration_predictions=pandas.DataFrame.from_dict(iteration_predictions)
    iteration_predictions.to_csv(output_file_name+'/iteration_predictions_test.csv',sep='\t',index=False)
    
    index_train, index_val = sklearn.model_selection.train_test_split(guideid_set, test_size=0.2,random_state=np.random.seed(111))
    X_train = X_df[X_df['guideid'].isin(index_train)]
    X_train=X_train[X_train['dataset_col'].isin(training_sets)]
    X_val = X_df[X_df['guideid'].isin(index_val)]
    X_val=X_val[X_val['dataset_col'].isin(training_sets)]
    y_train=X_train['activity']
    X_train=X_train[headers]
    y_val=X_val['activity']
    X_val=X_val[headers]
    #loader
    loader_train = CrisprDatasetTrain(X_train, y_train, header)
    loader_train = DataLoader(loader_train, batch_size=batch_size, num_workers = 6, shuffle = True, drop_last=True)
    dataset_val  = CrisprDatasetTrain(X_val, y_val, header)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers = 6, drop_last=True)
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
    filename_model = output_file_name + '/model_all' + ".ckpt"
    if choice=='cnn':
        estimator.fit(Crispr1DCNN(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val) 
    elif choice=='gru':
        estimator.fit(CrisprGRU(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
  
    


if __name__ == '__main__':
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
#%%
