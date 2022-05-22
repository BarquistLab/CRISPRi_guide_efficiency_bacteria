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
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import seed_everything
warnings.filterwarnings('ignore')
sns.set_palette('Set2')
dataset_labels=['E75 Rousset','E18 Cui','Wang']   
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
parser.add_argument("-c", "--choice", default="cnn", help="If train on CNN or GRU model, cnn/gru/crispron. default: cnn")
# parser.add_argument("-s", "--split", default='gene', help="train-test split stratege. gene. default: gene")
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
# split=args.split
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

def self_encode(sequence):#one-hot encoding for single nucleotide features
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

def dinucleotide(sequence):#encoding for dinucleotide features
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
    logging_file.write("Number of guides for essential genes: %s \n" % df.shape[0])    
    import statistics
    for dataset in range(len(datasets)):
        dataset_df=df[df['dataset']==dataset]
        for i in list(set(dataset_df['geneid'])):
            gene_df=dataset_df[dataset_df['geneid']==i]
            median=statistics.median(gene_df['log2FC'])
            for j in gene_df.index:
                df.at[j,'median']=median
                df.at[j,'activity_score']=median-df['log2FC'][j]
                df.at[j,'Nr_guide']=gene_df.shape[0]
    guide_sequence_set=list(dict.fromkeys(df['sequence']))
    for i in df.index:
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        df.at[i,'guideid']=guide_sequence_set.index(df['sequence'][i])
    
    df=df[df['Nr_guide']>=5]#keep only genes with more than 5 guides from each datasets
    logging_file.write("Number of guides after filtering: %s \n" % df.shape[0])
    
    # import scipy
    # print(time.asctime(),'Preprocessing...')
    # r75=df[df['dataset']==0]
    # c18=df[df['dataset']==1]
    # r75.index=r75['sequence']
    # r75=r75.loc[c18['sequence']]
    # c18.index=c18['sequence']
    # scaled_log2FC_rc=dict()
    # for i in r75.index:
    #     scaled_log2FC_rc[i]=np.mean([r75['log2FC'][i],c18['log2FC'][i]])
    # for i in df.index:
    #     if df['dataset'][i] in [0,1]:
    #         df.at[i,'scaled_log2FC']=scaled_log2FC_rc[df['sequence'][i]]
    # logging_file.write("Number of guides in E75 Rousset/E18 Cui: %s \n" % r75.shape[0])        
    # w=df[df['dataset']==2]
    # r75=df[df['dataset']==0]
    # w_overlap_log2fc=list()
    # r_overlap_log2fc=list()
    # w_overlap_seq=list()
    # for gene in list(set(w['geneid'])):
    #     if gene in list(r75['geneid']):
    #         w_gene=w[w['geneid']==gene]
    #         r_gene=r75[r75['geneid']==gene]
    #         overlap_pos=[pos for pos in list(w_gene['distance_start_codon']) if pos in list(r_gene['distance_start_codon'])]
    #         if len(overlap_pos)==0:
    #             continue
    #         for pos in overlap_pos:
    #             w_pos=w_gene[w_gene['distance_start_codon']==pos]
    #             r_pos=r_gene[r_gene['distance_start_codon']==pos]
    #             w_overlap_log2fc.append(sum(w_pos['log2FC']))
    #             r_overlap_log2fc.append(sum(r_pos['scaled_log2FC']))
    #             w_overlap_seq.append(list(w_pos['sequence'])[0])
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(w_overlap_log2fc,r_overlap_log2fc) 
    # logging_file.write("Number of guides in Wang: %s \n" % w.shape[0]) 
    # logging_file.write("Number of overlapping guides between Wang and Rousset/Cui: %s \n" % len(w_overlap_log2fc))  
    # logging_file.write("Slope and intercept of the regression line between logFC of Wang and averaged logFC of Rousset and Cui: %s , %s \n" % (round(slope,6),round(intercept,6)))      
    
    # slope=round(slope,6)
    # intercept=round(intercept,6)
    
    # plt.scatter(w_overlap_log2fc,r_overlap_log2fc,color='skyblue',edgecolors='white')
    # plt.plot(w_overlap_log2fc,np.array(w_overlap_log2fc)*slope+intercept,color='red')
    # plt.xlabel("logFC in Wang")
    # plt.ylabel("average logFC of E75 Rousset and E18 Cui")
    # plt.title("N = "+str(len(w_overlap_log2fc)))
    # plt.savefig(output_file_name+'/regress_wang.svg',dpi=150)
    # plt.close()
        
    # for i in df.index:
    #     if df['dataset'][i] in [0,1]:
    #         if df['dataset'][i]==0:
    #             df.at[i,'training']=1
    #         else:
    #             df.at[i,'training']=0
    #     else:
    #         df.at[i,'scaled_log2FC']=df['log2FC'][i]*slope+intercept
    #         if df['sequence'][i] not in w_overlap_seq:
    #             df.at[i,'training']=1
    #         else:
    #             df.at[i,'training']=0
    
    # for i in range(3):
    #     sns.distplot(df[df['dataset']==i]['activity_score'],label=dataset_labels[i],hist=False)
    # plt.legend()
    # plt.xlabel("Activity scores (before scaling)")
    # plt.savefig(output_file_name+"/activity_score_before.svg", dpi=150)
    # plt.close()

    # #calculate the activity scores for each gene in 3 datasets based on scaled logFC
    # for i in list(set(df['geneid'])):
    #     gene_df=df[df['geneid']==i]
    #     median=statistics.median(gene_df['scaled_log2FC'])
    #     for j in gene_df.index:
    #         df.at[j,'median']=median
    #         df.at[j,'activity_score']=median-df['scaled_log2FC'][j]
    
    # for i in range(3):
    #     sns.distplot(df[df['dataset']==i]['scaled_log2FC'],label=dataset_labels[i],hist=False)
    # plt.legend()
    # plt.xlabel("Scaled logFC")
    # plt.savefig(output_file_name+"/scaled_log2fc.png", dpi=150)
    # plt.close()
    
    # for i in range(3):
    #     sns.distplot(df[df['dataset']==i]['activity_score'],label=dataset_labels[i],hist=False)
    # plt.legend()
    # plt.xlabel("Activity scores (after scaling)")
    # plt.savefig(output_file_name+"/activity_score_after.svg", dpi=150)
    # plt.close()
    
    # training_tag=list(df['training'])
    # scaled_log2FC=np.array(df['scaled_log2FC'],dtype=float)
    # print(time.asctime(),'Done preprocessing...')
    geneids=list(df['geneid'])
    sequences=list(df['sequence'])
    log2FC=np.array(df['log2FC'],dtype=float)
    #define guideid based on chosen split method
    guideids=np.array(list(df['geneid']))
    
    # remove columns that are not used in training
    drop_features=['scaled_log2FC','training','std','Nr_guide','coding_strand','guideid',"intergenic","No.","genename","gene_biotype","gene_strand","gene_5","gene_3",
                   "genome_pos_5_end","genome_pos_3_end","guide_strand",'sequence','PAM','gene_essentiality',"geneid",
                   'off_target_90_100','off_target_80_90',	'off_target_70_80','off_target_60_70']
    for feature in drop_features:
        try:
            df=df.drop(feature,1)
        except KeyError:  
            pass
    y=np.array(df['activity_score'],dtype=float)
    median=np.array(df['median'],dtype=float)
    dataset_col=np.array(df['dataset'],dtype=float)
    X=df.drop(['log2FC','activity_score','median'],1)
    X['sequence_30nt'] = X.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1) #encode sequence features for DL 
    headers=list(X.columns.values)
    
    features=['dataset','geneid',"gene_5","gene_strand","gene_GC_content","distance_operon","distance_operon_perc","operon_downstream_genes","ess_gene_operon","gene_length","gene_expression_min","gene_expression_max"]#
    guide_features=[item for item in headers if item not in features]
    X=X[guide_features]
    headers=list(X.columns.values)
    
    logging_file.write("Number of features: %s\n" % len(headers))
    logging_file.write('Features: %s\n'%",".join(headers))
    X=pandas.DataFrame(data=X,columns=headers)
    return X, y, headers,dataset_col,log2FC,median, guideids ,sequences,geneids


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
def datafusion_scaling(df):
    import scipy,statistics
    logging_file= open(output_file_name + '/log.txt','a')
    print(time.asctime(),'Preprocessing...')
    r75=df[df['dataset']==0]
    c18=df[df['dataset']==1]
    r75.index=r75['sequence']
    r75=r75.loc[c18['sequence']] #align the gRNAs in two datasets
    c18.index=c18['sequence']
    scaled_log2FC_rc=dict()
    for i in r75.index:
        scaled_log2FC_rc[i]=np.mean([r75['log2FC'][i],c18['log2FC'][i]]) # calculate the mean logFC as sacled logFC
    for i in df.index:
        if df['dataset'][i] in [0,1]:
            df.at[i,'scaled_log2FC']=scaled_log2FC_rc[df['sequence'][i]]
    logging_file.write("Number of guides in E75 Rousset/E18 Cui: %s \n" % r75.shape[0])        
    w=df[df['dataset']==2]
    r75=df[df['dataset']==0]
    w_overlap_log2fc=list()
    r_overlap_log2fc=list()
    w_overlap_seq=list()
    for gene in list(set(w['geneid'])):
        if gene in list(r75['geneid']):
            w_gene=w[w['geneid']==gene]
            r_gene=r75[r75['geneid']==gene]
            overlap_pos=[pos for pos in list(w_gene['distance_start_codon']) if pos in list(r_gene['distance_start_codon'])] #record overlapping gRNAs in each gene
            if len(overlap_pos)==0:
                continue
            for pos in overlap_pos:
                w_pos=w_gene[w_gene['distance_start_codon']==pos]
                r_pos=r_gene[r_gene['distance_start_codon']==pos]
                w_overlap_log2fc.append(sum(w_pos['log2FC'])) #the logFC of overlapping gRNA in Wang
                r_overlap_log2fc.append(sum(r_pos['scaled_log2FC'])) #the scaled logFC of overlapping gRNA in Rousset
                w_overlap_seq.append(list(w_pos['sequence'])[0]) #record overlapping gRNAs in Wang to exclude them
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(w_overlap_log2fc,r_overlap_log2fc) # fit linear regression
    logging_file.write("Number of guides in Wang: %s \n" % w.shape[0]) 
    logging_file.write("Number of overlapping guides between Wang and Rousset/Cui: %s \n" % len(w_overlap_log2fc))  
    logging_file.write("Slope and intercept of the regression line between logFC of Wang and averaged logFC of Rousset and Cui: %s , %s \n" % (round(slope,6),round(intercept,6)))      
    
    slope=round(slope,6)
    intercept=round(intercept,6)
    
    plt.scatter(w_overlap_log2fc,r_overlap_log2fc,color='skyblue',edgecolors='white')
    plt.plot(w_overlap_log2fc,np.array(w_overlap_log2fc)*slope+intercept,color='red')
    plt.xlabel("logFC in Wang")
    plt.ylabel("average logFC of E75 Rousset and E18 Cui")
    plt.title("N = "+str(len(w_overlap_log2fc)))
    plt.savefig(output_file_name+'/regress_wang.svg',dpi=150)
    plt.close()    
    
    for i in df.index:
        if df['dataset'][i] in [0,1]:
            if df['dataset'][i]==0:
                df.at[i,'training']=1
            else:
                df.at[i,'training']=0
        else:
            df.at[i,'scaled_log2FC']=df['log2FC'][i]*slope+intercept
            if df['sequence'][i] not in w_overlap_seq:
                df.at[i,'training']=1
            else:
                df.at[i,'training']=0
    for i in list(set(df['geneid'])):
        gene_df=df[df['geneid']==i]
        median=statistics.median(gene_df['scaled_log2FC'])
        for j in gene_df.index:
            df.at[j,'median']=median
            df.at[j,'activity']=median-df['scaled_log2FC'][j]
    return df
def main():
    seed_everything(111,workers=True)
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
    df3=pandas.read_csv(datasets[2],sep="\t")
    df3 = df3.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    df3['dataset']=[2]*df3.shape[0]
    df2=df2.append(df3,ignore_index=True)  
    open(output_file_name + '/log.txt','a').write("Total number of guides in dataset %s: %s\n" % (datasets[2],df3.shape[0]))
    training_df=df1.append(df2,ignore_index=True)  
    training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    open(output_file_name + '/log.txt','a').write("Training dataset: %s\n"%training_set_list[tuple(training_sets)])
    #dropping unnecessary features and encode sequence features
    X,y,headers,dataset_col,log2FC,median,guideids ,sequences,geneids= DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    
    header=[i for i in headers if i !='sequence_30nt']
    filename = output_file_name+'/CRISPRi_headers.sav'
    pickle.dump(headers, open(filename, 'wb'))
    max_epochs = 500
    batch_size = 32
    patience = 5
    print(time.asctime(),'Start 10-fold CV...')
    #k-fold cross validation
    iteration_predictions=defaultdict(list)
    kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
    X_df=pandas.DataFrame(data=np.c_[X,y,log2FC,median,guideids,dataset_col,sequences,geneids],
                              columns=headers+['activity','log2FC','median','guideid','dataset','sequence','geneid'])
    fold_inner=0
    guideid_set=list(set(guideids))
    for train_index, test_index in kf.split(guideid_set):
        train_index = np.array(guideid_set)[train_index]
        test_index = np.array(guideid_set)[test_index]
        test = X_df[X_df['guideid'].isin(test_index)]
        y_test=test['activity']
        log2FC_test = np.array( test['log2FC'])
        # scaled_log2FC_test=np.array(test['scaled_log2FC'])
        # median_test =np.array( test['median'])
        X_test=test[headers]
        # train val split
        index_train, index_val = sklearn.model_selection.train_test_split(train_index, test_size=test_size,random_state=np.random.seed(111))
        X_train = X_df[X_df['guideid'].isin(train_index)]
        
        X_train=X_train[X_train['dataset'].isin(training_sets)]
        if len(training_sets)>1:
            X_train = datafusion_scaling(X_train)
        else:
            X_train['training']=[1]*X_train.shape[0]
        X_val = X_train[X_train['guideid'].isin(index_val)]
        X_train = X_train[X_train['guideid'].isin(index_train)]
        X_train=X_train[X_train['training']==1] #remove duplicate guides
        X_val=X_val[X_val['training']==1]
        X_val=X_val[X_val['dataset'].isin(training_sets)]
        y_train=X_train['activity']
        X_train=X_train[headers]
        y_val=X_val['activity']
        X_val=X_val[headers]
        
        #scaling
        SCALE = StandardScaler()
        X_train[header] = SCALE.fit_transform(X_train[header])
        X_val[header] = SCALE.transform(X_val[header])
        X_test[header] = SCALE.transform(X_test[header])
        
        #loader
        loader_train = CrisprDatasetTrain(X_train, y_train, header)
        loader_train = DataLoader(loader_train, batch_size=batch_size, shuffle = True,drop_last=True)
        dataset_val  = CrisprDatasetTrain(X_val, y_val, header)
        loader_val = DataLoader(dataset_val, batch_size=batch_size)
        loader_test = CrisprDatasetTrain(X_test, y_test, header)
        loader_test = DataLoader(loader_test, batch_size=X_test.shape[0])
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
                    verbose = True,
                    save_top_k = 1,
                    mode = 'min',)
        
        estimator = pl.Trainer(gpus=0, callbacks=[early_stop_callback,checkpoint_callback], max_epochs=max_epochs, check_val_every_n_epoch=1, logger=True,progress_bar_refresh_rate = 0, weights_summary=None)
        open(output_file_name + '/log.txt','a').write("Estimator:"+str(estimator)+"\n")
    
        from crispri_dl.architectures import Crispr1DCNN, CrisprGRU, CrisprOn1DCNN
        filename_model = output_file_name + '/model_'+str(fold_inner) + ".ckpt"
        
        #load trained model
        if choice=='cnn':
            estimator.fit(Crispr1DCNN(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
            trained_model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
        elif choice=='gru':
            estimator.fit(CrisprGRU(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
            trained_model = CrisprGRU.load_from_checkpoint(filename_model, num_features = len(header))
        elif choice=='crispron':
            estimator.fit(CrisprOn1DCNN(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
            trained_model = CrisprOn1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
    
    
        predictions_test = estimator.predict(
                                    model=trained_model,
                                    dataloaders=loader_test,
                                    return_predictions=True,
                                    ckpt_path=filename_model)
        predictions = predictions_test[0].cpu().numpy().flatten()
        fold_inner+=1
        iteration_predictions['log2FC'].append(list(log2FC_test))
        iteration_predictions['pred'].append(list(predictions))
        iteration_predictions['iteration'].append([fold_inner]*len(y_test))
        iteration_predictions['dataset'].append(list(test['dataset']))
        iteration_predictions['geneid'].append(list(test['guideid']))

            

    iteration_predictions=pandas.DataFrame.from_dict(iteration_predictions)
    iteration_predictions.to_csv(output_file_name+'/iteration_predictions.csv',sep='\t',index=False)
    print(time.asctime(),'Start saving model...')
    index_train, index_val = sklearn.model_selection.train_test_split(guideid_set, test_size=0.2,random_state=np.random.seed(111))
    
    X_train=X_df[X_df['dataset'].isin(training_sets)]
    if len(training_sets)>1:
        for i in range(3):
            sns.distplot(X_train[X_train['dataset']==i]['activity'],label=dataset_labels[i],hist=False)
        plt.legend()
        plt.xlabel("Activity scores (before scaling)")
        plt.savefig(output_file_name+"/activity_score_before.svg", dpi=150)
        plt.close()
        X_train = datafusion_scaling(X_train)
        for i in range(3):
            sns.distplot(X_train[X_train['dataset']==i]['activity'],label=dataset_labels[i],hist=False)
        plt.legend()
        plt.xlabel("Activity scores (before scaling)")
        plt.savefig(output_file_name+"/activity_score_after.svg", dpi=150)
        plt.close()
    else:
        X_train['training']=[1]*X_train.shape[0]
    X_val = X_train[X_train['guideid'].isin(index_val)]
    X_train = X_train[X_train['guideid'].isin(index_train)]
    X_train=X_train[X_train['training']==1] #remove duplicate guides
    X_val=X_val[X_val['dataset'].isin(training_sets)]
    X_val=X_val[X_val['training']==1]
    y_train=X_train['activity']
    X_train=X_train[headers]
    y_val=X_val['activity']
    X_val=X_val[headers]
    
    SCALE = StandardScaler()
    X_train[header] = SCALE.fit_transform(X_train[header])
    filename = output_file_name+'/SCALEr.sav'
    pickle.dump(SCALE, open(filename, 'wb'))
    X_val[header] = SCALE.transform(X_val[header])
    #loader
    loader_train = CrisprDatasetTrain(X_train, y_train, header)
    loader_train = DataLoader(loader_train, batch_size=batch_size, shuffle = True, drop_last=True)
    dataset_val  = CrisprDatasetTrain(X_val, y_val, header)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=True)
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
    if choice=='cnn':
        estimator.fit(Crispr1DCNN(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val) 
    elif choice=='gru':
        estimator.fit(CrisprGRU(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
    elif choice=='crispron':
        estimator.fit(CrisprOn1DCNN(len(header)), train_dataloader = loader_train, val_dataloaders = loader_val)  
    logging_file= open(output_file_name + '/log.txt','a')
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
    print(time.asctime(),'Done.')
if __name__ == '__main__':
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
#%%
