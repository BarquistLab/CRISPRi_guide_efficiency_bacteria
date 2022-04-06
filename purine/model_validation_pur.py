#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:07:25 2020

@author: yanying
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas
import itertools
import sklearn.model_selection
import sklearn.metrics
from scipy.stats import spearmanr,pearsonr,kendalltau,mannwhitneyu
import pickle
from collections import defaultdict
import statistics
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import matplotlib as mpl
from sklearn.metrics import ndcg_score
mpl.rcParams['figure.dpi'] = 400
import warnings
warnings.filterwarnings('ignore')

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
def DataFrame_input(df):
    PAM_encoded=[]
    sequence_encoded=[]
    dinucleotide_encoded=[]
    df['coding_strand']=[1]*df.shape[0]
    for i in df.index:
        PAM_encoded.append(self_encode(df['PAM'][i]))
        sequence_encoded.append(self_encode(df['sequence'][i]))
        dinucleotide_encoded.append(dinucleotide(df['sequence_30nt'][i]))
    df['sequence_30nt'] = df.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1)
    headers=df.columns.values.tolist()
    nts=['A','T','C','G']
    for i in range(20):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
    for i in range(3):
        for j in range(len(nts)):
            headers.append('PAM_%s_%s'%(i+1,nts[j]))
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    for i in range(30-1):
        for dint in dinucleotides:
            headers.append(dint+str(i+1)+str(i+2))
    
    df=pandas.DataFrame(data=np.c_[df,sequence_encoded,PAM_encoded,dinucleotide_encoded],columns=headers)
    return df


def pur(df,tests):
  
    genes=list(set(df['gene_name']))
    genes.sort()
    correlations=defaultdict(list)
 
    for OD in ['OD02','OD06','OD1']:#
        test=OD+"_edgeR.batch"
               
        for gene in genes:
            df_gene=df[df['gene_name']==gene]
            df_gene=df_gene.dropna(subset=[test])
            df_gene=df_gene.sort_values(by='distance_start_codon')
            for prediction in tests:
                predicted=MinMaxScaler().fit_transform(np.array(df_gene[prediction]).reshape(-1,1))
                predicted=np.hstack(predicted)
                Rs,ps=spearmanr(df_gene[test],-df_gene[prediction])
                correlations['spearmanr'].append(Rs)
                correlations['gene'].append(gene)
                correlations['timepoint'].append(OD)
                correlations['method'].append(prediction)
                
                
    for test in ['spearmanr']:#,'kendalltau','kendalltau_distance','pearsonr','mse']: #
        plt.figure()#figsize=(6,3)
        sns.set_palette('Set2',len(tests))
        sns.set_style("whitegrid")
    #     PROPS={'boxprops':{'edgecolor':'black','linewidth':0.5,},
    #            'medianprops':{'linewidth':0.5},
    # 'whiskerprops':{'linewidth':0.5},
    # 'capprops':{'linewidth':0.5}}
        ax=sns.boxplot(data=correlations,x='timepoint',y=test,hue='method')#,hue_order=['MERF_RF','RF','MERF_RF(R75)','MERF_RF(C18)','MERF_RF(W)','LASSO_hyperopt','pasteur_score'])
        sns.swarmplot(data=correlations,x='timepoint',y=test,hue='method',dodge=True,s=2,color='lightgrey')#,hue_order=['MERF_RF','RF','MERF_RF(R75)','MERF_RF(C18)','MERF_RF(W)','LASSO_hyperopt','pasteur_score'])
        plt.xlabel("")
        if test=='spearmanr':
            plt.ylabel("Spearman correlation",fontsize=14)
        plt.xticks([0,1,2],["OD 0.2","OD 0.6","OD 1"],fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc='lower right',fontsize='x-small')
        plt.subplots_adjust(right=0.8)
        handles, predictions = ax.get_legend_handles_labels()
        plt.legend(handles[:len(tests)], [labels[prediction] for prediction in predictions][:len(tests)],
                    fontsize=12, loc='lower left',
                    ncol=1,bbox_to_anchor=(1,0, 1, .102))
        plt.show()
        plt.close()  
    
    

labels={'cnn':'MS (CNN)','gru':'MS (GRU)','crispron':'MS (CRISPRon)'}
df=pandas.read_csv("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/figure3/pur_gRNAs.csv",sep='\t')
df=DataFrame_input(df)
    
for alg in ['cnn','gru','crispron']:
    
    filename_model = "/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/figure2/gene_split/%s/model_10.ckpt"%(alg)
    header=pickle.load(open("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/figure2/gene_split/%s/CRISPRi_headers.sav"%(alg),'rb'))
    SCALE=pickle.load(open("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/result/figure2/gene_split/%s/SCALEr.sav"%(alg),'rb'))
    df_sub=df[header]
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
    if alg=='gru':
        trained_model = CrisprGRU.load_from_checkpoint(filename_model, num_features = len(header))
    elif alg=='cnn':
        trained_model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(header))
    elif alg=='crispron':
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

    predictions_test = estimator.predict(
    model=trained_model,
    dataloaders=loader_test,
    return_predictions=True,
    ckpt_path=filename_model)
    #print(len(predictions_test))
    predictions = predictions_test[0].cpu().numpy().flatten()
    df[alg]=predictions
    
pur(df,['crispron','cnn','gru']) #['CNN','GRU']
