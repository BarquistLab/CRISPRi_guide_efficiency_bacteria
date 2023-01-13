#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:01:13 2023

@author: yan
"""

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr , pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import pandas as pd
data_path="data/"
cui2018data="https://gitlab.pasteur.fr/dbikard/badSeed_public/raw/master/screen_data.csv"
cui2018data=pd.read_csv(cui2018data,index_col=0)
cui2018data=cui2018data[cui2018data.ntargets==1]
cui2018dataGB=cui2018data.loc[cui2018data.coding==True,["gene","fit75"]].groupby("gene")
geneDF=cui2018dataGB.agg([len,np.median,np.mean,np.std])
print(geneDF.head())
from tqdm import tqdm
for g in tqdm(geneDF.index):
    cui2018data.loc[cui2018data.gene==g,"gene_median"]=geneDF.loc[g].fit75["median"]
    cui2018data.loc[cui2018data.gene==g,"gene_mean"]=geneDF.loc[g].fit75["mean"]
    cui2018data.loc[cui2018data.gene==g,"gene_std"]=geneDF.loc[g].fit75["std"]
responsive_genes=geneDF[(geneDF.fit75.len>4) & (geneDF.fit75["median"]<-2)].index
data3=cui2018data.loc[(cui2018data.coding==True) & cui2018data.gene.isin(responsive_genes)].copy()
print(data3.shape, responsive_genes.shape)
y=data3.gene_median-data3.fit75
data3["activity"]=y

#One-hot-encoding of the target sequence
bases=["A","T","G","C"]
def encode(seq):
    return np.array([[int(b==p) for b in seq] for p in bases])
    
def encode_seqarr(seq,r):
    '''One hot encoding of the sequence. r specifies the position range.'''
    X = np.array(
            [encode(''.join([s[i] for i in r])) for s in seq]
        )
    X = X.reshape(X.shape[0], -1)
    return X

def scorer_pearson(reg,X,y):
    return(pearsonr(reg.predict(X).flatten(),y)[0])

def scorer_spearman(reg,X,y):
    return(spearmanr(reg.predict(X).flatten(),y)[0])

# scaling the activity values
y=list(data3["activity"])
Xlin = encode_seqarr(data3["seq"],list(range(34,41))+list(range(43,59)))

# # Selecting alpha through CV
lassocv=LassoCV(cv=10,fit_intercept=False)
lassocv.fit(Xlin,y.ravel())
print(lassocv.alpha_)

#Final model
reg=Lasso(alpha=lassocv.alpha_,fit_intercept=False)
reg.fit(Xlin,y)

pickle.dump(reg,open("reg.pkl","bw"))