#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:26:36 2023

@author: yan
"""
#%%
### creating fasta file for cripsroff input  CRISPRoff version 1.1.2 RNAfold 2.2.5
##python crisproff/CRISPRspec_CRISPRoff_pipeline.py --guides Wang_dataset.fasta --no_azimuth --no_off_target_counts --CRISPRoff_scores_folder Wang/ --rnafold_x RNAfold2.2.5 --guide_params_out Wang/params_out --duplex_energy_params crisproff/energy_dics.pkl
path="/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/github_code"
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
crisproff_path="/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/Datasets/crisproff_result"
for dataset in datasets:
    
    df1=pandas.read_csv(dataset,sep="\t")
    name=dataset.split("/")[-1].split(".")[0]
    print(name)
    guides=open(crisproff_path+"/"+name+".fasta","w")
    print(df1.shape)
    print(len(set(df1['No.'])))
    if df1.shape[0] !=len(set(df1['No.'])):
        sys.exit()
    for i in df1.index:
        
        guides.write(">"+str(int(df1['No.'][i]))+"\n"+df1['sequence'][i]+df1['PAM'][i]+"\n")
    guides.close()

#%%
##combine crisproff features into dataset
path="/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/github_code"
datasets=[path+'/0_Datasets/E75_Rousset.csv',path+'/0_Datasets/E18_Cui.csv',path+'/0_Datasets/Wang_dataset.csv']
for dataset in datasets:
    
    df1=pandas.read_csv(dataset,sep="\t")
    name=dataset.split("/")[-1].split(".")[0]
    if name == "E18_Cui":
        off_result="/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/Datasets/crisproff_result/E75_Rousset/params_out"        
    else:
        off_result="/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/Datasets/crisproff_result/"+name+"/params_out"  
    print(name)
    off_result=pandas.read_csv(off_result,sep='\t')
    for i in df1.index:
        guide=off_result[off_result['guideID']==df1['No.'][i]]
        if guide.shape[0]==1 and list(guide['guideSeq'])[0][:20]==df1['sequence'][i]:
            df1.at[i,'spacer_self_fold']=float(list(guide['spacer_self_fold'])[0])
            df1.at[i,'RNA_DNA_eng']=float(list(guide['RNA_DNA_eng'])[0])
            df1.at[i,'DNA_DNA_opening']=float(list(guide['DNA_DNA_opening'])[0])
            df1.at[i,'CRISPRoff_score']=float(list(guide['CRISPRoff_score'])[0])
    df1.to_csv("/home/yan/Projects/CRISPRi_related/doc/CRISPRi_manuscript/Datasets/"+name+"_crisproff.csv",sep='\t',index=False)
