#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:45:36 2019

@author: yanying
"""

import matplotlib.pyplot as plt
import numpy as np
from Bio.Seq import Seq
import os
import time 
import seaborn as sns
import logging
import itertools
import pandas 
import pickle
import subprocess
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to design gRNAs from FASTA input file and predict gRNA efficiency  

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

def main():
    start_time=time.time()
    global  l, maxgc, mingc, PAM, SHAP_plots,t,output_file_name
    t=time.time()
    t=int(t)
    os.mkdir("static/%s" %(t))
    output_file_name="static/%s" %(t)
    genome_fasta=request.form['seq']
    
    genome_gff=None
    l=request.form['length']
    l=int(l)
    maxgc=request.form['maxgc']
    maxgc=int(maxgc)
    mingc=request.form['mingc']
    mingc=int(mingc)
    PAM=request.form['PAM']
    # blast_db=request.form['blast_db']
    # if blast_db == None:
    #     blast_db=genome_fasta.copy()
#    biotype=request.form['biotype']
#    if biotype == "all":
#            biotypes=["protein_coding","rRNA","tRNA","ncRNA","pseudogene"]
#    else:
#        biotypes=[biotype]
    SHAP_plots=request.form['shap']
    render_template('index.html',test=genome_fasta)
    error=ReferenceGenomeInfo(genome_fasta,genome_gff)
    if error != "":
        return render_template('index.html',test=genome_fasta,error =error)
    else:
        render_template('index.html',test=genome_fasta)
    library_guides=gRNA_search(reference_fasta)   
    guides=[]
    for gene in library_guides.keys():
        for guide in library_guides[gene]:
            guides.append(guide)
    if len(guides)==0:
        error="Error: No gRNA was found"
        return render_template('index.html',test=genome_fasta,error2 =error)
    
    guides_df=MachineLearning(guides)
    if SHAP_plots == 'yes':
        print("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))
        return render_template('result.html',test=genome_fasta,data=guides_df.to_html(index=False),total=guides_df.shape[0],seq_length=len(genome_fasta),length=l,maxgc=maxgc,mingc=mingc,PAM=PAM,t=t)   
    else:
        print("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))
        return render_template('result2.html',test=genome_fasta,data=guides_df.to_html(index=False),total=guides_df.shape[0],seq_length=len(genome_fasta),length=l,maxgc=maxgc,mingc=mingc,PAM=PAM,t=t)   

def ReferenceGenomeInfo(genome_fasta,genome_gff):
    global reference_fasta, reference_genes,taskname
    error=""
#    try: 
#        fasta_sequences = list(SeqIO.parse(open(genome_fasta),'fasta'))
#        if len(fasta_sequences) == 0:
#            error="Error: Please upload file in FASTA format."
#            return error
#        for fasta in fasta_sequences:  # input reference genome
#            reference_fasta=fasta.seq 
#            if genome_gff == None:
#                taskname=fasta.id
#            break
#    except:
    genome_fasta=genome_fasta.replace("\r","")
    if genome_fasta=="" or genome_fasta[0] != ">":
        genome_fasta=genome_fasta.replace("\n","")
        reference_fasta=Seq(genome_fasta.upper())
        taskname=""
    else:
        reference_fasta=Seq("".join(genome_fasta.split("\n")[1:]).upper())
        taskname=genome_fasta.split("\n")[0][1:]
    if reference_fasta == "":
        error="Error: Please input sequence in FASTA format or only sequence."
        return error
    if len(reference_fasta) > 25000:
        error="Error: Please input sequence shorter than 25,000 bp."
        return error
    if any(bp not in ['A','T','C','G'] for bp in reference_fasta):
        error="Error: There are non-ATCG characters in the input sequence. (Only one sequnce is accepted, please avoid multiple FASTA inputs.)"
        return error
#    if genome_gff != None:
#        reference_genes=[]
#        for line in open(genome_gff):
#           if "#" not in line and "Gene;gene" in line: ## input reference gff
#               line=line.replace("\n","")    
#               row=line.split("\t")
#               gene_biotype=row[8].split("gene_biotype=")[1].split(";")[0]
#               if gene_biotype in ["protein_coding","rRNA","tRNA","ncRNA","pseudogene"]:
#                   geneid=row[8].split(";")[0].split("-")[1]
#                   genename=row[8].split("Name=")[1].split(";")[0]
#                   start=int(row[3])
#                   end=int(row[4])
#                   strand=row[6]
#                   length=int(row[4])-int(row[3])+1
#                   seq=reference_fasta[int(start)-1:int(end)]
#                   GC_content = '{:.2f}'.format((seq.count('G') + seq.count('C')) / len(seq) * 100)
#                       
#                   if row[6]=="+":
#                       seq_flanking=reference_fasta[int(start)-1-20:int(end)]
#                       seq_5_3=seq
#                       seq_5_3_flanking=seq_flanking
#                       operon_5=start
#                       operon_3=end
#                   elif row[6]=="-":
#                       seq_flanking=reference_fasta[int(start)-1:int(end)+20]
#                       seq_5_3=seq.reverse_complement()
#                       seq_5_3_flanking=seq_flanking.reverse_complement()
#                       operon_5=end
#                       operon_3=start
#                   operon_downstream_genes=[]
#                   ess_gene_operon=[]
#                   reference_genes.append({"gene_name":genename,"geneid":geneid,"start":start,"end":end,"strand":strand,"length":length,"seq":seq_5_3,"seq_flanking":seq_5_3_flanking,"GC_content":GC_content,"biotype":gene_biotype,"operon_5":operon_5,"operon_3":operon_3,"operon_downstream_genes":operon_downstream_genes,"ess_gene_operon":ess_gene_operon})   
    return error

def gRNA_sequences(seq,l,mingc,maxgc,gene,reference_fasta,PAM):  ## seq is sense strand sequence from 5' to 3', 
    PAM=PAM[::-1] # look for reverse complement sequence of PAM in sense strand
    reverse_complement={"G":"C","C":"G","A":"T","T":"A"}
    PAM_rev_com=""
    for bp in PAM:
        if bp =="N":
            PAM_rev_com=PAM_rev_com+"N"
        else:
            PAM_rev_com=PAM_rev_com+reverse_complement[bp]
    PAM_s_rc=PAM_rev_com.replace('N','[ACGT]') 
    PAMs=PAM_s_rc
    guide=[]   
    if seq[:20]== 'N'*20:
        p=20
    else:
        p=0
    seq=str(seq)
    while p < len(seq):
        matches=list(re.finditer(PAMs,seq[p:])) ### search on sense strand for PAM
        if len(matches)>0:
            PAM_match=matches[0]
            PAM_pos=PAM_match.start()+p
            if len(seq)-(PAM_pos+3) >= l:
                gRNA_PAM=str(Seq(PAM_match.group()).reverse_complement())
                target_gene_pos=PAM_pos+3-20 # -20 for flanking upstream 20nt, start from reference start/end position
                if gene['strand']=="+":                     
                    target_genome_pos=int(gene["start"])+target_gene_pos
                    strand="-"
                elif gene['strand']=="-":
                    target_genome_pos=int(gene["end"])-target_gene_pos-19
                    strand="+"
                target_sequence=Seq(seq[PAM_pos+3:PAM_pos+23])   
                gRNA_seq=target_sequence.reverse_complement()
                GC_content = float((target_sequence.count('G') + target_sequence.count('C'))) / len(target_sequence) * 100
                if GC_content >= mingc and GC_content <= maxgc: # limit the number of guides in each gene, check GC content and restriction sites
                    if gene['strand']=="+":
                        target_sequence_full_length=reference_fasta[target_genome_pos-1:target_genome_pos-1+l]
                        gRNA_full_length_reverse=str(target_sequence_full_length.reverse_complement())
                    else:
                        target_sequence_full_length=reference_fasta[target_genome_pos-1-(l-20):target_genome_pos+19]
                        gRNA_full_length_reverse=str(target_sequence_full_length)
                    if taskname=="":
                        gRNA_ID='result_'+str(target_gene_pos+1)
                    else:
                        gRNA_ID=taskname+'_'+str(target_gene_pos+1)        
                    guide.append({"SequenceID":gene['SequenceID'],"start":str(gene['start']),"end":str(gene['end']),"gene_strand":gene['strand'],"length":gene['length'],"gene_GC_content":gene["GC_content"],"gRNA_ID":gRNA_ID,"gene_pos":target_gene_pos,"genome_pos":target_genome_pos,"seq_20nt":str(gRNA_seq),"seq_full_length": gRNA_full_length_reverse,"PAM":gRNA_PAM,"gRNA_strand":strand,"gRNA_GC_content":'{:.2F}'.format(GC_content)})    
                                                    
            p=PAM_pos+1
        else:
            break
    return guide


def gRNA_search(targeting_genes):
#    if type(targeting_genes)==list:
#        genes=[]
#        for gene in targeting_genes:
#            if type(gene)==str:
#                for GENE in reference_genes:
#                    if GENE['gene_name']== gene:
#                        genes.append(GENE)
#            elif type(gene)==dict:
#                genes.append(gene)
#        library_guides={}
#        for gene in genes:
#            library_guides[gene['gene_name']+"_"+str(gene['start'])+"_"+str(gene['end'])]=gRNA_sequences(gene["seq_flanking"],l,mingc,maxgc,gene,reference_fasta,PAM)  #gene["geneid"]+"_"+gene["start"] for pseudogenes with same locus tag and name but different position
#    else:
    library_guides={}
    gene={'SequenceID':taskname,"start":1,"end":len(targeting_genes),"strand":"+","length":len(targeting_genes),"GC_content":float((targeting_genes.count('G') + targeting_genes.count('C'))) / len(targeting_genes) * 100}
    library_guides[taskname]=gRNA_sequences("N"*20+targeting_genes,l,mingc,maxgc,gene,targeting_genes,PAM)
    return library_guides


def self_encode(sequence):
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        if sequence[i] != 'N':
            integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

def dinucleotide(sequence):
    nts=['A','T','C','G']
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    encoded=np.zeros([(len(nts)**2)*(len(sequence)-1)],dtype=np.float64)
    N_warning=0
    for nt in range(len(sequence)-1):
        if sequence[nt]=='N' or sequence[nt+1]=='N':
            N_warning=1
            continue
        else:
            encoded[nt*len(nts)**2+dinucleotides.index(sequence[nt]+sequence[nt+1])]=1
    return encoded,N_warning

def consecutive_nt_calculation(sequence):
    maxlen=0
    for k,g in itertools.groupby(sequence):
        group=list(g)
        if len(group)>maxlen:
            maxlen=len(group)
    return maxlen


def MFE_RNA_RNA_hybridization(sequence1,sequence2):
    with open(output_file_name+"/MFE_hybridization.fasta","w") as MFE_hybridization_fasta:
        MFE_hybridization_fasta.write(">s1\n"+sequence1+"\n>s2\n"+sequence2+"\n")
    hybridization_file=open(output_file_name + '/hybridization.txt',"w")
    hybridization_fasta=open(output_file_name+"/MFE_hybridization.fasta",'r')
    subprocess.run(["RNAduplex2.4.14"],stdin=hybridization_fasta,stdout=hybridization_file)
    for line in open(output_file_name + '/hybridization.txt'):
        if ":" in line:
            MFE=line.split(":")[1].split("(")[1].split(")")[0]  
    subprocess.run(["rm",output_file_name + '/hybridization.txt',output_file_name+"/MFE_hybridization.fasta"])
    return MFE

def MFE_RNA_DNA_hybridization(sequence1,sequence2):
    with open(output_file_name+"/MFE_hybridization_DNA.fasta","w") as MFE_hybridization_fasta:
        MFE_hybridization_fasta.write(">s1_RNA\n"+sequence1+"\n>s2_DNA\n"+sequence2+"\n")  # first RNA and then DNA
    hybridization_file=open(output_file_name + '/hybridization.txt',"w")
    hybridization_fasta=open(output_file_name+"/MFE_hybridization_DNA.fasta",'r')
    subprocess.run(["RNAduplex2.1.9h"],stdin=hybridization_fasta,stdout=hybridization_file)
    for line in open(output_file_name + '/hybridization.txt'):
        if ":" in line:
            MFE=line.split(":")[1].split("(")[1].split(")")[0]  
    subprocess.run(["rm",output_file_name + '/hybridization.txt',output_file_name+"/MFE_hybridization_DNA.fasta"])
    return MFE

def MFE_folding(sequence):
    with open(output_file_name+"/MFE_folding.fasta","w") as MFE_folding_fasta:
        MFE_folding_fasta.write(">s\n"+sequence+"\n")
    folding_file=open(output_file_name + '/folding.txt',"w")
    subprocess.run(["RNAfold2.4.14","--noPS","-i",output_file_name+"/MFE_folding.fasta"],stdout=folding_file)
    for line in open(output_file_name + '/folding.txt'):
        if "-" in line or "+" in line or "0.00" in line:
            MFE=line.split("(")[-1].split(")")[0]
    subprocess.run(["rm",output_file_name + '/folding.txt',output_file_name+"/MFE_folding.fasta"])
    return MFE


def MachineLearning(guides):
    MachineLearning_ModelTraining()
    guides_df=MachineLearning_Transform(guides)
    guides_df=MachineLearning_Predict(guides_df)
    guides_df["distance_start_codon"] += 1
    chosen_header=["gRNA_ID","SequenceID","distance_start_codon","distance_start_codon_perc","guide_GC_content","coding_strand","seq_20nt","seq_full_length","PAM","predicted_log2FC","Warning"]
    if l == 20:
        chosen_header.remove("seq_full_length")
    if taskname=="":
        chosen_header.remove("SequenceID")
    if all(guides_df['Warning']==""):
        chosen_header.remove("Warning")
    guides_df=guides_df[chosen_header]
    guides_df=guides_df.astype({"distance_start_codon":int,"coding_strand":str,"distance_start_codon_perc":float})
    guides_df=guides_df.round({"distance_start_codon_perc":2,"predicted_log2FC":4})
    guides_df=guides_df.rename(columns={"distance_start_codon":"Distance to start codon (bp)","distance_start_codon_perc":"Distance to start codon relative to the sequence length (%)",\
                                "guide_GC_content":"GC content (%)","coding_strand":"If it targets coding strand","predicted_log2FC":"Activity score",\
                                "seq_20nt":"gRNA sequence (20nt)"})
    if "seq_full_length" in guides_df.columns.values.tolist():
        guides_df=guides_df.rename(columns={"seq_full_length":"gRNA sequence with desired length"})
    guides_df=guides_df.sort_values(by="Activity score",ascending=True)
    guides_df.to_csv("static/%s/"%t+"gRNAs.csv",sep='\t',index=False)
    return guides_df

def MachineLearning_ModelTraining():
    
    global estimator,headers
    headers=pickle.load(open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/CRISPRi_headers.sav','rb'))
    estimator=pickle.load(open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/CRISPRi_model.sav','rb'))

def SHAP(estimator,X_unscaled,X,headers,name):
    import shap
    # import scipy.cluster
    if os.path.isdir(name)==False:
        os.mkdir(name)
    # y = estimator.predict(X)
    X=pandas.DataFrame(X,columns=headers)
    X=X.astype(float)
    X=X[:3000]
    explainer = shap.TreeExplainer(estimator)
    # pickle.dump(explainer,open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/SHAPexplainer.sav','wb'))
    # explainer=pickle.load(open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/SHAPexplainer.sav','rb'))
    shap_values = explainer.shap_values(X)
    values=pandas.DataFrame(shap_values,columns=headers,index=map(str,X_unscaled['gRNA_ID'][:3000]))
    values.to_csv("static/"+name.split("/")[1]+"/shap_values.csv",index=True,sep='\t')
    if os.path.isdir(name)==False:
        os.mkdir(name)
    shap.save_html(name+'/all_force_plots.html',shap.force_plot(explainer.expected_value, shap_values, X,show= False))
    
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.yticks(fontsize='small')
    plt.subplots_adjust(left=0.25, top=0.95)
    plt.savefig(name+"/shap_value_bar.png",dpi=400)
    plt.close()
    plt.figure()
    shap.summary_plot(shap_values, X,show=False,max_display=10)
    plt.subplots_adjust(left=0.25, top=0.95)
    plt.yticks(fontsize='small')
    plt.savefig(name+"/shap_value.png",dpi=400)
    plt.close()
    
#    genes=list(set(X_unscaled['gene_name']))
    # D=pandas.DataFrame(shap_values,columns=headers)
    # values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    # values=values.sort_values(by=["shap_values"],ascending=False)
    # features=list(values["features"].head(20))
    # D=D[features]
    # D=D.transpose()
    # plt.figure()
    # d = scipy.spatial.distance.pdist(shap_values, 'sqeuclidean')
    # clustOrder = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.complete(d))
    # logging.info('cluster order: %s\n'%clustOrder)
    # sns.heatmap(D[clustOrder],yticklabels=1,xticklabels=False,cmap="vlag",vmin=-0.5,vmax=0.5,cbar_kws={'label': 'SHAP value'})
    # plt.subplots_adjust(left=0.35,right=0.9)
    # plt.xlabel('samples')
    # plt.savefig(name+'/heatmap.png',dpi=400)
    # plt.close()
    # D=D.transpose()
    # D['pos']=[ "%s_%s_%s"%(a,b,c) for a,b,c in zip(list(map(str,X_unscaled['SequenceID'])),list(map(int,X_unscaled['distance_start_codon'])),list(np.around(y,decimals=2)))]
    # D.set_index('pos',inplace=True)
    subprocess.run(['tar','cvzf','static/%s/SHAP_plots.tar.gz'%t,name])     
    
def MachineLearning_Predict(guides_df):
    guide_df_sub=guides_df[headers]
    prediction=estimator.predict(guide_df_sub)
    guides_df['predicted_log2FC']=prediction
    if SHAP_plots == "yes":
        SHAP(estimator,guides_df,guide_df_sub,headers,'static/%s/SHAP_plots'%t)
    cols=guides_df.columns.values.tolist()
    cols=['predicted_log2FC']+cols[:-1]
    guides_df=guides_df[cols]
    return guides_df 

def MachineLearning_Transform(guides):
    index=range(len(guides))
    column=["gRNA_ID","SequenceID","start","end","gene_strand","length","genome_pos","seq_20nt","seq_full_length","PAM","Warning","coding_strand"]+headers
    guides_df=pandas.DataFrame(index=index,columns=column)
    # subprocess.run(["makeblastdb","-in",blast_db,"-dbtype","nucl"],stdout=subprocess.DEVNULL)
    for k in range(len(guides)):
        guide=guides[k]
        transformed_guide={}
        sequence=guide['seq_20nt']
        target_seq=str(Seq(sequence).reverse_complement())
        PAM=guide['PAM']
        guide_GC_content='{:.2f}'.format((sequence.count('G') + sequence.count('C')) / len(sequence) * 100)
        if guide['gRNA_strand']=="+":
            genome_pos_5_end=int(guide['genome_pos'])
            genome_pos_3_end=genome_pos_5_end+len(sequence)-1
            guide_strand=1  
            if genome_pos_5_end < 5:
                sequence_30nt='N'* (5-genome_pos_5_end)+str(reference_fasta[0:genome_pos_3_end+6])
            else:
                sequence_30nt=reference_fasta[genome_pos_5_end-5:genome_pos_3_end+6]
    
        elif guide['gRNA_strand']=="-":
            genome_pos_3_end=int(guide['genome_pos'])
            genome_pos_5_end=genome_pos_3_end+len(sequence)-1
            guide_strand=0
            if genome_pos_3_end < 7:
                sequence_30nt='N'* (7-genome_pos_3_end)+str(reference_fasta[0:genome_pos_5_end+4])
            else:
                sequence_30nt=reference_fasta[genome_pos_3_end-7:genome_pos_5_end+4].reverse_complement()
        PAM_encoded=self_encode(PAM)
        sequence_encoded=self_encode(sequence)
        dinucleotide_encoded,N_warning=dinucleotide(sequence_30nt)
        if N_warning == 0:
            N_warning = ""
        else:
            N_warning = "N is found in extended 30nt sequence"
        sequences=np.hstack((sequence_encoded,PAM_encoded,dinucleotide_encoded))
        PAM_len=len(PAM)
        sequence_len=len(sequence)
        dinucleotide_len=len(sequence_30nt)
        sequence_header=[]
        nts=['A','T','C','G']
        for i in range(sequence_len):
            for j in range(len(nts)):
                sequence_header.append('sequence_%s_%s'%(i+1,nts[j]))
        for i in range(PAM_len):
            for j in range(len(nts)):
                sequence_header.append('PAM_%s_%s'%(i+1,nts[j]))
        items=list(itertools.product(nts,repeat=2))
        dinucleotides=list(map(lambda x: x[0]+x[1],items))
        for i in range(dinucleotide_len-1):
            for dint in dinucleotides:
                sequence_header.append(dint+str(i+1)+str(i+2)) 
        sequence_dict=dict(zip(sequence_header,sequences))
        ### gene info and distance to start codon/operon    
        gene_length=guide['length']
        if guide['gene_strand']=='+':
            gene_strand=1
        else:
            gene_strand=0
        if gene_strand==guide_strand:
            coding_strand="No"
        else:
            coding_strand="Yes"
        distance_start_codon=guide['gene_pos']
        distance_start_codon_perc=distance_start_codon/gene_length*100
        #homopolymers"
        homopolymers=consecutive_nt_calculation(sequence)
        
        #"MFE_hybrid_full","MFE_hybrid_seed","MFE_mono_guide","MFE_dimer_guide"
        MFE_hybrid_full=MFE_RNA_DNA_hybridization(sequence.replace("T","U"),target_seq)
        MFE_hybrid_seed=MFE_RNA_DNA_hybridization(sequence[-8:].replace("T","U"),target_seq[:8])
        MFE_homodimer_guide=MFE_RNA_RNA_hybridization(sequence,sequence)
        MFE_monomer_guide=MFE_folding(sequence)
        
        transformed_guide={'Warning':N_warning,'coding_strand':coding_strand,  'distance_start_codon':distance_start_codon,\
                           'distance_start_codon_perc':distance_start_codon_perc,  'guide_GC_content':guide_GC_content,\
                           "MFE_homodimer_guide":MFE_homodimer_guide,"MFE_monomer_guide":MFE_monomer_guide,"MFE_hybrid_full":MFE_hybrid_full,"MFE_hybrid_seed":MFE_hybrid_seed,
                           "homopolymers":homopolymers}
        guide.update(transformed_guide)
        guide.update(sequence_dict)
        guide_df=pandas.DataFrame(guide,index=[k])
        guides_df.update(guide_df)
    return guides_df

@app.errorhandler(404)
def error_404(error):
	return render_template('errors/404.html'), 404


@app.errorhandler(403)
def error_403(error):
	return render_template('errors/403.html'), 403


@app.errorhandler(500)
def error_500(error):
	return render_template('errors/500.html'), 500

if __name__ == '__main__':
    app.run(debug=False)
    
