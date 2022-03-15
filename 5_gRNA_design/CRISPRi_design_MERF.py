#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:45:36 2019

@author: yanying
"""

import matplotlib.pyplot as plt
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
import regex as re
import os
import time 
import logging
import itertools
import pandas 
import pickle
import subprocess
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')
start_time=time.time()
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to design gRNAs for FASTA input files and predict gRNA efficiency using trained MERF model.

Example: python CRISPRi_design_MERF.py test.fasta -o test

It also supports to select multiple genes in a genome for design by input reference genome fasta as FASTA input and the gff3 file for the reference genome.   

Example: python CRISPRi_design_MERF.py NC_000913.3.fasta -gff NC_000913.3.gff3 -targeting_genes purA,b1131 -shap no -o purs
                  """)
parser.add_argument("fasta", help="fasta file")
parser.add_argument("-gff", default=None,help="gff file")
parser.add_argument("-targeting_genes",default=None,help="gene name or gene ID of targeting genes (such as thrL, b0001), multiple genes separated by ','. If None, target all genes in gff file. Default: None")
parser.add_argument("-o", "--output", default="results", help="output folder name. Default: results")
parser.add_argument("-l","--length", type=int, default=20, help="Length of gRNA (bp). Default: 20")
parser.add_argument("-maxgc", type=float, default=85, help="Maximal GC content of gRNA. Default: 85")
parser.add_argument("-mingc", type=float, default=30, help="Minimal GC content of gRNA. Default: 30")
parser.add_argument("-b","--biotype", type=str, default="all", help="Targeting gene type. all/protein_coding/rRNA/tRNA/ncRNA/pseudogene. Default: all")
parser.add_argument("-PAM", type=str,default='NGG', help="PAM sequence. Default: NGG")
parser.add_argument("-shap", type=str,default='no', help="If output SHAP value summary plots, yes/no. (Calculating SHAP values takes much longer.) Default: no")
args = parser.parse_args()
fasta=args.fasta
genome_gff=args.gff
targeting_genes=args.targeting_genes
if targeting_genes != None:
    targeting_genes=targeting_genes.split(",")
        
output_file_name = args.output
l=args.length
maxgc = args.maxgc 
mingc = args.mingc
biotype=args.biotype
PAM=args.PAM
SHAP_plots=args.shap
if biotype == "all":
    biotypes=["protein_coding","rRNA","tRNA","ncRNA","pseudogene"]
else:
    if biotype not in ["protein_coding","rRNA","tRNA","ncRNA","pseudogene"]:
        print("Please choose ONE of the biotype or all..\nAbort.")   
        sys.exit()
    biotypes=[biotype]

if SHAP_plots not in ['yes','no']:
    print("Please input valid choice for -shap..\nAbort.")   
    sys.exit()
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
def ReferenceGenomeInfo(fasta,genome_gff):
    if genome_gff == None:
        global tasknames
        tasknames=dict()
        fasta_sequences = list(SeqIO.parse(open(fasta),'fasta'))
        if len(fasta_sequences) == 0:
            print("Error: No sequences found in input file. Please input file in FASTA format.")
            sys.exit()
        for fasta in fasta_sequences:  # input reference genome
            tasknames.update({fasta.id:fasta.seq })
        
    
    if genome_gff != None:
        global reference_genes,reference_FASTA
        fasta_sequences = list(SeqIO.parse(open(fasta),'fasta'))
        for fasta in fasta_sequences:  # input reference genome
            reference_fasta=fasta.seq 
        reference_FASTA=Seq(re.sub('[^ATCG]','N',str(reference_fasta).upper()))
        reference_genes=[]
        for line in open(genome_gff):
           if "#" not in line and "Gene;gene" in line: ## input reference gff
               line=line.replace("\n","")    
               row=line.split("\t")
               gene_biotype=row[8].split("gene_biotype=")[1].split(";")[0]
               if gene_biotype in biotypes:
                   geneid=row[8].split(";")[0].split("-")[1]
                   genename=row[8].split("Name=")[1].split(";")[0]
                   start=int(row[3])
                   end=int(row[4])
                   strand=row[6]
                   length=int(row[4])-int(row[3])+1
                   seq=reference_FASTA[int(start)-1:int(end)]
                   GC_content = '{:.2f}'.format((seq.count('G') + seq.count('C')) / len(seq) * 100)
                   if row[6]=="+":
                       seq_flanking=reference_FASTA[int(start)-1-20:int(end)]
                       seq_5_3=seq
                       seq_5_3_flanking=seq_flanking
                   elif row[6]=="-":
                       seq_flanking=reference_FASTA[int(start)-1:int(end)+20]
                       seq_5_3=seq.reverse_complement()
                       seq_5_3_flanking=seq_flanking.reverse_complement()
                   reference_genes.append({"gene_name":genename,"geneid":geneid,"start":start,"end":end,"strand":strand,"length":length,"seq":seq_5_3,"seq_flanking":seq_5_3_flanking,"GC_content":GC_content,"biotype":gene_biotype})   


def gRNA_sequences(seq,l,mingc,maxgc,gene,reference_fasta,PAM,taskname):  ## seq is sense strand sequence from 5' to 3', 
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
    guide=list()
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
    if type(targeting_genes)==list:
        genes=[]
        for gene in targeting_genes:
            if type(gene)==str:
                for GENE in reference_genes:
                    if GENE['gene_name']== gene or GENE['geneid']==gene:
                        genes.append(GENE)
        library_guides={}
        for gene in genes:
            gene.update({'SequenceID':gene['gene_name']})
            library_guides[gene['gene_name']+"_"+str(gene['start'])+"_"+str(gene['end'])]=gRNA_sequences(gene["seq_flanking"],l,mingc,maxgc,gene,reference_FASTA,PAM,gene['gene_name']+"_"+str(gene['start'])+"_"+str(gene['end']))  #gene["geneid"]+"_"+gene["start"] for pseudogenes with same locus tag and name but different position
            logging.info("The number of gRNAs for %s: %s"%(gene['gene_name'], len(library_guides[gene['gene_name']+"_"+str(gene['start'])+"_"+str(gene['end'])])))
            print("Done designing gRNAs for %s, number of gRNAs: %s"%(gene['gene_name'], len(library_guides[gene['gene_name']+"_"+str(gene['start'])+"_"+str(gene['end'])])))
    else:
        tasknames=targeting_genes
        library_guides={}
        for taskname in tasknames.keys():
            gene={'SequenceID':taskname,"start":1,"end":len(tasknames[taskname]),"strand":"+","length":len(tasknames[taskname]),"GC_content":float((tasknames[taskname].count('G') + tasknames[taskname].count('C'))) / len(tasknames[taskname]) * 100}
            library_guides[taskname]=gRNA_sequences("N"*20+tasknames[taskname],l,mingc,maxgc,gene,tasknames[taskname],PAM,taskname)
            logging.info("The number of gRNAs for %s: %s"%(taskname, len(library_guides[taskname])))
            print("Done designing gRNAs for %s, number of gRNAs: %s"%(taskname, len(library_guides[taskname])))
    return library_guides

###functions to encode sequence features and calculate features necessary for prediction of gRNA efficiency scores
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

## predict gRNA efficiency
def MachineLearning(guides):
    MachineLearning_ModelTraining()
    guides_df=MachineLearning_Transform(guides)
    guides_df=MachineLearning_Predict(guides_df)
    guides_df["distance_start_codon"] += 1
    chosen_header=["gRNA_ID","SequenceID","distance_start_codon","distance_start_codon_perc","guide_GC_content","coding_strand","seq_20nt","seq_full_length","PAM","predicted_log2FC","Warning"]
    if l == 20:
        chosen_header.remove("seq_full_length")
    if all(guides_df['Warning']==""):
        chosen_header.remove("Warning")
    guides_df=guides_df[chosen_header]
    guides_df=guides_df.astype({"distance_start_codon":int,"coding_strand":str,"distance_start_codon_perc":float})
    guides_df=guides_df.round({"distance_start_codon_perc":2,"predicted_log2FC":4})
    guides_df=guides_df.rename(columns={"distance_start_codon":"Distance to start codon (bp)","distance_start_codon_perc":"Distance to start codon relative to the sequence length (%)",\
                                "guide_GC_content":"GC content (%)","coding_strand":"coding strand","predicted_log2FC":"Activity score",\
                                "seq_20nt":"gRNA sequence (20nt)"})
    if "seq_full_length" in guides_df.columns.values.tolist():
        guides_df=guides_df.rename(columns={"seq_full_length":"gRNA sequence with desired length"})
    guides_df=guides_df.sort_values(by="Activity score",ascending=True)
    guides_df.to_csv(output_file_name+"/"+"gRNAs.csv",sep='\t',index=False)
    return guides_df

def MachineLearning_ModelTraining():
    
    global estimator,headers
    headers=pickle.load(open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/CRISPRi_headers.sav','rb'))
    estimator=pickle.load(open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/CRISPRi_model.sav','rb'))

    
def MachineLearning_Transform(library):
    guides=[]
    for gene in library.keys():
        for guide in library[gene]:
            guides.append(guide)
    index=range(len(guides))
    if len(guides)==0:
        print("Error: No gRNA was found")
        sys.exit()
    column=["gRNA_ID","SequenceID","start","end","gene_strand","length","genome_pos","seq_20nt","seq_full_length","PAM","Warning","coding_strand"]+headers
    guides_df=pandas.DataFrame(index=index,columns=column)
    for k in range(len(guides)):
        guide=guides[k]
        transformed_guide={}
        sequence=guide['seq_20nt']
        target_seq=str(Seq(sequence).reverse_complement())
        PAM=guide['PAM']
        guide_GC_content='{:.2f}'.format((sequence.count('G') + sequence.count('C')) / len(sequence) * 100)
        if genome_gff == None:
            reference_fasta=tasknames[guide['SequenceID']]
        else:
            reference_fasta=reference_FASTA
        if guide['gRNA_strand']=="+":
            genome_pos_5_end=int(guide['genome_pos'])
            genome_pos_3_end=genome_pos_5_end+len(sequence)-1
            guide_strand=1  
            if genome_pos_5_end < 5:
                sequence_30nt='N'* (5-genome_pos_5_end)+str(reference_fasta[0:genome_pos_3_end+6])
            else:
                sequence_30nt=reference_fasta[genome_pos_5_end-5:genome_pos_3_end+6]
            if len(sequence_30nt)<30:
                sequence_30nt=sequence_30nt+"N"*(30-len(sequence_30nt))
        elif guide['gRNA_strand']=="-":
            genome_pos_3_end=int(guide['genome_pos'])
            genome_pos_5_end=genome_pos_3_end+len(sequence)-1
            guide_strand=0
            if genome_pos_3_end < 7:
                sequence_30nt='N'* (7-genome_pos_3_end)+str(reference_fasta[0:genome_pos_5_end+4])
            else:
                sequence_30nt=reference_fasta[genome_pos_3_end-7:genome_pos_5_end+4].reverse_complement()
            if len(sequence_30nt)<30:
                sequence_30nt=sequence_30nt+"N"*(30-len(sequence_30nt))
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
    
def MachineLearning_Predict(guides_df):
    guide_df_sub=guides_df[headers]
    prediction=estimator.predict(guide_df_sub)
    guides_df['predicted_log2FC']=prediction
    if SHAP_plots == "yes":
        print("Start SHAP interpretation at %s"%time.asctime())
        SHAP(estimator,guides_df,guide_df_sub,headers,output_file_name+'/SHAP_plots')
    cols=guides_df.columns.values.tolist()
    cols=['predicted_log2FC']+cols[:-1]
    guides_df=guides_df[cols]
    return guides_df 

def SHAP(estimator,X_unscaled,X,headers,name):
    import shap
    if os.path.isdir(name)==False:
        os.mkdir(name)
    X=pandas.DataFrame(X,columns=headers)
    X=X.astype(float)
    X=X[:3000]
    explainer = shap.TreeExplainer(estimator)
    # pickle.dump(explainer,open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/SHAPexplainer.sav','wb'))
    # explainer=pickle.load(open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/saved_model/SHAPexplainer.sav','rb'))
    shap_values = explainer.shap_values(X)
    values=pandas.DataFrame(shap_values,columns=headers,index=map(str,X_unscaled['gRNA_ID'][:3000]))
    values.to_csv(name+"/shap_values.csv",index=True,sep='\t')
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
if __name__ == '__main__':
    logging_file= output_file_name + '/log.txt'
    logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info("Python script: %s\n"%sys.argv[0])
    logging.info("Parsed arguments: %s\n"%args)
    print("Start designing at %s"%time.asctime())
    ReferenceGenomeInfo(fasta,genome_gff)
    if targeting_genes != None and genome_gff == None:
        print('Error: GFF file must be uploaded for selecting targeting genes')
        sys.exit()
    if targeting_genes != None and genome_gff != None:
        
        library_guides=gRNA_search(targeting_genes)
    elif targeting_genes == None and genome_gff != None:
        library_guides=gRNA_search(reference_genes)
    elif targeting_genes == None and genome_gff == None:
        library_guides=gRNA_search(tasknames) 
    if len(library_guides)==0:
        print("Error: No gRNA was found")
        sys.exit()
    print("Start gRNA efficiency prediction at %s"%time.asctime())
    MachineLearning(library_guides)
    logging.info("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))
    print("Done at %s"%time.asctime())
    
