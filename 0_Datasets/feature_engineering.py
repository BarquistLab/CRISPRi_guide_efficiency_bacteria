#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:16:59 2019

@author: yanying
"""

import numpy as np
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import subprocess
import os
import time 
import datetime
import logging
import itertools
import pandas
import sys


start_time=time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s gRNA CSV file [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to compute extensive features for gRNAs. Including: 4 thermodynamic features (MFE features), gene information (gene essentiality, operon position, number of downstream genes in the operon, number of downstream essential genes in the operon, etc.)

For the input file, CSV format is accepted. Columns with name 'seq' for gRNA sequence and 'geneid' for the locus tag (in the format b0001) are required.  

2 versions of Vienna packages are required. For installation, please check the information in the repository.

Example: python feature_engineering.py test.csv -o test
                  """)
parser.add_argument("library", help="gRNA library csv file")
parser.add_argument("-o", "--output", default="results", help="output file name")
parser.add_argument("-l", "--expression_level_lower", default="01", 
                    help="""the LOWER boundary of the range of OD value for expression level, which cannot be higher than the upper boundary
01: OD 0.1
02: OD 0.2
03: OD 0.3
04: OD 0.4
08: OD 0.8
14: OD 1.4
16: OD 1.6
15: 15min after stationary phase
30: 30min after stationary phase
180: 180min after stationary phase
default: 01
""")
parser.add_argument("-u", "--expression_level_upper", default="14", 
                    help="""the UPPER boundary of the range of OD value for expression level, which cannot be lower than the lower boundary
01: OD 0.1
02: OD 0.2
03: OD 0.3
04: OD 0.4
08: OD 0.8
14: OD 1.4
16: OD 1.6
15: 15min after stationary phase
30: 30min after stationary phase
180: 180min after stationary phase
default: 14
""")
args = parser.parse_args()
library_df=args.library
output_file_name = args.output
expression_level_lower=args.expression_level_lower
expression_level_upper=args.expression_level_upper

### predefiined files for reference genome
operon_file="OperonSet.txt"
expression_level_file="Expression_level_TPM.csv"
expression_levels=pandas.read_csv(expression_level_file,sep='\t',index_col=0)
timpepoints=expression_levels.columns.values.tolist()
OD_range=timpepoints[timpepoints.index("WT_"+expression_level_lower):timpepoints.index("WT_"+expression_level_upper)+1]
if len(OD_range)==0:
    print("Please reselect the boundaries for the expression level values")
    sys.exit()
expression_levels=expression_levels[OD_range]
reference_fasta_file="NC_000913.3.fasta"
reference_gff="NC_000913.3.gff3"

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


def main():
    open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
    open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
    Keio_essential_genes=['alaS', 'btuB', 'coaA', 'coaE', 'djlB', 'dnaG', 'folP', 'glmM', 'glyS', 'groL', 'hemE', 'ileS', 'lptB', 'parC', 'polA', 'prfB', 'priB', 'rho', 'rplK', 'rpoD', 'rpsU', 'tpr', 'yiaD', 'accA', 'accB', 'accC', 'accD', 'acpP', 'acpS', 'adk', 'alsK', 'argS', 'asd', 'asnS', 'aspS', 'bamA', 'bamD', 'bcsB', 'birA', 'can', 'cca', 'cdsA', 'chpS', 'coaD', 'csrA', 'cydA', 'cydC', 'cysS', 'dapA', 'dapB', 'dapD', 'dapE', 'def', 'degS', 'der', 'dfp', 'dicA', 'dnaA', 'dnaB', 'dnaC', 'dnaE', 'dnaN', 'dnaX', 'dut', 'dxr', 'dxs', 'eno', 'entD', 'era', 'erpA', 'fabA', 'fabB', 'fabD', 'fabG', 'fabI', 'fabZ', 'fbaA', 'ffh', 'fldA', 'fmt', 'folA', 'folC', 'folD', 'folE', 'folK', 'frr', 'ftsA', 'ftsB', 'ftsE', 'ftsH', 'ftsI', 'ftsK', 'ftsL', 'ftsN', 'ftsQ', 'ftsW', 'ftsX', 'ftsY', 'ftsZ', 'fusA', 'gapA', 'glmS', 'glmU', 'glnS', 'gltX', 'glyQ', 'gmk', 'gpsA', 'groS', 'grpE', 'gyrA', 'gyrB', 'hemA', 'hemB', 'hemC', 'hemD', 'hemG', 'hemH', 'hemL', 'hisS', 'holA', 'holB', 'igaA', 'infA', 'infB', 'infC', 'ispA', 'ispB', 'ispD', 'ispE', 'ispF', 'ispG', 'ispH', 'kdsA', 'kdsB', 'lepB', 'leuS', 'lexA', 'lgt', 'ligA', 'lnt', 'lolA', 'lolB', 'lolC', 'lolD', 'lolE', 'lptA', 'lptC', 'lptD', 'lptE', 'lptF', 'lptG', 'lpxA', 'lpxB', 'lpxC', 'lpxD', 'lpxH', 'lpxK', 'lspA', 'map', 'mazE', 'metG', 'metK', 'minD', 'minE', 'mlaB', 'mqsA', 'mraY', 'mrdA', 'mrdB', 'mreB', 'mreC', 'mreD', 'msbA', 'mukB', 'mukE', 'mukF', 'murA', 'murB', 'murC', 'murD', 'murE', 'murF', 'murG', 'murI', 'murJ', 'nadD', 'nadE', 'nadK', 'nrdA', 'nrdB', 'nusA', 'nusG', 'obgE', 'orn', 'parE', 'pgk', 'pgsA', 'pheS', 'pheT', 'plsB', 'plsC', 'ppa', 'prfA', 'prmC', 'proS', 'prs', 'psd', 'pssA', 'pth', 'purB', 'pyrG', 'pyrH', 'racR', 'ribA', 'ribB', 'ribC', 'ribD', 'ribE', 'ribF', 'rnc', 'rne', 'rnpA', 'rplB', 'rplC', 'rplD', 'rplE', 'rplF', 'rplJ', 'rplL', 'rplM', 'rplN', 'rplO', 'rplP', 'rplQ', 'rplR', 'rplS', 'rplT', 'rplU', 'rplV', 'rplW', 'rplX', 'rpmA', 'rpmB', 'rpmC', 'rpmD', 'rpmH', 'rpoA', 'rpoB', 'rpoC', 'rpoE', 'rpoH', 'rpsA', 'rpsB', 'rpsC', 'rpsD', 'rpsE', 'rpsG', 'rpsH', 'rpsJ', 'rpsK', 'rpsL', 'rpsN', 'rpsP', 'rpsR', 'rpsS', 'rseP', 'rsmI', 'secA', 'secD', 'secE', 'secF', 'secM', 'secY', 'serS', 'spoT', 'ssb', 'suhB', 'tadA', 'tdcF', 'thiL', 'thrS', 'tilS', 'tmk', 'topA', 'trmD', 'trpS', 'tsaB', 'tsaC', 'tsaD', 'tsaE', 'tsf', 'tyrS', 'ubiA', 'ubiB', 'ubiD', 'ubiV', 'uppS', 'valS', 'waaA', 'waaU', 'wzyE', 'yabQ', 'yafF', 'yagG', 'yceQ', 'ydfB', 'ydiL', 'yefM', 'yejM', 'yhhQ', 'yibJ', 'yidC', 'yihA', 'ymfK', 'yqgD', 'yqgF', 'zipA']
    ## import operons
    operons={}
    for line in open(operon_file):
        if "#" not in line:
            row=line.replace("\n","").split()
            if int(row[4])>1:
                operons.update({row[0]:{"left":int(row[1]),"right":int(row[2]),"strand":row[3],"Genes_in_operon":row[5].split(",")}})
    ## import reference annotation
    fasta_sequences = SeqIO.parse(open(reference_fasta_file),'fasta')    
    for fasta in fasta_sequences:  # input reference genome
        reference_fasta=fasta.seq 
    GFF=dict()
    for line in open(reference_gff):
        if "#" not in line and "Gene;gene" in line:
            line=line.replace("\n","")
            row=line.split()
            geneid=row[8].split(";")[0].split("-")[1]
            genename=row[8].split(";")[4].split("=")[1]
            length=int(row[4])-int(row[3])+1
            gene_biotype=row[8].split(";")[5].split("=")[1]
            seq=str(reference_fasta[int(row[3])-1:int(row[4])])
            if genename in Keio_essential_genes:
                gene_essentiality=1
            else:
                gene_essentiality=0
            if row[6]=="+":
                strand=1
#                seq_5_3=seq    
                start=int(row[3])
                end=int(row[4])      
            elif row[6]=="-":
                strand=-1
#                seq_5_3=seq.reverse_complement()   
                start=int(row[4])
                end=int(row[3])    
            GC_content = round((seq.count('G') + seq.count('C')) / len(seq) * 100,2)
            operon_5=start
            operon_3=end
            operon_downstream_genes=[]
            ess_gene_operon=[]
            for key in operons.keys():   
                if genename in operons[key]['Genes_in_operon']:
                    if operons[key]['strand']=='forward':
                        operon_5=operons[key]['left']
                        operon_3=operons[key]['right']
                    else:
                        operon_5=operons[key]['right']
                        operon_3=operons[key]['left']
                    operon_downstream_genes=operons[key]['Genes_in_operon'][operons[key]['Genes_in_operon'].index(genename)+1:]
                    ess_gene_operon=[item for item in operons[key]['Genes_in_operon'][operons[key]['Genes_in_operon'].index(genename)+1:] if item in Keio_essential_genes]
                    break
            
            GFF.update({geneid:{"genename":genename,"geneid":geneid,"start":start,"end":end,"GC_content":GC_content,"strand":strand,"length":length,"biotype":gene_biotype,"gene_essentiality":gene_essentiality,"operon_5":operon_5,"operon_3":operon_3,"operon_downstream_genes":operon_downstream_genes,"ess_gene_operon":ess_gene_operon}})
    
    
    
    ### import library 
    library=pandas.read_csv(library_df,sep="\t",dtype={'geneid':str,'seq':str})
    for i in library.index:
        sequence=library['seq'][i]
        geneid=library['geneid'][i]
        
        if geneid not in GFF.keys():
            print(geneid,"not in NC_000913.3.gff3 file, please check the input. The example for geneid: b0001" )
            library.at[i,'Error']="Invalid gene ID"
            continue
        else:
            gene=GFF[geneid]
        library.at[i,'genename']=gene["genename"]
        library.at[i,'gene_biotype']=gene['biotype']
        library.at[i,'gene_essentiality']=gene['gene_essentiality']
        library.at[i,'gene_length']=gene['length']
        library.at[i,'gene_GC_content']=gene['GC_content']
        library.at[i,'gene_strand']=gene["strand"]
        if gene["strand"] ==1:
            distance_operon=gene['start']-gene['operon_5']
        else:
            distance_operon=gene['operon_5']-gene['start']
        library.at[i,'distance_operon']=distance_operon
        library.at[i,'distance_operon_perc']=distance_operon/(abs(gene['operon_5']-gene['operon_3']))*100
        library.at[i,'operon_downstream_genes']=len(gene['operon_downstream_genes'])   
        library.at[i,'ess_gene_operon']=len(gene['ess_gene_operon'])
        
        try:
            library.at[i,'gene_expression_min']=min(expression_levels.loc[geneid])
            library.at[i,'gene_expression_max']=max(expression_levels.loc[geneid])
        except KeyError:
            library.at[i,'gene_expression_min']=np.nan
            library.at[i,'gene_expression_max']=np.nan            
        target_seq=str(Seq(sequence).reverse_complement())
        library.at[i,'guide_GC_content']='{:.2f}'.format((sequence.count('G') + sequence.count('C')) / len(sequence) * 100)
        #"MFE_hybrid_full","MFE_hybrid_seed","MFE_mono_guide","MFE_dimer_guide"
        library.at[i,'MFE_hybrid_full']=MFE_RNA_DNA_hybridization(sequence.replace("T","U"),target_seq)
        library.at[i,'MFE_hybrid_seed']=MFE_RNA_DNA_hybridization(sequence[-8:].replace("T","U"),target_seq[:8])
        library.at[i,'MFE_homodimer_guide']=MFE_RNA_RNA_hybridization(sequence,sequence)
        library.at[i,'MFE_monomer_guide']=MFE_folding(sequence)
        #"consective_nts"
        library.at[i,'homopolymers']=consecutive_nt_calculation(sequence)
    library.to_csv(output_file_name+"/gRNAs.csv",sep='\t',index=False)   
    subprocess.run(["rm",output_file_name+"/MFE_hybridization.fasta",output_file_name + '/hybridization.txt',output_file_name+"/MFE_hybridization_DNA.fasta"]) 
if __name__ == '__main__':
    logging_file= output_file_name + '/log.txt'
    logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
    logging.info("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
