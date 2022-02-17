############################################
# imports
############################################

import pandas as pd
import numpy as np

from Bio import SeqIO
import itertools
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold


############################################
# load tabular data
############################################

def preprocess_input_data(file):
    data = pd.read_csv(file,header=0, index_col=0, sep="\t")
    data = data[data['gene_essentiality']==1]
    data = data[data['coding_strand']==1]
    data = data[data['intergenic']==0]
    data["gene_expression_min"] = np.log2(data["gene_expression_min"]+1)
    #data["gene_expression_min"] = data["gene_expression_min"]
    data["gene_expression_max"] = np.log2(data["gene_expression_max"]+1)
    #data["gene_expression_max"] = data["gene_expression_max"]
    data = data.dropna().reset_index(drop=True)
    return data


############################################
# filter data set
############################################

def get_high_guide_genes(data):
    
    gene_count = data["geneid"].value_counts().to_frame()
    median_log2FC = data[["log2FC","geneid"]].groupby(['geneid']).median()
    
    for g in tqdm(data["geneid"].unique()):
        data.loc[data.geneid==g,"number_guides"]=gene_count.loc[g,"geneid"]
        data.loc[data.geneid==g,"gene_median"]=median_log2FC.loc[g,"log2FC"]
        
    # Pasteur model filtering
    index = data[(data["number_guides"] > 4) & (data["gene_median"] < -2)].index
    genes_to_keep = data.loc[index]["geneid"].unique()
    return genes_to_keep


############################################

def filter_by_variance(data, var_thr):

    constant_filter = VarianceThreshold(threshold=var_thr)
    constant_filter.fit(data)
    
    features_to_keep = data.columns[constant_filter.get_support()]
    
    return features_to_keep


############################################
# calculate sequence features
############################################

def get_sequence(data,file_genome,upstream=20,downstream=20):
    
    fasta_sequences = SeqIO.parse(open(file_genome),'fasta')
    
    for fasta in fasta_sequences:  # input reference genome
        reference_fasta = fasta.seq 
    
    extended_seq = []
    
    for i in data.index.values:

        if data["genome_pos_5_end"][i] > data["genome_pos_3_end"][i]:
            start = max(0,data["genome_pos_3_end"][i] - 1 - downstream)
            end = data["genome_pos_5_end"][i] + upstream
            start = int(start)
            end = int(end)
            extended_seq.append(str(reference_fasta[start:end].reverse_complement()))
        else:
            start = max(0,data["genome_pos_5_end"][i] - 1 - upstream)
            end = data["genome_pos_3_end"][i] + downstream
            start = int(start)
            end = int(end)
            extended_seq.append(str(reference_fasta[start:end]))
        
    return extended_seq


############################################
# encode ML sequence features
############################################

def encode_nucleotides(sequence):
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    encoded_sequence = integer_encoded.flatten()
    return encoded_sequence


############################################

def encode_dinucleotide(sequence):
    nts=['A','T','C','G']
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    encoded_sequence = np.zeros([(len(nts)**2)*(len(sequence)-1)],dtype=np.float64)
    for nt in range(len(sequence)-1):
        if sequence[nt]=='N' or sequence[nt+1] =='N':
            continue
        encoded_sequence[nt*len(nts)**2+dinucleotides.index(sequence[nt]+sequence[nt+1])]=1
    return encoded_sequence


############################################

def create_sequence_header(PAM_len,sequence_len,dinucleotide_len):

    nts=['A','T','C','G']
    sequence_header = []

    for i in range(PAM_len):
        for j in range(len(nts)):
            sequence_header.append('PAM_%s_%s'%(i+1,nts[j]))

    for i in range(sequence_len):
        for j in range(len(nts)):
            sequence_header.append('sequence_%s_%s'%(i+1,nts[j]))

    items = list(itertools.product(nts,repeat=2))
    dinucleotides = list(map(lambda x: x[0]+x[1],items))
    for i in range(dinucleotide_len-1):
        for dint in dinucleotides:
            sequence_header.append('sequence_%s_%s'%(str(i+1)+"-"+str(i+2),dint))
    
    return sequence_header


############################################

def one_hot_encode_ML(data):
    pam = pd.DataFrame.from_records(data.apply(lambda row : encode_nucleotides(row['PAM']), axis = 1))
    sequence_20nt = pd.DataFrame.from_records(data.apply(lambda row : encode_nucleotides(row['sequence']), axis = 1))
    #sequence_30nt = pd.DataFrame.from_records(data.apply(lambda row : encode_dinucleotide(row['sequence_30nt']), axis = 1))
    sequence_40nt = pd.DataFrame.from_records(data.apply(lambda row : encode_dinucleotide(row['sequence_40nt']), axis = 1))
    
    one_hot_encoded_sequences = pd.concat([pam,sequence_20nt,sequence_40nt],axis=1)
    one_hot_encoded_sequences.columns = create_sequence_header(3,20,40)
    #one_hot_encoded_sequences["geneid"] = data["geneid"]

    return one_hot_encoded_sequences


############################################
# encode DL sequence features
############################################

def encode_sequence(sequence):
   
    alphabet = 'AGCT'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    #encoded_sequence_old = tf.keras.utils.to_categorical(integer_encoded, num_classes=4)
    encoded_sequence = np.eye(4)[integer_encoded]
    return encoded_sequence


############################################

def one_hot_encode_DL(data):
  
    pam = pd.DataFrame(data.apply(lambda row : encode_sequence(row['PAM']), axis = 1))
    sequence_20nt = pd.DataFrame(data.apply(lambda row : encode_sequence(row['sequence']), axis = 1))
    sequence_30nt = pd.DataFrame(data.apply(lambda row : encode_sequence(row['sequence_30nt']), axis = 1))
    sequence_40nt = pd.DataFrame(data.apply(lambda row : encode_sequence(row['sequence_40nt']), axis = 1))

    one_hot_encoded_sequences = pd.concat([pam,sequence_20nt,sequence_30nt,sequence_40nt],axis=1)
    one_hot_encoded_sequences.columns = ["pam","sequence_20nt","sequence_30nt","sequence_40nt"]
    #one_hot_encoded_sequences["geneid"] = data["geneid"]

    return one_hot_encoded_sequences


############################################
# get kmer features
############################################

def count_kmers(read, k):

    # Start with an empty dictionary
    counts = {}
    # Calculate how many kmers of length k there are
    num_kmers = len(read) - k + 1
    # Loop over the kmer start positions
    for i in range(num_kmers):
        # Slice the string to get the kmer
        kmer = read[i:i+k]
        # Add the kmer to the dictionary if it's not there
        if kmer not in counts:
            counts[kmer] = 0
        # Increment the count for this kmer
        counts[kmer] += 1
    # Return the final counts
    return counts


############################################

def get_kmer_table(data,k):

    bases = ['A','T','G','C']
    columns = [''.join(p) for p in itertools.product(bases, repeat=k)]
    table_kmer = pd.DataFrame(columns = columns,index=range(data.shape[0]))
    
    
    seq_list = data["sequence_40nt"].to_list()
    for i in range(len(seq_list)):
        seq = seq_list[i]
        count_table = pd.DataFrame(count_kmers(seq, k),index=[0])
        table_kmer.loc[i, count_table.columns] = count_table.iloc[0]
    
    table_kmer = table_kmer.fillna(0)
    return table_kmer


############################################
# estimate gene effect
############################################


def compute_gene_effect_median(data):

    median_log2FC = data[["log2FC","geneid"]].groupby(['geneid']).median()
    #var_log2FC = data[["log2FC","geneid"]].groupby(['geneid']).var()
    
    for g in tqdm(data["geneid"].unique()):
        data.loc[data.geneid==g,"log2FC_gene_median"]=median_log2FC.loc[g,"log2FC"]
        #data.loc[data.geneid==g,"var_log2FC"]=var_log2FC.loc[g,"log2FC"]
    return data


############################################

def compute_gene_effect_normalized_rank(data):

    data["rank"] = data.groupby('geneid')['log2FC'].rank(ascending=False)
    
    gene_count = data["geneid"].value_counts().to_frame()
    for g in tqdm(data["geneid"].unique()):
        data.loc[data.geneid==g,"number_guides"]=gene_count.loc[g,"geneid"]
    
    data["log2FC_normalized_rank"] = data["rank"] / data["number_guides"]
    data = data.drop(['rank', 'number_guides'], axis=1)
    
    return data


############################################