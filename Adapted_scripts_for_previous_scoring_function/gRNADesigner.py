#Calculates the Rule set 2 score for the given 30-mer
#Input: 1. 30mer sgRNA+context sequence, NNNN[sgRNA sequence]NGGNNN
#       2. Amino acid cut position, for full model prediction only
#       3. Percent peptide, for full model prediction only
#Output: Rule set 2 score

import pandas as pd
import csv, argparse, sys
import pickle
import model_comparison

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq',
        type=str,
        help='30-mer')
    parser.add_argument('--aa-cut',
        type=int,
        default=-1,
        help='Amino acid cut position of sgRNA')
    parser.add_argument('--per-peptide',
        type=float,
        default=-1,
        help='Percentage of protein cut by sgRNA')
    return parser

if __name__ == '__main__':
    import math
    # args = get_parser().parse_args()
    # seq = args.seq.upper()
    seqs=[]
    aa_cuts=[]
    per_peptides=[]
    df1=pd.read_csv('TABLE_for_gRNAs.csv',sep='\t')
    scores=[]
    for i in df1.index:
        seqs.append(df1['sequence_30nt'][i])
        aa_cuts.append(math.ceil(int(df1['distance_start_codon'][i])/3))
        per_peptides.append(df1['distance_start_codon_perc'][i])
    file=open('OUTPUT_SCORE_FILE','w')
    for seq, aa_cut, per_peptide in zip(seqs, aa_cuts, per_peptides) :
        # aa_cut= -1
        # per_peptide = -1
        if len(seq)!=30: 
            print "Please enter a 30mer sequence."
            sys.exit(1)
        # aa_cut = args.aa_cut
        # per_peptide = args.per_peptide
        model_file_1 = 'PATH_TO_CODE/Code/Rule_Set_2_scoring_v1/saved_models/V3_model_nopos.pickle'
        model_file_2 = 'PATH_TO_CODE/Code/Rule_Set_2_scoring_v1/saved_models/V3_model_full.pickle'
        if (aa_cut == -1) or (per_peptide == -1):
            model_file = model_file_1
        else:
            model_file = model_file_2
        try:
            with open(model_file, 'rb') as f:
                model= pickle.load(f)    
        except:
            raise Exception("could not find model stored to file %s" % sys.exc_info()[1])
        if seq[25:27] == 'GG':
            score = model_comparison.predict(seq, aa_cut, per_peptide, model=model)
            file.writelines(str(score)+'\n')
            # print score
        else:
            print >> sys.stderr, 'Calculates on-target scores for sgRNAs with NGG PAM only.'
    file.close()
