import argparse
import random
import numpy as np
import os
from shutil import copyfile
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def cut_no_overlap(length, kmer=1, max_prob=0.5):
    cuts = []
    while length:
        if length <= 509 + kmer:
            cuts.append(length)
            break
        else:
            if random.random() > max_prob:
                cut = max(int(random.random() * (509 + kmer)), 5)
            else:
                cut = 509 + kmer
            cuts.append(cut)
            length -= cut

    return cuts


def sampling(length, kmer=1, sampling_rate=1):
    times = int(length * sampling_rate / 256)
    starts = []
    ends = []
    for i in range(times):
        cut = max(int(random.random() * (509 + kmer)), 5)
        start = np.random.randint(length - kmer)
        starts.append(start)
        ends.append(start + cut)

    return starts, ends


def sampling_fix(length, kmer=1, sampling_rate=1, fix_length=10245):
    times = int(length * sampling_rate / fix_length)
    starts = []
    ends = []
    for i in range(times):
        cut = fix_length
        start = np.random.randint(length - 6 - fix_length)
        starts.append(start)
        ends.append(start + cut)

    return starts, ends


def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string) - kmer:
        sentence += original_string[i:i + kmer] + " "
        i += stride

    return sentence[:-1].strip("\"")


def get_kmer_sequence(original_string, kmer=1):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string) - kmer):
        sequence.append(original_string[i:i + kmer])

    sequence.append(original_string[-kmer:])
    return sequence


def Process(args):
    old_file = open(args.file_path, "r")
    if args.output_path == None:
        args.output_path = args.file_path

    if args.sampling_rate != 1.0:
        new_file_path = args.output_path + "_sam" + str(args.kmer)
    else:
        new_file_path = args.output_path + "_cut" + str(args.kmer)
    new_file = open(new_file_path, "w")
    line = old_file.readline()
    while line:
        line_length = len(line)
        if args.sampling_rate != 1.0:
            starts, ends = sampling_fix(length=line_length, kmer=args.kmer, sampling_rate=args.sampling_rate,
                                        fix_length=args.length)
            for i in range(len(starts)):
                new_line = line[starts[i]:ends[i]]
                sentence = get_kmer_sentence(new_line, kmer=args.kmer)
                new_file.write(sentence + "\n")

        else:
            cuts = cut_no_overlap(length=line_length, kmer=args.kmer)
            start = 0
            for cut in cuts:
                new_line = line[start:start + cut]
                sentence = get_kmer_sentence(new_line, kmer=args.kmer)
                start += cut
                new_file.write(sentence + "\n")

        line = old_file.readline()
def pad_seq(left, right, seq):
    if int(left)<0:
        seq = seq[-left:]
    else:
        seq = 'N'*left + seq
    if int(right)<0:
        seq = seq[:right]
    else:
        seq = seq + 'N'*right
    return seq
    
def my_process2(old_file_path, dmr,  cluster, num):

    df = pd.read_csv(old_file_path, sep='\t')

    tumor_df = df[(df['group']==1)&(df['dmr']==dmr)]
    normal_df = df[(df['group']!=1)&(df['dmr']==dmr)]

    tumor_df['pad_seq'] = tumor_df.apply(lambda x: pad_seq(x['num_left_zero'],x['num_right_zero'],x['data']), axis=1)
    normal_df['pad_seq'] = normal_df.apply(lambda x: pad_seq(x['num_left_zero'],x['num_right_zero'],x['data']), axis=1)

    tumor_old_seq = list(tumor_df['pad_seq'])
    normal_old_seq = list(normal_df['pad_seq'])
    tumor_new_seq, normal_new_seq = [], []
    tumor_cluster, normal_cluster = [], []
    for seq in tumor_old_seq:
        line_length = len(seq.strip())
        sentence = ''
        for i in range(line_length-num+1):
            sentence += seq[i:i + num] + " "
        tumor_new_seq.append(sentence)
        tumor_cluster.append(cluster)
    for seq in normal_old_seq:
        line_length = len(seq.strip())
        sentence = ''
        for i in range(line_length-num+1):
            sentence += seq[i:i + num] + " "
        normal_new_seq.append(sentence)
        normal_cluster.append(cluster)
    tumor_df['sequence'] = tumor_new_seq
    normal_df['sequence'] = normal_new_seq
    
    tumor_df['cluster'] = tumor_cluster
    normal_df['cluster'] = normal_cluster
    return tumor_df, normal_df

if __name__ == "__main__":
    new_root = "/model/sequence_data/"
    dmr = sys.argv[1].split(',')
    print(dmr)
    k=5
    i= 0
    all_tumor_df = pd.DataFrame()
    all_normal_df = pd.DataFrame()
    cluster_dmr = pd.read_csv('/model/cluster_dmr.csv')
    for d in dmr:
        cluster = list(cluster_dmr.loc[cluster_dmr['dmr']==d]['cluster'])[0]
        file='/model/ft_data_for_train/esophagus/add_4_healthy_10cluster/dmr_{}.txt'.format(cluster)
        tumor_df, normal_df = my_process2(file, d, cluster, k)
        
        all_tumor_df = pd.concat([all_tumor_df, tumor_df], axis=0)
        all_normal_df = pd.concat([all_normal_df, normal_df], axis=0)
        
    all_tumor_df.to_csv(new_root+'tumor.csv')
    all_normal_df.to_csv(new_root+'normal.csv')

    print("done")

