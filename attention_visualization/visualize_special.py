import torch
#import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel, DNATokenizer
import random
import glob
import json
import logging
import os
import re
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy
import time

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)
    
def softmax(x):
    
    # x为一维数据时
    if x.ndim == 1:
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
    
    # x为二维数据时
    elif x.ndim == 2:
        val_num = np.zeros_like(x)
        for i in range(len(x)):
            part_x = x[i]
            val_num[i] = np.exp(part_x-np.max(part_x))/np.sum(np.exp(part_x-np.max(part_x)))
        return val_num

def get_attention_dna(model, tokenizer, sentence_a, layer_num):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b=None, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention = model(input_ids)[-1]
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 
    attn = format_attention(attention)
    print(attn.detach().numpy().shape)
    attn_score = [float(attn[layer_num,:,0,i].mean()) for i in range(attn.detach().numpy().shape[3])]
    return np.array(attn_score), tokens

def get_real_score(attention_scores, kmer, metric):
    counts = np.zeros([len(attention_scores)+kmer-1])
    real_scores = np.zeros([len(attention_scores)+kmer-1])

    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score

        real_scores = real_scores/counts
    else:
        pass

    return real_scores
 
def Visualize(model_name, data, layer_num,metric):
    # load model and calculate attention
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = DNATokenizer.from_pretrained(model_name, do_lower_case=False)
    sequence, dmr, label, pad_seq, cluster = list(data['sequence']), list(data['dmr']), list(data['group']), list(data['pad_seq']), list(data['cluster'])
    atten_scores = []
    dna_sequence = []
    new_dmr = []
    new_label = []
    new_pad_seq, new_cluster = [], []
    for i in range(len(sequence)):
        x = sequence[i].split()
        if len(x)<512:
            atten_score, tokens = get_attention_dna(model,tokenizer,x,layer_num)
            atten_scores.append(atten_score)
            dna_sequence.append(tokens)
            new_dmr.append(dmr[i])
            new_label.append(label[i])
            new_pad_seq.append(pad_seq[i])
            new_cluster.append(cluster[i])
    tt = pd.DataFrame()
    tt['scores'] = atten_scores
    tt['dna_sequence'] = dna_sequence
    tt['label'] = new_label
    tt['origin_sequence'] = new_pad_seq
    tt['dmr'] = new_dmr
    tt['cluster'] = new_cluster
    #data['atten_score']=atten_scores
    #data['dna_seq']=dna_sequence
    
    return tt
        

if __name__ == "__main__":
    
    data_path = "/model/sequence_data/"
    files = ['tumor.csv','normal.csv']
    layer_num = 11
    metric = 'mean'
    all_df = pd.DataFrame()
    for f in files:
        data_name = data_path + f
        data = pd.read_csv(data_name)
        cluster = list(set(list(data['cluster'])))
        for c in cluster:
            c_data = data.loc[data['cluster']==c]
            model_name = "/model/fine_tuning_model/esophagus_paper/add_4_healthy_10cluster/dmr_{}/best_specificity".format(c)
            attention_df = Visualize(model_name, c_data,layer_num,metric)
            all_df = pd.concat([all_df, attention_df], axis=0)
    all_df = all_df.reset_index()
    all_df.to_csv('dmr_attention_score.csv')
