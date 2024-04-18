import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, ZeroPadding1D
from keras import backend as K
import pandas as pd
import sys
from DISMIR_training_yqren import build_model, lstm_seq, conv_onehot
from collections import defaultdict
import tensorflow as tf
from tqdm import tqdm, trange
import pandas as pd
from pandas import DataFrame
from keras.models import load_model
from typing import List

def dmr_seq_pad(dmr_seqs, pad_lefts, pad_rights):
    ret = []
    for seq, pad_left, pad_right in zip(dmr_seqs, pad_lefts, pad_rights):
        if pad_left < 0:
            seq = seq[-pad_left:]
            pad_left = 0
        if pad_right < 0:
            seq = seq[:pad_right]
            pad_right = 0
        
        res = 'N'*pad_left + seq + 'N'*pad_right
        ret.append(res)
    return ret

def filter_N(dmr_seqs: List[str], start, end):
    ret = []
    for seq in dmr_seqs:
        if 'N' in seq[start:end]:
            continue
        else:
            ret.append(seq)
    return ret


def min_max_normalization(matrix, min_value=0, max_value=1):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val) * (max_value - min_value) + min_value
    return normalized_matrix


# 加载模型
onehot = True
mode_list = ['CNN','CNN+LSTM','CNN+GRU']
max_words = 300
top_words = 9
embed_dim = 20
num_filter = 100
if onehot:
    embed_dim = 8
num_labels = 1
epochs = 10
batch = 64 
cluster_id = '01'


# 读取数据
df = pd.read_csv("/path/to/data/dmr_cluster01.txt", sep='\t')
# 找出长度小于300的dmr的样本
newdf = df[((df["num_left_zero"] + df["num_right_zero"] + df["read_length"])<300)&(df["group"]==1)]
# newdf = df[((df["num_left_zero"] + df["num_right_zero"] + df["read_length"])<300)]
newdf = newdf.reset_index(drop=True)
remain_dmr = list(set(newdf["dmr"].tolist()))


dev_seq = list(newdf['data'])
dev_seq = lstm_seq(dev_seq,max_words)
dev_seq_one_hot = conv_onehot(dev_seq,max_words)

output_mofit_index = None

for mode in tqdm(mode_list, desc = "model"):
    model = load_model('./gpu_train_result/{}_{}_{}_weight.h5'.format(cluster_id,mode,num_filter))
    layer_to_visualize = model.layers[1]
    activation_model = Model(inputs=model.input, outputs=layer_to_visualize.output)
    predict_result = model.predict(dev_seq_one_hot, verbose=0)
    # 找出结果>0.9的result所在的DMR
    num_reads_use = (predict_result>0.9).sum()
    index_with_high_predict = np.argsort(predict_result, axis=0)[::-1][:num_reads_use].squeeze()
    high_predict_dmr = list(set(newdf.loc[index_with_high_predict]["dmr"].tolist()))
    # 每个DMR内部补齐，计算激活值
    max_position_all_dmr = []
    max_position_value_all_dmr = []
    for dmr_name in tqdm(high_predict_dmr, desc="dmr"):
        dmr_df = newdf[newdf["dmr"] == dmr_name]
        dmr_seq = dmr_df["data"].tolist()
        dmr_left_pad_num = dmr_df["num_left_zero"].tolist()
        dmr_right_pad_num = dmr_df["num_right_zero"].tolist()
        dmr_padded_seq = dmr_seq_pad(dmr_seq, dmr_left_pad_num, dmr_right_pad_num)
        dmr_padded_seq_ids = lstm_seq(dmr_padded_seq, max_words)
        dmr_padded_seq_embeds = conv_onehot(dmr_padded_seq_ids, max_words)
        # 计算激活值, [batch_size, 300, 100]
        # [B, S, kernel_num]
        activation_model_output = activation_model(dmr_padded_seq_embeds)
        output_np = activation_model_output.numpy()
        # 找出每个kernel最大值的位置
        max_position = output_np.reshape(-1, output_np.shape[-1]).argmax(axis=0) % 300
        max_position_value = output_np.reshape(-1, output_np.shape[-1]).max(axis=0)
        max_position_all_dmr.append(max_position)
        max_position_value_all_dmr.append(max_position_value)
    # break
    # [71, 100]
    max_position_value_all_dmr = np.stack(max_position_value_all_dmr)
    max_position_all_dmr = np.stack(max_position_all_dmr)
# 找出前1k个最高的，对应的某个卷积核，某个DMR
    # 以下代码尚未修正
    topk = 1000
    top_indices = np.argpartition(max_position_value_all_dmr.flatten(), -topk)[-topk:]
    row_indices, col_indices = np.unravel_index(top_indices, max_position_all_dmr.shape)
    # row_indices: dmr
    # col_indices: kernel
    # 开始自己算卷积了
    stride, embed_dim, kernel_num = layer_to_visualize.kernel.shape
    motif_index = 0
    for row, col in tqdm(zip(row_indices, col_indices), desc="draw", total=len(row_indices)):
        dmr_name = high_predict_dmr[row]
        dmr_index = row
        kernel_index = col
        motif_position = max_position_all_dmr[row, col]
        kernel_weight = layer_to_visualize.kernel[:, :, kernel_index]

        dmr_df = newdf[newdf["dmr"] == dmr_name]
        dmr_seq = dmr_df["data"].tolist()
        dmr_left_pad_num = dmr_df["num_left_zero"].tolist()
        dmr_right_pad_num = dmr_df["num_right_zero"].tolist()
        dmr_padded_seq = dmr_seq_pad(dmr_seq, dmr_left_pad_num, dmr_right_pad_num)
        dmr_padded_seq = filter_N(dmr_padded_seq, motif_position, motif_position + stride)
        if len(dmr_padded_seq) == 0:
            # print('1')
            continue
        dmr_padded_seq_ids = lstm_seq(dmr_padded_seq, max_words)
        # [B, S, E]
        dmr_padded_seq_embeds = conv_onehot(dmr_padded_seq_ids, max_words)
        if dmr_padded_seq_embeds[:,motif_position:motif_position + stride, :].shape[1] < stride:
            # print('2')
            continue
        act = kernel_weight * dmr_padded_seq_embeds[:,motif_position:motif_position + stride, :]
        # print(f"{type(act)=}, {act.shape=}")
        act:np.ndarray = act.numpy()
        token_weight = act.sum(axis=-1)
        # 全部归一化
        token_weight = min_max_normalization(token_weight, 1,2)
        # 求每个位置每种碱基的均值
        dmr_padded_seq_codes = [list(s) for s in dmr_padded_seq]
        dmr_padded_seq_codes = np.array(dmr_padded_seq_codes)[:, motif_position:motif_position + stride]
        if dmr_padded_seq_codes.shape[1] < stride:
            # print('3')
            continue
        {'A':1,'T':2,'C':3,'G':4,'M':5,'L':6,'N':7}
        token_mask = {}
        for code in ['A', 'T', 'C', 'G', 'M', 'L', 'N']:
            token_mask[code] = dmr_padded_seq_codes == code
        
        # group-wise的平均值
        token_probs_matrix = np.zeros((4, stride))
        for code in ['A', 'T', 'C', 'G', 'M', 'L', 'N']:
            token_conut = token_mask[code].sum(axis=0)
            token_probs = (token_mask[code] * token_weight).sum(axis=0)
            token_conut = np.clip(token_conut, 1, np.max([token_conut.max(), 1]))
            if any(token_conut==0):
                raise
            token_probs = token_probs / token_conut

            if code == 'A':
                token_probs_matrix[0,:] += token_probs
            if code == 'C' or code == 'M':
                token_probs_matrix[1,:] += token_probs
            if code == 'G' or code == 'L':
                token_probs_matrix[2,:] += token_probs
            if code == 'T':
                token_probs_matrix[3,:] += token_probs
        
        if np.any(token_probs_matrix.sum(axis=0) == 0):
            raise
        
        token_probs_matrix = token_probs_matrix / token_probs_matrix.sum(axis=0)
        token_probs_matrix = token_probs_matrix.T
        # 转换成txt

        prefix = f"""MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.250 C 0.250 G 0.250 T 0.250 

MOTIF {dmr_name}_{kernel_index}_{motif_position}
letter-probability matrix: alength=4 w=20 nsites=20 E=0"""
        other_prefix = f"""
MOTIF {dmr_name}_{kernel_index}_{motif_position}
letter-probability matrix: alength=4 w=20 nsites=20 E=0"""
        if motif_index == 0:
            os.makedirs("visual_new_result", exist_ok=True)
            file_name = f"visual_new_result/{mode}_{topk}_cluster{cluster_id}.txt"
            f = open(file_name, 'wb')
            np.savetxt(f, token_probs_matrix, fmt='%.6f', header=prefix,comments='')
            f.close()
            f=open(file_name, 'ab')
        else:
            
            np.savetxt(f, token_probs_matrix, fmt='%.6f', header=other_prefix,comments='')
        
        motif_index += 1
    output_mofit_index = motif_index

