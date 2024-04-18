import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm, trange
import pandas as pd
from pandas import DataFrame
from typing import List
import torch
from transformers import BertConfig, BertForSequenceClassification, DNATokenizer, BertForSequenceClassificationVisualize


device = "cuda" if torch.cuda.is_available() else "cpu"

def dmr_seq_pad(dmr_seqs, pad_lefts, pad_rights, pad_token_id, cls_token_id, sep_token_id):
    ret = []
    for seq, pad_left, pad_right in zip(dmr_seqs, pad_lefts, pad_rights):
        if pad_left <= 0:
            seq = seq[-pad_left+1:]
            pad_left = 1
        
        # if pad_right < 0:
        #     seq = seq[:pad_right-1]
        #     pad_right = 1
        
        pad_left -= 1
        pad_right = 300 - pad_left - len(seq) - 2
        if pad_right < 0:
            seq = seq[:pad_right]
            pad_right = 0
        res = [pad_token_id]*pad_left + [cls_token_id] +  seq + [sep_token_id] + [pad_token_id]*pad_right
        ret.append(res)
    return ret

def filter_N(dmr_seqs: List[List[int]], tokenizer, start, end):
    ret = []
    def any_in(pattern, target):
        for p in pattern:
            if p in target:
                return True      
        return False
    special_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    for seq in dmr_seqs:
        if any_in(special_ids, seq[start:end]):
            continue
        else:
            ret.append(seq)
    return ret

def min_max_normalization(matrix, min_value=0, max_value=1):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val) * (max_value - min_value) + min_value
    return normalized_matrix

def predict_pad(input_ids, left = False):
    if left:
        pass
    else:
        input_ids = input_ids + [0] *(300-len(input_ids))
    
    return input_ids



def convert_ids_to_raw_seq(tokenizer, input_ids) -> List[str]:
    res = []
    for ids in input_ids:
        one_str = []
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for idx, token in enumerate(tokens):
            if idx == 0:
                one_str.append(token)
            else:
                one_str.append(token[-1])
        one_str = ''.join(one_str)
        res.append(one_str)
    return res

@torch.no_grad()
def batch_forward(model, input_ids, attention_mask, return_emb, result_tensor = None) -> torch.Tensor:
    batch_size = len(input_ids)
    for i in range(batch_size):
        output = model(input_ids=input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), return_emb=return_emb)
        if result_tensor is None:
            result_tensor = torch.zeros((batch_size, *output.shape[1:]), device=output.device, dtype=output.dtype)
        result_tensor[i, ...] = output.detach().squeeze(0)
    return result_tensor


if __name__ == '__main__':

    # 加载模型
    MODEL_CLASSES = {
        "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    }

    model_path = '/path/to/model'
    data_path = '/path/to/tokenized/pred.tsv'
    cluster_id = "01"
    tokenizer = DNATokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(
                    model_path,
                    num_labels=2,
                    finetuning_task='dnaprom',
                    cache_dir=None,
                )
    model = BertForSequenceClassification.from_pretrained(model_path, config=config).to(device)
    activation_model = BertForSequenceClassificationVisualize.from_pretrained(model_path, config=config).to(device)

    # 准备数据

    df = pd.read_csv("/path/to/not/tokenized/dmr_cluster01.txt", sep='\t')
    data = pd.read_csv(data_path, sep='\t', header=0)
    bert_df = data
    bert_df[df.columns[1:]] = df[df.columns[1:]]
    df = bert_df

    # 找出长度小于300的dmr的样本
    newdf = df[((df["num_left_zero"] + df["num_right_zero"] + df["read_length"])<300)&(df["group"]==1)]
    # newdf = df[((df["num_left_zero"] + df["num_right_zero"] + df["read_length"])<300)]
    newdf = newdf.reset_index(drop=True)
    remain_dmr = list(set(newdf["dmr"].tolist()))

    # 找出预测结果>0.9的DMR
    batch_size = len(newdf)
    model.eval()
    activation_model.eval()

    predict_result = []
    for i in trange(len(newdf)):
        data_point = newdf['sequence'][i]
        
        inputs = tokenizer.encode_plus(data_point)
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        input_ids = predict_pad(input_ids)
        attention_mask = predict_pad(attention_mask)

        input_ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device="cuda")
        
        with torch.no_grad():
            output = model(**{"input_ids": input_ids, "attention_mask": attention_mask})
            logits = output[0]
            prob = torch.softmax(logits, dim=-1)
            predict_result.append(prob[0][1].item())



    mode = 'BERT'
    predict_result = np.array(predict_result)
    num_reads_use = (predict_result>0.9).sum()
    index_with_high_predict = np.argsort(predict_result, axis=0)[::-1][:num_reads_use].squeeze()
    high_predict_dmr = list(set(newdf.loc[index_with_high_predict]["dmr"].tolist()))

    # 计算所有DMR所有卷积核的最大激活值
    max_position_all_dmr = []
    max_position_value_all_dmr = []

    for dmr_name in tqdm(high_predict_dmr, desc="dmr"):
        dmr_df = newdf[newdf["dmr"] == dmr_name]
        dmr_seq = dmr_df["sequence"].tolist()
        dmr_left_pad_num = dmr_df["num_left_zero"].tolist()
        dmr_right_pad_num = dmr_df["num_right_zero"].tolist()

        dmr_seq_inputs = tokenizer.batch_encode_plus(dmr_seq)
        dmr_seq_input_ids, dmr_seq_attention_mask = dmr_seq_inputs["input_ids"], dmr_seq_inputs["attention_mask"]
        
        dmr_padded_seq_ids = dmr_seq_pad(dmr_seq_input_ids, dmr_left_pad_num, dmr_right_pad_num, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)
        dmr_padded_seq_attention_masks = dmr_seq_pad(dmr_seq_attention_mask, dmr_left_pad_num, dmr_right_pad_num, 0,1,1)
        # dmr_padded_seq_ids = lstm_seq(dmr_padded_seq, max_words)
        # dmr_padded_seq_embeds = conv_onehot(dmr_padded_seq_ids, max_words)
        # 计算激活值, [batch_size, 300, 100]
        # [B, kernel_num, S]
        input_ids = torch.tensor(dmr_padded_seq_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(dmr_padded_seq_attention_masks, dtype=torch.long, device=device)
        activation_model_output = batch_forward(activation_model, input_ids, attention_mask, False)
        # activation_model_output = activation_model(**{"input_ids": input_ids, "attention_mask": attention_mask, "return_emb": False})

        # 以下待修改
        # [B, S, kernel_num]
        output_np = activation_model_output.squeeze(-1).transpose(1,2).cpu().numpy()
        # 找出每个kernel最大值的位置
        max_position = output_np.reshape(-1, output_np.shape[-1]).argmax(axis=0) % 300
        max_position_value = output_np.reshape(-1, output_np.shape[-1]).max(axis=0)
        max_position_all_dmr.append(max_position)
        max_position_value_all_dmr.append(max_position_value)
    
    # 收集这些激活值并找top1000
    max_position_value_all_dmr = np.stack(max_position_value_all_dmr)
    max_position_all_dmr = np.stack(max_position_all_dmr)
    top_indices = np.argpartition(max_position_value_all_dmr.flatten(), -1000)[-1000:]
    row_indices, col_indices = np.unravel_index(top_indices, max_position_all_dmr.shape)
    stride, embed_dim, kernel_num = activation_model.conv.weight.shape[2],activation_model.conv.weight.shape[3], activation_model.conv.weight.shape[0]

    # 这top1000的激活值对应的位置开始画图了
    motif_index = 0
    for row, col in tqdm(zip(row_indices, col_indices), desc="draw", total=len(row_indices)):
        dmr_name = high_predict_dmr[row]
        dmr_index = row
        kernel_index = col
        motif_position = max_position_all_dmr[row, col]
        kernel_weight = activation_model.conv.weight[kernel_index, ...].squeeze()


        dmr_df = newdf[newdf["dmr"] == dmr_name]
        dmr_seq = dmr_df["sequence"].tolist()
        dmr_left_pad_num = dmr_df["num_left_zero"].tolist()
        dmr_right_pad_num = dmr_df["num_right_zero"].tolist()

        dmr_seq_inputs = tokenizer.batch_encode_plus(dmr_seq)
        dmr_seq_input_ids, dmr_seq_attention_mask = dmr_seq_inputs["input_ids"], dmr_seq_inputs["attention_mask"]
        dmr_padded_seq_ids = dmr_seq_pad(dmr_seq_input_ids, dmr_left_pad_num, dmr_right_pad_num, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)
        dmr_padded_seq_attention_masks = dmr_seq_pad(dmr_seq_attention_mask, dmr_left_pad_num, dmr_right_pad_num, 0,1,1)

        dmr_padded_seq_ids = filter_N(dmr_padded_seq_ids,tokenizer,  motif_position, motif_position + stride)
        dmr_padded_seq_attention_masks = filter_N(dmr_padded_seq_attention_masks, tokenizer, motif_position, motif_position + stride)
        input_ids = torch.tensor(dmr_padded_seq_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(dmr_padded_seq_attention_masks, dtype=torch.long, device=device)

        
        if len(dmr_padded_seq_ids) == 0:
            continue

        if input_ids[:,motif_position:motif_position + stride].shape[1] < stride:
            continue

        # [B, S, E]
        dmr_padded_seq_embeds = batch_forward(activation_model, input_ids, attention_mask, True).squeeze(1)
        # dmr_padded_seq_embeds = activation_model({"input_ids": input_ids, "attention_mask": attention_mask}).squeeeze(1)
        act = kernel_weight * dmr_padded_seq_embeds[:, motif_position:motif_position + stride:, :]
        # [B, stride]
        act:np.ndarray = act.detach().cpu().numpy()
        token_weight = act.sum(axis=-1)
        token_weight = min_max_normalization(token_weight, 1,2)
        batch_size, seq_len = token_weight.shape
        token_weight_extend = np.zeros((batch_size, seq_len+4))
        for col_idx in range(seq_len):
            token_weight_extend[:, col_idx:col_idx+5] += np.expand_dims(token_weight[:, col_idx], -1)
        
        token_weight = token_weight_extend
        
        token_letters = convert_ids_to_raw_seq(tokenizer, input_ids[:, motif_position:motif_position + stride])
        token_letters = [list(s) for s in token_letters]
        token_letters = np.array(token_letters)
        dmr_padded_seq_codes = token_letters

        token_mask = {}
        for code in ['A', 'T', 'C', 'G', 'M', 'L', 'N']:
            token_mask[code] = dmr_padded_seq_codes == code
        
        token_probs_matrix = np.zeros((4, token_weight.shape[-1]))
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
            file_name = f"visual_new_result/{mode}_cluster{cluster_id}.txt"
            f = open(file_name, 'wb')
            # os.makedirs(f"visual_new_result/{mode}", exist_ok=True)
            np.savetxt(f, token_probs_matrix, fmt='%.6f', header=prefix,comments='')
            f.close()
            f = open(file_name, 'ab')
        else:
            np.savetxt(f, token_probs_matrix, fmt='%.6f', header=other_prefix,comments='')
        motif_index += 1