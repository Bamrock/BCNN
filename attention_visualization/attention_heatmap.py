import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
## 根据模型获取的attention weight计算每个碱基对应的attention weight，对出现在多个token中的相同位置碱基取平均
def generate_weight(data):
    attention_weights = []
    dna_sequence = data['origin_sequence']
    for sequence,score in zip(data['origin_sequence'],data['scores']):
        new_score = []
        score = score.split()[1:-1]
        score = [float(i) for i in score]
        for number,seq in enumerate(sequence):
            if number == 0:
                new_score.append(np.average(score[number]))
            elif number == len(sequence)-1:
                new_score.append(np.average(score[-1]))
            elif number in [1,2,3]:
                new_score.append(np.average(score[:number+1]))
            elif number in [len(sequence)-2, len(sequence)-3, len(sequence)-4]:
                new_score.append(np.average(score[number-4:]))
            else:
                new_score.append(np.average(score[number-4:number+1])) 
        attention_weights.append(new_score)
    return dna_sequence, np.array(attention_weights)
    
def calculate_char_percentage(text, char):
    """
    计算指定字符在字符串中的占比
    :param text: 输入字符串
    :param char: 要统计的字符
    :return: 占比百分比（0-100）
    """
    # 参数校验
    if not text:
        return 0.0
    
    # 计算字符出现次数
    count = text.count(char)
    
    # 计算占比
    percentage = count / len(text)
    return round(percentage, 2)  # 保留两位小数
    
## 筛选并过滤N较多的数据
def filter_sequence(dna_sequence, attention_weights, threshold):
    keep_sequence, keep_weights = [], []
    for seq,weights in zip(dna_sequence, attention_weights):
        seq, weights = seq[-100:], weights[-100:]        ## 只保留后100个碱基
        if calculate_char_percentage(seq,'N') < threshold:  ##去掉N占比超过设定阈值的数据
            keep_sequence.append(seq)
            keep_weights.append(weights)
    return keep_sequence, np.array(keep_weights)
    
## 根据数据标签，数据所属的DMR检索需要的数据，过滤其中N较多的数据
def heatmap_data(normal_data, tumor_data, select_dmr, threshold):
    ## 为每个碱基分配attention weight
    normal_dna_sequence, normal_attention_weights = generate_weight(normal_data.loc[normal_data['dmr']==select_dmr])
    tumor_dna_sequence, tumor_attention_weights = generate_weight(tumor_data.loc[tumor_data['dmr']==select_dmr])
    ## 过滤N出现较多的数据
    normal_keep_sequence, normal_keep_weights = filter_sequence(normal_dna_sequence, normal_attention_weights, threshold)
    tumor_keep_sequence, tumor_keep_weights = filter_sequence(tumor_dna_sequence, tumor_attention_weights, threshold)
    ## 不同类型的数据量保持一致
    sample_number = len(tumor_keep_sequence)
    normal_keep_sequence, normal_keep_weights = normal_keep_sequence[:sample_number], normal_keep_weights[:sample_number]
    ## attention weights归一化处理
    #tumor_keep_weights = tumor_keep_weights / tumor_keep_weights.sum(axis=0, keepdims=True) ## 对attention weights进行归一化处理
    #normal_keep_weights = normal_keep_weights / normal_keep_weights.sum(axis=0, keepdims=True) ## 对attention weights进行归一化处理
    print('{}经过筛选过滤后可视化的数据量为{}'.format(select_dmr, sample_number))
    return normal_keep_sequence, normal_keep_weights, tumor_keep_sequence, tumor_keep_weights

def save_seq2logo_file(select_dmr, sample_type, keep_sequence):
    ## ATCGML
    nu_weight = np.zeros((len(keep_sequence[0]),6),dtype=int)
    nu = list('ATCGML')
    for seq in keep_sequence:
        for n,i in enumerate(seq):
            if i in nu:
                nu_weight[n][nu.index(i)] += 1
    eps = 1e-10  # 极小值，防止除以零
    percentage_arr = nu_weight / (nu_weight.sum(axis=1, keepdims=True) + eps)## 计算每个碱基在该位置的占比

    with open('/SISDC_GPFS/Home_SE/KNOWN/test2-NN/BERT/model/heatmap_result/{}_{}_seq2logo.txt'.format(select_dmr, sample_type), "w", encoding="utf-8") as keep_sequence_file:
        keep_sequence_file.write("A T C G M L\n")  # 写入一行
        for i in percentage_arr:
            keep_sequence_file.write(' '.join(list(map(str, i)))+'\n')
            
# 创建热力图
def create_heatmap(select_dmr, sample_type, keep_weights):
    plt.figure(figsize=(100, 30))
    # 2. 创建自定义黄蓝渐变颜色映射
    colors = ["#FFFF00", "#0080FF"]  # 黄色到蓝色
    cmap = LinearSegmentedColormap.from_list("yellow_blue", colors)
    # 使用seaborn绘制热力图
    ax = sns.heatmap(
        keep_weights,
        annot=False,            # 在单元格中显示数值
        cmap=cmap,         # 颜色映射
        #vmin=0,
        #vmax=0.07,
        xticklabels=False,    # 关闭x轴标签
        yticklabels=False,    # 关闭y轴标签
        rasterized=True       # 加速渲染
    )
    # 4. 添加标签和标题
    
    # 5. 添加网格线（浅灰色）
    ax.grid(True, color='gray', linestyle='-', linewidth=0.1, alpha=0.5)
    
    # 7. 保存和显示
    plt.tight_layout()
    
    plt.savefig('/model/heatmap_result/{}_{}_attention_heatmap.png'.format(select_dmr, sample_type), dpi=200, bbox_inches='tight')
    plt.show()

## 保存可视化的数据及对应的attention weight
def save_heatmapdata(keep_sequence, keep_weights, sample_type):
    heat_data = pd.DataFrame()
    heat_data['data'], heat_data['weight'] = keep_sequence, list(keep_weights)
    heat_data.to_csv('/model/heatmap_result/{}_{}_heatmapdata.csv'.format(select_dmr, sample_type))
    
    
if __name__ == "__main__":
    
    dmr = sys.argv[1].split(',')
    threshold = 1   #设定的N占比阈值，若为1则不过滤
    data = pd.read_csv('/model/dmr_attention_score.csv')
    normal_data = data.loc[data['label']!=1]     
    tumor_data = data.loc[data['label']==1]
    for select_dmr in dmr:
        normal_keep_sequence, normal_keep_weights, tumor_keep_sequence, tumor_keep_weights = heatmap_data(normal_data, tumor_data, select_dmr, threshold)
        #save_heatmapdata(normal_keep_sequence, normal_keep_weights, 'normal')
        #save_heatmapdata(tumor_keep_sequence, tumor_keep_weights, 'tumor')
        for sample_type in ['normal', 'tumor']:
            if sample_type == 'normal':
                keep_sequence = normal_keep_sequence
                keep_weights = normal_keep_weights
            else:
                keep_sequence = tumor_keep_sequence
                keep_weights = tumor_keep_weights
            save_seq2logo_file(select_dmr, sample_type, keep_sequence)
            create_heatmap(select_dmr, sample_type, keep_weights)