# BCNN
Here we present the codes of BCNN.

This repository includes: source codes of pre-trainng, finetune, prediction, visualization and data samples. This package is still under development, as more features will be included gradually.

## Requirements and Setup
Python version >= 3.6

PyTorch version >= 1.10.0
```
# clone the repository
git clone https://github.com/Bamrock/BCNN.git
cd BCNN
pip install -r requirements.txt
```

(Optional, install apex for fp16 training)

```
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Pre-training

```
python -m torch.distributed.launch --nproc_per_node= \
    --nnodes= --node_rank= --master_addr= \
    --master_port= \
    pretrain.py \
```

Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).


## Finetune 

```

python -m torch.distributed.launch --nproc_per_node= \
    --nnodes= --node_rank= --master_addr= \
    --master_port= \
    finetune.py \
```

Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).


## Prediction

```

python prediction.py
```
Add --fp16 tag if you want to perfrom mixed precision. (You have to install the 'apex' from source first).

## Kernel Visualization

```

python BERT_visualization.py  # Visualization of Convolutional Kernels in BCNN
python CNN_series_visualization.py # Visualization of Convolutional Kernels in CNN+LSTM, CNN+GRU

```

## Attention Visualization

```

sh visualization/visual.py  # Visualization of Attention weights in BCNN

```


Our repository references the code of DNABERT: https://github.com/jerryji1993/DNABERT

