#!/bin/bash

dmr=chr19_57560102_57560299,chr11_59502640_59502795
python prediction_data_special.py ${dmr}
bsub -n 20 -q HPC.S1.GPU.X785.sha -o output.%J python visualize_special.py
python attention_heatmap.py ${dmr}
