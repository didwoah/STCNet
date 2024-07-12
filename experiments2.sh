#!/bin/bash

gammas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for gamma in "${gammas[@]}"; do
    python sst_ce.py --dataset nina1 --cosine --encoder C:/Users/PC/Desktop/MSC-HGR/save/Multi/nina1_models/lr_0.05_decay_0.0001_bsz_1024_temp_0.07_tri_0_gamma_${gamma}_cos_warm/last.pth
done

for gamma in "${gammas[@]}"; do
    python sst_ce.py --dataset nina2 --cosine --sampled --encoder C:/Users/PC/Desktop/MSC-HGR/save/Multi/nina2_models/lr_0.05_decay_0.0001_bsz_1024_temp_0.07_tri_0_gamma_${gamma}_cos_samp_warm/last.pth 
done

for gamma in "${gammas[@]}"; do
    python sst_ce.py --dataset nina4 --cosine --sampled --batch_size 32 --encoder C:/Users/PC/Desktop/MSC-HGR/save/Multi/nina4_models/lr_0.05_decay_0.0001_bsz_512_temp_0.07_tri_0_gamma_${gamma}_cos_samp_warm/last.pth 
done