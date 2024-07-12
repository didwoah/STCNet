#!/bin/bash

gammas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# for gamma in "${gammas[@]}"; do
#     python sst_sac.py --dataset nina1 --batch_size 1024 --cosine --gamma ${gamma}
# done

for gamma in "${gammas[@]}"; do
    python sst_sac.py --dataset nina2 --sampled --batch_size 1024 --cosine --gamma ${gamma}
done

for gamma in "${gammas[@]}"; do
    python sst_sac.py --dataset nina4 --sampled --batch_size 512 --cosine --gamma ${gamma}
done