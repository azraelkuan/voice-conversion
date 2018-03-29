#!/bin/bash

#SBATCH -p fatq
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -o log/pre.log


source activate dl

python preprocess.py \
    --num_workers 10 \
    --name cmu_arctic \
    --in_dir /mnt/lustre/sjtu/users/kc430/data/sjtu/tts/voice-conversion/arctic \
    --out_dir /mnt/lustre/sjtu/users/kc430/data/my/vc/cmu_arctic


