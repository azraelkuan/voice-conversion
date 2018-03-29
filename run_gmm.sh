#!/bin/bash

#SBATCH -p fatq
#SBATCH -n 1
#SBATCH -c 15
#SBATCH -o log/gmm.log


source activate dl

python train_gmm.py


