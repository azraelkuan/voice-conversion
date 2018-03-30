#!/bin/bash

source activate dl

python test.py \
    --ssp bdl \
    --tsp slt \
    --cpt_path checkpoints/bdl-slt-dual-rnn.cpt \
    --save_path wavs/dual \
    --dual true
