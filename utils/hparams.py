# -*- coding: utf-8 -*-
import pyworld
import pysptk
from os.path import join, expanduser
import numpy as np


# data path
DATA_ROOT = join("/mnt/lustre/sjtu/shared/", 'data/tts/voice-conversion', 'arctic')

# parameters
fs = 16000
fft_len = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
order = 24
frame_period = 5
hop_length = int(fs * (frame_period * 0.001))
test_size = 0.03

max_files = 100  # number of utterances to be used.
test_size = 0.03
use_delta = True

if use_delta:
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
else:
    windows = [
        (0, 0, np.array([1.0])),
    ]

