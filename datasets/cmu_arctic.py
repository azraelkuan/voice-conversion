# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

import pysptk
import pyworld
from nnmnkwii.datasets import cmu_arctic
from nnmnkwii.preprocessing import trim_zeros_frames
import os
import librosa
import config


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    speakers = cmu_arctic.available_speakers

    wd = cmu_arctic.WavFileDataSource(in_dir, speakers=speakers)
    wav_paths = wd.collect_files()
    speaker_ids = wd.labels

    for index, (speaker_id, wav_path) in enumerate(
            zip(speaker_ids, wav_paths)):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_id, wav_path, "N/A")))
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    x, fs = librosa.load(wav_path, sr=config.fs)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=config.frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    spectrogram = trim_zeros_frames(spectrogram)
    mc = pysptk.sp2mc(spectrogram, order=config.order, alpha=config.alpha)
    timesteps = mc.shape[0]
    wav_id = wav_path.split("/")[-1].split('.')[0]
    mc_name = '{}-mc.npy'.format(wav_id)
    np.save(os.path.join(out_dir, mc_name), mc, allow_pickle=False)

    # compute lf0
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    # Return a tuple describing this training example:
    return mc_name, timesteps, text, speaker_id, lf0.tolist()

