# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pyworld
import pysptk
import os
import librosa

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from nnmnkwii.preprocessing import trim_zeros_frames
from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from utils import hparams
from tqdm import tqdm


available_speakers = ["awb", "bdl", "clb", "jmk", "ksp", "rms", "slt"]


def str_to_bool(bool_string):
    if bool_string == "true":
        return True
    else:
        return False


def get_args():
    parser = argparse.ArgumentParser("pre process the arctic data")
    parser.add_argument('--speaker', type=str, default='clb', help="the speaker u want to deal")
    parser.add_argument('--all', type=str_to_bool, default="false", help="whether deal all the speaker")
    parser.add_argument('--output_dir', type=str, default=None, help='the dir to save feature')
    args = parser.parse_args()
    return args


def build_from_path(files, output_dir, num_workers=4):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    for path in files:
        futures.append(executor.submit(partial(get_feature, path, output_dir)))

    return [future.result() for future in tqdm(futures)]


def get_feature(path, output_dir):
    x, fs = librosa.load(path, sr=hparams.fs)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=hparams.frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    spectrogram = trim_zeros_frames(spectrogram)
    mc = pysptk.sp2mc(spectrogram, order=hparams.order, alpha=hparams.alpha)

    wav_id = path.split('/')[-1].split(".")[0]
    speaker = wav_id.split('_')[3]
    os.makedirs(os.path.join(output_dir, speaker), exist_ok=True)
    np.save(os.path.join(output_dir, speaker, wav_id), mc)


def main():
    args = get_args()

    if args.output_dir is None:
        raise ValueError("the output dir is not None!")

    if args.all:
        speakers = available_speakers
    else:
        speakers = [args.speaker]

    data_source = CMUArcticWavFileDataSource(data_root=hparams.DATA_ROOT, speakers=speakers)
    files = data_source.collect_files()
    build_from_path(files, args.output_dir, num_workers=4)


if __name__ == '__main__':
    main()