import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
import config
import numpy as np


def preprocess(mod, in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    lf0s = {}
    for m in metadata:
        if m[3] not in lf0s:
            lf0s[m[3]] = m[4]
        else:
            lf0s[m[3]].extend(m[4])
    with open(os.path.join(out_dir, 'norm.txt'), 'w', encoding='utf-8') as f:
        for k, v in lf0s.items():
            v = np.asarray(v)
            nonzero_indices = np.nonzero(v)
            mean = np.mean(v[nonzero_indices])
            std = np.std(v[nonzero_indices])
            f.write('{} {} {}\n'.format(k, mean, std))

    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m[:-1]]) + '\n')
    frames = sum([m[1] for m in metadata])
    sr = config.fs
    hours = frames * config.frame_period / 1000 / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--in_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=str, default=None)
    args = parser.parse_args()

    name = args.name
    in_dir = args.in_dir
    out_dir = args.out_dir
    num_workers = args.num_workers
    num_workers = cpu_count() if num_workers is None else int(num_workers)

    print("Sampling frequency: {}".format(config.fs))

    assert name in ["cmu_arctic"]
    mod = importlib.import_module('datasets.{}'.format(name))
    preprocess(mod, in_dir, out_dir, num_workers)
