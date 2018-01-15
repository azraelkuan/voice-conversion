# -*- coding: utf-8 -*-

import numpy as np
from os.path import join, splitext, isdir
from os import listdir

from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from sklearn.model_selection import train_test_split
from utils import hparams


class MyFileDataSource(CMUArcticWavFileDataSource):
    def __init__(self, *args, **kwargs):
        super(MyFileDataSource, self).__init__(*args, **kwargs)
        self.test_paths = None

    def collect_files(self):
        speaker_dirs = list(
            map(lambda x: join(self.data_root, x),
                self.speakers))
        paths = []
        labels = []

        if self.max_files is None:
            max_files_per_speaker = None
        else:
            max_files_per_speaker = self.max_files // len(self.speakers)
        for (i, d) in enumerate(speaker_dirs):
            if not isdir(d):
                raise RuntimeError("{} doesn't exist.".format(d))
            files = [join(speaker_dirs[i], f) for f in listdir(d)]
            files = list(filter(lambda x: splitext(x)[1] == ".npy", files))
            files = sorted(files)
            files = files[:max_files_per_speaker]
            for f in files[:max_files_per_speaker]:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[i]])

        self.labels = np.array(labels, dtype=np.int32)

        paths_train, paths_test = train_test_split(paths, test_size=hparams.test_size, random_state=1234)
        self.test_paths = paths_test
        return paths_train

    def collect_features(self, path):
        feature = np.load(path)
        return feature


