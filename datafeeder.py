import os
import numpy as np
import torch
from torch.utils.data import Dataset
from fastdtw import fastdtw
from nnmnkwii.metrics import melcd


class McepDataSet(Dataset):
    """
    only get path
    """

    def __init__(self, ssp, tsp, data_root, scp):
        super(McepDataSet, self).__init__()
        self.ssp = ssp
        self.tsp = tsp
        self.data_root = data_root

        # get the dataset ids
        with open(scp, 'r', encoding='utf-8') as f:
            self.wav_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.wav_ids)

    def __getitem__(self, idx):
        wav_id = self.wav_ids[idx]

        source_path = os.path.join(self.data_root, "cmu_us_arctic_{0}_{1}-mc.npy".format(self.ssp, wav_id))
        target_path = os.path.join(self.data_root, "cmu_us_arctic_{0}_{1}-mc.npy".format(self.tsp, wav_id))

        source_data = np.load(source_path)
        target_data = np.load(target_path)
        return source_data, target_data


def collate_fn(batch):
    """
    apply dtw into a batch
    :param batch:
    :return:
    """
    inputs = []
    outputs = []
    lengths = []
    for each in batch:
        x = each[0]
        y = each[1]
        _, path = fastdtw(x, y, dist=melcd, radius=1)
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x, y = x[pathx], y[pathy]
        inputs.append(x)
        outputs.append(y)
        lengths.append(len(x))

    # pad zero
    max_len = max(lengths)
    for i in range(len(batch)):
        inputs[i] = np.pad(inputs[i], [(0, max_len - len(inputs[i])), (0, 0)], mode='constant')
        outputs[i] = np.pad(outputs[i], [(0, max_len - len(outputs[i])), (0, 0)], mode='constant')

    inputs, outputs, lengths = np.asarray(inputs).astype(np.float32), np.asarray(outputs).astype(np.float32), \
                               np.asarray(lengths).astype(np.int16)
    return torch.from_numpy(inputs), torch.from_numpy(outputs), torch.from_numpy(lengths)
