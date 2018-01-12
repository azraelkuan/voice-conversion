# -*- coding: utf-8 -*-
import pyworld
import pysptk
import numpy as np
import pickle

from nnmnkwii.baseline.gmm import MLPG
from nnmnkwii.datasets import PaddedFileSourceDataset
from nnmnkwii.metrics import melcd
from nnmnkwii.preprocessing import delta_features
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.util import apply_each2d_trim
from pysptk.synthesis import MLSADF, Synthesizer
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture

from datasets import MyFileDataSource
from utils import hparams


def train(source_dataset, target_dataset):
    source_data = PaddedFileSourceDataset(source_dataset, 1200).asarray()
    target_data = PaddedFileSourceDataset(target_dataset, 1200).asarray()

    # Drop 1st dimension
    source_data, target_data = source_data[:, :, 1:], target_data[:, :, 1:]

    source_data_aligned, target_data_aligned = DTWAligner(verbose=0, dist=melcd).transform((source_data, target_data))

    static_dim = source_data_aligned.shape[-1]
    if hparams.use_delta:
        source_data_aligned = apply_each2d_trim(delta_features, source_data_aligned, hparams.windows)
        target_data_aligned = apply_each2d_trim(delta_features, target_data_aligned, hparams.windows)

    final_data = np.concatenate((source_data_aligned, target_data_aligned), axis=-1).reshape(-1, source_data_aligned.shape[-1]*2)
    gmm = GaussianMixture(n_components=64, covariance_type="full", max_iter=100, verbose=1)
    gmm.fit(final_data)

    return gmm, static_dim


def test_one_utt(gmm, static_dim, src_path, tgt_path, disable_mlpg=False, diffvc=True):
    # GMM-based parameter generation is provided by the library in `baseline` module
    if disable_mlpg:
        # Force disable MLPG
        paramgen = MLPG(gmm, windows=[(0, 0, np.array([1.0]))], diff=diffvc)
    else:
        paramgen = MLPG(gmm, windows=hparams.windows, diff=diffvc)

    fs, x = wavfile.read(src_path)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=hparams.frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

    mc = pysptk.sp2mc(spectrogram, order=hparams.order, alpha=hparams.alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    if hparams.use_delta:
        mc = delta_features(mc, hparams.windows)

    mc = paramgen.transform(mc)
    if disable_mlpg and mc.shape[-1] != static_dim:
        mc = mc[:, :static_dim]
    assert mc.shape[-1] == static_dim
    mc = np.hstack((c0[:, None], mc))
    if diffvc:
        mc[:, 0] = 0  # remove power coefficients
        engine = Synthesizer(MLSADF(order=hparams.order, alpha=hparams.alpha), hopsize=hparams.hop_length)
        b = pysptk.mc2b(mc.astype(np.float64), alpha=hparams.alpha)
        waveform = engine.synthesis(x, b)
    else:
        spectrogram = pysptk.mc2sp(
            mc.astype(np.float64), alpha=hparams.alpha, fftlen=hparams.fft_len)
        waveform = pyworld.synthesize(
            f0, spectrogram, aperiodicity, fs, hparams.frame_period)

    return waveform


def main():

    print("*" * 25, "Begin to load data", "*" * 25)
    clb_source = MyFileDataSource(data_root=hparams.DATA_ROOT, speakers=["clb"], max_files=hparams.max_files)
    slt_source = MyFileDataSource(data_root=hparams.DATA_ROOT, speakers=["slt"], max_files=hparams.max_files)
    print("*" * 25, "Finsh to load data", "*" * 25)

    print("*" * 25, "Begin to train", "*" * 25)
    gmm, static_dim = train(clb_source, slt_source)
    print("*" * 25, "Finish to train", "*" * 25)

    # save gmm model
    with open('model/gmm/baseline.model', 'wb') as f:
        pickle.dump(gmm, f)

    print("*" * 25, "Begin to test", "*" * 25)
    for i, (src_path, tgt_path) in enumerate(zip(clb_source.test_paths, slt_source.test_paths)):
        print("{}-th sample".format(i + 1))
        wo_MLPG = test_one_utt(gmm, static_dim, src_path, tgt_path, disable_mlpg=True)
        w_MLPG = test_one_utt(gmm, static_dim, src_path, tgt_path, disable_mlpg=False)
        wavfile.write("wav/gmm/w_MLPG_{}.wav".format(i + 1), rate=16000, data=w_MLPG)
        wavfile.write("wav/gmm/wo_MLPG_{}.wav".format(i + 1), rate=16000, data=wo_MLPG)
    print("*" * 25, "Finish to test", "*" * 25)


if __name__ == '__main__':
    main()
