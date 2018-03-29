from nnmnkwii.datasets import PaddedFileSourceDataset
from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
from nnmnkwii.baseline.gmm import MLPG

import pickle
import config
import librosa
import os

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer


class MyFileDataSource(CMUArcticWavFileDataSource):
    def __init__(self, *args, **kwargs):
        super(MyFileDataSource, self).__init__(*args, **kwargs)
        self.test_paths = None

    def collect_files(self):
        paths = super(MyFileDataSource, self).collect_files()
        paths_train, paths_test = train_test_split(
            paths, test_size=config.test_size, random_state=1234)

        # keep paths for later testing
        self.test_paths = paths_test

        return paths_train

    def collect_features(self, path):
        x, fs = librosa.load(path, sr=config.fs)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=config.frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        spectrogram = trim_zeros_frames(spectrogram)
        mc = pysptk.sp2mc(spectrogram, order=config.order, alpha=config.alpha)
        return mc


def test_one_utt(src_path, tgt_path, disable_mlpg=False, diffvc=True):
    # GMM-based parameter generation is provided by the library in `baseline` module
    if disable_mlpg:
        # Force disable MLPG
        paramgen = MLPG(gmm, windows=[(0, 0, np.array([1.0]))], diff=diffvc)
    else:
        paramgen = MLPG(gmm, windows=config.windows, diff=diffvc)

    x, fs = librosa.load(src_path, sr=config.fs)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=config.frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

    mc = pysptk.sp2mc(spectrogram, order=config.order, alpha=config.alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    if config.use_delta:
        mc = delta_features(mc, config.windows)
    mc = paramgen.transform(mc)
    if disable_mlpg and mc.shape[-1] != static_dim:
        mc = mc[:, :static_dim]
    assert mc.shape[-1] == static_dim
    mc = np.hstack((c0[:, None], mc))
    if diffvc:
        mc[:, 0] = 0 # remove power coefficients
        engine = Synthesizer(MLSADF(order=config.order, alpha=config.alpha), hopsize=config.hop_length)
        b = pysptk.mc2b(mc.astype(np.float64), alpha=config.alpha)
        waveform = engine.synthesis(x, b)
    else:
        spectrogram = pysptk.mc2sp(
            mc.astype(np.float64), alpha=config.alpha, fftlen=config.fftlen)
        waveform = pyworld.synthesize(
            f0, spectrogram, aperiodicity, fs, config.frame_period)

    return waveform


clb_source = MyFileDataSource(data_root=config.data_root,
                              speakers=["bdl"], max_files=config.max_files)
slt_source = MyFileDataSource(data_root=config.data_root,
                              speakers=["slt"], max_files=config.max_files)

X = PaddedFileSourceDataset(clb_source, 1200).asarray()
Y = PaddedFileSourceDataset(slt_source, 1200).asarray()

# Alignment
X_aligned, Y_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y))

# Drop 1st (power) dim
X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]

# apply MLPG
static_dim = X_aligned.shape[-1]
if config.use_delta:
    X_aligned = apply_each2d_trim(delta_features, X_aligned, config.windows)
    Y_aligned = apply_each2d_trim(delta_features, Y_aligned, config.windows)

XY = np.concatenate((X_aligned, Y_aligned), axis=-1).reshape(-1, X_aligned.shape[-1]*2)
# remove zero padding
XY = remove_zeros_frames(XY)

# train gmm

gmm = GaussianMixture(n_components=64, covariance_type="full", max_iter=100, verbose=1)
gmm.fit(XY)

os.makedirs("checkpoints", exist_ok=True)
# save gmm model
with open("checkpoints/gmm.cpt", 'wb') as f:
    pickle.dump(gmm, f)

# test
os.makedirs("wavs/gmm", exist_ok=True)

for i, (src_path, tgt_path) in enumerate(zip(clb_source.test_paths, slt_source.test_paths)):
    print("{}-th sample".format(i+1))
    wo_MLPG = test_one_utt(src_path, tgt_path, disable_mlpg=True)
    w_MLPG = test_one_utt(src_path, tgt_path, disable_mlpg=False)

    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav('wavs/gmm/w_MLPG_{}.wav'.format(i), (w_MLPG*maxv).astype(np.int16), config.fs)
    librosa.output.write_wav('wavs/gmm/wo_MLPG_{}.wav'.format(i), (wo_MLPG*maxv).astype(np.int16), config.fs)




