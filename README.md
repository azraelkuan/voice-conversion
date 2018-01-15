# pytorch_voice_conversion

> use pytorch to implement voice conversion

## ToDo List
- [x] GMM Baseline
- [ ] LSTM Result 
- [ ] Dual Result
- [ ] Gan Result


## First
substitute the arctic data dir in `utils/hparams.py`


## Pre process
Since the process of getting mgc feature is too slow, so we preprocess is to numpy array and save it.

> for single speaker

`python preprocess.py --speaker slt --output_dir /your_arctic_npy_dir/`

> for all speaker

`python preprocess.py --all true --output_dir /your_arctic_npy_dir/`


## GMM Baseline
the gmm baseline is copied from the [nnmnkwii gitpage](https://r9y9.github.io/nnmnkwii/latest/nnmnkwii_gallery/notebooks/vc/01-GMM%20voice%20conversion%20(en).html)

1. `python train_gmmm.py | tee log/gmm.log`
2. check the generated wav in `wav/gmm`


## LSTM 

