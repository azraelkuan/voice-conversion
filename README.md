# pytorch_voice_conversion

> use pytorch to implement voice conversion

## ToDo List
- [x] GMM Baseline
- [x] LSTM Result
- [x] Dual Result
- [ ] Gan Result


## First
check the data root in `config.py` and replace it by yours

## GMM Baseline
the gmm baseline is copied from the [nnmnkwii gitpage](https://r9y9.github.io/nnmnkwii/latest/nnmnkwii_gallery/notebooks/vc/01-GMM%20voice%20conversion%20(en).html)

## LSTM
### Prepare Data
replace the `in_dir` and `out_dir` in `run_pre.sh` and then run it.
> the `in_dir` contains wavs like `xx/arctic/cmu_us_bdl_arctic/wav/xx.wav`

### Train
replace the `data_root` in `run_rnn.sh` and choose the source(target) speaker, then run it.
> finally the checkpoint will be saved in the dir `checkpoints`

### Test
replace the ssp tsp checkpoint_path in `test.py`, the result wav will be under the dir `wavs`

### Dual
the train and test process is the same of LSTM except add the args `dual`

