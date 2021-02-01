
# Automated Model Design and Benchmarking of 3D Deep Learning Models for COVID-19 Detection with Chest CT Scans

Accepted in AAAI-2021.

```bib
@article{He2021CovidNet3D, 
  title={Automated Model Design and Benchmarking of 3D Deep Learning Models for COVID-19 Detection with Chest CT Scans}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={He, Xin and Wang, Shihao and Chu, Xiaowen and Shi, Shaohuai and Tang, Jiangping and Liu, Xin and Yan, Chenggang and Zhang, Jiyong and Ding, Guiguang}, 
  year={2021}
}
```

# Dependences

```python3
pip install -r requirements.txt
```

# search

```bash
bash scripts/search_ct.sh
```

A logger directory will be created according to the `logger.name` in config file, with the following structure:

Supporse `logger.name=MyExp`：
```bash
|_output
    |_MyExp
        |_version_0 ()
            |_epoch_0.json
            |_last.pth
            |_best_acc{}_epoch{}.pth
            |_log.txt
            |_search_ct.yaml
        |_version_1()
```

- `epoch_0.json, epoch_1.json, ..., epoch_N.json` are the architectures of different epochs.
- `last.pth` is the latest checkpoint
- `best_acc{}_epoch{}.pth` is the best checkpoint
- `log.txt`
- `search_ct.yaml` is the backup config file, which will be used in the retraining stage


# retrain

```bash
bash scripts/retrain_ct.sh
```

The commands in `retrain_ct.sh` are as follows:

```bash
srun -n 1 --cpus-per-task 2 python -m ipdb retrain.py \
--config_file outputs/checkpoint/version_0/search_ct.yaml \
--arc_path outputs/checkpoint/version_0/epoch_0.json  \
input.size [128,128]
```

You should manually set `config_file` and `arc_path`. The image size in the search stage is 64x64. Here, in the retraining stage, you should specify a larger image size.

`arc_path` indicates which architecture you want to retrain. You can select it based on their perfomance in the search stage.

The following directory will be created:

```python
|_output
    |_MyExp
        |_version_0 (search stage)
            |_epoch_0.json
            |_last.pth
            |_
        |_version_0_retrain_0 (retraining stage)
            |_last.pth
            |_best_acc0.96_epoch13.pth (file name records the best acc and the corresponding epoch)
            |_othe files
        |_version_0_retrain_1 (results of other architectures if you select other architecture json file.)
```


# Q&A

- `ModuleNotFoundError: No module named 'sklearn.neighbors._base'`

You may need to upgrade your `scikit-learn` lib.
