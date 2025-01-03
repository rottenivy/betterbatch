# Better Batch for Deep Probabilistic Time Series Forecasting (AISTATS 2024)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 1.13](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

This repository contains the PyTorch implementation for the paper
[Better Batch for Deep Probabilistic Time Series Forecasting](https://proceedings.mlr.press/v238/zheng24a/zheng24a.pdf) by [Vincent Z. Zheng](https://vincent-zheng.com/) and [Lijun Sun](https://lijunsun.github.io/). This work has been accepted at AISTATS 2024.

<p align="center">
  <img width="600" src="imgs/minibatch_new.jpg">
</p>

## Requirements

The code requires Python 3.10 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Training and Evaluation

### 1. Obtain the Dataset

Use the `generate_datasets.ipynb` notebook located in the `exps` directory to transform datasets from GluonTS and .h5 files into the `TimeSeriesDataSet` format used by PyTorch Forecasting.

### 2. Train the Model

To train a baseline model on the `m4_hourly` dataset, execute the following command:

```bash
python train_baseline.py --model deepar --dataset m4_hourly
```

To train the proposed model on the `m4_hourly` dataset, execute the following command:

```bash
python train_deepar_batch.py --model deepar --dataset m4_hourly --loss kernel --num_mixture 4
```

## Reference
```bash
@inproceedings{zheng2024better,
  title={Better Batch for Deep Probabilistic Time Series Forecasting},
  author={Zheng, Zhihao and Choi, Seongjin and Sun, Lijun},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={91--99},
  year={2024},
  organization={PMLR}
}
```
