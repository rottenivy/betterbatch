import warnings
warnings.filterwarnings("ignore")
import pickle
import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
import pmdarima as pm

import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    NaNLabelEncoder,
)

matplotlib.use("Agg")
pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default="arima")
parser.add_argument('--dataset', type=str, default="toy_example")
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


with open('../../data/pytorch_forecsating_datasets/pred_horizon_dict.pkl', 'rb') as f:
    pred_horizon_dict = pickle.load(f)
with open('../../data/pytorch_forecsating_datasets/pred_rolling_dict.pkl', 'rb') as f:
    pred_rolling_dict = pickle.load(f)
with open('../../data/pytorch_forecsating_datasets/dataset_freq.pkl', 'rb') as f:
    dataset_freq_dict = pickle.load(f)
args.prediction_horizon = pred_horizon_dict[args.dataset]
args.num_pred_rolling = pred_rolling_dict[args.dataset] if args.dataset in pred_rolling_dict.keys() else 1


def main():
    ################################## Load Data ##################################
    data = pd.read_csv("../../data/pytorch_forecsating_datasets/%s.csv"%(args.dataset))
    if dataset_freq_dict[args.dataset] in ['30min', '5min', 'H', 'T']:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['tod'] = (data['datetime'].values - data['datetime'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        data['dow'] = data['datetime'].dt.weekday
        time_varying_known_cats = ['tod', 'dow']
        data = data.astype(dict(sensor=str, tod=str, dow=str))
    elif dataset_freq_dict[args.dataset] in ['B', 'D']:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['dow'] = data['datetime'].dt.weekday
        time_varying_known_cats = ['dow']
        data = data.astype(dict(sensor=str, dow=str))
    else:
        time_varying_known_cats = []
        data = data.astype(dict(sensor=str))

    ################################## Create Dataloaders ##################################
    validation_cutoff = data["time_idx"].max() - args.prediction_horizon - args.num_pred_rolling + 1
    training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        target_normalizer=None,
        categorical_encoders={"series": NaNLabelEncoder().fit(data.sensor)},
        group_ids=["sensor"],
        static_categoricals=["sensor"],
        time_varying_known_categoricals=time_varying_known_cats,
        time_varying_unknown_reals=["value"],
        max_encoder_length=args.prediction_horizon*10,
        max_prediction_length=args.prediction_horizon,
        allow_missing_timesteps=False,
    )

    testing = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=validation_cutoff + 1)
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0
    )

    ################################## Evaluate Model ##################################
    # metrics = []
    # for i in range(3):
    actuals = []
    raw_predictions = []
    for x, _ in test_dataloader:
        # train = testing.target_normalizer.inverse_transform(x["encoder_cont"][...,0])
        # test = testing.target_normalizer.inverse_transform(x["decoder_cont"][...,0])
        train = x["encoder_cont"][...,0]
        test = x["decoder_cont"][...,0]
        preds = []
        for i in range(train.shape[0]):
            preds.append(pm.auto_arima(train[i].numpy()).predict(n_periods=test.shape[1]))
        preds = np.stack(preds, axis=0)

        actuals.append(test.numpy())
        raw_predictions.append(preds)

    actuals = np.concatenate(actuals)
    raw_predictions = np.concatenate(raw_predictions)

    ql05 = np.abs(raw_predictions-actuals)/2
    p05_risk = ql05.sum()/actuals.sum()
    # mse
    mse = np.mean((raw_predictions-actuals)**2)
    metrics = [[p05_risk], [mse]]
        # metrics.append([p05_risk])

    metrics = np.array(metrics)
    # metrics = np.concatenate([metrics.mean(0).reshape(-1, 1), metrics.std(0).reshape(-1, 1)], axis=1)

    if not os.path.isdir("./metrics/%s"%(args.model)):
        os.makedirs("./metrics/%s"%(args.model))

    with open('./metrics/%s/%s.txt'%(args.model, args.dataset), 'w') as f:
        for i in range(metrics.shape[0]):
            if i != metrics.shape[0]-1:
                f.write('& %.4f '%(metrics[i, 0]))
            else:
                f.write('& %.4f \n'%(metrics[i, 0]))


if __name__ == "__main__":
    main()