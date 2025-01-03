import warnings
warnings.filterwarnings("ignore")
import pickle
import argparse
import os
import glob
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib

# from model import DeepAR_nopred
import pytorch_lightning as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting.metrics.quantile import QuantileLoss
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
)

import properscoring as ps
from pyro.ops.stats import crps_empirical
from model import ARTransformer

matplotlib.use("Agg")
pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default="deepar")
parser.add_argument('--dataset', type=str, default="m4_hourly")
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


with open('./datasets/pred_horizon_dict.pkl', 'rb') as f:
    pred_horizon_dict = pickle.load(f)
with open('./datasets/pred_rolling_dict.pkl', 'rb') as f:
    pred_rolling_dict = pickle.load(f)
with open('./datasets/dataset_freq.pkl', 'rb') as f:
    dataset_freq_dict = pickle.load(f)
args.prediction_horizon = pred_horizon_dict[args.dataset]
args.num_pred_rolling = pred_rolling_dict[args.dataset] if args.dataset in pred_rolling_dict.keys() else 1


def main():
    ################################## Load Data ##################################
    data = pd.read_csv("./datasets/%s.csv"%(args.dataset))
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
    context_length = args.prediction_horizon
    validation_cutoff = data["time_idx"].max() - args.prediction_horizon - args.num_pred_rolling + 1
    training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        target_normalizer=GroupNormalizer(groups=["sensor"], transformation=None),
        categorical_encoders={"series": NaNLabelEncoder().fit(data.sensor)},
        group_ids=["sensor"],
        static_categoricals=["sensor"],
        time_varying_known_categoricals=time_varying_known_cats,
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=args.prediction_horizon,
        allow_missing_timesteps=False,
    )

    testing = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=validation_cutoff + 1)
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0
    )

    ################################## Initialize Model ##################################
    logger = TensorBoardLogger(save_dir="logs", name=args.model, version="%s_vanilla"%(args.dataset))
    best_model_path = glob.glob(os.path.join('.', logger.log_dir, 'checkpoints', '*'))[0]

    if args.model == "deepar":
        model = DeepAR
    elif args.model == "gpt":
        model = ARTransformer
    else:
        raise Exception("Choose among deepar, lstm, and tft")

    ################################## Evaluate Model ##################################
    best_model = model.load_from_checkpoint(best_model_path)

    actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])

    metrics = []
    for i in range(3):
        raw_predictions = best_model.predict(test_dataloader, mode="raw", n_samples=100)

        # test wtih closed-form CRPS
        mu = raw_predictions['prediction'].mean(-1)
        sig = raw_predictions['prediction'].std(-1)
        crps = ps.crps_gaussian(actuals, mu=mu, sig=sig)
        crps_sum = (crps.sum()/actuals.sum()).item()

        # test with empirical CRPS
        # crps = crps_empirical(raw_predictions['prediction'].permute(2, 0, 1), actuals)
        # crps_sum = (crps.sum()/actuals.sum()).item()

        ql = QuantileLoss(quantiles=[0.5, 0.9])
        ql05 = ql.loss(raw_predictions['prediction'], actuals)[...,0]
        ql09 = ql.loss(raw_predictions['prediction'], actuals)[...,1]

        p05_risk = (ql05.sum()/actuals.sum()).item()
        p09_risk = (ql09.sum()/actuals.sum()).item()

        mse = torch.mean((raw_predictions['prediction'].mean(-1)-actuals)**2)

        metrics.append([crps_sum, p05_risk, p09_risk, mse])

    metrics = np.array(metrics)
    metrics = np.concatenate([metrics.mean(0).reshape(-1, 1), metrics.std(0).reshape(-1, 1)], axis=1)

    if not os.path.isdir("./metrics/%s"%(args.model)):
        os.makedirs("./metrics/%s"%(args.model))

    with open('./metrics/%s/%s_vanilla.txt'%(args.model, args.dataset), 'w') as f:
        for i in range(metrics.shape[0]):
            if i != metrics.shape[0]-1:
                f.write('& %.4f$\pm$%.4f'%(metrics[i, 0], metrics[i, 1]))
            else:
                f.write('& %.4f$\pm$%.4f \n'%(metrics[i, 0], metrics[i, 1]))


if __name__ == "__main__":
    main()
