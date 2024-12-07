import warnings
warnings.filterwarnings("ignore")
import pickle
import argparse
import os
import yaml
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting.metrics.distributions import NormalDistributionLoss
from pytorch_forecasting.metrics.quantile import QuantileLoss
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
)

from pyro.ops.stats import crps_empirical
from model import LSTMModel, TemporalFusionTransformer, ARTransformer

# import wandb
# wandb.login()

matplotlib.use("Agg")
pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default="deepar")
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

    validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.time_idx <= validation_cutoff], min_prediction_idx=training_cutoff + 1)
    testing = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=validation_cutoff + 1)

    train_dataloader = training.to_dataloader(
        train=True, batch_size=args.batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0
    )
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0
    )

    ################################## Initialize Model ##################################
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_top_k=1, monitor="val_loss", mode="min")

    f = open("./config/%s.yaml"%(args.model))
    configs = yaml.load(f, Loader=yaml.Loader)
    f.close()

    logger = TensorBoardLogger(save_dir="logs", name=args.model, version="%s_vanilla_pt10"%(args.dataset))

    trainer = pl.Trainer(
        logger=logger,
        # max_epochs=1,
        max_epochs=configs['train']['max_epochs'],
        accelerator='gpu',
        devices=[args.device],
        enable_model_summary=True,
        gradient_clip_val=configs['train']['gradient_clip_val'],
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_train_batches=configs['train']['limit_train_batches'],
        enable_checkpointing=True,
    )

    if args.model == "deepar":
        model = DeepAR
        net = model.from_dataset(
            training,
            cell_type=configs['model']['cell_type'],
            hidden_size=configs['model']['hidden_size'],
            rnn_layers=configs['model']['rnn_layers'],
            dropout=configs['model']['dropout'],
            learning_rate=configs['model']['learning_rate'],
            loss=NormalDistributionLoss(),
            optimizer="adam"
        )
    elif args.model == "lstm":
        model = LSTMModel
        net = model.from_dataset(
            training,
            hidden_size=configs['model']['hidden_size'],
            n_layers=configs['model']['n_layers'],
            dropout=configs['model']['dropout'],
            learning_rate=configs['model']['learning_rate'],
            loss=NormalDistributionLoss(),
            optimizer="adam"
        )
    elif args.model == "tft":
        if len(time_varying_known_cats) == 0:
            sys.exit(0)
        model = TemporalFusionTransformer
        net = model.from_dataset(
            training,
            learning_rate=configs['model']['learning_rate'],
            hidden_size=configs['model']['hidden_size'],
            attention_head_size=configs['model']['attention_head_size'],
            dropout=configs['model']['dropout'],
            hidden_continuous_size=configs['model']['hidden_continuous_size'],
            loss=NormalDistributionLoss(),
            optimizer="adam",
        )
    elif args.model == "gpt":
        model = ARTransformer
        net = model.from_dataset(
            training,
            n_heads=configs['model']['n_heads'],
            hidden_size=configs['model']['hidden_size'],
            rnn_layers=configs['model']['rnn_layers'],
            dropout=configs['model']['dropout'],
            learning_rate=configs['model']['learning_rate'],
            loss=NormalDistributionLoss(),
            optimizer="adam"
        )
    else:
        raise Exception("Choose among deepar, lstm, and tft")

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    ################################## Evaluate Model ##################################
    # best_model = model.load_from_checkpoint("./logs/gpt/m4_hourly_vanilla/checkpoints/epoch=97-val_loss=1.10.ckpt")
    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

    actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])

    metrics = []
    for i in range(3):
        raw_predictions = best_model.predict(test_dataloader, mode="raw", n_samples=100)

        crps = crps_empirical(raw_predictions['prediction'].permute(2, 0, 1), actuals)
        crps_mean = crps.mean()
        crps_sum = (crps.sum()/actuals.sum()).item()

        ql = QuantileLoss(quantiles=[0.5, 0.9])
        ql05 = ql.loss(raw_predictions['prediction'], actuals)[...,0]
        ql09 = ql.loss(raw_predictions['prediction'], actuals)[...,1]

        p05_risk = (ql05.sum()/actuals.sum()).item()
        p09_risk = (ql09.sum()/actuals.sum()).item()

        metrics.append([crps_mean, crps_sum, p05_risk, p09_risk])

    metrics = np.array(metrics)
    metrics = np.concatenate([metrics.mean(0).reshape(-1, 1), metrics.std(0).reshape(-1, 1)], axis=1)

    if not os.path.isdir("./metrics/%s"%(args.model)):
        os.makedirs("./metrics/%s"%(args.model))

    with open('./metrics/%s/%s_vanilla_pt10.txt'%(args.model, args.dataset), 'w') as f:
        for i in range(metrics.shape[0]):
            if i != metrics.shape[0]-1:
                f.write(r'& %.4f$\pm$%.4f'%(metrics[i, 0], metrics[i, 1]))
            else:
                f.write(r'& %.4f$\pm$%.4f \n'%(metrics[i, 0], metrics[i, 1]))

    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    # wandb.init(config=args)
    score = main()
    # wandb.log({'val_loss': score})
    # wandb.finish()
