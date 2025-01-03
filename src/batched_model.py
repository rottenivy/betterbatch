from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from pytorch_forecasting import DeepAR
from pytorch_forecasting.models.nn import HiddenState
from pytorch_forecasting.models.base_model import _torch_cat_na, _concatenate_output
from pytorch_forecasting.metrics.base_metrics import DistributionLoss
from pytorch_forecasting.metrics import MultiLoss
from pytorch_forecasting.utils import apply_to_list, to_list

from pytorch_forecasting.optim import Ranger
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    DistributionLoss,
    MultiLoss,
)
from pytorch_forecasting.optim import Ranger
from pytorch_forecasting.utils import (
    apply_to_list,
    create_mask,
    move_to_device,
    to_list,
)

from model import ARTransformer
from loss import BatchMGD_Kernel


class BatchDeepAREstimator(DeepAR):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.loss.K > 1:
            if isinstance(self.loss, BatchMGD_Kernel):
                # self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K+1), nn.Softmax(dim=-1))
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K+1), nn.Softmax(dim=-1))  # TODO: better than ELU
            else:
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K), nn.Softmax(dim=-1))
        elif not self.loss.static:
            self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, 1), nn.Sigmoid())

    def configure_optimizers(self):
        # either set a schedule of lrs or find it dynamically
        if self.hparams.optimizer_params is None:
            optimizer_params = {}
        else:
            optimizer_params = self.hparams.optimizer_params
        # set optimizer
        lrs = self.hparams.learning_rate
        if isinstance(lrs, (list, tuple)):
            lr = lrs[0]
        else:
            lr = lrs

        # assign parameter groups
        params = list(self.named_parameters())
        # grouped_parameters = [
        # {"params": [p for n, p in params if n.split('.')[0] == 'loss'], 'lr': self.loss.lr},
        # {"params": [p for n, p in params if n.split('.')[0] != 'loss'], 'lr': lr}]
        grouped_parameters = [
        {"params": [p for n, p in params if n.split('.')[0] in ['loss', 'mixture_projector']], 'lr': self.loss.lr, 'weight_decay': self.loss.lr/10},  # TODO add mixture projecter to this
        {"params": [p for n, p in params if n.split('.')[0] not in ['loss', 'mixture_projector']], 'lr': lr}]

        if callable(self.optimizer):
            try:
                optimizer = self.optimizer(
                    grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            except TypeError:  # in case there is no weight decay
                optimizer = self.optimizer(grouped_parameters, lr=lr, **optimizer_params)
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif self.hparams.optimizer == "ranger":
            optimizer = Ranger(grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif hasattr(torch.optim, self.hparams.optimizer):
            try:
                optimizer = getattr(torch.optim, self.hparams.optimizer)(
                    grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            except TypeError:  # in case there is no weight decay
                optimizer = getattr(torch.optim, self.hparams.optimizer)(grouped_parameters, lr=lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer of self.hparams.optimizer={self.hparams.optimizer} unknown")

        # set scheduler
        if isinstance(lrs, (list, tuple)):  # change for each epoch
            # normalize lrs
            lrs = np.array(lrs) / lrs[0]
            scheduler_config = {
                "scheduler": LambdaLR(optimizer, lambda epoch: lrs[min(epoch, len(lrs) - 1)]),
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }
        elif self.hparams.reduce_on_plateau_patience is None:
            scheduler_config = {}
        else:  # find schedule based on validation loss
            scheduler_config = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=1.0 / self.hparams.reduce_on_plateau_reduction,
                    patience=self.hparams.reduce_on_plateau_patience,
                    cooldown=self.hparams.reduce_on_plateau_patience,
                    min_lr=self.hparams.reduce_on_plateau_min_lr,
                ),
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn(x, hidden_state, lengths=lengths, enforce_sorted=False)
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output, decoder_output, hidden_state

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            """
            output: the output features h_t from the last layer of the LSTM, for each t: (N, L, H_out)
            h_n: the final hidden state for each element in the sequence: (num_layers, N, H_out)
            c_n: the final cell state for each element in the sequence: (num_layers, N, H_cell)
            """
            output, decoder_output, _ = self.decode_all(input_vector, hidden_state, lengths=decoder_lengths)
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = lagged_targets[-1]
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, _, hidden_state = self.decode_all(x, hidden_state)
                prediction = apply_to_list(prediction, lambda x: x[:, 0])  # select first time step
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
                n_samples=n_samples,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1))
        return output, decoder_output

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        outputs = []
        decoder_outputs = []
        for i in range(x["decoder_lengths"][0]):
            x["encoder_cat"] = torch.cat([x["encoder_cat"][:,i:], x["decoder_cat"][:,:i]], dim=1)
            x["encoder_cont"] = torch.cat([x["encoder_cont"][:,i:], x["decoder_cont"][:,:i]], dim=1)
            hidden_state = self.encode(x) # (num_layer, batch_size, H)
            # decode
            input_vector = self.construct_input_vector(
                x["decoder_cat"][:,i:i+1],
                x["decoder_cont"][:,i:i+1],
                one_off_target=x["encoder_cont"][
                    torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                    x["encoder_lengths"] - 1,
                    self.target_positions.unsqueeze(-1),
                ].T.contiguous(),
            )  # (batch_size (N in DeepAR), Q, H)

            if self.training:
                assert n_samples is None, "cannot sample from decoder when training"
            output, decoder_output = self.decode(
                input_vector,
                decoder_lengths=torch.ones_like(x["decoder_lengths"], dtype=torch.int),
                target_scale=x["target_scale"],
                hidden_state=hidden_state,
                n_samples=n_samples,
            ) # (batch_size (N in DeepAR), Q, dist_proj (loc_scaler, scale_scaler, loc, scale)
            # return relevant part
            outputs.append(output)
            decoder_outputs.append(decoder_output)
        output = torch.cat(outputs, dim=1)

        decoder_output = torch.cat(decoder_outputs, dim=1)

        if not self.loss.static:
            mixture_weights = self.mixture_projector(decoder_output)
            output = torch.cat([output, mixture_weights], dim=-1)

        return self.to_network_output(prediction=output)


class BatchDeepARPredictor(DeepAR):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.loss.K > 1:
            if isinstance(self.loss, BatchMGD_Kernel):
                # self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K+1), nn.Softmax(dim=-1))
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K+1), nn.Softmax(dim=-1))
            else:
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K), nn.Softmax(dim=-1))
        elif not self.loss.static:
            self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, 1), nn.Sigmoid())

    def output_to_prediction(
        self,
        normalized_prediction_parameters: torch.Tensor,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int = 1,
        pre_normed_prediction_params: torch.Tensor = None,
        x_1: torch.Tensor = None,  # pre_normed_outputs
        current_decoder_output: torch.Tensor = None,
        **kwargs,
        ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        single_prediction = to_list(normalized_prediction_parameters)[0].ndim == 2
        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space
        prediction_parameters = self.transform_output(
            prediction=normalized_prediction_parameters, target_scale=target_scale, **kwargs
        )

        corr_list = [self.loss.get_corr(self.loss.c_list[i]) for i in range(self.loss.K)]

        if self.loss.K > 1:
            if isinstance(self.loss, BatchMGD_Kernel):
                corr_list += [self.loss.identity]
            mixture_weights = self.mixture_projector(current_decoder_output)
            corr = sum([corr_list[i].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0)*mixture_weights[...,i:i+1] for i in range(len(corr_list))])
        else:
            if self.loss.static:
                sigma = F.sigmoid(self.loss.sigma)
                corr = (1-sigma)*corr_list[0] + sigma*self.loss.identity
            else:
                sigma = self.mixture_projector(current_decoder_output)
                corr = corr_list[0].unsqueeze(0).repeat_interleave(sigma.shape[0], 0)
                corr = (1-sigma)*corr + sigma*self.loss.identity

        pre_prediction_parameters = self.transform_output(
        prediction=pre_normed_prediction_params, target_scale=target_scale, **kwargs
        )
        prediction_params_all = torch.cat([pre_prediction_parameters, prediction_parameters], dim=1)

        cov_mat = torch.diag_embed(prediction_params_all[..., 3])@corr@torch.diag_embed(prediction_params_all[..., 3])
        cov_21 = cov_mat[:, -1:, :-1]
        cov_11 = cov_mat[:, :-1, :-1]
        cov_22 = cov_mat[:, -1:, -1:]

        mu_1 = prediction_params_all[:, :-1, 2:3]
        mu_2 = prediction_params_all[:, -1:, 2:3]

        # use all avaiable observations
        mu_21 = mu_2 + cov_21@torch.inverse(cov_11)@(x_1-mu_1)
        sigma_21 = cov_22 - cov_21@torch.inverse(cov_11)@cov_21.mT

        prediction_parameters[...,2] = mu_21[...,0]
        # prediction_parameters[...,3] = torch.sqrt(sigma_21[...,0])
        prediction_parameters[...,3] = torch.nan_to_num(torch.sqrt(sigma_21[...,0]), nan=0.00000000001)

        # todo: handle classification
        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            # todo: handle mixed losses
            if n_samples > 1:
                prediction_parameters = apply_to_list(
                    prediction_parameters, lambda x: x.reshape(int(x.size(0) / n_samples), n_samples, -1)
                )
                prediction = self.loss.sample(prediction_parameters, 1)
                prediction = apply_to_list(prediction, lambda x: x.reshape(x.size(0) * n_samples, 1, -1))
            else:
                prediction = self.loss.sample(normalized_prediction_parameters, 1)

        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)
        if isinstance(normalized_prediction, list):
            input_target = torch.cat(normalized_prediction, dim=-1)
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction

        # remove time dimension
        if single_prediction:
            prediction = apply_to_list(prediction, lambda x: x.squeeze(1))
            input_target = input_target.squeeze(1)

        if self.loss.static:
            return prediction, input_target, torch.ones_like(prediction, device=prediction.device)
        else:
            if self.loss.K > 1:
                return prediction, input_target, mixture_weights.squeeze(1)
            else:
                return prediction, input_target, sigma.squeeze(1)

    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        first_hidden_state: Any,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        pre_normed_outputs: torch.Tensor = None,
        pre_normed_prediction_params: torch.Tensor = None,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        # make predictions which are fed into next step
        output = []
        weights = []
        current_target = first_target
        current_hidden_state = first_hidden_state

        normalized_output = [first_target]

        # pre_normed_prediction_params = []
        for idx in range(n_decoder_steps):
            # get lagged targets
            normed_prediction_params, current_decoder_output, current_hidden_state = decode_one(
                idx, lagged_targets=normalized_output, hidden_state=current_hidden_state, **kwargs
            )  # current target: (N*n_sample, (mu, sigma))

            prediction, current_target, mixture_weights = self.output_to_prediction(
                    normed_prediction_params, target_scale=target_scale, n_samples=n_samples, pre_normed_prediction_params=pre_normed_prediction_params, x_1=pre_normed_outputs,
                    current_decoder_output=current_decoder_output
                )

            # save normalized output for lagged targets
            normalized_output.append(current_target)

            pre_normed_outputs = torch.cat([pre_normed_outputs[:,1:], current_target.unsqueeze(1)], dim=1)
            pre_normed_prediction_params = torch.cat([pre_normed_prediction_params[:,1:], normed_prediction_params.unsqueeze(1)], dim=1)

            output.append(prediction)
            weights.append(mixture_weights)
        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
            weights = torch.stack(weights, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output, weights

    def encode(self, x: Dict[str, torch.Tensor]) -> HiddenState:
        """
        Encode sequence into hidden state
        """
        # encode using rnn
        assert x["encoder_lengths"].min() > 0
        encoder_lengths = x["encoder_lengths"] - 1
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        encoder_output, hidden_state = self.rnn(
            input_vector, lengths=encoder_lengths, enforce_sorted=False
        )  # second ouput is not needed (hidden state)
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(encoder_output)
        else:
            output = [projector(encoder_output) for projector in self.distribution_projector]
        return output, hidden_state

    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn(x, hidden_state, lengths=lengths, enforce_sorted=False)
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output, decoder_output, hidden_state

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
        encoder_output: torch.Tensor = None,
        encoder_dist_params: torch.Tensor = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:  # trainig
            output, _, _ = self.decode_all(input_vector, hidden_state, lengths=decoder_lengths)  # use real values as decoder input
            output = self.transform_output(output, target_scale=target_scale)
        else:  # validating and testing
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)  # (batch_size*n_sample, prediction_length, n_features)
            hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            encoder_output = encoder_output.repeat_interleave(n_samples, 0)
            encoder_dist_params = encoder_dist_params.repeat_interleave(n_samples, 0)

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = lagged_targets[-1]
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, decoder_output, hidden_state = self.decode_all(x, hidden_state)  # prediction: (batch_size*n_sample, 1, dist_parameters), hidden_state: (n_layers, batch_size*n_sample, hidden_size)
                prediction = apply_to_list(prediction, lambda x: x[:, 0])  # select first time step
                return prediction, decoder_output, hidden_state

            # make predictions which are fed into next step
            output, weights = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
                n_samples=n_samples,
                pre_normed_outputs=encoder_output,
                pre_normed_prediction_params=encoder_dist_params,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1))
            weights = apply_to_list(weights, lambda x: x.reshape(-1, n_samples, input_vector.size(1), weights.shape[-1]).permute(0, 2, 3, 1))
        return output, weights

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        encoder_dist_params, hidden_state = self.encode(x) # (num_layer, batch_size, H)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=x["encoder_cont"][
                torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                x["encoder_lengths"] - 1,
                self.target_positions.unsqueeze(-1),
            ].T.contiguous(),
        )  # (batch_size (N in DeepAR), Q, H)

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output, weights = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=hidden_state,
            n_samples=n_samples,
            encoder_output=x["encoder_cont"][:,-self.loss.batch_cov_horizon+1:],
            encoder_dist_params=encoder_dist_params[:,-self.loss.batch_cov_horizon+1:],
        ) # (batch_size (N in DeepAR), Q, dist_proj (loc_scaler, scale_scaler, loc, scale)
        # return relevant part
        return self.to_network_output(prediction=output), self.to_network_output(prediction=weights)

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        return_w: bool = False,
        mode_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Run inference / prediction.

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles", or "raw", or tuple ``("raw", output_name)`` where output_name is
                a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index (in the same order as the output, i.e. the row of the
                dataframe corresponds to the first dimension of the output and the given time index is the time index
                of the first prediction)
            return_decoder_lengths: if to return decoder_lengths (in the same order as the output
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs (in the same order as prediction output)
            mode_kwargs (Dict[str, Any]): keyword arguments for ``to_prediction()`` or ``to_quantiles()``
                for modes "prediction" and "quantiles"
            **kwargs: additional arguments to network's forward method

        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        # convert to dataloader
        if isinstance(data, pd.DataFrame):
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        if isinstance(data, TimeSeriesDataSet):
            dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
        else:
            dataloader = data

        # mode kwargs default to None
        if mode_kwargs is None:
            mode_kwargs = {}

        # ensure passed dataloader is correct
        assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

        # prepare model
        self.eval()  # no dropout, etc. no gradients

        # run predictions
        output = []
        decode_lenghts = []
        x_list = []
        index = []
        w_list = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
        with torch.no_grad():
            for x, _ in dataloader:
                # move data to appropriate device
                data_device = x["encoder_cont"].device
                if data_device != self.device:
                    x = move_to_device(x, self.device)

                # make prediction
                out, w = self(x, **kwargs)  # raw output is dictionary

                lengths = x["decoder_lengths"]
                if return_decoder_lengths:
                    decode_lenghts.append(lengths)
                nan_mask = create_mask(lengths.max(), lengths)
                if isinstance(mode, (tuple, list)):
                    if mode[0] == "raw":
                        out = out[mode[1]]
                    else:
                        raise ValueError(
                            f"If a tuple is specified, the first element must be 'raw' - got {mode[0]} instead"
                        )
                elif mode == "prediction":
                    out = self.to_prediction(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask, torch.tensor(float("nan"))) if o.dtype == torch.float else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:  # only floats can be filled with nans
                        out = out.masked_fill(nan_mask, torch.tensor(float("nan")))
                elif mode == "quantiles":
                    out = self.to_quantiles(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                            if o.dtype == torch.float
                            else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:
                        out = out.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                elif mode == "raw":
                    pass
                else:
                    raise ValueError(f"Unknown mode {mode} - see docs for valid arguments")

                out = move_to_device(out, device="cpu")

                output.append(out)
                if return_x:
                    x = move_to_device(x, "cpu")
                    x_list.append(x)
                if return_index:
                    index.append(dataloader.dataset.x_to_index(x))
                if return_w:
                    w_list.append(w)
                progress_bar.update()
                if fast_dev_run:
                    break

        # concatenate output (of different batches)
        if isinstance(mode, (tuple, list)) or mode != "raw":
            if isinstance(output[0], (tuple, list)) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
                output = [_torch_cat_na([out[idx] for out in output]) for idx in range(len(output[0]))]
            else:
                output = _torch_cat_na(output)
        elif mode == "raw":
            output = _concatenate_output(output)

        # generate output
        if return_x or return_index or return_decoder_lengths or return_w:
            output = [output]
        if return_x:
            output.append(_concatenate_output(x_list))
        if return_index:
            output.append(pd.concat(index, axis=0, ignore_index=True))
        if return_decoder_lengths:
            output.append(torch.cat(decode_lenghts, dim=0))
        if return_w:
            output.append(_concatenate_output(w_list))
        return output


class BatchGPTEstimator(ARTransformer):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.loss.K > 1:
            if isinstance(self.loss, BatchMGD_Kernel):
                # self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K+1), nn.Softmax(dim=-1))
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K+1), nn.Softmax(dim=-1))  # TODO: better than ELU
            else:
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K), nn.Softmax(dim=-1))
        elif not self.loss.static:
            self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, 1), nn.Sigmoid())

    def configure_optimizers(self):
        # either set a schedule of lrs or find it dynamically
        if self.hparams.optimizer_params is None:
            optimizer_params = {}
        else:
            optimizer_params = self.hparams.optimizer_params
        # set optimizer
        lrs = self.hparams.learning_rate
        if isinstance(lrs, (list, tuple)):
            lr = lrs[0]
        else:
            lr = lrs

        # assign parameter groups
        params = list(self.named_parameters())
        # grouped_parameters = [
        # {"params": [p for n, p in params if n.split('.')[0] == 'loss'], 'lr': self.loss.lr},
        # {"params": [p for n, p in params if n.split('.')[0] != 'loss'], 'lr': lr}]
        grouped_parameters = [
        {"params": [p for n, p in params if n.split('.')[0] in ['loss', 'mixture_projector']], 'lr': self.loss.lr, 'weight_decay': self.loss.lr/10},  # TODO add mixture projecter to this
        {"params": [p for n, p in params if n.split('.')[0] not in ['loss', 'mixture_projector']], 'lr': lr}]

        if callable(self.optimizer):
            try:
                optimizer = self.optimizer(
                    grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            except TypeError:  # in case there is no weight decay
                optimizer = self.optimizer(grouped_parameters, lr=lr, **optimizer_params)
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif self.hparams.optimizer == "ranger":
            optimizer = Ranger(grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
            )
        elif hasattr(torch.optim, self.hparams.optimizer):
            try:
                optimizer = getattr(torch.optim, self.hparams.optimizer)(
                    grouped_parameters, lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            except TypeError:  # in case there is no weight decay
                optimizer = getattr(torch.optim, self.hparams.optimizer)(grouped_parameters, lr=lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer of self.hparams.optimizer={self.hparams.optimizer} unknown")

        # set scheduler
        if isinstance(lrs, (list, tuple)):  # change for each epoch
            # normalize lrs
            lrs = np.array(lrs) / lrs[0]
            scheduler_config = {
                "scheduler": LambdaLR(optimizer, lambda epoch: lrs[min(epoch, len(lrs) - 1)]),
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }
        elif self.hparams.reduce_on_plateau_patience is None:
            scheduler_config = {}
        else:  # find schedule based on validation loss
            scheduler_config = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=1.0 / self.hparams.reduce_on_plateau_reduction,
                    patience=self.hparams.reduce_on_plateau_patience,
                    cooldown=self.hparams.reduce_on_plateau_patience,
                    min_lr=self.hparams.reduce_on_plateau_min_lr,
                ),
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def decode_all(
        self,
        input_vector: torch.Tensor,
        decoder_length: torch.Tensor = None,
    ):
        src = self.add_input_vector(input_vector)
        # src = self.pos_encoder(input_vector.permute(1,0,2)).permute(1,0,2)
        decoder_output = self.rnn(src, mask=nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device))
        decoder_output = decoder_output[:,-decoder_length:]
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output, decoder_output

    def decode(
        self,
        input_vector: torch.Tensor,
        decoder_length: int,
        target_scale: torch.Tensor,
        n_samples: int = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            output, decoder_output = self.decode_all(input_vector, decoder_length)
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                decoder_length
            ):
                x = input_vector[:, :decoder_length+idx]
                lagged_targets = torch.stack(lagged_targets, dim=1)
                x[:, decoder_length-1:, target_pos] = lagged_targets
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction = self.decode_all(x, lagged_targets.shape[1])
                prediction = apply_to_list(prediction, lambda x: x[:, -1])  # select first time step
                return prediction

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, -decoder_length, target_pos],
                target_scale=target_scale,
                n_decoder_steps=decoder_length,
                n_samples=n_samples,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, decoder_length).permute(0, 2, 1))
        return output, decoder_output

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        # encoder_output = self.encode(x)
        # encoder_length = encoder_output.shape[1]

        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)

        input_vector = self.construct_input_vector(x_cat, x_cont)
        decoder_length = int(x["decoder_lengths"].max())

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"

        output, decoder_output = self.decode(
            input_vector,
            decoder_length=decoder_length,
            target_scale=x["target_scale"],
            n_samples=n_samples,
        )

        if not self.loss.static:
            mixture_weights = self.mixture_projector(decoder_output)
            output = torch.cat([output, mixture_weights], dim=-1)

        return self.to_network_output(prediction=output)


class BatchGPTPredictor(ARTransformer):
    """
    1-step DeepAR Model, with conditional sampling
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.loss.K > 1:
            if isinstance(self.loss, BatchMGD_Kernel):
                # self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, self.loss.K+1), nn.Softmax(dim=-1))
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K+1), nn.Softmax(dim=-1))
            else:
                self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, self.loss.K), nn.Softmax(dim=-1))
        elif not self.loss.static:
            self.mixture_projector = nn.Sequential(nn.Linear(self.hparams.hidden_size, 20), nn.ELU(), nn.Linear(20, 1), nn.Sigmoid())

    def output_to_prediction(
        self,
        normalized_prediction_parameters: torch.Tensor,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int = 1,
        pre_normed_prediction_params: torch.Tensor = None,
        x_1: torch.Tensor = None,  # pre_normed_outputs
        current_decoder_output: torch.Tensor = None,
        **kwargs,
        ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        single_prediction = to_list(normalized_prediction_parameters)[0].ndim == 2
        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space
        prediction_parameters = self.transform_output(
            prediction=normalized_prediction_parameters, target_scale=target_scale, **kwargs
        )

        corr_list = [self.loss.get_corr(self.loss.c_list[i]) for i in range(self.loss.K)]

        if self.loss.K > 1:
            if isinstance(self.loss, BatchMGD_Kernel):
                corr_list += [self.loss.identity]
            mixture_weights = self.mixture_projector(current_decoder_output)
            corr = sum([corr_list[i].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0)*mixture_weights[...,i:i+1] for i in range(len(corr_list))])
        else:
            if self.loss.static:
                sigma = F.sigmoid(self.loss.sigma)
                corr = (1-sigma)*corr_list[0] + sigma*self.loss.identity
            else:
                sigma = self.mixture_projector(current_decoder_output)
                corr = corr_list[0].unsqueeze(0).repeat_interleave(sigma.shape[0], 0)
                corr = (1-sigma)*corr + sigma*self.loss.identity

        pre_prediction_parameters = self.transform_output(
        prediction=pre_normed_prediction_params, target_scale=target_scale, **kwargs
        )
        prediction_params_all = torch.cat([pre_prediction_parameters, prediction_parameters], dim=1)

        cov_mat = torch.diag_embed(prediction_params_all[..., 3])@corr@torch.diag_embed(prediction_params_all[..., 3])
        cov_21 = cov_mat[:, -1:, :-1]
        cov_11 = cov_mat[:, :-1, :-1]
        cov_22 = cov_mat[:, -1:, -1:]

        mu_1 = prediction_params_all[:, :-1, 2:3]
        mu_2 = prediction_params_all[:, -1:, 2:3]

        # use all avaiable observations
        mu_21 = mu_2 + cov_21@torch.inverse(cov_11)@(x_1-mu_1)
        sigma_21 = cov_22 - cov_21@torch.inverse(cov_11)@cov_21.mT

        prediction_parameters[...,2] = mu_21[...,0]
        # prediction_parameters[...,3] = torch.sqrt(sigma_21[...,0])
        prediction_parameters[...,3] = torch.nan_to_num(torch.sqrt(sigma_21[...,0]), nan=0.00000000001)

        # todo: handle classification
        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            # todo: handle mixed losses
            if n_samples > 1:
                prediction_parameters = apply_to_list(
                    prediction_parameters, lambda x: x.reshape(int(x.size(0) / n_samples), n_samples, -1)
                )
                prediction = self.loss.sample(prediction_parameters, 1)
                prediction = apply_to_list(prediction, lambda x: x.reshape(x.size(0) * n_samples, 1, -1))
            else:
                prediction = self.loss.sample(normalized_prediction_parameters, 1)

        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)
        if isinstance(normalized_prediction, list):
            input_target = torch.cat(normalized_prediction, dim=-1)
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction

        # remove time dimension
        if single_prediction:
            prediction = apply_to_list(prediction, lambda x: x.squeeze(1))
            input_target = input_target.squeeze(1)

        if self.loss.static:
            return prediction, input_target, torch.ones_like(prediction, device=prediction.device)
        else:
            if self.loss.K > 1:
                return prediction, input_target, mixture_weights.squeeze(1)
            else:
                return prediction, input_target, sigma.squeeze(1)

    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        pre_normed_outputs: torch.Tensor = None,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:

        # make predictions which are fed into next step
        output = []
        weights = []
        normalized_output = [first_target]
        for idx in range(n_decoder_steps):
            # get lagged targets
            normed_prediction_params, current_decoder_output = decode_one(
                idx, lagged_targets=normalized_output, decoder_length=n_decoder_steps, **kwargs
            )

            # get prediction and its normalized version for the next step
            prediction, current_target, mixture_weights = self.output_to_prediction(
                normalized_prediction_parameters=normed_prediction_params[:, -1],
                target_scale=target_scale,
                n_samples=n_samples, pre_normed_prediction_params=normed_prediction_params[:, :-1][:,-self.loss.batch_cov_horizon+1:],
                x_1=pre_normed_outputs,
                current_decoder_output=current_decoder_output
            )
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            pre_normed_outputs = torch.cat([pre_normed_outputs[:,1:], current_target.unsqueeze(1)], dim=1)

            output.append(prediction)
            weights.append(mixture_weights)
        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
            weights = torch.stack(weights, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output, weights

    def decode_all(
        self,
        input_vector: torch.Tensor,
        decoder_length: int,
    ):
        src = self.add_input_vector(input_vector)
        # src = self.pos_encoder(input_vector.permute(1,0,2)).permute(1,0,2)
        decoder_output = self.rnn(src, mask=nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(src.device))
        if isinstance(self.hparams.target, str):  # single target
            output = self.distribution_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.distribution_projector]
        return output, decoder_output[:,-decoder_length:]

    def decode(
        self,
        input_vector: torch.Tensor,
        decoder_length: int,
        target_scale: torch.Tensor,
        n_samples: int = None,
        encoder_output: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decode hidden state of RNN into prediction. If n_smaples is given,
        decode not by using actual values but rather by
        sampling new targets from past predictions iteratively
        """
        if n_samples is None:
            output, decoder_output = self.decode_all(input_vector, decoder_length)
            output = self.transform_output(output, target_scale=target_scale)
            return output, decoder_output
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            target_scale = apply_to_list(target_scale, lambda x: x.repeat_interleave(n_samples, 0))

            encoder_output = encoder_output.repeat_interleave(n_samples, 0)

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                decoder_length
            ):
                x = input_vector[:, :decoder_length+idx]
                lagged_targets = torch.stack(lagged_targets, dim=1)
                x[:, decoder_length-1:, target_pos] = lagged_targets
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, decoder_output = self.decode_all(x, lagged_targets.shape[1])
                return prediction, decoder_output[:, -1:]

            # make predictions which are fed into next step
            output, weights = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, -decoder_length, target_pos],
                target_scale=target_scale,
                n_decoder_steps=decoder_length,
                n_samples=n_samples,
                pre_normed_outputs=encoder_output,
            )
            output = apply_to_list(output, lambda x: x.reshape(-1, n_samples, decoder_length).permute(0, 2, 1))
            weights = apply_to_list(weights, lambda x: x.reshape(-1, n_samples, decoder_length, weights.shape[-1]).permute(0, 2, 3, 1))
            return output, weights

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)

        input_vector = self.construct_input_vector(x_cat, x_cont)
        decoder_length = int(x["decoder_lengths"].max())

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"

        output, weights = self.decode(
            input_vector,
            decoder_length=decoder_length,
            target_scale=x["target_scale"],
            n_samples=n_samples,
            encoder_output=x["encoder_cont"][:,-self.loss.batch_cov_horizon+1:],
        )

        return self.to_network_output(prediction=output), self.to_network_output(prediction=weights)

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        return_w: bool = False,
        mode_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        # convert to dataloader
        if isinstance(data, pd.DataFrame):
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        if isinstance(data, TimeSeriesDataSet):
            dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
        else:
            dataloader = data

        # mode kwargs default to None
        if mode_kwargs is None:
            mode_kwargs = {}

        # ensure passed dataloader is correct
        assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

        # prepare model
        self.eval()  # no dropout, etc. no gradients

        # run predictions
        output = []
        decode_lenghts = []
        x_list = []
        index = []
        w_list = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
        with torch.no_grad():
            for x, _ in dataloader:
                # move data to appropriate device
                data_device = x["encoder_cont"].device
                if data_device != self.device:
                    x = move_to_device(x, self.device)

                # make prediction
                out, w = self(x, **kwargs)  # raw output is dictionary

                lengths = x["decoder_lengths"]
                if return_decoder_lengths:
                    decode_lenghts.append(lengths)
                nan_mask = create_mask(lengths.max(), lengths)
                if isinstance(mode, (tuple, list)):
                    if mode[0] == "raw":
                        out = out[mode[1]]
                    else:
                        raise ValueError(
                            f"If a tuple is specified, the first element must be 'raw' - got {mode[0]} instead"
                        )
                elif mode == "prediction":
                    out = self.to_prediction(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask, torch.tensor(float("nan"))) if o.dtype == torch.float else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:  # only floats can be filled with nans
                        out = out.masked_fill(nan_mask, torch.tensor(float("nan")))
                elif mode == "quantiles":
                    out = self.to_quantiles(out, **mode_kwargs)
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                            if o.dtype == torch.float
                            else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:
                        out = out.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                elif mode == "raw":
                    pass
                else:
                    raise ValueError(f"Unknown mode {mode} - see docs for valid arguments")

                out = move_to_device(out, device="cpu")

                output.append(out)
                if return_x:
                    x = move_to_device(x, "cpu")
                    x_list.append(x)
                if return_index:
                    index.append(dataloader.dataset.x_to_index(x))
                if return_w:
                    w_list.append(w)
                progress_bar.update()
                if fast_dev_run:
                    break

        # concatenate output (of different batches)
        if isinstance(mode, (tuple, list)) or mode != "raw":
            if isinstance(output[0], (tuple, list)) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
                output = [_torch_cat_na([out[idx] for out in output]) for idx in range(len(output[0]))]
            else:
                output = _torch_cat_na(output)
        elif mode == "raw":
            output = _concatenate_output(output)

        # generate output
        if return_x or return_index or return_decoder_lengths or return_w:
            output = [output]
        if return_x:
            output.append(_concatenate_output(x_list))
        if return_index:
            output.append(pd.concat(index, axis=0, ignore_index=True))
        if return_decoder_lengths:
            output.append(torch.cat(decode_lenghts, dim=0))
        if return_w:
            output.append(_concatenate_output(w_list))
        return output

