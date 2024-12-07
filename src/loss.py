import warnings
import torch
from torch import distributions, nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics import DistributionLoss


def toeplitz(c, r):
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)


class BatchMGD_Kernel(DistributionLoss):
    """
    with BatchCovLoss only, using SE kernel for parametrizing corr.
    """
    def __init__(
        self,
        D: int = 12,
        K: int = 1,  # number of mixturex
        l: int = 2,
        lr: float = 0.001,  # individual learning rate
        static: bool = True,
        static_l: bool = True,
    ):
        super().__init__()
        self.batch_distribution = distributions.MultivariateNormal
        self.distribution_class = distributions.Normal
        self.distribution_arguments = ["loc", "scale"]

        self.lr = lr
        self.batch_cov_horizon = D
        self.K = K
        self.static = static
        self.static_l = static_l

        self.dist = nn.Parameter(torch.range(0, self.batch_cov_horizon-1), requires_grad=False)

        if self.K > 1:  # dynamic mixture
            # cand_l = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            cand_l = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            self.c_list = nn.ParameterList([nn.Parameter(torch.tensor(cand_l[i]), requires_grad=False) for i in range(self.K)])
        else: 
            self.c_list = nn.ParameterList([nn.Parameter(torch.tensor(float(l)), requires_grad=not self.static_l)])

            if self.static:  # static adjustment with identity matrix
                # self.sigma = nn.Parameter(torch.tensor(5.0), requires_grad=True)
                self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)  # TODO: change to randn, randn is worse than rand

        self.identity = nn.Parameter(torch.eye(self.batch_cov_horizon), requires_grad=False)

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        """
        used in decoder to concat network outpout with scaler
        """
        self._transformation = encoder.transformation
        loc = parameters[..., 0]
        scale = F.softplus(parameters[..., 1])
        return torch.concat(
            [target_scale.unsqueeze(1).expand(-1, loc.size(1), -1), loc.unsqueeze(-1), scale.unsqueeze(-1)], dim=-1
        )
    
    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        distr = self.distribution_class(loc=x[..., 2], scale=x[..., 3])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def kernel_fun(self, l):
        return torch.exp(-self.dist**2/F.relu(l)**2)  #TODO: relu will have problem for learnable lengthscale
    
    def get_corr(self, l):
        dist = self.kernel_fun(l)
        corr = toeplitz(dist, dist)
        return corr

    def get_cov(self, x: torch.Tensor, mixture_weights: torch.Tensor = None):
        corr_list = [self.get_corr(self.c_list[i]) for i in range(self.K)]

        if self.K > 1:
            corr_list += [self.identity]
            corr = sum([corr_list[i].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0)*mixture_weights[:,-1:,i:i+1] for i in range(len(corr_list))])
        else:
            if self.static:
                sigma = F.sigmoid(self.sigma)
                corr = (1-sigma)*corr_list[0] + sigma*self.identity
            else:
                corr = (1-mixture_weights[:,-1:])*corr_list[0].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0) + mixture_weights[:,-1:]*self.identity

        cov_mat = torch.diag_embed(x[..., 3])@corr@torch.diag_embed(x[..., 3])
        return cov_mat

    def map_x_to_distribution_batch(self, x: torch.Tensor, mixture_weights: torch.Tensor = None) -> distributions.MultivariateNormal:
        cov_mat = self.get_cov(x, mixture_weights)
        distr = self.batch_distribution(loc=x[..., 2], covariance_matrix=cov_mat)
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )
    
    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        losses = self.mask_losses(losses, lengths-self.batch_cov_horizon+1)
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = lengths
            else:
                self.losses = torch.cat([self.losses, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses = losses.sum()
            if not torch.isfinite(losses):
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        loss function: BatchCovLoss
        y_pred: (batch_size (N), Q, n_params), params for normed data
        y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
        """
        N = y_pred.shape[1]//self.batch_cov_horizon
        y_pred = y_pred[:,:self.batch_cov_horizon*N]
        y_actual = y_actual[:,:self.batch_cov_horizon*N]
        y_actual = y_actual.reshape(y_actual.shape[0], N, self.batch_cov_horizon)
        if self.static:
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)
            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_distribution_batch(y_pred[:, i]).log_prob(y_actual[:, i]).unsqueeze(-1))
            loss = torch.cat(loss, dim=-1)
        else:
            mixture_weights = y_pred[..., 4:]
            y_pred = y_pred[..., :4]

            mixture_weights = mixture_weights[:,:self.batch_cov_horizon*N]
            mixture_weights = mixture_weights.reshape(mixture_weights.shape[0], N, self.batch_cov_horizon, -1)
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)

            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_distribution_batch(y_pred[:, i], mixture_weights[:, i]).log_prob(y_actual[:, i]).unsqueeze(-1))
            loss = torch.cat(loss, dim=-1)

        return loss
    
    # def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
    #     """
    #     loss function: BatchCovLoss
    #     y_pred: (batch_size (N), Q, n_params), params for normed data
    #     y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
    #     """
    #     if self.static:
    #         loss = []
    #         for i in range(y_pred.shape[1]-self.batch_cov_horizon+1):
    #             loss.append(-self.map_x_to_distribution_batch(y_pred[:, i:i+self.batch_cov_horizon]).log_prob(y_actual[:, i:i+self.batch_cov_horizon]).unsqueeze(-1))
    #         loss = torch.cat(loss, dim=-1)
    #     else:
    #         mixture_weights = y_pred[..., 4:]
    #         y_pred = y_pred[..., :4]

    #         loss = []
    #         for i in range(y_pred.shape[1]-self.batch_cov_horizon+1):
    #             loss.append(-self.map_x_to_distribution_batch(y_pred[:, i:i+self.batch_cov_horizon], mixture_weights[:, i:i+self.batch_cov_horizon]).log_prob(y_actual[:, i:i+self.batch_cov_horizon]).unsqueeze(-1))
    #         loss = torch.cat(loss, dim=-1)

    #     return loss
 

    """
    with BatchCovLoss only, using toeplitz for parametrizing corr.
    """
    def __init__(
        self,
        D: int = 12,  # autocorrelation horizon
        K: int = 1,  # number of mixturex
        rho: float = 0.0,  # l_1 penalty on the Toeplitz
        lr: float = 0.001,  # individual learning rate
        static: bool = True,
    ):
        super().__init__()
        self.batch_distribution = distributions.MultivariateNormal
        self.distribution_class = distributions.Normal
        self.distribution_arguments = ["loc", "scale"]

        self.rho = rho
        self.lr = lr
        self.batch_cov_horizon = D
        self.K = K
        self.static = static

        self.c_list = nn.ParameterList([nn.Parameter(torch.randn(2*D-1), requires_grad=True) for i in range(K)])
        # self.c_list = nn.ParameterList([nn.Parameter(torch.rand(2*D-1), requires_grad=True) for i in range(K)])
        # self.c_list = nn.ParameterList([nn.Parameter(torch.normal(0, 1, size=(1, 2*D-1)).flatten(), requires_grad=True) for i in range(K)])

        if self.K == 1:
            if self.static:
                # self.sigma = nn.Parameter(torch.tensor(5.0), requires_grad=True)
                self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)
            self.identity = nn.Parameter(torch.eye(self.batch_cov_horizon), requires_grad=False)

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        """
        used in decoder to concat network outpout with scaler
        """
        self._transformation = encoder.transformation
        loc = parameters[..., 0]
        scale = F.softplus(parameters[..., 1])
        return torch.concat(
            [target_scale.unsqueeze(1).expand(-1, loc.size(1), -1), loc.unsqueeze(-1), scale.unsqueeze(-1)], dim=-1
        )
    
    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        distr = self.distribution_class(loc=x[..., 2], scale=x[..., 3])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def get_corr(self, c):
        c = torch.fft.irfft(torch.mul(torch.conj(torch.fft.rfft(c)), torch.fft.rfft(c)))
        c = c[:self.batch_cov_horizon]
        c = c/c.max()
        corr = toeplitz(c, c)
        return corr

    def get_cov(self, x: torch.Tensor, mixture_weights: torch.Tensor = None):
        corr_list = [self.get_corr(self.c_list[i]) for i in range(self.K)]

        if self.K > 1:
            corr = sum([corr_list[i].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0)*mixture_weights[:,-1:,i:i+1] for i in range(self.K)])
        else:
            if self.static:
                sigma = F.sigmoid(self.sigma)
                corr = (1-sigma)*corr_list[0] + sigma*self.identity
            else:
                corr = (1-mixture_weights[:,-1:])*corr_list[0].unsqueeze(0).repeat_interleave(mixture_weights.shape[0], 0) + mixture_weights[:,-1:]*self.identity

        cov_mat = torch.diag_embed(x[..., 3])@corr@torch.diag_embed(x[..., 3])  # (B, D, D)@(B, D, D)@(B, D, D)
        return cov_mat

    def get_reg(self,):
        return sum([self.get_corr(self.c_list[i])[1:,0].abs().mean() for i in range(self.K)])/self.K

    def map_x_to_distribution_batch(self, x: torch.Tensor, mixture_weights: torch.Tensor = None) -> distributions.MultivariateNormal:
        cov_mat = self.get_cov(x, mixture_weights)
        distr = self.batch_distribution(loc=x[..., 2], covariance_matrix=cov_mat)
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )
    
    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        losses = self.mask_losses(losses, lengths-self.batch_cov_horizon+1)
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = lengths
            else:
                self.losses = torch.cat([self.losses, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses = losses.sum()
            if not torch.isfinite(losses):
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        loss function: BatchCovLoss
        y_pred: (batch_size (N), Q, n_params), params for normed data
        y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
        """
        N = y_pred.shape[1]//self.batch_cov_horizon
        y_pred = y_pred[:,:self.batch_cov_horizon*N]
        y_actual = y_actual[:,:self.batch_cov_horizon*N]
        y_actual = y_actual.reshape(y_actual.shape[0], N, self.batch_cov_horizon)
        if self.static:
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)
            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_distribution_batch(y_pred[:, i]).log_prob(y_actual[:, i]).unsqueeze(-1))
            loss = torch.cat(loss, dim=-1) + self.rho*self.get_reg()
        else:
            mixture_weights = y_pred[..., 4:]
            y_pred = y_pred[..., :4]

            mixture_weights = mixture_weights[:,:self.batch_cov_horizon*N]
            mixture_weights = mixture_weights.reshape(mixture_weights.shape[0], N, self.batch_cov_horizon, -1)
            y_pred = y_pred.reshape(y_pred.shape[0], N, self.batch_cov_horizon, -1)

            loss = []
            for i in range(y_pred.shape[1]):
                loss.append(-self.map_x_to_distribution_batch(y_pred[:, i], mixture_weights[:, i]).log_prob(y_actual[:, i]).unsqueeze(-1))
            loss = torch.cat(loss, dim=-1) + self.rho*self.get_reg()

        return loss
    
    # def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
    #     """
    #     loss function: BatchCovLoss
    #     y_pred: (batch_size (N), Q, n_params), params for normed data
    #     y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
    #     """

    #     if self.static:
    #         loss = []
    #         for i in range(y_pred.shape[1]-self.batch_cov_horizon+1):
    #             loss.append(-self.map_x_to_distribution_batch(y_pred[:, i:i+self.batch_cov_horizon]).log_prob(y_actual[:, i:i+self.batch_cov_horizon]).unsqueeze(-1))
    #         loss = torch.cat(loss, dim=-1) + self.rho*self.get_reg()
    #     else:
    #         mixture_weights = y_pred[..., 4:]
    #         y_pred = y_pred[..., :4]

    #         loss = []
    #         for i in range(y_pred.shape[1]-self.batch_cov_horizon+1):
    #             loss.append(-self.map_x_to_distribution_batch(y_pred[:, i:i+self.batch_cov_horizon], mixture_weights[:, i:i+self.batch_cov_horizon]).log_prob(y_actual[:, i:i+self.batch_cov_horizon]).unsqueeze(-1))
    #         loss = torch.cat(loss, dim=-1) + self.rho*self.get_reg()

    #     return loss

        # batch_distribution  = self.map_x_to_distribution_batch(y_pred)
        # loss = -batch_distribution.log_prob(y_actual) + self.rho*self.get_reg()
        # return loss.unsqueeze(-1)  # (B, Q)


    """
    with BatchCovLoss only, using toeplitz for parametrizing corr.
    """
    def __init__(
        self,
        hidden_size: int = 40,
        D: int = 12,  # autocorrelation horizon
    ):
        super().__init__()
        self.batch_distribution = distributions.MultivariateNormal
        self.distribution_class = distributions.Normal
        self.distribution_arguments = ["loc", "scale"]

        self.ar_projector = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2*D-1))

        self.batch_cov_horizon = D

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        """
        used in decoder to concat network outpout with scaler
        """
        self._transformation = encoder.transformation
        loc = parameters[..., 0]
        scale = F.softplus(parameters[..., 1])
        return torch.concat(
            [target_scale.unsqueeze(1).expand(-1, loc.size(1), -1), loc.unsqueeze(-1), scale.unsqueeze(-1)], dim=-1
        )
    
    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        distr = self.distribution_class(loc=x[..., 2], scale=x[..., 3])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )

    def get_corr(self, c):
        c = torch.fft.irfft(torch.mul(torch.conj(torch.fft.rfft(c)), torch.fft.rfft(c)))
        c = c[:self.batch_cov_horizon]
        c = c/c.max()
        corr = toeplitz(c, c)
        return corr

    def get_cov(self, x: torch.Tensor, c: torch.Tensor = None):
        corr = torch.stack([self.get_corr(c[b,-1,:]) for b in range(c.shape[0])], dim=0)
        cov_mat = torch.diag_embed(x[..., 3])@corr@torch.diag_embed(x[..., 3])
        return cov_mat

    def map_x_to_distribution_batch(self, x: torch.Tensor, c: torch.Tensor = None) -> distributions.MultivariateNormal:
        cov_mat = self.get_cov(x, c)
        distr = self.batch_distribution(loc=x[..., 2], covariance_matrix=cov_mat)
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr, [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]]
            )
    
    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        losses = self.mask_losses(losses, lengths-self.batch_cov_horizon+1)
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = lengths
            else:
                self.losses = torch.cat([self.losses, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses = losses.sum()
            if not torch.isfinite(losses):
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        loss function: BatchCovLoss
        y_pred: (batch_size (N), Q, n_params), params for normed data
        y_actual: (batch_size (N), Q), this comes from dataloader y of (x, y), in original scale
        """
        decoder_output = y_pred[..., 4:]
        c = self.ar_projector(decoder_output)
        y_pred = y_pred[..., :4]

        loss = []
        for i in range(y_pred.shape[1]-self.batch_cov_horizon+1):
            loss.append(-self.map_x_to_distribution_batch(y_pred[:, i:i+self.batch_cov_horizon], c[:, i:i+self.batch_cov_horizon]).log_prob(y_actual[:, i:i+self.batch_cov_horizon]).unsqueeze(-1))
        loss = torch.cat(loss, dim=-1)

        return loss