from enum import Enum

import torch
import torch.nn.functional as F


class TauType(Enum):
    fixed = 0,
    random = 1,

    def get_tau(self, states, action_size, num_quantiles, device, batch_size):
        """
        Adapted from https://github.com/xtma/dsac/blob/master/rlkit/torch/dsac/dsac.py, August 2021

        Parameters
        ----------
        states
        actions

        Returns
        -------
            tau, tau_hat, presum_tau
        """
        if self == TauType.fixed:
            # for C51, we'd get [1/51, 1/51, 1/51, ...], with size 51, for each action.
            presum_tau = torch.zeros(action_size, num_quantiles).to(device) + 1. / num_quantiles
        elif self == TauType.random:
            # for IQN, we can sample random quantile fractions 'tau' for each action
            presum_tau = torch.rand(action_size, num_quantiles)
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        else:
            raise NotImplemented("Tau generation not defined for {}".format(self.name))

        # presum_tau sums to 1 in the tau dimension. Convert each tau dimension into a CDF
        tau = torch.cumsum(presum_tau, dim=1).to(device)  # (N, T), note that they are tau1...tauN in the paper

        with torch.no_grad():
            # let's make some target tau's
            tau_hat = torch.zeros_like(tau).to(device)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.

        tau = tau.expand((batch_size, *tau.shape))
        tau_hat = tau_hat.expand((batch_size, *tau_hat.shape))
        presum_tau = presum_tau.expand((batch_size, *presum_tau.shape))

        return tau, tau_hat, presum_tau


def quantile_regression_loss(input, target, tau, weight, kappa=1):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, beta=kappa, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()
