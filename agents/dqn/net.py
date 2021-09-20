import numpy as np
import torch
from torch import nn
from utils.layers import NoisyLinear


class MLP(nn.Module):
    def __init__(self, n_in, n_out, layer_sizes=(64, 64), activation=nn.ReLU):
        super().__init__()

        layers = []
        layers.append(nn.Linear(n_in, layer_sizes[0]))

        for i in range(len(layer_sizes) - 1):
            layers.append(activation())
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

        layers.append(activation())
        layers.append(nn.Linear(layer_sizes[-1], n_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DuellingDQN(nn.Module):
    def __init__(self, n_in, n_out, hidden_size=16, num_hidden=1, activation=nn.ReLU):

        super().__init__()

        if num_hidden > 0:
            processing = [nn.Linear(n_in, hidden_size), activation()]

            for _ in range(num_hidden):
                processing.extend([nn.Linear(hidden_size, hidden_size), activation()])

            self.processing = nn.Sequential(*processing)

            self.value = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                activation(),
                nn.Linear(hidden_size, 1)
            )

            self.advantage = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                activation(),
                nn.Linear(hidden_size, n_out)
            )
        else:
            self.processing = None

            self.value = nn.Sequential(
                nn.Linear(n_in, hidden_size, bias=True),
                activation(),
                nn.Linear(hidden_size, 1)
            )

            self.advantage = nn.Sequential(
                nn.Linear(n_in, hidden_size, bias=True),
                activation(),
                nn.Linear(hidden_size, n_out)
            )

    def forward(self, x: torch.Tensor):
        """
        All classes which extend nn.Module have to implement a forward(x), which defines how a input vector is fed
        through the network.
        :param x: input vector
        :return: output vector
        """

        if self.processing is not None:
            x = self.processing(x)

        v = self.value(x)

        advantage = self.advantage(x)

        return v + (advantage - advantage.mean())


class NoisyDuellingDQN(nn.Module):
    def __init__(self, n_in, n_out, hidden_size=16, num_hidden=1, activation=nn.ReLU, noisy_sigma_0=0.5):

        super().__init__()

        if num_hidden > 0:
            processing = [NoisyLinear(n_in, hidden_size, sigma_0=noisy_sigma_0), activation()]

            for _ in range(num_hidden):
                processing.extend([NoisyLinear(hidden_size, hidden_size, sigma_0=noisy_sigma_0), activation()])

            self.processing = nn.Sequential(*processing)

            self.value = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size, sigma_0=noisy_sigma_0),
                activation(),
                NoisyLinear(hidden_size, 1)
            )

            self.advantage = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size, sigma_0=noisy_sigma_0),
                activation(),
                NoisyLinear(hidden_size, n_out, sigma_0=noisy_sigma_0)
            )
        else:
            self.processing = None

            self.value = nn.Sequential(
                NoisyLinear(n_in, hidden_size, sigma_0=noisy_sigma_0),
                activation(),
                NoisyLinear(hidden_size, 1, sigma_0=noisy_sigma_0)
            )

            self.advantage = nn.Sequential(
                NoisyLinear(n_in, hidden_size, sigma_0=noisy_sigma_0),
                activation(),
                NoisyLinear(hidden_size, n_out, sigma_0=noisy_sigma_0)
            )

    def forward(self, x: torch.Tensor):
        """
        All classes which extend nn.Module have to implement a forward(x), which defines how a input vector is fed
        through the network.
        :param x: input vector
        :return: output vector
        """
        if self.processing is not None:
            x = self.processing(x)

        v = self.value(x)

        advantage = self.advantage(x)

        return v + (advantage - advantage.mean())

    def update_noise(self):
        for module in self.value.modules():
            if isinstance(module, NoisyLinear):
                module.update_noise()

        for module in self.advantage.modules():
            if isinstance(module, NoisyLinear):
                module.update_noise()

        # for module in self.processing.modules():
        #     if isinstance(module, NoisyLinear):
        #         module.update_noise()


# Adapted from https://github.com/xtma/dsac/blob/master/rlkit/torch/dsac/networks.py
class IQNNet(nn.Module):
    def __init__(
            self,
            S,
            hidden_sizes,
            A,
            device,
            embedding_size=64,
            num_quantiles=32,
            layer_norm=True,
            **kwargs,
    ):
        super().__init__()
        self.layer_norm = layer_norm

        self.base_fc = []
        last_size = S
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], A)
        self.const_vec = torch.from_numpy(np.arange(1, 1 + self.embedding_size)).to(device)

        self.to(device)

    def forward(self, state, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """

        h = self.base_fc(state)  # (N, C)

        x = torch.cos(tau * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        return output

    def q_vals(self, state, tau_hat, presum_tau):
        zs = self(state, tau_hat)
        q = (zs * presum_tau).sum(dim=-1)

        return q
