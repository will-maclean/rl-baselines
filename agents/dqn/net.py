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
    def __init__(self, n_in, n_out, activation=nn.ReLU):

        super().__init__()

        # self.processing = nn.Sequential(
        #     nn.Linear(n_in, 64),
        #     activation()
        # )

        self.value = nn.Sequential(
            nn.Linear(n_in, 16, bias=True),
            activation(),
            nn.Linear(16, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(n_in, 16, bias=True),
            activation(),
            nn.Linear(16, n_out)
        )

    def forward(self, x: torch.Tensor):
        """
        All classes which extend nn.Module have to implement a forward(x), which defines how a input vector is fed
        through the network.
        :param x: input vector
        :return: output vector
        """

        # x = self.processing(x)

        v = self.value(x)

        advantage = self.advantage(x)

        return v + (advantage - advantage.mean())


class NoisyDuellingDQN(nn.Module):
    # todo - could be larger for Atari modules
    def __init__(self, n_in, n_out, activation=nn.ReLU, noisy_sigma_0=0.5):

        super().__init__()

        # self.processing = nn.Sequential(
        #     NoisyLinear(n_in, 64, sigma_0=noisy_sigma_0),
        #     activation()
        # )

        self.value = nn.Sequential(
            NoisyLinear(n_in, 16, sigma_0=noisy_sigma_0),
            activation(),
            NoisyLinear(16, 1, sigma_0=noisy_sigma_0)
        )

        self.advantage = nn.Sequential(
            NoisyLinear(n_in, 16, sigma_0=noisy_sigma_0),
            activation(),
            NoisyLinear(16, n_out, sigma_0=noisy_sigma_0)
        )

    def forward(self, x: torch.Tensor):
        """
        All classes which extend nn.Module have to implement a forward(x), which defines how a input vector is fed
        through the network.
        :param x: input vector
        :return: output vector
        """

        # x = self.processing(x)

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


