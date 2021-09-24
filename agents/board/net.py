import torch
import numpy as np
from torch import nn


S = (119, 8, 8)
A = 4672


class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, ks=3, stride=1, pad=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            bias=False
        )

        # Optionally include activations
        # If BatchNorm is True, use nn.BatchNorm2d to include a Batch Normalisation
        self.bn = nn.BatchNorm2d(out_planes)
        self.activ = nn.ReLU()

        # If activ is True, use nn.ReLU to include a ReLU

    def forward(self, x: torch.Tensor):
        # Run through convolution then BatchNorm then ReLU
        x = self.conv(x)

        x = self.bn(x) if self.bn else x
        x = self.activ(x) if self.activ else x

        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = Conv(in_planes, out_planes)
        self.conv2 = Conv(out_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        x += identity
        x = self.relu(x)

        return x


class ValueHead(nn.Module):
    def __init__(self, standard_input, hidden_size=256):
        super().__init__()

        self.bn = nn.BatchNorm2d(standard_input.shape[1])
        self.fc1 = nn.Linear(standard_input.flatten(start_dim=1).shape[1], out_features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features=1)

    def forward(self, x: torch.Tensor):
        x = self.bn(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)

        return x


class PolicyHead(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.conv = Conv(in_planes, 256)
        self.policy = nn.Conv2d(256, 73, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return torch.softmax(self.policy(x).flatten(start_dim=1), dim=1)


class AlphaZeroNet(nn.Module):
    """
    Net used in AlphaZero algorithm
    """

    def __init__(self, device, res_blocks=19):
        """

        :param device: torch device
        """
        super().__init__()

        self.device = device

        self.conv = Conv(119, 256)

        self.reslayers = nn.Sequential(*[ResidualLayer(256, 256) for _ in range(res_blocks)])

        self.policyhead = PolicyHead(256)

        standard_input = torch.zeros((1, *S))  # observation space from paper
        standard_value_input = self.reslayers.forward(self.conv.forward(standard_input))
        self.valuehead = ValueHead(standard_value_input)

        self.to(self.device)

    def forward(self, x: torch.Tensor or np.array):
        """
        If state passed in is a np.array, assume that results should be converted to np.arrays as well
        :param x: state
        :return: value and policy
        """
        if isinstance(x, np.ndarray):
            convert = True
            x = torch.from_numpy(x).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            convert = False
        x = self.conv(x)
        x = self.reslayers(x)
        value = self.valuehead(x)
        policy = self.policyhead(x)

        if convert:
            value = value.cpu().item()
            policy = policy.cpu().detach().numpy().squeeze()

        return policy, value
