import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F


class ContinuousPolicy(nn.Module):
    def __init__(self, S, A, device):
        super().__init__()

        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(S, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(64, A),
        )

        self.sigma = nn.Sequential(
            nn.Linear(64, A),
        )

        self.S = S
        self.A = A

    def forward(self, x):
        return self.layers(x)

    def act(self, s: torch.Tensor):
        if isinstance(s, np.ndarray):
            was_numpy = True
            s = torch.from_numpy(s).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            was_numpy = False

        if s.isnan().any():
            print("problem")

        x = self(s.detach())

        if x.isnan().any():
            print("problem")

        mu = self.mu(x)
        log_sd = self.sigma(x)

        log_sd = torch.clamp(log_sd, -20, 2)
        sd = torch.exp(log_sd)  # todo - experiment with .abs() instead of .exp()

        dist = Normal(loc=mu, scale=sd)

        a = dist.rsample()

        logp_pi = dist.log_prob(a).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=1)
        logp_pi = logp_pi.unsqueeze(-1)

        a = torch.tanh(a)

        if was_numpy:
            a = a.detach().cpu().numpy()

        return a, logp_pi


class ContinuousCritic(nn.Module):
    def __init__(self, S, A_size=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(S + A_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.layers(x)
