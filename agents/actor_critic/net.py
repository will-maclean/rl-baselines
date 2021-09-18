import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ContinuousPolicy(nn.Module):
    def __init__(self, S, A, device):
        super().__init__()

        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(S, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2 * A),
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

        mu = x[:, :self.A]
        sd = x[:, self.A:]

        sd = sd.exp()

        dist = Normal(loc=mu, scale=sd)

        a = dist.rsample()
        a = torch.tanh(a)
        ln_pi = dist.log_prob(a)

        if was_numpy:
            a = a.detach().cpu().numpy()

        return a, ln_pi


class ContinuousCritic(nn.Module):
    def __init__(self, S, A_size=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(S + A_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.layers(x)
