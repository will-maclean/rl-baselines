import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical
import torch.nn.functional as F


class ContinuousPolicy(nn.Module):
    def __init__(self, S, A, device, hidden_size=64, num_hidden=2, activation=nn.ReLU):
        super().__init__()

        self.device = device

        layers = [
            nn.Linear(S, hidden_size),
            activation(),
        ]

        for _ in range(num_hidden):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                activation(),
            ])

        self.layers = nn.Sequential(*layers)

        self.mu = nn.Sequential(
            nn.Linear(hidden_size, A),
        )

        self.sigma = nn.Sequential(
            nn.Linear(hidden_size, A),
        )

        self.S = S
        self.A = A

    def forward(self, x):
        return self.layers(x)

    def act(self, s: torch.Tensor, log_pi=True):
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
        sd = torch.exp(log_sd)

        dist = Normal(loc=mu, scale=sd)

        a = dist.rsample()

        if log_pi:
            logp_pi = dist.log_prob(a).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=1)
            logp_pi = logp_pi.unsqueeze(-1)
        else:
            logp_pi = dist.log_prob(a).exp()

        a = torch.tanh(a)

        if was_numpy:
            a = a.detach().cpu().numpy()

        return a, logp_pi

    def dist(self, x):
        mu = self.mu(x)
        log_sd = self.sigma(x)

        log_sd = torch.clamp(log_sd, -20, 2)
        sd = torch.exp(log_sd)

        return Normal(loc=mu, scale=sd)


class DiscretePolicy(nn.Module):
    def __init__(self, S, A, device, hidden_size=64):
        super().__init__()

        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(S[0], hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, A),
            nn.Softmax(dim=1),
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

        x = self(s)

        dist = Categorical(probs=x)

        a = dist.sample()  # todo - unsqueeze?

        logp_pi = dist.log_prob(a)

        if was_numpy:
            a = a.detach().cpu().item()

        return a, logp_pi


class DiscreteCNNPolicy(nn.Module):
    def __init__(self, S, A, device, hidden_size=64):
        super().__init__()

        self.device = device

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=S[0], out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        sample_input = torch.zeros((1, *S))
        cnn_output_shape = self.cnn(sample_input).flatten(1).shape[1]

        self.layers = nn.Sequential(
            nn.Linear(cnn_output_shape, 512),
            nn.ReLU(),
            nn.Linear(512, A),
            nn.Softmax(dim=1)
        )

        self.S = S
        self.A = A

    def forward(self, x):
        x = self.cnn(x).flatten(1)
        return self.layers(x)

    def act(self, s: torch.Tensor):
        if isinstance(s, np.ndarray):
            was_numpy = True
            s = torch.from_numpy(s).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            was_numpy = False

        x = self(s)

        dist = Categorical(probs=x)

        a = dist.sample()  # todo - unsqueeze?

        logp_pi = dist.log_prob(a)

        if was_numpy:
            a = a.detach().cpu().item()

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


class ContinuousV(nn.Module):
    def __init__(self, S):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(S, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s):
        return self.layers(s)


class DiscreteCritic(nn.Module):
    def __init__(self, S, A, hidden_size=64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(S[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, A),
        )

    def forward(self, s):
        return self.layers(s)


class DiscreteCNNCritic(nn.Module):
    def __init__(self, S, A, hidden_size=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=S[0], out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        sample_input = torch.zeros((1, *S))
        cnn_output_shape = self.cnn(sample_input).flatten(1).shape[1]

        self.layers = nn.Sequential(
            nn.Linear(cnn_output_shape, 512),
            nn.ReLU(),
            nn.Linear(512, A),
        )

    def forward(self, s):
        s = self.cnn(s).flatten(1)
        return self.layers(s)


class ContinuousMLPAC(nn.Module):
    def __init__(self, S, A, device,
                 hidden_size=64,
                 num_hidden=2,
                 activation=nn.ReLU):
        super().__init__()

        self.device = device

        layers = [
            nn.Linear(S, hidden_size),
            activation(),
        ]

        for _ in range(num_hidden):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                activation(),
            ])

        self.layers = nn.Sequential(*layers)

        self.pi = ContinuousPolicy(hidden_size, A, device, hidden_size, 0)

        self.v = ContinuousV(hidden_size)

    def forward(self, x):
        x = self.layers(x)

        dist = self.pi.dist(x)

        return dist, self.v(x)

    def act(self, s: torch.Tensor, v=False):
        if isinstance(s, np.ndarray):
            was_numpy = True
            s = torch.from_numpy(s).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            was_numpy = False

        if s.isnan().any():
            print("problem")

        x = self.layers(s.detach())

        if v:
            return self.pi.act(x, log_pi=False), (self.v,)
        else:
            return self.pi.act(x, log_pi=False)
