import numpy as np
import torch
from torch.optim import Adam

from agents.agent import OfflineAgent
from utils.distrib import TauType
from utils.replay import StandardReplayBuffer, ReplayBuffer
from .net import IQNNet


def huber(delta, tau, kappa: float = 1.):
    L = torch.where(delta.abs() <= kappa, 0.5 * delta.pow(2), kappa * (delta.abs() - 0.5 * kappa))
    rho = (tau - (delta < 0).float()).abs() * L/kappa

    return rho


class IQNAgent(OfflineAgent):
    def __init__(self, env,
                 device,
                 gamma=0.99,
                 name="IQN",
                 net_class=IQNNet,
                 lr=3e-4,
                 max_steps=10_000,
                 polyak_tau=0.05,
                 hard_update_freq=500,
                 replay_buffer_type=StandardReplayBuffer,
                 max_memory=10_000,
                 soft_update_freq=None,
                 reward_scale=1,
                 batch_size=32,
                 num_quantiles=32,
                 hidden_size=32,
                 num_hidden=3,
                 huber_kappa=1.0
                 ):
        memory = replay_buffer_type(max_memory)
        super().__init__(env, name, memory=memory, batch_size=batch_size, device=device)
        self.device = device
        self.gamma = gamma
        self.max_steps = max_steps
        self.polyak_tau = polyak_tau
        self.hard_update_freq = hard_update_freq
        self.soft_update_freq = soft_update_freq
        self.lr = lr
        self.reward_scale = reward_scale
        self.num_quantiles = num_quantiles
        self.tau = TauType.random
        self.huber_kappa = huber_kappa
        self.A = env.action_space.n
        self.S = env.reset().shape[0]

        self.memory: ReplayBuffer = memory

        hidden_sizes = [hidden_size for _ in range(num_hidden)]

        sample_input = env.reset()
        self.net = net_class(S=sample_input.shape[0],
                             hidden_sizes=hidden_sizes,
                             A=env.action_space.n,
                             num_quantiles=num_quantiles,
                             device=device).to(device)
        self.target_net = net_class(S=sample_input.shape[0],
                                    hidden_sizes=hidden_sizes,
                                    A=env.action_space.n,
                                    num_quantiles=num_quantiles,
                                    device=device).to(device)

        self.optim = Adam(params=self.net.parameters(), lr=lr)

        self.target_net.load_state_dict(self.net.state_dict())

    def act(self, state, env, step=-1):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0).to(dtype=torch.float32, device=self.device)

        tau, tau_hat, presum_tau = self.tau.get_tau(self.S, self.A, self.num_quantiles, self.device, state.shape[0])

        with torch.no_grad():
            return self.net.q_vals(state, tau_hat, presum_tau).squeeze().argmax().cpu().item()

    def train_step(self, step):
        states, actions, next_states, rewards, dones = self.get_batch()

        # build targets
        with torch.no_grad():
            tau, tau_hat, presum_tau = self.tau.get_tau(self.S, self.A, self.num_quantiles, self.device, self.batch_size)
            next_z = self.target_net(next_states, tau_hat)
            next_q = self.target_net.q_vals(next_states, tau_hat, presum_tau)
            next_a = next_q.argmax(dim=1).squeeze()
            next_z_a = next_z[:, next_a, :]

        # build prediction
        tau, tau_hat, presum_tau = self.tau.get_tau(self.S, self.A, self.num_quantiles, self.device, self.batch_size)
        z_pred = self.net(states, tau_hat)
        z_pred_a = z_pred[:, actions.squeeze(), :]

        delta = rewards + self.gamma * next_z_a - z_pred_a

        loss = huber(delta, tau, kappa=self.huber_kappa).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.detach().cpu().item()

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, log_dict):
        pass

    def config(self):
        return {
            "name": self.name,
            'gamma': self.gamma,
            "net_class": self.net.__class__,
            "lr": self.lr,
            "polyak_tau": self.polyak_tau,
            "hard_update_freq": self.hard_update_freq,
            "soft_update_freq": self.soft_update_freq,
            "reward_scale": self.reward_scale,
            "memory_type": self.memory.__class__,
            "num_quantiles": self.num_quantiles,
            "tau_type": self.tau.name,
            "huber_kappa": self.huber_kappa,
        }
