import random

import numpy as np
import torch
from torch.optim import Adam

from agents.agent import OfflineAgent
from .net import DuellingDQN
from utils.replay import StandardReplayBuffer, ReplayBuffer
from utils import epsilon_decay, copy_weights


class DQNAgent(OfflineAgent):
    def __init__(self, env,
                 device,
                 gamma=0.99,
                 name="DDQN",
                 net_class=DuellingDQN,
                 lr=3e-4,
                 max_steps=10_000,
                 polyak_tau=0.05,
                 hard_update_freq=None,
                 replay_buffer_type=StandardReplayBuffer,
                 max_memory=10_000,
                 soft_update_freq=None,
                 reward_scale=1,
                 batch_size=32,
                 hidden_size=16,
                 num_hidden=1,
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
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden

        self.memory: ReplayBuffer = memory

        sample_input = env.reset()
        self.net = net_class(
            sample_input.shape[0],
            env.action_space.n,
            hidden_size=self.hidden_size,
            num_hidden=self.num_hidden,
        ).to(device)
        self.target_net = net_class(
            sample_input.shape[0],
            env.action_space.n,
            hidden_size=self.hidden_size,
            num_hidden=self.num_hidden,
        ).to(device)

        self.optim = Adam(params=self.net.parameters(), lr=lr)

        self.target_net.load_state_dict(self.net.state_dict())

    def act(self, state, env, step=-1):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=torch.float32)

        eps = epsilon_decay(step, 1, 0.01, 40_000)

        if random.random() < eps:
            return env.action_space.sample(), eps
        else:
            qs = self.net(state).squeeze()
            return qs.argmax().detach().cpu().item(), eps

    def train_step(self, step):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device,
                                                                                        dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([np.float32(state) for state in next_states])).to(self.device,
                                                                                                  dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        loss = self._ddqn_loss(states, actions, next_states, rewards, dones)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.hard_update_freq is not None and step % self.hard_update_freq == 0:
            copy_weights(self.net, self.target_net, polyak=None)

        if self.soft_update_freq is not None and step % self.soft_update_freq == 0:
            copy_weights(self.net, self.target_net, polyak=self.polyak_tau)

        return {"loss": loss.detach().cpu().item()}

    def _ddqn_loss(self, states, actions, next_states, rewards, dones, reduction="mean"):
        q_values = self.net(states)
        actions = actions.unsqueeze(1)
        q_values = q_values.gather(1, actions).squeeze(1)

        next_actions = self.target_net(next_states).argmax(dim=1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        td_target = self.reward_scale * rewards + self.gamma * next_q_values * (1 - dones)
        loss = torch.nn.functional.smooth_l1_loss(q_values, td_target, reduction=reduction)
        loss = loss.mean()

        return loss

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, log_dict):
        if isinstance(log_dict, float):
            return {"epsilon": log_dict}

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
            "hidden_size": self.hidden_size,
            "num_hidden": self.num_hidden,
        }