import random

import numpy as np
import torch
from torch.optim import Adam

from agents.agent import RLAgent
from net import DuellingDQN, NoisyDuellingDQN
from utils import epsilon_decay, copy_weights


class DQNAgent(RLAgent):
    def __init__(self, env,
                 device,
                 gamma,
                 name="DDQN",
                 net_class=DuellingDQN,
                 lr=3e-4,
                 max_steps=10_000,
                 polyak_tau=0.05,
                 hard_update_freq=None,
                 soft_update_freq=None,
                 reward_scale=1,
                 ):
        super().__init__(env, name)
        self.device = device
        self.gamma = gamma
        self.max_steps = max_steps
        self.polyak_tau = polyak_tau
        self.hard_update_freq = hard_update_freq
        self.soft_update_freq = soft_update_freq
        self.lr = lr
        self.reward_scale = 1

        sample_input = env.reset()
        self.net = net_class(sample_input.shape[0], env.action_space.n).to(device)
        self.target_net = net_class(sample_input.shape[0], env.action_space.n).to(device)

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

    def train_step(self, batch, step):
        states, actions, next_states, rewards, dones = batch

        states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device,
                                                                                        dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([np.float32(state) for state in next_states])).to(self.device,
                                                                                                  dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.net(states)
        actions = actions.unsqueeze(1)
        q_values = q_values.gather(1, actions).squeeze(1)

        next_actions = self.target_net(next_states).argmax(dim=1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        td_target = self.reward_scale * rewards + self.gamma * next_q_values * (1 - dones)
        loss = torch.nn.functional.smooth_l1_loss(q_values, td_target)
        loss = loss.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.hard_update_freq is not None and step % self.hard_update_freq == 0:
            copy_weights(self.net, self.target_net, polyak=None)

        if self.soft_update_freq is not None and step % self.soft_update_freq == 0:
            copy_weights(self.net, self.target_net, polyak=self.polyak_tau)

        return {"loss": loss.detach().cpu().item()}

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
        }


class RainbowDQNAgent(DQNAgent):
    # Rainbow list:
    # 1. Duelling (X)
    # 2. Double (X)
    # 3. Noisy (X)
    # 4. Prioritised ( )
    # 5. N-Step ( )
    # 6. Distributed ( )

    def __init__(self, env,
                 device,
                 gamma,
                 name="RainbowDQN",
                 net_class=NoisyDuellingDQN,
                 lr=3e-4,
                 max_steps=10_000,
                 polyak_tau=0.05,
                 hard_update_freq=None,
                 soft_update_freq=None,
                 reward_scale=1,
                 ):
        super(RainbowDQNAgent, self).__init__(env=env,
                                              device=device,
                                              gamma=gamma,
                                              name=name,
                                              net_class=net_class,
                                              lr=lr,
                                              max_steps=max_steps,
                                              polyak_tau=polyak_tau,
                                              hard_update_freq=hard_update_freq,
                                              soft_update_freq=soft_update_freq,
                                              reward_scale=reward_scale)

    def act(self, state, env, step=-1):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=torch.float32)

        qs = self.net(state).squeeze()
        return qs.argmax().detach().cpu().item(), None

    def train_step(self, batch, step):
        loss = super().train_step(batch, step)
        self.net.update_noise()
        self.target_net.update_noise()
        return loss

    def log(self, log_dict):
        return None
