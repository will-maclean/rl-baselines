import numpy as np
import torch
from torch.optim import Adam
from numpy.random import default_rng

from agents.agent import OfflineAgent
from .utils.mcts import MCTS
from utils.replay import StandardReplayBuffer
from .net import AlphaZeroNet


class AlphaZeroAgent(OfflineAgent):
    def __init__(self, env, device,
                 batch_size=32,
                 max_memory=100_000,
                 rollouts=500,
                 lr=1e-4,
                 pi_temp: float = 1.0,
                 res_layers: int = 19
                 ):
        memory = StandardReplayBuffer(max_memory)
        super().__init__(
            env=env,
            name="AlphaZero",
            memory=memory,
            batch_size=batch_size,
            device=device,
        )
        self.rollouts = rollouts
        self.pi_temp = pi_temp
        self.max_memory = max_memory
        self.lr = lr
        self.A = env.action_space.n
        self.res_layers = res_layers

        self.rng = default_rng()
        self.net = AlphaZeroNet(device, res_blocks=self.res_layers).to(device)
        self.optim = Adam(self.net.parameters(), lr=self.lr)

    def train_step(self, step):
        states, rewards, mcts_pis = self.memory.sample(self.batch_size)

        states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device,
                                                                                        dtype=torch.float32)
        mcts_pis = torch.from_numpy(np.stack([np.float32(mcts_pi) for mcts_pi in mcts_pis])).to(self.device,
                                                                                                  dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        pi, v = self.net(states)

        q_loss = (rewards - v).pow(2).mean()
        pi_loss = (mcts_pis * pi.log()).mean()
        loss = q_loss + pi_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "train/q_loss": q_loss.detach().cpu().item(),
            "train/pi_loss": pi_loss.detach().cpu().item(),
            "train/loss": loss.detach().cpu().item(),
        }

    def act(self, state, env, step=-1):
        mcts = MCTS(env, state, self.rollouts, self.device, self.net, self.pi_temp)
        pi = mcts.act()
        a = self.rng.choice(np.arange(self.A), p=pi)
        a = int(a)
        return a, pi  # todo - sample properly

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, log_dict):
        pass

    def config(self):
        return {
            "batch_size": self.batch_size,
            "max_memory": self.max_memory,
            "rollouts": self.rollouts,
            "pi_temp": self.pi_temp,
            "lr": self.lr,
            "res_layers": self.res_layers,
        }
