import numpy as np
import torch
import wandb
from torch.optim import Adam
from numpy.random import default_rng

from agents.agent import OfflineAgent
from .utils.mcts import MCTS
from .utils.adversary import RandomAdversary
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
        self.eval_adversary = RandomAdversary()

        self.net = AlphaZeroNet(device, res_blocks=self.res_layers).to(device)
        self.optim = Adam(self.net.parameters(), lr=self.lr)
        self.mcts = MCTS(game=env,
                         state=env.reset(),
                         rollouts=self.rollouts,
                         device=self.device,
                         net=self.net,
                         pi_temp=self.pi_temp,
                         A=self.A)

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
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.3)
        self.optim.step()

        return {
            "train/q_loss": q_loss.detach().cpu().item(),
            "train/pi_loss": pi_loss.detach().cpu().item(),
            "train/loss": loss.detach().cpu().item(),
        }

    def act(self, state, env, step=-1):
        a, pi = self.mcts.act()
        return a, pi

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, log_dict):
        wandb.log(log_dict)

    def reset(self, env=None, reset_state=None):
        self.mcts = MCTS(game=env,
                         state=reset_state,
                         rollouts=self.rollouts,
                         device=self.device,
                         net=self.net,
                         pi_temp=self.pi_temp,
                         A=self.A)

    def config(self):
        return {
            "batch_size": self.batch_size,
            "max_memory": self.max_memory,
            "rollouts": self.rollouts,
            "pi_temp": self.pi_temp,
            "lr": self.lr,
            "res_layers": self.res_layers,
        }

    def wandb_watch(self):
        wandb.watch(self.net)

    def evaluate(self, env):
        state = env.reset()
        self.reset(env, state)
        done = False
        player = np.random.choice([-1, 1])
        ep_length = 0
        reward = 0
        max_len = 200  # if a game goes for 200 moves, presume a draw

        while not done:
            if player == 1:
                a, *_ = self.act(state, env)
            else:
                a = self.eval_adversary.act(env, state)
                self.opponent_action(a)

            state, reward, done, _ = env.step(a)

            player *= -1
            ep_length += 1

            if done:
                break
            if ep_length > max_len:
                reward = 0
                break

        return {
            "eval/ep_length": ep_length,
            "eval/return": player * reward,
            "eval/adversary": self.eval_adversary.__class__,
        }

    def opponent_action(self, a):
        self.mcts.opponent_action(a)
