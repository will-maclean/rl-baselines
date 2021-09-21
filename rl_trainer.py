from abc import ABC, abstractmethod

import gym
import torch.cuda

import wandb
from tqdm import tqdm

from agents import DQNAgent, RainbowDQNAgent, RLAgent, DiscreteSACAgent
from agents.actor_critic import SACAgent
from agents.agent import OfflineAgent, OnlineAgent
from utils.env import ParallelEnv
from utils.replay import StandardReplayBuffer
from utils.wrappers import wrap_dqn, WrapPendulum


class RLTrainer(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def config(self):
        pass


class OfflineTrainer(RLTrainer):
    def __init__(self,
                 env,
                 agent,
                 env_steps: int = 50_000,
                 train_steps_per_env_step: int = 1,
                 burn_in: int = 2_000,
                 batch_size: int = 32,
                 render: bool = False,
                 train_every: int = 1,
                 ):
        self.env = env
        self.agent: OfflineAgent = agent
        self.env_steps = env_steps
        self.train_steps_per_env_step = train_steps_per_env_step
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.render = render
        self.train_every = train_every

    def train(self):
        # setup Weights and Biases
        wandb.init(project="rl-baselines",
                   config={**self.config(), **self.agent.config()},
                   group=self.agent.name
                   )

        self.agent.wandb_watch()

        state = self.env.reset()
        step = 0
        env_return = 0
        episodes = 0
        ep_length = 0

        while step < self.env_steps:
            action, act_info = self.agent.act(state, self.env, step)
            next_state, reward, done, info = self.env.step(action)

            if self.render:
                env.render()

            env_return += reward
            step += 1
            ep_length += 1

            self.agent.memory.append(state, action, next_state, reward, done)

            log_dict = self.agent.log(act_info)
            if log_dict is not None:
                wandb.log(log_dict)

            state = next_state

            if step > self.burn_in and step % self.train_every == 0:
                for _ in range(self.train_steps_per_env_step):
                    log_dict = self.agent.train_step(step)
                    if log_dict is not None:
                        wandb.log(log_dict)

            if done:
                state = self.env.reset()

                wandb.log({
                    "episode": episodes,
                    "return": env_return,
                    "episode_length": ep_length,
                })

                episodes += 1
                env_return = 0
                ep_length = 0

    def config(self):
        return {
            "env_name": self.env.spec.id,
            "env_steps": self.env_steps,
            "train_steps_per_env_step": self.train_steps_per_env_step,
            "burn_in": self.burn_in,
            "batch_size": self.batch_size,
            "train_every": self.train_every,
        }


class OnlineTrainer(RLTrainer):
    def __init__(self,
                 env,
                 agent,
                 env_steps: int = 50_000,
                 n_envs: int = 4,
                 steps_per_env=32,
                 eval_every: int = 2,
                 ):
        self.env = env
        self.agent: OnlineAgent = agent
        self.env_steps = env_steps
        self.n_envs = n_envs
        self.steps_per_env = steps_per_env
        self.eval_every = eval_every

    def train(self):
        # setup Weights and Biases
        wandb.init(project="rl-baselines",
                   config={**self.config(), **self.agent.config()},
                   group=self.agent.name
                   )

        self.agent.wandb_watch()

        train_count = 0
        for step in tqdm(range(0, self.env_steps, self.n_envs * self.steps_per_env)):
            sample = self.agent.sample()
            log_dict = self.agent.train_step(sample)

            if log_dict is not None:
                wandb.log(log_dict)

            train_count += 1

            if train_count > self.eval_every:
                # online agents generally don't complete episodes in such a way that we can track them, as they often
                # use parallel environments. Instead of arbitrarily choosing an environment to track, we'll just run an
                # evaluation episode to see how our agent performs.

                result = self.evaluate()
                wandb.log(result)
                train_count = 0

    def config(self):
        return {}

    def evaluate(self):
        ep_return = 0
        state = self.env.reset()
        done = False
        while not done:
            a, _ = self.agent.act(state, self.env)
            state, reward, done, _ = self.env.step(a[0].detach().cpu().numpy())

            ep_return += reward

        return {
            "eval_score": ep_return,
        }


if __name__ == "__main__":
    env = gym.make("MsPacman-ramNoFrameskip-v0")
    env = wrap_dqn(env)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_steps = 1_000_000

    agent = DiscreteSACAgent(env, device,
                             gamma=0.99,
                             lr_pi=3e-4,
                             lr_q=3e-4,
                             lr_a=3e-4,
                             max_memory=500_000,
                             trainable_alpha=True,
                             reward_scale=0.1,
                             alpha=6,
                             polyak_tau=0.005,
                             pi_hidden_size=64,
                             q_hidden_size=64,
                             min_alpha=0.001,
                             )

    trainer = OfflineTrainer(env,
                             agent,
                             env_steps=max_steps,
                             batch_size=32,
                             burn_in=20_000,
                             train_steps_per_env_step=1,
                             render=False,
                             train_every=4,
                             )
    trainer.train()
