from abc import ABC, abstractmethod

import gym
import torch.cuda

import wandb

from agents import DQNAgent, RainbowDQNAgent, RLAgent
from agents.actor_critic import SACAgent
from agents.agent import OfflineAgent
from replay import ReplayBuffer, StandardReplayBuffer
from wrappers import wrap_dqn, WrapPendulum


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
                 ):
        self.env = env
        self.agent: OfflineAgent = agent
        self.env_steps = env_steps
        self.train_steps_per_env_step = train_steps_per_env_step
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.render = render

    def train(self):
        # setup Weights and Biases
        wandb.login(key="c9501a5fdcab6428101221c42b3e8b34f942538f")
        run = wandb.init(entity="thirstCrusher",
                         project="rl-baselines",
                         config={**self.config(), **self.agent.config()},
                         group=self.agent.name
                         )

        state = self.env.reset()
        step = 0
        env_return = 0
        episodes = 0
        ep_length = 0

        while step < self.env_steps:
            action, act_info = self.agent.act(state, env, step)
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

            if step > self.burn_in:
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
        }


class OnlineTrainer(RLTrainer):
    def __init__(self,
                 env,
                 agent,
                 env_steps: int = 50_000,
                 batch_size: int = 32,
                 render: bool = False,
                 ):
        self.env = env
        self.agent: RLAgent = agent
        self.env_steps = env_steps
        self.batch_size = batch_size
        self.render = render

    def train(self):
        # setup Weights and Biases
        wandb.login(key="c9501a5fdcab6428101221c42b3e8b34f942538f")
        run = wandb.init(entity="thirstCrusher",
                         project="rl-baselines",
                         config={**self.config(), **self.agent.config()},
                         group=self.agent.name
                         )

        state = self.env.reset()
        step = 0
        env_return = 0
        episodes = 0
        ep_length = 0

        ep_data = []

        while step < self.env_steps:
            action, act_info = self.agent.act(state, env, step)
            next_state, reward, done, info = self.env.step(action)

            if self.render:
                env.render()

            env_return += reward
            step += 1
            ep_length += 1

            ep_data.append((state, action, next_state, reward, done))

            log_dict = self.agent.log(act_info)
            if log_dict is not None:
                wandb.log(log_dict)

            state = next_state

            if len(ep_data) >= self.batch_size:
                log_dict = self.agent.train_step(ep_data, step)
                wandb.log(log_dict)
                ep_data = []

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
            "batch_size": self.batch_size,
        }


if __name__ == "__main__":
    env = WrapPendulum(gym.make("Pendulum-v0"))
    # env = wrap_dqn(env)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_steps = 500_000

    agent = SACAgent(env, device,
                     gamma=0.99,
                     lr_pi=0.00003,
                     lr_q=0.00003,
                     lr_a=0.00003,
                     max_memory=400_000,
                     trainable_alpha=False,
                     reward_scale=0.0015,
                     alpha=0.2,
                     polyak_tau=0.005,
                     )
    trainer = OfflineTrainer(env,
                             agent,
                             env_steps=max_steps,
                             batch_size=32,
                             burn_in=500,
                             train_steps_per_env_step=2,
                             render=False,
                             )
    trainer.train()
