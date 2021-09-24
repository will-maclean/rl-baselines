import argparse

import yaml

import gym
import gym_chess

from agents.agent import OfflineAgent, OnlineAgent
from utils import config_argparse
from utils.wrappers import *
from agents import AgentType, str_to_agent_type
from rl_trainer import OfflineTrainer, OnlineTrainer, AlphaZeroTrainer


class RLRunner:
    def __init__(self, env_name, cfg_file):
        # load config
        with open(cfg_file, "r") as stream:
            config = yaml.safe_load(stream)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        env = gym.make(env_name)

        config = config_argparse(config)

        # wrap env if required
        if not config["env_wrapper"] == "no_wrapper":
            wrapper_func = globals()[config["env_wrapper"]]
            env = wrapper_func(env)
        else:
            wrapper_func=None

        # create trainer
        if config["trainer"] == "offline":

            # create agent
            agent_type = str_to_agent_type(config['agent'])
            agent: OfflineAgent = agent_type.make_agent(env=env, device=device, **config["agent_params"])
            self.trainer = OfflineTrainer(env=env, agent=agent, **config["trainer_params"])
        elif config["trainer"] == "online":
            agent_type = str_to_agent_type(config['agent'])
            agent: OnlineAgent = agent_type.make_agent(env=env, device=device, env_name=env_name,
                                                       env_wrappers=wrapper_func, **config["agent_params"])
            self.trainer = OnlineTrainer(env, agent=agent, **config["trainer_params"])
        elif config["trainer"] == "alphazero_trainer":
            # create agent
            agent_type = str_to_agent_type(config['agent'])
            agent: OfflineAgent = agent_type.make_agent(env=env, device=device, **config["agent_params"])
            self.trainer = AlphaZeroTrainer(env=env, agent=agent, **config["trainer_params"])
        else:
            raise NotImplementedError

    def start(self):
        self.trainer.train()


if __name__ == "__main__":
    runner = RLRunner("ChessAlphaZero-v0", "config/AlphaZeroAgent/ChessAlphaZero-v0.yaml")
    runner.start()
