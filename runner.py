import yaml

import gym

from agents.agent import OfflineAgent, OnlineAgent
from utils.wrappers import *
from agents import AgentType, str_to_agent_type
from rl_trainer import OfflineTrainer, OnlineTrainer


class RLRunner:
    def __init__(self, env_name, cfg_file):
        # load config
        with open(cfg_file, "r") as stream:
            config = yaml.safe_load(stream)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        env = gym.make(env_name)

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
        else:
            agent_type = str_to_agent_type(config['agent'])
            agent: OnlineAgent = agent_type.make_agent(env=env, device=device, env_name=env_name,
                                                       env_wrappers=wrapper_func, **config["agent_params"])
            self.trainer = OnlineTrainer(env, agent=agent, **config["trainer_params"])

    def start(self):
        self.trainer.train()


if __name__ == "__main__":
    runner = RLRunner("Pendulum-v0", "config/PPOAgent/Pendulum-v0.yaml")
    runner.start()
