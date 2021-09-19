import yaml

import gym

from utils.wrappers import *
from agents import AgentType, str_to_agent_type
from rl_trainer import OfflineTrainer


class RLRunner:
    def __init__(self, env_name, cfg_file):
        # load config
        with open(cfg_file, "r") as stream:
            config = yaml.safe_load(stream)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        env = gym.make(env_name)

        # wrap env if required
        if not config["env_wrapper"] == "no_wrapper":
            env = locals()[config["env_wrapper"]](env)

        # create agent
        agent_type = str_to_agent_type(config['agent'])
        agent = agent_type.make_agent(env=env, device=device, **config["agent_params"])

        # create trainer
        if config["trainer"] == "offline":
            self.trainer = OfflineTrainer(env=env, agent=agent, **config["trainer_params"])
        else:
            raise NotImplementedError

    def start(self):
        self.trainer.train()


if __name__ == "__main__":
    runner = RLRunner("CartPole-v0", "config/RainbowDQNAgent/CartPole-v0.yaml")
    runner.start()
