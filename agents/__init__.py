from enum import Enum

from .dqn import DQNAgent, RainbowDQNAgent
from .actor_critic import SACAgent, DiscreteSACAgent
from .policy_gradient import PPOAgent
from .agent import RLAgent


class AgentType(Enum):
    dqn = 0,
    rainbow_dqn = 1,
    sac = 2,
    discrete_sac = 3,

    def make_agent(self, *args, **kwargs):
        if self == AgentType.dqn:
            return DQNAgent(*args, **kwargs)
        elif self == AgentType.rainbow_dqn:
            return RainbowDQNAgent(*args, **kwargs)
        elif self == AgentType.sac:
            return SACAgent(*args, **kwargs)
        elif self == AgentType.discrete_sac:
            return DiscreteSACAgent(*args, **kwargs)
        else:
            raise NotImplementedError


def str_to_agent_type(name):
    if name == "dqn":
        return AgentType.dqn
    elif name == "rainbow_dqn":
        return AgentType.rainbow_dqn
    elif name == "sac":
        return AgentType.sac
    elif name == "discrete_sac":
        return AgentType.discrete_sac
