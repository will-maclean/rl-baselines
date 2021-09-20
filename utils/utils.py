import torch
from torch import nn
import numpy as np


def linear_decay(step, start_val, final_val, final_steps):
    fraction = min(float(step) / final_steps, 1.0)
    return start_val + fraction * (final_val - start_val)


def epsilon_decay(step, start_val, final_val, decay):
    return max(final_val + (start_val - final_val) * np.exp(-1 * step / decay), final_val)


def copy_weights(copy_from: nn.Module, copy_to: nn.Module, polyak=None):
    """
    Copy weights from one network to another. Optionally copies with Polyak averaging.

    Parameters
    ----------
    copy_from - net to copy from
    copy_to - net to copy to
    polyak - if None, then don't do Polyak averaging (i.e. directly copy weights). If you want Polyak averaging, then
    set polyak to your tau constant (usually 0.01).

    Returns
    -------
    None
    """
    if polyak is not None:
        for target_param, param in zip(copy_to.parameters(), copy_from.parameters()):
            target_param.data.copy_(polyak * param + (1 - polyak) * target_param)
    else:
        copy_to.load_state_dict(copy_from.state_dict())


def gae(done, rewards, values, n_envs, steps_per_env, gamma, gae_lambda, device):
    """
    Call method for the GAE to calculate/return advantages

    Parameters
    ----------
    done: Tensor[num_workers, num_steps]
    rewards: Tensor [num_workers, num_steps]
    values: Tensor [num_workers, num_steps + 1]

    Returns
    -------
    advantages: Tensor [num_workers, num_steps]
    """
    advantages = torch.zeros((n_envs, steps_per_env, 1), dtype=torch.float, device=device)
    last_advantage = 0
    for state in reversed(range(steps_per_env)):
        error = rewards[:, state] + gamma * values[:, state + 1] * (~done[:, state]) - values[:, state]
        last_advantage = (error + gamma * gae_lambda * last_advantage * (~done[:, state]))

        advantages[:, state] = last_advantage

    return advantages
