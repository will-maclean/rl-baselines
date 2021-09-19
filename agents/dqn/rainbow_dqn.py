import numpy as np
import torch
import wandb

from .net import NoisyDuellingDQN
from .dqn import DQNAgent
from utils.replay import StandardReplayBuffer


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
                 replay_buffer_type=StandardReplayBuffer,
                 lr=3e-4,
                 max_steps=10_000,
                 polyak_tau=0.05,
                 hard_update_freq=None,
                 soft_update_freq=None,
                 reward_scale=1,
                 batch_size=32,
                 max_memory=10_000,
                 hidden_size=16,
                 num_hidden=0,
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
                                              reward_scale=reward_scale,
                                              replay_buffer_type=replay_buffer_type,
                                              batch_size=batch_size,
                                              max_memory=max_memory,
                                              hidden_size=hidden_size,
                                              num_hidden=num_hidden,
                                              )

    def act(self, state, env, step=-1):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=torch.float32)

        qs = self.net(state).squeeze()
        return qs.argmax().detach().cpu().item(), None

    def train_step(self, step):
        loss = super().train_step(step)

        self.net.update_noise()
        self.target_net.update_noise()

        return loss

    # def train_step(self, step):
    #     weights, sampled_exp, indexes = self.memory.sample(self.batch_size, step)
    #
    #     states, actions, next_states, rewards, dones = zip(*sampled_exp)
    #
    #     states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device)
    #     next_states = torch.from_numpy(np.stack([np.float32(state) for state in next_states])).to(self.device)
    #     actions = torch.tensor(actions, dtype=torch.long, device=self.device)
    #     rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
    #     dones = torch.tensor(dones, dtype=torch.float, device=self.device)
    #
    #     q_values = self.net(states)
    #     actions = actions.unsqueeze(1)
    #     q_values = q_values.gather(1, actions).squeeze(1)
    #
    #     next_actions = self.net(next_states).argmax(dim=1).unsqueeze(1)
    #     next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
    #
    #     td_target: torch.Tensor = rewards + self.gamma * next_q_values * (1 - dones)
    #     td_errors: torch.Tensor = torch.nn.functional.smooth_l1_loss(td_target, q_values, reduction="none").to(
    #         self.device)
    #
    #     losses = td_errors * torch.tensor(weights).to(self.device)
    #     loss = losses.mean()
    #
    #     self.optim.zero_grad()
    #     loss.backward()
    #     self.optim.step()
    #
    #     self.memory.update_priorities(losses.detach().cpu().numpy() + 1e-5, indexes)
    #
    #     self.net.update_noise()
    #     self.target_net.update_noise()
    #
    #     return {
    #         "loss": loss
    #     }

    def log(self, log_dict):
        return None

    def wandb_watch(self):
        wandb.watch(self.net)
