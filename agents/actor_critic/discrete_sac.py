import numpy as np
import torch
import torch.nn.functional as F

from utils.replay import StandardReplayBuffer
from .net import DiscretePolicy, DiscreteCritic
from .sac import SACAgent


class DiscreteSACAgent(SACAgent):
    def __init__(self,
                 env,
                 device,
                 name="DiscreteSAC",
                 replay_buffer_type=StandardReplayBuffer,
                 max_memory=10_000,
                 batch_size=32,
                 gamma=0.99,
                 alpha=0.2,
                 trainable_alpha=True,
                 lr_pi=3e-4,
                 lr_q=3e-4,
                 lr_a=3e-4,
                 polyak_tau=5e-3,
                 reward_scale=1,
                 pi_class=DiscretePolicy,
                 critic_class=DiscreteCritic,
                 pi_hidden_size=64,
                 q_hidden_size=64,
                 min_alpha=None
                 ):
        super().__init__(
            env=env,
            device=device,
            name=name,
            replay_buffer_type=replay_buffer_type,
            max_memory=max_memory,
            batch_size=batch_size,
            gamma=gamma,
            alpha=alpha,
            trainable_alpha=trainable_alpha,
            lr_pi=lr_pi,
            lr_q=lr_q,
            lr_a=lr_a,
            polyak_tau=polyak_tau,
            reward_scale=reward_scale,
            pi_class=pi_class,
            critic_class=critic_class,
            pi_hidden_size=pi_hidden_size,
            q_hidden_size=q_hidden_size,
            min_alpha=min_alpha,
        )

    def calculate_actor_loss(self, state):
        policy = self.pi(state)
        with torch.no_grad():
            qf1_pi = self.q1(state)
            qf2_pi = self.q2(state)
            min_qf = torch.min(qf1_pi, qf2_pi)

        ln_policy = torch.clamp(policy.log(), -20, 0)

        policy_loss = (policy * (self.alpha * ln_policy - min_qf)).sum(dim=1).mean()
        return policy_loss, ln_policy

    def calculate_alpha_loss(self, ln_policy):
        alpha_loss = -(self.log_alpha * (ln_policy + self.target_entropy).detach()).mean()
        return alpha_loss

    def calculate_critic_losses(self, states, actions, next_states, rewards, dones):
        with torch.no_grad():
            qf1_next_target = self.q1_targ(next_states)
            qf2_next_target = self.q2_targ(next_states)
            next_state_policy = self.pi(next_states)

            next_ln_pi = torch.clamp(next_state_policy.log(), -20, 0)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_ln_pi
            min_qf_next_target = next_state_policy * min_qf_next_target
            next_q_value = self.reward_scale * rewards + (1.0 - dones) * self.gamma * min_qf_next_target

        qf1 = self.q1(states)
        qf2 = self.q2(states)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        return qf1_loss, qf2_loss

    def get_batch(self, batch_size):
        states, actions, next_states, rewards, dones, = self.memory.sample(batch_size)

        states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device,
                                                                                        dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([np.float32(state) for state in next_states])).to(self.device,
                                                                                                  dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, next_states, rewards, dones