import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

from .net import ContinuousPolicy, ContinuousCritic
from agents.agent import OfflineAgent
from utils.replay import StandardReplayBuffer
from utils import copy_weights


class SACAgent(OfflineAgent):
    def __init__(self,
                 env,
                 device,
                 name="SAC",
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
                 pi_class=ContinuousPolicy,
                 critic_class=ContinuousCritic,
                 pi_hidden_size=64,
                 q_hidden_size=64,
                 min_alpha=None
                 ):

        memory = replay_buffer_type(max_memory)
        super(SACAgent, self).__init__(
            env=env,
            name=name,
            memory=memory,
            batch_size=batch_size,
            device=device
        )
        self.s = env.reset().shape[0]
        try:
            self.a = env.action_space.shape[0]
        except IndexError:
            self.a = env.action_space.n

        self.gamma = gamma
        self.lr_pi = lr_pi
        self.lr_q = lr_q
        self.lr_a = lr_a
        self.polyak_tau = polyak_tau
        self.reward_scale = reward_scale
        self.pi_hidden_size = pi_hidden_size
        self.q_hidden_size = q_hidden_size
        self.min_alpha = min_alpha

        self.trainable_alpha = trainable_alpha

        if self.trainable_alpha:
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, dtype=torch.float32).to(self.device)
            self.target_entropy = torch.tensor(-0.98*np.log(1/self.a), dtype=torch.float32).to(self.device)
            self.alpha = self.log_alpha.exp()
        else:
            self.log_alpha = None
            self.target_entropy = None
            self.alpha = alpha

        self.pi = pi_class(self.s, self.a, device, hidden_size=pi_hidden_size).to(self.device)
        self.q1 = critic_class(self.s, self.a, hidden_size=q_hidden_size).to(self.device)
        self.q2 = critic_class(self.s, self.a, hidden_size=q_hidden_size).to(self.device)
        self.q1_targ = critic_class(self.s, self.a, hidden_size=q_hidden_size).to(self.device)
        self.q2_targ = critic_class(self.s, self.a, hidden_size=q_hidden_size).to(self.device)

        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # Freeze all the parameters in the network
        for param in self.q1_targ.parameters():
            param.requires_grad = False

        for param in self.q2_targ.parameters():
            param.requires_grad = False

        self.pi_optim = Adam(params=self.pi.parameters(), lr=self.lr_pi)
        self.q1_optim = Adam(params=self.q1.parameters(), lr=self.lr_q)
        self.q2_optim = Adam(params=self.q2.parameters(), lr=self.lr_q)

        if self.trainable_alpha:
            self.a_optim = Adam(params=[self.log_alpha], lr=self.lr_a)

    def act(self, state, env, step=-1):
        a, _ = self.pi.act(state)

        return a, None

    def get_batch(self, batch_size):
        states, actions, next_states, rewards, dones = self.memory.sample(batch_size)

        states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device,
                                                                                        dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([np.float32(state) for state in next_states])).to(self.device,
                                                                                                  dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device).squeeze(-1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, next_states, rewards, dones

    def train_step(self, step):
        states, actions, next_states, rewards, dones = self.get_batch(self.batch_size)

        q1_loss, q2_loss = self.calculate_critic_losses(states, actions, next_states, rewards, dones)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5)
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5)
        self.q2_optim.step()

        pi_loss, ln_pi = self.calculate_actor_loss(states)

        self.pi_optim.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 5)
        self.pi_optim.step()

        log_dict = {
            "pi_loss": pi_loss.detach().cpu().item(),
            "q1_loss": q1_loss.detach().cpu().item(),
            "q2_loss": q2_loss.detach().cpu().item(),
            "alpha": self.alpha,
            "log_alpha": self.log_alpha,
        }

        if self.trainable_alpha:
            alpha_loss = self.calculate_alpha_loss(ln_pi)

            self.a_optim.zero_grad()
            alpha_loss.backward()
            self.a_optim.step()

            self.alpha = self.log_alpha.exp()

            if self.min_alpha is not None:
                self.alpha = max(self.alpha, self.min_alpha)

            log_dict["alpha_loss"] = alpha_loss.detach().cpu().item()

        copy_weights(copy_from=self.q1, copy_to=self.q1_targ, polyak=self.polyak_tau)
        copy_weights(copy_from=self.q2, copy_to=self.q2_targ, polyak=self.polyak_tau)

        return log_dict

    def calculate_actor_loss(self, state):
        action, log_pi = self.pi.act(state)
        with torch.no_grad():
            qf1_pi = self.q1(state, action)
            qf2_pi = self.q2(state, action)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_alpha_loss(self, log_pi):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def calculate_critic_losses(self, states, actions, next_states, rewards, dones):
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.pi.act(next_states)
            qf1_next_target = self.q1_targ(next_states, next_state_action)
            qf2_next_target = self.q2_targ(next_states, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = self.reward_scale * rewards + (1.0 - dones) * self.gamma * min_qf_next_target

        qf1 = self.q1(states, actions)
        qf2 = self.q2(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, log_dict):
        pass

    def config(self):
        return {
            "lr_pi": self.lr_pi,
            "lr_q": self.lr_q,
            "lr_a": self.lr_a,
            "alpha0": self.alpha,
            "gamma": self.gamma,
            "polyak_tau": self.polyak_tau,
            "pi_class": self.pi.__class__,
            "q_class": self.q1.__class__,
            "reward_scale": self.reward_scale,
            "q_hidden_size": self.q_hidden_size,
            "pi_hidden_size": self.pi_hidden_size,
            "min_alpha": self.min_alpha,
        }

