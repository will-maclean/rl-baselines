import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam

from agents.actor_critic.net import ContinuousPolicy, ContinuousCritic, ContinuousMLPAC
from agents.agent import OnlineAgent
from utils.env import ParallelEnv
from utils.replay import StandardReplayBuffer
from utils.utils import gae


def ppo_pi_clip_loss(ln_pi, old_ln_pi, adv, eps=0.2):
    r = torch.exp(ln_pi - old_ln_pi)
    ratio_advantage = r * adv
    clipped_horatio = torch.clamp(r, 1 - eps, 1 + eps)
    clipped_ratio_advantage = clipped_horatio * adv

    # Compute Loss
    ppo_loss = torch.min(ratio_advantage, clipped_ratio_advantage)

    return -ppo_loss.mean()  # gradient ascent


def ppo_v_loss(values, returns, old_values, eps=0.2, loss_type="huber"):
    if loss_type == "huber":
        loss = F.smooth_l1_loss(values, returns)

    elif loss_type == "clipped":
        clipped_values = old_values + torch.clamp(values - old_values, -eps, eps)
        clipped_value_loss = (clipped_values - returns) ** 2
        value_loss = (values - returns) ** 2
        loss = torch.max(clipped_value_loss, value_loss)
    else:
        raise NotImplementedError

    return loss.mean()


class PPOAgent(OnlineAgent):

    def __init__(self,
                 env,
                 env_name,
                 device,
                 env_wrappers=None,
                 name="PPO",
                 hidden_size=32,
                 num_hidden=2,
                 steps_per_env=32,
                 n_envs=4,
                 n_mini_batch=4,
                 lr=3e-4,
                 epochs=4,
                 gamma=1.,
                 gae_lambda=0.5,
                 critic_coefficient=0.5,
                 entroy_coefficient=0.2
                 ):

        memory = StandardReplayBuffer(steps_per_env * n_envs)
        super().__init__(
            env=env,
            device=device,
            name=name,
            memory=memory,
        )

        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.steps_per_env = steps_per_env
        self.n_envs = n_envs
        self.n_mini_batch = n_mini_batch
        self.lr = lr
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.critic_coefficient = critic_coefficient
        self.entropy_coefficient = entroy_coefficient
        self.env_name = env_name

        assert self.n_envs * self.steps_per_env % self.n_mini_batch == 0

        self.batch_size = steps_per_env * n_envs
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        self.S = env.reset().shape[0]
        self.A = env.action_space.shape[0]

        self.net = ContinuousMLPAC(
            S=self.S,
            A=self.A,
            device=device,
            hidden_size=self.hidden_size,
            num_hidden=self.num_hidden,
        ).to(device)

        self.optim = Adam(self.net.parameters(), self.lr)

        self.obs_shape = env.reset().shape
        self.a_size = env.action_space.shape[0]  # todo - set up for discrete envs as well

        self.obs = torch.zeros((self.n_envs, *self.obs_shape), dtype=torch.float32)

        def make_env():
            if env_wrappers is None:
                return gym.make(env_name)
            else:
                return env_wrappers(gym.make(env_name))

        self.envs = [make_env() for _ in range(self.n_envs)]

        for w, env in enumerate(self.envs):
            self.obs[w] = torch.tensor(env.reset(), device=self.device, dtype=torch.float)

    def act(self, state, env, step=-1):
        a, v = self.net.act(state, v=True)
        return a, v

    def train_step(self, samples):
        # we want to train on the data self.epochs times
        log_dicts = []
        for _ in range(self.epochs):

            # shuffle sample ID's
            shuffled_idx = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                # we want a shuffled mini batch
                end = start + self.mini_batch_size
                mini_batch_idx = shuffled_idx[start:end]  # [3, 6, 1, 2, 10, 4, 6] -> [3, 6, 1]
                mini_batch = {}
                for k, v in samples.items():
                    # mini_batch is a Dict of experience attributes randomly indexed with size mini_batch_size
                    mini_batch[k] = v[mini_batch_idx]

                # loss & found
                loss, log_dict = self._calc_loss(sample=mini_batch)

                log_dicts.append(log_dict)

                self.optim.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optim.step()

        return log_dicts[0]

    def sample(self):
        rewards = torch.zeros((self.n_envs, self.steps_per_env, 1), dtype=torch.float32, device=self.device)
        actions = np.zeros((self.n_envs, self.steps_per_env, self.a_size), dtype=np.float32)
        done = torch.zeros((self.n_envs, self.steps_per_env, 1), dtype=torch.bool, device=self.device)
        obs = torch.zeros((self.n_envs, self.steps_per_env, *self.obs_shape), dtype=torch.float32,
                          device=self.device)
        log_probs = torch.zeros((self.n_envs, self.steps_per_env, self.a_size), dtype=torch.float32,
                                device=self.device)
        values = torch.zeros((self.n_envs, self.steps_per_env + 1, 1), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # sample worker_steps from each worker
            for t in range(self.steps_per_env):

                # self.obs contains the last observation from each worker
                self.obs = self.obs.to(self.device)
                obs[:, t] = self.obs

                # sample actions from old policy for each worker
                dist, v = self.net(self.obs)
                values[:, t] = v
                a = dist.sample()
                log_probs[:, t] = dist.log_prob(a)
                actions[:, t] = a.clone().detach().cpu().numpy()

                # have each worker execute the selected action
                for w, env in enumerate(self.envs):
                    next_state, rewards[w, t], done[w, t], _ = env.step(actions[w, t])

                _, v = self.net(self.obs)
                values[:, self.steps_per_env] = v

            advantages = gae(done, rewards, values, self.n_envs, self.steps_per_env,
                             self.gamma, self.gae_lambda, self.device)

            actions = torch.from_numpy(actions).to(device=self.device)
            samples = {
                'obs': obs,
                'actions': actions,
                'values': values[:, :-1],
                'log_pis': log_probs,
                'advantages': advantages
            }

            samples_flat = {}
            for k, v in samples.items():
                samples_flat[k] = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])

            return samples_flat

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, log_dict):
        pass

    def config(self):
        return {
            "env_name": self.env_name,
             "name": self.name,
            "hidden_size": self.hidden_size,
            "num_hidden": self.num_hidden,
            "steps_per_env": self.steps_per_env,
            "n_envs": self.n_envs,
            "n_mini_batch": self.n_mini_batch,
            "replay_buffer_type": self.memory.__class__,
            "lr": self.lr,
            "epochs": self.epochs,
            "gamma": self.gamma,
            "critic_coefficient": self.critic_coefficient,
            "entropy_coefficient": self.entropy_coefficient,
        }

    def ready_to_train(self):
        return self.memory.full()

    @staticmethod
    def _normalise(to_normalise: torch.Tensor):
        return (to_normalise - to_normalise.mean()) / (to_normalise.std() + 1e-10)

    def _calc_loss(self, sample):
        new_dist, new_val = self.net(sample['obs'])
        new_log_prob = new_dist.log_prob(sample['actions'])

        sample['advantages'] = self._normalise(sample['advantages'])
        sampled_return = sample['values'] + sample['advantages']

        ppo_loss = ppo_pi_clip_loss(new_log_prob, sample['log_pis'], sample['advantages'])
        critic_loss = ppo_v_loss(new_val, sampled_return,
                                 sample['values'])
        entropy = new_dist.entropy().mean()
        loss = ppo_loss + self.critic_coefficient * critic_loss - self.entropy_coefficient * entropy

        # FIXME - should I be replacing the current log probs in the sample with the new calculated log probs?
        # sample['log_pis'] = sampled_log_prob.detach()
        # return loss, sample

        return loss, {
            "loss": loss.item(),
            "policy_loss": ppo_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

    def wandb_watch(self):
        wandb.watch(self.net)
