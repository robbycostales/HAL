# Modified from https://github.com/ShangtongZhang/DeepRL
from ..utils import to_np, epsilon_greedy, close_obj, tensor
from ..component.envs import LazyFrames
from ..utils.config import Config
from ..utils.torch_utils import range_tensor
from .BaseAgent import BaseAgent, BaseActor

import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import psutil


class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _set_up(self):
        if self.config.c51:
            self.config.atoms = tensor(self.config.atoms)

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()

        config = self.config
        if config.noisy_linear:
            self._network.reset_noise()
        with config.lock:
            prediction = self._network(config.state_normalizer(self._state))
        q_values = self.compute_q(prediction)

        epsilon = config.random_action_prob()  # always want to iterate schedule
        if config.noisy_linear:
            epsilon = 0
        elif self._total_steps < config.exploration_steps:
            epsilon = 1
        action = epsilon_greedy(epsilon, q_values)
        next_state, reward, done, info = self._task.step(action)
        entry = [self._state, action, reward, next_state, done, info]
        self._total_steps += config.num_envs
        self._state = next_state
        return entry

    def compute_q(self, prediction):
        if self.config.c51:
            q_values = to_np((prediction['prob'] * self.config.atoms).sum(-1))
        else:
            q_values = to_np(prediction['q'])
        return q_values


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        if config.c51:
            print(config.categorical_n_atoms, config.categorical_v_min,
                  config.categorical_v_max)
            config.atoms = np.linspace(config.categorical_v_min,
                                       config.categorical_v_max,
                                       config.categorical_n_atoms)

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)
        self.total_steps = 0
        self.n_updates = 0

        if config.c51:
            self.batch_indices = range_tensor(config.batch_size)
            self.atoms = tensor(config.atoms)
            self.delta_atom = (config.categorical_v_max -
                               config.categorical_v_min) / \
                               float(config.categorical_n_atoms - 1)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        if self.config.c51:
            prediction = self.network(state)
            prediction = (prediction['prob'] * self.atoms).sum(-1)
        else:
            prediction = self.network(state)['q']
        q = self.actor.compute_q(prediction)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def reduce_c51_loss(self, loss):
        return loss.mean()

    def reduce_loss(self, loss):
        return loss.pow(2).mul(0.5).mean()

    def compute_c51_loss(self, states, next_states, masks, rewards, actions):
        config = self.config
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)
        with torch.no_grad():
            prob_next = self.target_network(next_states)['prob']
            q_next = (prob_next * self.atoms).sum(-1)
            if config.double_q:
                a_next = torch.argmax((self.network(next_states)['prob'] *
                                       self.atoms).sum(-1), dim=-1)
            else:
                a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices, a_next, :]

        # Unsqueeze to match atom dimensions
        rewards = rewards.unsqueeze(-1)
        masks = masks.unsqueeze(-1)
        atoms_target = rewards + self.config.discount ** config.n_step * \
            masks * self.atoms.view(1, -1)
        atoms_target.clamp_(self.config.categorical_v_min,
                            self.config.categorical_v_max)
        atoms_target = atoms_target.unsqueeze(1)
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() /
                       self.delta_atom).clamp(0, 1) * prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.network(states)['log_prob']
        actions = actions.long()
        log_prob = log_prob[self.batch_indices, actions, :]
        KL = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1)
        return KL

    def compute_loss(self, states, next_states, masks, rewards, actions):
        config = self.config
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)

        with torch.no_grad():
            q_next = self.target_network(next_states)['q'].detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states)['q'],
                                            dim=-1)
                q_next = q_next.gather(1,
                                       best_actions.unsqueeze(-1)).squeeze(1)
            else:
                q_next = q_next.max(1)[0]

        gamma = config.discount
        q_target = rewards + gamma ** config.n_step * q_next * masks
        q = self.network(states)['q']
        q = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = q_target - q
        return loss

    def update(self):
        """Update agent parameters
        """
        config = self.config

        transitions = self.replay.sample()

        # Reset noise in noisy layers each iteration
        if config.noisy_linear:
            self.target_network.reset_noise()
            self.network.reset_noise()

        # Prepare variables for loss computation
        state = {'img': transitions.im_state,
                 'vec': transitions.vec_state}
        next_state = {'img': transitions.next_im_state,
                      'vec': transitions.next_vec_state}
        states = tensor(state)
        next_states = tensor(next_state)
        masks = tensor(transitions.mask)
        rewards = tensor(transitions.reward)
        actions = tensor(transitions.action)
        # Actions are discrete (cast to int64)
        actions = actions.type(torch.LongTensor).to(Config.DEVICE)

        # Compute loss
        if config.c51:
            loss = self.compute_c51_loss(states, next_states, masks, rewards,
                                         actions)
        else:
            loss = self.compute_loss(states, next_states, masks, rewards,
                                     actions)

        # Update priorities in prioritized replay
        if config.prior_replay:
            priorities = loss.abs().add(config.replay_eps)\
                .pow(config.replay_alpha)
            idxs = tensor(transitions.idx).long()
            self.replay.update_priorities(zip(to_np(idxs),
                                          to_np(priorities)))
            sampling_probs = tensor(transitions.sampling_prob)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6)\
                .pow(-config.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        # Compute gradients
        if config.c51:
            loss = self.reduce_c51_loss(loss)  # reduce loss to a float
        else:
            loss = self.reduce_loss(loss)
        self.optimizer.zero_grad()  # reset the gradients
        loss.backward()  # compute new gradients
        # Clip gradients
        grad_norm = \
            nn.utils.clip_grad_norm_(self.network.parameters(),
                                     self.config.gradient_clip)
        # Apply the gradients through optimizer
        with config.lock:
            self.optimizer.step()

        self.logger.add_scalar('losses/q_loss', loss)
        self.logger.add_scalar('gradients/grad_norm', grad_norm)

        self.n_updates += 1

    def step(self):
        """Training step"""
        config = self.config
        # Take environmental step
        transitions = self.actor.step()
        # Feed transitions from step into replay
        for states, actions, rewards, next_states, dones, info in transitions:
            self.record_online_return(info)
            self.total_steps += config.num_envs
            self.replay.feed(dict(
                im_state=
                    np.array([s['img'][-1] if isinstance(s['img'],
                              LazyFrames) else s['img'] for s
                              in states]),
                vec_state=np.array([s['vec'] for s in states]),
                action=actions,
                reward=[config.reward_normalizer(r) for r in rewards],
                mask=1 - np.asarray(dones, dtype=np.int32),
                timeout=np.array([i['timeout'] for i in info])
            ))

        # Need enough samples (from exploration) before training components
        if self.total_steps <= self.config.exploration_steps:
            return

        # Perform training step
        self.update()

        # Dump logging scalars at a certain frequency
        if self.n_updates % self.config.log_interval == 0:
            # Monitor memory usage and replay capacity
            # Percentage of total ram used
            perc_ram = psutil.virtual_memory().percent
            self.logger.add_scalar('memory/perc_ram', perc_ram)

            # Monitor replay capacities
            self.logger.add_scalar('memory/replay_size', self.replay.size())

            self.logger.dump_scalars(self.n_updates)

        # Update target network at the appropriate frequency
        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
