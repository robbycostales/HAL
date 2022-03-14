# Almost unchanged from https://github.com/ShangtongZhang/DeepRL
import torch
import numpy as np
import pickle
from ..utils import *
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(wb=config.wb, tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        if hasattr(self, 'task'):
            close_obj(self.task)
        if hasattr(self, 'replay'):
            close_obj(self.replay)
        if hasattr(self, 'actor'):
            close_obj(self.actor)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        episodic_success = []
        for ep in range(self.config.eval_episodes):
            info = self.eval_episode()
            total_rewards = info['ret']
            success = info['success']
            episodic_returns.append(np.sum(total_rewards))
            episodic_success.append(int(success))
        if hasattr(self, 'n_updates'):
            steps = self.n_updates
        else:
            steps = self.total_steps
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.log_scalar('episodic_return_test', np.mean(episodic_returns), steps)
        self.logger.log_scalar('success_rate_test', np.mean(episodic_success), steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
            'success_rate_test': np.mean(episodic_success)
        }

    def record_online_return(self, info, offset=0):
        if hasattr(self, 'n_updates'):
            steps = self.n_updates
        else:
            steps = self.total_steps
        if isinstance(info, dict):
            info = (info,)

        # episodic_return is not None only when episode is finished
        rets = [i['episodic_return'] for i in info if i['episodic_return'] is not None]
        successes = [i['success'] for i in info if i['episodic_return'] is not None]
        ep_len = []
        for curr_info in info:
            if curr_info['episodic_return'] is not None:
                if curr_info['impossible']:
                    # don't want to conflate shorter ep due to success w/
                    # impossibility
                    curr_ep_len = curr_info['steps'] + curr_info['steps_remaining']
                else:
                    curr_ep_len = curr_info['steps']
                ep_len.append(curr_ep_len)
        if len(rets) > 0:
            ret = np.mean(rets)
            self.logger.log_scalar('episodic_return_train', ret, steps + offset)
            self.logger.info('steps %d, episodic_return_train %s' % (steps + offset, ret))
            success_rate = np.mean(successes)
            self.logger.log_scalar('success_rate', success_rate, steps + offset)
            self.logger.info('steps %d, success_rate %s' % (steps + offset, success_rate))
            mean_ep_len = np.mean(ep_len)
            self.logger.log_scalar('episode_length', mean_ep_len, steps + offset)
            self.logger.info('steps %d, episode_length %s' % (steps + offset, mean_ep_len))

    def switch_task(self):
        '''
        Prepares environment for next rollout (only resets if finished)
        '''
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self._pipe, self._worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency // self.config.num_envs):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self._worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self._worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self._worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self._pipe.send([self.STEP, None])
        return self._pipe.recv()

    def close(self):
        self._pipe.send([self.EXIT, None])
        self._pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self._pipe.send([self.NETWORK, net])
