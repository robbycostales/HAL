# Modified from https://github.com/ShangtongZhang/DeepRL
import gym
import numpy as np
from gym.spaces import Dict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from multiprocessing import Process, Pipe

from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import VecEnv, CloudpickleWrapper

from ..utils import *


def make_env(env_id, seed, rank, gridrl_params=None):
    def _thunk():
        # identify task and create environment
        if 'MiniGrid' in env_id:
            env = gym.make(env_id, gridrl_params=gridrl_params)
        else:
            env = gym.make(env_id)

        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.total_rewards += reward

        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        if type(self.observation_space) is Dict:
            new_spaces = {}
            for name, space in self.observation_space.spaces.items():
                if len(space.shape) == 3:  # image
                    new_spaces[name] = Box(
                        space.low[0, 0, 0],
                        space.high[0, 0, 0],
                        [space.shape[2], space.shape[1], space.shape[0]],
                        dtype=space.dtype)
                else:
                    new_spaces[name] = space
            self.observation_space = Dict(new_spaces)
        else:
            obs_shape = self.observation_space.shape
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0],
                [obs_shape[2], obs_shape[1], obs_shape[0]],
                dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.
        """
        self._frames = frames

    def __array__(self, dtype=None, frame_inds=None):
        if frame_inds is None:
            frame_inds = range(len(self._frames))
        elif type(frame_inds) is int:
            frame_inds = [frame_inds]
        out = np.concatenate([self._frames[i] for i in frame_inds], axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        """
        Previous version of this code stacked frames and indexed the channels.
        We want to index by *frames* not channels (important if we're using RGB
        instead of grayscale images).
        """
        return self.__array__(frame_inds=i)


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns, auto_reset=True):
        self.auto_reset = auto_reset
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                if self.auto_reset:
                    obs = self.envs[i].reset()
                # Dealing with timeout / done distinction
                if info.get('timeout', False):
                    done = False
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close_extras(self):
        self.closed = True
        for env in self.envs:
            env.close()


def worker(remote, parent_remote, env_fn_wrapper, auto_reset):
    parent_remote.close()
    env = env_fn_wrapper.x()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    if auto_reset:
                        ob = env.reset()
                    # Dealing with timeout / done distinction
                    if info.get('timeout', False):
                        done = False
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                if data is not None:
                    remote.send(env.render(mode='rgb_array', **data))
                else:
                    remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    Modified from: https://github.com/openai/baselines/blob/8e56ddeac296deab3cc0adc79c84a5abb59d7a3a/baselines/common/vec_env/subproc_vec_env.py
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, auto_reset=True):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.auto_reset = auto_reset
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), auto_reset))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self, data=None):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', data))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 seed=None,
                 gridrl_params=None,  # passed to Minigrid
                 auto_reset=True):

        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, gridrl_params=gridrl_params) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs, auto_reset=auto_reset)
        self.name = name
        self.observation_space = self.env.observation_space
        if type(self.env.observation_space) is Box:
            self.state_dim = int(np.prod(self.env.observation_space.shape))
        elif type(self.env.observation_space) is Dict:
            # We are assuming now that there are `img` and `vec` components
            self.vec_dim = int(np.prod(self.env.observation_space['vec'].shape))
            self.img_dim = self.env.observation_space['img'].shape
            self.state_dim = [self.vec_dim, self.img_dim]
        else:
            self.state_dim = None

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

    def close(self):
        self.env.close()

    def get_images(self, data=None):
        return self.env.get_images(data=data)
