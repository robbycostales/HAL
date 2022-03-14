# Modified from https://github.com/ShangtongZhang/DeepRL
import torch
import numpy as np
import random
from collections import namedtuple, defaultdict
import torch.multiprocessing as mp

from ..utils import SumTree
from ..utils.torch_utils import tensor

# Legacy
Transition = namedtuple('Transition', ['state', 'action', 'reward',
                                       'next_state', 'mask'])
PrioritizedTransition = namedtuple('Transition',
                                   ['state', 'action', 'reward', 'next_state',
                                    'mask', 'sampling_prob', 'idx'])


def get_transition_class(prioritized=False, recurrent=False, algorithm='dqn'):
    transition_entries = ['action', 'reward', 'mask']
    state_types = ['im_state', 'vec_state']

    transition_entries += state_types
    for stype in state_types:
        transition_entries += ['next_%s' % stype]

    if algorithm in ['hdqn-c']:
        transition_entries += ['goal']

    if algorithm in ['hdqn-mc']:
        transition_entries += ['next_affordance']

    transition_entries += ['timeout']

    if recurrent:  # refesh hidden states a la R2D2
        transition_entries += ['prev_hidden', 'data_idx']

    if prioritized:
        transition_entries += ['sampling_prob', 'idx']

    return namedtuple('Transition', list(set(transition_entries)))


class Storage:
    def __init__(self, memory_size, keys=None):
        if keys is None:
            keys = []
            keys = \
                keys + ['state', 'im_state', 'vec_state', 'action', 'reward',
                        'mask', 'v', 'q', 'pi', 'log_pi', 'entropy',
                        'advantage', 'ret', 'q_a', 'log_pi_a', 'mean',
                        'next_state', 'next_im_state', 'next_vec_state',
                        'nstep_im_state', 'nstep_vec_state', 'hidden',
                        'demo', 'timeout', 'reward_prime',
                        'h', 'dists', 'prev_task', 'next_task', 'goal',
                        'affordance']
        self.keys = keys
        self.memory_size = memory_size
        self.reset()

    def feed(self, data):
        '''
        Append `data` values to corresponding keys in storage
        '''
        # Assumes keys are fed in one at a time
        for k, v in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key: ' + str(k))
            getattr(self, k).append(v)
        self._size += 1

    def placeholder(self):
        '''
        Fill in `None` values for all unused keys
        '''
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.memory_size)

    def reset(self):
        '''
        Empty stored data for all keys
        '''
        for key in self.keys:
            setattr(self, key, [])
        self.pos = 0
        self._size = 0

    def extract(self, keys):
        '''
        Retrieve all stored data for specified keys
        '''
        data = [getattr(self, k)[:self.memory_size] for k in keys]
        data = map(lambda x: torch.cat(x, dim=0), data)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))


class ClassifierStorage:
    '''Designed with affordance classifier in mind. A unique instance will be
    created for each milestone.

    All methods are modified from Storage to support separate storage of data
    points by class, allowing for class balancing.
    '''

    def __init__(self, memory_size, batch_size, n_classes, keys, msi=None,
                 dist_scale=0.7):
        '''Initialization

        Args:
            memory_size: size alottod to each class before data begins to be
                overwritten
            batch_size: default batch size for self.sample()
            n_classes: the number of classes, which will each have a separately
                allocated stoarge so sampling can be more even across classes
            dist_scale: normal distribution std deviation for segment sampling
        '''
        self.data_class = namedtuple('Data', list(set(keys)))

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.keys = keys
        self.msi = msi
        self.dist_scale = dist_scale

        # Flags indicating if true / false negatives exist
        self.fn_exists = False
        self.tn_exists = False

        # Flags indicating parities exist
        self.parity_exists = [0, 0]
        self.both_parities_exist = False

        self.reset()

    def get_datum(self, cls, i, og=None):
        '''Get datum from class (cls) and index (i) specs.

        Args:
            og (int): index of original point, if finding pair;
                ensure that it does not cross the current index of overwriting
        '''
        # If index out of range
        if i >= self._size[cls]:
            return None
        # If similar sample will not be sampled from same trajectory
        if og is not None and (og <= self._i[cls] - 1 <= i or
                               i <= self._i[cls] - 1 <= og):
            return None

        datum = dict()
        for k in self.keys:
            datum[k] = getattr(self, k)[cls][i]
        return self.data_class(**datum)

    def get_clr_datum(self, cls, i):
        '''Get anchor and positive example for contrastive learning.
        Anchor is from class (cls) and index (i).

        Args:
            cls (int): class index
            i (int): data point index
        '''
        # Get data point
        anc_datum = self.get_datum(cls, i)

        # Get positive data point
        irange = getattr(self, 'irange')[cls][i]
        if irange[0] == irange[1]:
            # If only one point in trajectory, can't generate positive example
            return None, None
        # Sample from distribution until we get point within range that is not
        # equal to original data point
        while True:
            # Construct probability distribution over nearest samples
            k = np.random.normal(loc=0.0, scale=self.dist_scale)
            k = int(np.rint(k))
            if irange[0] <= k <= irange[1] and k != 0:
                break
        # Get positive example
        pos_datum = self.get_datum(cls, i+k, og=i)

        return anc_datum, pos_datum

    def sample(self, batch_size=None):
        '''Randomly sample elements from the replay
        '''
        if batch_size is None:
            batch_size = self.batch_size

        # If any of the classes have no elements, or if both parities do not
        # exist for positive class, refuse to sample
        if min(self._size) == 0 or not self.both_parities_exist:
            return None

        # Sample evenly from each class
        sampled_data = []
        while len(sampled_data) < batch_size:
            cls = np.random.randint(0, self.n_classes)
            i = np.random.randint(0, self._size[cls])
            d = self.get_datum(cls, i)
            sampled_data.append(d)

        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        sampled_data = self.data_class(*sampled_data)

        return sampled_data

    def sample_class(self, cls, batch_size=None, negtype=None, parity=None):
        """Sample from class `cls`

        Args:
            cls (int): class index
            batch_size (int):  batch size
            negtype (bool, None): True (sample true negatives), False (false
                negatives), or None (either)
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self._size[cls] == 0:
            return None

        # If negtype doesn't exist (by checking flags)
        if negtype is not None and ((not self.tn_exists and negtype) or
                                    (not self.fn_exists and not negtype)):
            return None
        # If either parity doesn't exist (by checking flag)
        if parity is not None and not self.both_parities_exist:
            return None

        # Sample continuously from single class
        sampled_data = []
        count = 0
        while len(sampled_data) < batch_size:
            count += 1
            i = np.random.randint(0, self._size[cls])
            d = self.get_datum(cls, i)
            # Deal with negtype, if applicable
            if negtype is not None:
                real_negtype = d.truth == d.label
                # Only allow sample to be added if negtypes match
                if negtype != real_negtype:
                    continue
            # Deal with parity, if applicable
            if parity is not None:
                parity_matches = parity == d.parity
                # Only allow sample to be added if parities match
                if not parity_matches:
                    continue
            sampled_data.append(d)

        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        sampled_data = self.data_class(*sampled_data)

        return sampled_data

    def feed(self, data):
        '''Append `data` values to corresponding keys in storage

        Currently assumes data points are fed one at a time and that
            'label' key corresponds to class identity
        '''
        cls = data['label']
        parity = data['parity']
        truth = data['truth']
        for k, v in data.items():
            # If encountered key does not exist, raise error
            if k not in self.keys:
                raise RuntimeError('Undefined key: ' + str(k))

            # Set negtype flags
            if cls == 0:
                if cls == truth:
                    self.tn_exists = True
                else:
                    self.fn_exists = True
            # Set parity flags, only referring to positive (gridRL specific)
            if cls == 1:
                self.parity_exists[parity] = 1
                if sum(self.parity_exists) == len(self.parity_exists):
                    self.both_parities_exist = True

            if self._ow[cls]:
                getattr(self, k)[cls][self._i[cls]] = v
            else:
                getattr(self, k)[cls].insert(self._i[cls], v)

        if not self._ow[cls]:
            self._size[cls] += 1
            self._total_size += 1

        self._i[cls] += 1
        if self._i[cls] == self.memory_size:
            print("Overwriting!")
            print(self._size)
            self._ow[cls] = True
            self._i[cls] = 0

    def placeholder(self):
        '''Fill in `None` values for all unused keys (not applicable to
        classifier).
        '''
        raise NotImplementedError

    def reset(self):
        '''Empty stored data for all keys
        '''
        for key in self.keys:
            setattr(self, key, [[] for _ in range(self.n_classes)])
        self._i = [0 for _ in range(self.n_classes)]
        self._size = [0 for _ in range(self.n_classes)]
        self._ow = [False for _ in range(self.n_classes)]
        self._total_size = 0


class UniformReplay(Storage):
    def __init__(self, memory_size, batch_size, n_step=1, num_envs=1,
                 discount=1, history_length=1,
                 trajectory_length=None, keys=None,
                 recurrent=False, algorithm='dqn'):
        super(UniformReplay, self).__init__(memory_size, keys)
        self.base_TransitionCLS = \
            get_transition_class(prioritized=False, recurrent=recurrent,
                                 algorithm=algorithm)

        self.batch_size = batch_size
        self.n_step = n_step
        self.num_envs = num_envs
        self.total_skip = num_envs
        self.discount = discount
        if trajectory_length is not None:
            assert history_length == 1, "Frame stack not supported for \
                trajectory sampling"
            assert n_step == 1, "n_step > 1 not supported for \
                trajectory sampling"
        self.history_length = history_length
        self.trajectory_length = trajectory_length
        self.recurrent = recurrent
        self.algorithm = algorithm
        self.pos = 0
        self._size = 0
        self.demo_end = None

    def compute_valid_indices(self):
        indices = []
        indices.extend(list(range(self.history_length - 1,
                                  self.pos - self.n_step)))
        indices.extend(list(range(self.pos + self.history_length - 1,
                                  self.size() - self.n_step)))
        return np.asarray(indices)

    def mark_demo_end(self):
        self.demo_end = self.pos

    def feed(self, data):
        for k, vs in data.items():
            if k not in self.keys:
                print(k)
                raise RuntimeError('Undefined key')
            storage = getattr(self, k)
            pos = self.pos
            size = self.size()
            for v in vs:
                if pos >= len(storage):
                    storage.append(v)
                    size += 1
                else:
                    storage[self.pos] = v
                pos = (pos + 1) % self.memory_size
                # If position was just reset to 0, switch it to be the end of
                #   demonstrations, so we don't overwrite expert data
                if pos == 0 and self.demo_end is not None:
                    # assumes demonstrations fill beginning of buffer
                    self.pos = self.demo_end
        self.pos = pos
        self._size = size

    def sample(self, batch_size=None, min_perc_demos=0.5):
        if batch_size is None:
            batch_size = self.batch_size
        sampled_data = []
        while len(sampled_data) < batch_size:
            # If replay contains demonstrations, and we have an unbalanced
            # amount of non-demonstrations (such that min_perc_demos won't
            # be satisfied), sample from demos / non-demos separately to meet
            # ratio!
            if self.demo_end is not None and \
                    self.size() / self.demo_end > min_perc_demos and \
                    self.size() > self.demo_end + 24:
                demo_idx = np.random.randint(0, self.demo_end)
                agnt_idx = np.random.randint(self.demo_end, self.size())
                p = min_perc_demos
                idx = np.random.choice([demo_idx, agnt_idx], p=[p, 1-p])
            # If ratio isn't a problem, sample uniformly over demos / non
            else:
                idx = np.random.randint(0, self.size())
            transition = self.construct_transition(idx)
            # Construct transition, and if valid, add to sampled data
            if transition is not None:
                sampled_data.append(transition)
        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return self.base_TransitionCLS(*sampled_data)

    def extract(self):
        '''
        Extract all elements
        '''
        data = []
        for i in range(self.size()):
            transition = self.construct_transition(i)
            if transition is not None:
                data.append(transition)
        data = zip(*data)
        data = list(map(lambda x: np.asarray(x), data))
        return self.base_TransitionCLS(*data)

    def valid_index(self, index, n_step, traj_len):
        """
        Checks if `index` is valid given sampling preferences
        """
        try:
            if self.timeout[index]:
                # If episode timed out (didn't finish) we'd need to bootstrap.
                # Can't bootstrap bc we don't have the next state
                return False
        except IndexError:
            return False

        # We feed data from envs in parallel, also taking into account
        #   counterfactual transitions
        step_mult = self.total_skip

        # Given history length, n_step, and step multiplication by number of
        #   environments, calculate starting and ending indices
        start_ind = index - ((self.history_length - 1) * step_mult)
        end_ind = index + ((n_step + traj_len - 1) * step_mult)

        # Test if either:
        # 1) falls between beginning of replay and current position
        if start_ind >= 0 and end_ind < self.pos:
            return True
        # or 2) falls between current position and end of replay
        if start_ind >= self.pos and end_ind < self.size():
            return True
        # Otherwise (if straddles current pos or out of bounds), invalid
        return False

    def _cutoff_ind_range(self, ind_range, cutoff):
        """
        replaces indices < cutoff w/ cutoff
        """
        if cutoff is None:
            return ind_range
        return np.maximum(ind_range, cutoff)

    def construct_transition(self, index, r_dict=False):
        """
        Construct a transition sequence (S, A, R, S', etc.) at a given index.
        Supports n-step returns, as well as trajectory sampling (the latter for
            methods that require sampling contiguous trajectories of
            demonstration data).

        Args:
            index: (int) position in replay
            r_dict: (bool) return type (dict) instead of replay CLS object
        """
        # Test if pre-specified type of sampling is possible at given `index`
        n_step = self.n_step
        traj_len = self.trajectory_length if self.trajectory_length is \
            not None else 1
        if not self.valid_index(index, n_step, traj_len):
            return None

        # We feed data from environments in parallel (num_envs steps at a time)
        step_mult = self.num_envs

        # Check history for any episode termination (done or timeout), so we
        #   don't stack frames from previous episodes
        cutoff = None
        # Frame stack and n_step > 1 not supported for trajectory sampling
        if self.trajectory_length is None:
            for i in reversed(range(index - ((self.history_length - 1) *
                                    step_mult), index, step_mult)):
                # Detect termination by 1) timeout or 2) natural end of ep.
                terminated = 0
                terminated += self.timeout[i]
                terminated += (1 - self.mask[i])
                # If termination, set cutoff at this point
                if terminated >= 1 and cutoff is None:
                    cutoff = i + step_mult
                    break
            # Check future for timeouts so not inter-episodic
            for i in range(index + step_mult,
                           index + (n_step * step_mult + 1), step_mult):
                if self.timeout[i]:
                    n_step = (i - index) // step_mult
                    break

        # Calculate start and end indices of sequence, and verify
        s_start = index - ((self.history_length - 1) * step_mult)
        s_end = index + (traj_len - 1) * step_mult
        if s_start < 0:
            raise RuntimeError('Invalid index')
        if s_end < self.pos and s_end + (n_step * step_mult) >= self.pos:
            raise RuntimeError('Invalid index')

        # Begin building transition dictionary
        transition_dict = {}

        state_types = ['im_state', 'vec_state']
        # Calculate start of S' (as in S,A,R,S')
        next_s_start = s_start + (n_step * step_mult)
        next_s_end = s_end + (n_step * step_mult)

        # For each state type, calculate S and S' and assign to appropriate
        #   variables
        # State types variable is either ['state'] or ['im_state', 'vec_state']
        for stype in state_types:
            st_shape = getattr(self, stype)[0].shape
            if len(st_shape) >= 2:  # image
                if len(st_shape) == 2:
                    st_shape = (1, *st_shape)
                if self.history_length > 1:
                    # Stack frames on channel dimension, add dummy dim for time
                    combine = lambda x: np.concatenate(x, axis=0)[None]
                else:
                    # Stack frames on new time dimension
                    #   (we remove at end if traj_len == 1)
                    combine = lambda x: np.stack(x, axis=0)
                transition_dict[stype] = combine(
                    [getattr(self, stype)[i].reshape(st_shape) for i in
                     self._cutoff_ind_range(
                        range(s_start, s_end + 1, step_mult), cutoff)])
                transition_dict['next_%s' % stype] = combine(
                    [getattr(self, stype)[i].reshape(st_shape) for i in
                     self._cutoff_ind_range(
                        range(next_s_start, next_s_end + 1, step_mult),
                        cutoff)])
            else:
                # Ignore history_length for non-image states
                transition_dict[stype] = \
                    getattr(self, stype)[s_start + ((self.history_length - 1) *
                                         step_mult):s_end + 1:step_mult]
                transition_dict['next_%s' % stype] = \
                    getattr(self, stype)[next_s_start + ((self.history_length
                                         - 1) * step_mult):next_s_end
                                         + 1:step_mult]

        # Calculate actions and add to transition object
        transition_dict['action'] = np.array(self.action[index:s_end +
                                                         1:step_mult])

        # Create array of rewards used for further calculations
        reward = np.array([[self.reward[i] for i in range(ind, ind + (n_step *
                            step_mult), step_mult)] for ind in
                           range(index, s_end + 1, step_mult)])
        mask = np.array([[self.mask[i] for i in range(ind, ind + (n_step *
                          step_mult), step_mult)] for ind in
                         range(index, s_end + 1, step_mult)])

        # Create cumulative reward / mask
        cum_r = np.zeros(traj_len)
        cum_mask = np.ones(traj_len)
        for i in reversed(np.arange(n_step)):
            cum_r = reward[:, i] + mask[:, i] * self.discount * cum_r
            cum_mask = np.logical_and(cum_mask, mask[:, i]).astype(np.int32)

        # Assign rest of transition object values
        transition_dict['reward'] = cum_r
        transition_dict['mask'] = cum_mask

        # Recurrent models
        if self.recurrent:
            transition_dict['data_idx'] = \
                list(range(index, s_end + 1, step_mult))
            transition_dict['prev_hidden'] = self.hidden[index - step_mult]

        # Include 'timeout' field
        transition_dict['timeout'] = \
            np.array(self.timeout[index:s_end + 1:step_mult])

        # Affordances
        if self.algorithm in ['hdqn-mc']:
            transition_dict['next_affordance'] = \
                np.array(self.affordance[index+step_mult:s_end+step_mult + 1:step_mult])

        # Goals
        if self.algorithm in ['hdqn-c']:
            transition_dict['goal'] = \
                np.array(self.goal[index:s_end + 1:step_mult])

        # If not doing trajectory sampling, remove time index
        if self.trajectory_length is None:
            for k, v in transition_dict.items():
                try:
                    transition_dict[k] = v[0]
                except:
                    print(k, v)

        # Return dictionary format
        if r_dict:
            return transition_dict
        # Return transition CLS format
        else:
            transition = self.base_TransitionCLS(**transition_dict)
            return transition

    def size(self):
        return self._size

    def full(self):
        return self._size == self.memory_size

    def update_hidden(self, info):
        for idx, hidden in info:
            self.hidden[idx] = hidden

    def update_priorities(self, info):
        raise NotImplementedError

    def update_goal_masks(self, info):
        """
        Update goal masks for specified transitions
        Args: (contained in info tuple)
            envi: (int) which parallel environment this update corresponds to
            msi: (int) which incorrect milestone are we adding to goal mask
            n: (int) number of option steps (how many updates to perform
        """
        # The only data we are manipulating is the 'goal' key
        storage = getattr(self, 'goal')
        # Extract elements from tuple
        (envi, msi, n) = info
        num_envs = self.num_envs
        idx = len(storage) - (num_envs - envi)
        for _ in range(n):
            if idx < 0:
                # This will happen when we purge the replay, and it still wants
                # to update old data
                break
            storage[idx][msi] = 1.0
            idx -= num_envs


class PrioritizedReplay(UniformReplay):
    def __init__(self, memory_size, batch_size, n_step=1, num_envs=1,
                 discount=1, history_length=1,
                 trajectory_length=None, keys=None, recurrent=False,
                 algorithm='dqn'):
        super(PrioritizedReplay, self).__init__(memory_size, batch_size,
                                                n_step, num_envs,
                                                discount,
                                                history_length,
                                                trajectory_length, keys,
                                                recurrent=recurrent,
                                                algorithm=algorithm)
        self.prior_TransitionCLS = \
            get_transition_class(prioritized=True, recurrent=recurrent,
                                 algorithm=algorithm)
        self.tree = SumTree(memory_size)
        self.max_priority = 1

        self.unif_idx_counts = []
        self.prior_idx_counts = []

    def feed(self, data):
        super().feed(data)
        # Figure out length of data, to make sure we add the right number of
        # elements to our tree

        # Get sample key
        k = list(data.keys())[0]
        # Get length of data from getting length of dictionary entry
        n = len(data[k])

        # Insert a tree item *for each* sample that is being fed
        for _ in range(n):
            self.tree.add(self.max_priority, None)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        segment = self.tree.total() / batch_size

        sampled_data = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data_index) = self.tree.get(s)

            data_index_prime = np.random.randint(0, super().size())

            self.prior_idx_counts.append(data_index)
            self.unif_idx_counts.append(data_index_prime)

            transition = super().construct_transition(data_index, r_dict=True)
            if transition is None:
                continue
            sampled_data.append(self.prior_TransitionCLS(
                **transition,
                sampling_prob=p / self.tree.total(),
                idx=idx,
            ))
        while len(sampled_data) < batch_size:
            # This should rarely happen
            sampled_data.append(random.choice(sampled_data))

        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        sampled_data = self.prior_TransitionCLS(*sampled_data)
        return sampled_data

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)


class ReplayWrapper(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    UPDATE_PRIORITIES = 3
    MARK_DEMO_END = 4
    UPDATE_HIDDEN = 5
    EXTRACT = 6
    UPDATE_GOAL_MASKS = 8
    SIZE = 9
    RESET = 10

    def __init__(self, replay_cls, replay_kwargs, _async=True):
        mp.Process.__init__(self)

        self.TransitionCLS = \
            get_transition_class(prioritized=(replay_cls is PrioritizedReplay),
                                 algorithm=replay_kwargs['algorithm'],
                                 recurrent=replay_kwargs['recurrent'])
        self.replay_kwargs = replay_kwargs
        self.replay_cls = replay_cls
        self.cache_len = 2
        if _async:
            self.pipe, self.worker_pipe = mp.Pipe()
            self.start()
        else:
            self.replay = replay_cls(**replay_kwargs)
            self.sample = self.replay.sample
            self.extract = self.replay.extract
            self.feed = self.replay.feed
            self.update_priorities = self.replay.update_priorities
            self.update_hidden = self.replay.update_hidden
            self.mark_demo_end = self.replay.mark_demo_end

    def run(self):
        replay = self.replay_cls(**self.replay_kwargs)

        cache = []

        cache_initialized = False
        cur_cache = 0

        def set_up_cache():
            batch_data = replay.sample()
            if batch_data is None:
                return False
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.cache_len):
                cache.append([x.clone() for x in batch_data])
                for x in cache[i]:
                    x.share_memory_()
            sample(0)
            sample(1)
            return True

        def sample(cur_cache):
            batch_data = replay.sample()
            batch_data = [tensor(x) for x in batch_data]
            for cache_x, x in zip(cache[cur_cache], batch_data):
                cache_x.copy_(x)

        count = 0
        while True:
            op, data = self.worker_pipe.recv()
            count += 1
            if op == self.FEED:
                replay.feed(data)
            elif op == self.SAMPLE:
                if cache_initialized:
                    self.worker_pipe.send([cur_cache, None])
                else:
                    cache_initialized = set_up_cache()
                    self.worker_pipe.send([cur_cache, cache])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache)
            elif op == self.UPDATE_PRIORITIES:
                replay.update_priorities(data)
            elif op == self.UPDATE_HIDDEN:
                replay.update_hidden(data)
            elif op == self.MARK_DEMO_END:
                replay.mark_demo_end()
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            elif op == self.EXTRACT:
                replay.extract()
            elif op == self.UPDATE_GOAL_MASKS:
                replay.update_goal_masks(data)
            elif op == self.SIZE:
                size = replay._size
                self.worker_pipe.send([size])
            elif op == self.RESET:
                replay.reset()
            else:
                raise Exception('Unknown command')

    def mark_demo_end(self):
        self.pipe.send([self.MARK_DEMO_END, None])

    def feed(self, exp):
        self.pipe.send([self.FEED, exp])

    def reset(self):
        self.pipe.send([self.RESET, None])

    def sample(self, min_perc_demos=0.0):
        self.pipe.send([self.SAMPLE, min_perc_demos])
        cache_id, data = self.pipe.recv()
        # When cache is just initialized, set self.cache = (local) cache,
        # for later reference
        if data is not None:
            self.cache = data
        # Once self.cache is set, no longer need to pass data (will be None)
        return self.TransitionCLS(*self.cache[cache_id])

    def extract(self):
        self.pipe.send([self.EXTRACT, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.cache = data
        return self.TransitionCLS(*self.cache[cache_id])

    def update_goal_masks(self, info):
        self.pipe.send([self.UPDATE_GOAL_MASKS, info])

    def update_priorities(self, info):
        self.pipe.send([self.UPDATE_PRIORITIES, info])

    def update_hidden(self, info):
        self.pipe.send([self.UPDATE_HIDDEN, info])

    def size(self):
        self.pipe.send([self.SIZE, None])
        size = self.pipe.recv()
        return size

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()


class ClassifierStorageWrapper(mp.Process):
    """Wrapper for ClassifierStorage, allowing for async sampling for faster
    run-time. Implemented specifically with affordance classifier and
    contrastive learning of representations (for classifier) in mind; not
    a general implementation.
    """
    FEED = 0
    SAMPLE_ACI = 1
    SAMPLE_CLR = 2
    SAMPLE_NEG = 3
    SAMPLE_POS = 4
    SAMPLE_UNIF = 5
    EXIT = 6
    SIZE = 7
    SAMPLE_AC = 8
    RESET = 9

    def __init__(self, storage_cls, storage_kwargs, _async=True, m=1):
        mp.Process.__init__(self)

        if 'keys' in storage_kwargs:
            keys = storage_kwargs['keys']
            self.data_class = namedtuple('Data', list(set(keys)))
        else:
            # Currently, keys cannot be constructed dynamically
            raise NotImplementedError

        self.storage_cls = storage_cls
        self.storage_kwargs = storage_kwargs
        self.m = m
        self.cache_len = 2
        self.batch_size = storage_kwargs['batch_size']
        self.n_classes = storage_kwargs['n_classes']

        # Multiple, so have to initialize before individual elements can be
        # referenced (different from self.cache in other wrapper)
        self.aci_cache = dict()
        self.ac_cache = dict()
        self.pos_cache = dict()
        self.unif_cache = dict()

        if _async:
            self.pipe, self.worker_pipe = mp.Pipe()
            self.start()
        else:
            # self.storages = [storage_cls(**storage_kwargs, msi=i) for i in
                    # range(self.m)]

            # There is currently no non-async support for ClassifierStorage,
            # but it would be trivial to implement if necessary.
            # Could simply add functions that are currently part of
            # ClassifierStorageWrapper to ClassifierStorage directly
            # and then set variables here pointing to them like in
            # ReplayWrapper's __init__
            raise NotImplementedError

    def run(self):
        storages = [self.storage_cls(**self.storage_kwargs, msi=i) for i in
                    range(self.m)]

        ac_cache = [[None for _ in range(self.m)]
                    for _ in range(self.cache_len)]
        aci_cache = defaultdict(list)
        clr_cache = defaultdict(list)
        pos_cache = defaultdict(list)
        unif_cache = defaultdict(list)

        ac_cache_initialized = False
        ac_cache_status = [False for _ in range(self.m)]
        aci_cache_initialized = defaultdict(lambda: False)
        clr_cache_initialized = False
        pos_cache_initialized = defaultdict(lambda: False)
        unif_cache_initialized = defaultdict(lambda: False)

        cur_ac_cache = 0
        cur_aci_cache = defaultdict(lambda: 0)
        cur_clr_cache = 0
        cur_pos_cache = defaultdict(lambda: 0)
        cur_unif_cache = defaultdict(lambda: 0)

        def set_up_ac_cache():
            placeholder = None
            for msi in range(self.m):
                batch_data = storages[msi].sample()
                if batch_data is not None:
                    ac_cache_status[msi] = True
                    batch_data = [tensor(x) for x in batch_data]
                    placeholder = batch_data
                    for i in range(self.cache_len):
                        ac_cache[i][msi] = [x.clone() for x in batch_data]
                        for x in ac_cache[i][msi]:
                            x.share_memory_()

            # If any were initialized, create placeholder data for others
            if placeholder is not None:
                batch_data = placeholder
                for msi in range(self.m):
                    if ac_cache_status[msi] is True:
                        continue
                    for i in range(self.cache_len):
                        ac_cache[i][msi] = [x.clone() for x in batch_data]
                        for x in ac_cache[i][msi]:
                            x.share_memory_()
                for i in range(self.cache_len):
                    sample_ac(i)
                return True
            else:
                return False

        def sample_ac(cur_ac_cache):
            for msi in range(self.m):
                batch_data = storages[msi].sample()
                if batch_data is None:
                    continue
                else:
                    ac_cache_status[msi] = True
                    batch_data = [tensor(x) for x in batch_data]
                    for cache_x, x in zip(ac_cache[cur_ac_cache][msi],
                                          batch_data):
                        cache_x.copy_(x)

        def set_up_aci_cache(msi):
            batch_data = storages[msi].sample()
            if batch_data is None:
                return False
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.cache_len):
                aci_cache[msi].append([x.clone() for x in batch_data])
                for x in aci_cache[msi][i]:
                    x.share_memory_()

            for i in range(self.cache_len):
                sample_aci(msi, i)

            return True

        def sample_aci(msi, cur_aci_cache):
            batch_data = storages[msi].sample()
            if batch_data is None:
                return False
            batch_data = [tensor(x) for x in batch_data]
            for cache_x, x in zip(aci_cache[msi][cur_aci_cache], batch_data):
                cache_x.copy_(x)

        def set_up_clr_cache():
            batch_data = sample_clr(None, setup=True)
            if batch_data is None:
                return False

            for i in range(self.cache_len):
                for j in range(3):
                    clr_cache[j].append([x.clone() for x in batch_data[j]])
                    for x in clr_cache[j][i]:
                        x.share_memory_()

            for i in range(self.cache_len):
                sample_clr(i)

            return True

        def sample_clr(cur_clr_cache, setup=False):
            # Unlike ac (at least, for the moment), clr data must be sampled
            # across storages for all milestones
            anc_data = []
            pos_data = []
            neg_data = []
            # NOTE: Could make this faster by sampling neg. data in other loop,
            # since it is indep. of anc/pos (and we will encounter less fails)
            # but this does not appear to be a bottleneck
            while len(anc_data) < self.batch_size:
                # Sample evenly across non-empty classes for positive example
                # (non-uniformly)
                while True:
                    msi = np.random.randint(self.m)
                    cls = np.random.randint(self.n_classes)
                    size = storages[msi]._size[cls]
                    if size > 0:
                        break
                i = np.random.randint(size)
                anc_datum, pos_datum = storages[msi].get_clr_datum(cls, i)
                if anc_datum is None or pos_datum is None:
                    continue

                # Likewise, for negative example
                while True:
                    msi = np.random.randint(self.m)
                    cls = np.random.randint(self.n_classes)
                    size = storages[msi]._size[cls]
                    if size > 0:
                        break
                i = np.random.randint(size)
                neg_datum = storages[msi].get_datum(cls, i)
                if neg_datum is None:
                    raise IndexError

                anc_data.append(anc_datum)
                pos_data.append(pos_datum)
                neg_data.append(neg_datum)

            anc_data = zip(*anc_data)
            anc_data = list(map(lambda x: np.asarray(x), anc_data))
            anc_data = self.data_class(*anc_data)
            anc_data = [tensor(x) for x in anc_data]

            pos_data = zip(*pos_data)
            pos_data = list(map(lambda x: np.asarray(x), pos_data))
            pos_data = self.data_class(*pos_data)
            pos_data = [tensor(x) for x in pos_data]

            neg_data = zip(*neg_data)
            neg_data = list(map(lambda x: np.asarray(x), neg_data))
            neg_data = self.data_class(*neg_data)
            neg_data = [tensor(x) for x in neg_data]

            batch_data = [anc_data, pos_data, neg_data]

            if setup:
                return batch_data

            for i in range(3):
                for cache_x, x in zip(clr_cache[i][cur_clr_cache],
                                      batch_data[i]):
                    cache_x.copy_(x)

        def set_up_pos_cache(msi, n, parity):
            batch_data = \
                storages[msi].sample_class(1, batch_size=n, parity=parity)
            if batch_data is None:
                return False
            batch_data = [tensor(x) for x in batch_data]
            for i in range(self.cache_len):
                pos_cache[(msi, parity)].append(
                    [x.clone() for x in batch_data])
                for x in pos_cache[(msi, parity)][i]:
                    x.share_memory_()

            for i in range(self.cache_len):
                sample_pos(msi, n, i, parity)

            return True

        def sample_pos(msi, n, cur_pos_cache, parity):
            batch_data = \
                storages[msi].sample_class(1, batch_size=n, parity=parity)
            batch_data = [tensor(x) for x in batch_data]
            for cache_x, x in \
                    zip(pos_cache[(msi, parity)][cur_pos_cache], batch_data):
                cache_x.copy_(x)

        def set_up_unif_cache(n):
            batch_data = sample_unif(n, None, setup=True)
            if batch_data is None:
                return False

            for i in range(self.cache_len):
                unif_cache[n].append([x.clone() for x in batch_data])
                for x in unif_cache[n][i]:
                    x.share_memory_()

            for i in range(self.cache_len):
                sample_unif(n, i)

            return True

        def sample_unif(n, cur_unif_cache, setup=False):
            data = []
            while len(data) < self.batch_size:
                # Sample uniformly across all milestones and classes
                indices = [(msi, cls) for msi in range(self.m)
                           for cls in range(self.n_classes)]
                sizes = [storages[msi]._size[cls] for (msi, cls) in indices]
                sizes = np.array(sizes)
                p = sizes / sizes.sum()  # Turn into uniform distribution
                idx = np.random.choice(len(indices), p=p)
                (msi, cls) = indices[idx]
                i = np.random.randint(sizes[idx])
                datum = storages[msi].get_datum(cls, i)
                if datum is None:
                    continue
                data.append(datum)

            data = zip(*data)
            data = list(map(lambda x: np.asarray(x), data))
            data = self.data_class(*data)
            data = [tensor(x) for x in data]

            batch_data = data

            if setup:
                return batch_data

            for cache_x, x in zip(unif_cache[n][cur_unif_cache], batch_data):
                cache_x.copy_(x)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.FEED:
                (msi, data) = data
                storages[msi].feed(data)
            elif op == self.SAMPLE_AC:
                # In this case, each ms will have a different cache status.
                # However, to properly share memory with other process, we
                # should use first successful sampling of a single milestone as
                # placeholder data for others, and be sure to pass on info
                # on what data is legitimate vs placeholder.
                if ac_cache_initialized:
                    self.worker_pipe.send([cur_ac_cache, None,
                                           ac_cache_status])
                else:
                    ret = set_up_ac_cache()
                    # Cache successfully initialized (enough data)
                    if ret:
                        ac_cache_initialized = ret
                        self.worker_pipe.send([cur_ac_cache, ac_cache,
                                               ac_cache_status])
                    # Cache not initialized (not enough data)
                    else:
                        self.worker_pipe.send([None, None, ac_cache_status])
                # Update cache if already or newly initialized
                if ac_cache_initialized:
                    cur_ac_cache = (cur_ac_cache + 1) % 2
                    sample_ac(cur_ac_cache)
            elif op == self.SAMPLE_ACI:
                msi = data
                # If cache already initialized, send over current data
                if aci_cache_initialized[msi]:
                    self.worker_pipe.send([cur_aci_cache[msi], None])
                else:
                    ret = set_up_aci_cache(msi)
                    # Cache successfully initialized (enough data)
                    if ret:
                        aci_cache_initialized[msi] = ret
                        self.worker_pipe.send([cur_aci_cache[msi],
                                               aci_cache[msi]])
                    # Cache not initialized (not enough data)
                    else:
                        self.worker_pipe.send([None, None])
                # Update cache if already or newly initialized
                if aci_cache_initialized[msi]:
                    cur_aci_cache[msi] = (cur_aci_cache[msi] + 1) % 2
                    sample_aci(msi, cur_aci_cache[msi])
            elif op == self.SAMPLE_CLR:
                if clr_cache_initialized:
                    self.worker_pipe.send([cur_clr_cache, None])
                else:
                    clr_cache_initialized = set_up_clr_cache()
                    self.worker_pipe.send([cur_clr_cache, clr_cache])
                cur_clr_cache = (cur_clr_cache + 1) % 2
                sample_clr(cur_clr_cache)
            elif op == self.SAMPLE_NEG:
                # NOTE: No cache for negative sampling, since it is not
                # yet used online (will very slow if you try to use it)
                (negtype, msi, n) = data
                batch_data = storages[msi].sample_class(0, negtype=negtype,
                                                        batch_size=n)
                if batch_data is None:
                    self.worker_pipe.send([None, None])
                else:
                    batch_data = [tensor(x) for x in batch_data]
                    negs = [x.clone() for x in batch_data]
                    for x in negs:
                        x.share_memory_()
                    self.worker_pipe.send([None, negs])
            elif op == self.SAMPLE_POS:
                (msi, n, parity) = data
                if pos_cache_initialized[(msi, parity)]:
                    self.worker_pipe.send([cur_pos_cache[(msi, parity)], None])
                else:
                    pos_cache_initialized[(msi, parity)] = \
                        set_up_pos_cache(msi, n, parity)
                    self.worker_pipe.send([cur_pos_cache[(msi, parity)],
                                           pos_cache[(msi, parity)]])
                cur_pos_cache[(msi, parity)] = \
                    (cur_pos_cache[(msi, parity)] + 1) % 2
                sample_pos(msi, n, cur_pos_cache[(msi, parity)], parity)
            elif op == self.SAMPLE_UNIF:
                # NOTE: Cache is indexed by 'n', so we can potentially sample
                # different numbers of points without issue
                n = data
                if unif_cache_initialized[n]:
                    self.worker_pipe.send([cur_unif_cache[n], None])
                else:
                    unif_cache_initialized[n] = set_up_unif_cache(n)
                    self.worker_pipe.send([cur_unif_cache[n], unif_cache[n]])
                cur_unif_cache[n] = (cur_unif_cache[n] + 1) % 2
                sample_unif(n, cur_unif_cache[n])
            elif op == self.SIZE:
                size = 0
                for s in storages:
                    size += s._total_size
                self.worker_pipe.send([size])
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            elif op == self.RESET:
                for i in storages:
                    i.reset()
            else:
                raise Exception('Unknown command')

    def feed(self, exp):
        self.pipe.send([self.FEED, exp])

    def reset(self):
        self.pipe.send([self.RESET, None])

    def sample_ac(self):
        """Samples even split of affordance classifier data for all msis
        If there is no positive (or negative) data yet, will return None, and
        delay initialization of the cache (for given index).
        """
        self.pipe.send([self.SAMPLE_AC, None])
        cache_id, data, status = self.pipe.recv()
        if data is not None:
            self.ac_cache = data
        # If cache is initialized (enough data)
        if cache_id is not None:
            datas = []
            for msi in range(len(status)):
                if self.ac_cache[cache_id][msi] is not None:
                    datas\
                        .append(self.data_class(*self.ac_cache[cache_id][msi]))
                else:
                    datas.append(None)
            return datas, status
        # If cache is uninitialized (not enough data)
        else:
            return None, status

    def sample_aci(self, msi):
        """Samples even split of affordance classifier data for a given
        milestone indexed by `msi`.
        If there is no positive (or negative) data yet, will return None, and
        delay initialization of the cache.
        """
        self.pipe.send([self.SAMPLE_ACI, msi])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.aci_cache[msi] = data
        # If cache is initialized (enough data)
        if cache_id is not None:
            return self.data_class(*self.aci_cache[msi][cache_id])
        # If cache is uninitialized (not enough data)
        else:
            return None

    def sample_clr(self):
        """Samples data for contrastive (triplet) loss.
        Assumes that there is enough data to do such sampling.
        """
        self.pipe.send([self.SAMPLE_CLR, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.clr_cache = data

        anc_data = self.data_class(*self.clr_cache[0][cache_id])
        pos_data = self.data_class(*self.clr_cache[1][cache_id])
        neg_data = self.data_class(*self.clr_cache[2][cache_id])
        return (anc_data, pos_data, neg_data)

    def sample_neg(self, negtype, msi, n):
        self.pipe.send([self.SAMPLE_NEG, (negtype, msi, n)])
        _, data = self.pipe.recv()
        if data is None:
            return None
        return self.data_class(*data)

    def sample_pos(self, msi, n, parity):
        self.pipe.send([self.SAMPLE_POS, (msi, n, parity)])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.pos_cache[(msi, parity)] = data
        return self.data_class(*self.pos_cache[(msi, parity)][cache_id])

    def sample_unif(self, n):
        """Samples uniformly across milestones and classes
        """
        self.pipe.send([self.SAMPLE_UNIF, None])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.unif_cache[n] = data

        return self.data_class(*self.unif_cache[n][cache_id])

    def size(self):
        self.pipe.send([self.SIZE, None])
        size = self.pipe.recv()
        return size

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()
