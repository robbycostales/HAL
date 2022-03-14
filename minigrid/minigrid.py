import math
import cv2
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import spaces
import time
import os
from gym.utils import seeding

from .rendering import point_in_rect, fill_coords, point_in_circle, \
                       point_in_triangle, rotate_fn, \
                       highlight_img, downsample
from .world_objs import *


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, env_no, width, height):

        assert width >= 3
        assert height >= 3

        self.env_no = env_no
        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        if self.width > i >= 0 and self.height > j >= 0:
            return self.grid[j * self.width + i]
        else:
            raise IndexError

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(self.env_no))

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(self.env_no))

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.env_no, self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(self.env_no, width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(self.env_no)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
        fully_obs=False,
        env_no=ENV_TO_IDX['craft']
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        # img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3),
        #                dtype=np.uint8)
        # Change blank squares to be white instead of black

        fill = 255
        img = np.full(shape=(tile_size * subdivs, tile_size * subdivs, 3),
                      fill_value=fill,
                      dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            if fully_obs:
                # Render as block in fully observable setting
                fill_coords(img, point_in_rect(0, 1, 0, 1), (255, 0, 0))
            else:
                # Render agent as triangle in partially observable setting
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5,
                                   theta=0.5*math.pi*agent_dir)
                fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None,
        fully_obs=False
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height),
                                      dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.ones(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)  # returns object at cell location

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    fully_obs=fully_obs,
                    env_no=self.env_no
                )
                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def int_encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        The "int" encoding refers to the fact that this representation uses
            integers to represent objects, rather than separate layers or
            RBG values
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j] = OBJECT_TO_IDX[self.env_no]['empty']
                    else:
                        array[i, j] = OBJECT_TO_IDX[self.env_no][v.type]

        return array

    def sparse_encode(self, vis_mask=None, agent_pos=None, fully_obs=False):
        """
        Produce a compact numpy encoding of the grid
        The "sparse" encoding refers to the fact that this representation uses
            separate layers to represent objects (one-hot-style), rather than
            integers or RBG values
        """
        en = self.env_no
        N = NO_OBJECTS[en]

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        # Craft is the only environment that does not need attributes
        if en == ENV_TO_IDX['craft']:
            array = np.zeros((N, self.width, self.height),
                             dtype='uint8')
            attributes = False
        else:
            array = np.zeros((N + len(STATE_TO_IDX), self.width, self.height),
                             dtype='uint8')
            attributes = True

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    # Place normal objects
                    v = self.get(i, j)
                    if v is None:
                        # Empty space
                        array[OBJECT_TO_IDX[en]['empty'], i, j] = 1
                    else:
                        # Not empty object
                        array[OBJECT_TO_IDX[en][v.type], i, j] = 1
                        if attributes:
                            # Place attributes
                            state = v.encode()[-1]
                            array[state + N, i, j] = 1

        # Place agent (handled differently for full and partial obs)
        if fully_obs and agent_pos is not None:
            array[OBJECT_TO_IDX[self.env_no]['agent'], agent_pos[0],
                  agent_pos[1]] = 1
        else:
            # window size
            ws = int(np.sum(vis_mask)**(1/2))
            array[OBJECT_TO_IDX[self.env_no]['agent'], ws//2, ws//2] = 1

        return array

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX[self.env_no]['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array, env_no):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = Grid(env_no, width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state, env_no)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX[env_no]['unseen'])

        return grid, vis_mask

    @staticmethod  # MOD
    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                if j > 0:
                    mask[i+1, j-1] = True
                    mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j] = True
                if j > 0:
                    mask[i-1, j-1] = True
                    mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6

    # Enumeration of possible actions for Minecraft
    class CraftEnvActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        up = 2
        down = 3

        # Mine
        mine = 4

        # Craft
        craft_wood = 5
        craft_stick = 6
        craft_pickaxe = 7
        #  craft_wood_pickaxe = 7
        #  craft_stone_pickaxe = 8
        #  craft_iron_pickaxe = 9
        craft_iron = 8
        craft_crafting_bench = 9
        craft_furnace = 10

    # Enumeration of possible actions for Treasure Navigation
    class TreasureEnvActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        up = 2
        down = 3

        # Grab objects, etc.
        interact = 4

    class TreasureEnvActionsWithDrop(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        up = 2
        down = 3

        # Grab/use held object
        interact = 4
        # Drop currently held object
        drop = 5

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=True,
        seed=1337,
        agent_view_size=7,
        final_item=None,
        rewards="sparse",
        fully_obs=False,
        env_no=ENV_TO_IDX['general'],
        gridrl_params=None,
    ):
        # For CraftEnv (rewarded, episode ends with collection of final item -
        # during pre-training, we start with the most difficult task to prevent
        # the episodes from ending pre-maturely)
        self.task_final_item = final_item
        if IDX_TO_ENV[env_no] == 'craft':
            self.final_item = 'diamond'
        elif IDX_TO_ENV[env_no] == 'treasure':
            self.final_item = 'treasure'
        elif IDX_TO_ENV[env_no] == 'general':
            self.final_item = final_item
        # Removes orientation (default facing is up) and need for mining action
        self.fully_obs = fully_obs
        # Indicates reward structure to be used ("sparse" or "dense")
        self.rewards = rewards
        self.env_no = env_no
        # Stochasticity
        self.start_key = None

        # Extract parameter dictionary items
        self.gridrl_params = gridrl_params
        if gridrl_params is not None:
            # Default milestone set in use
            self.dms = gridrl_params['dms']
            self.print_affordances = gridrl_params['print_affordances']
            self.task_agnostic_steps = gridrl_params['task_agnostic_steps']
            self.agent_view_size = gridrl_params['agent_view_size']
            agent_view_size = self.agent_view_size
            self.penalty_scaling = gridrl_params['penalty_scaling']
            self.single_obj_inventory = gridrl_params['single_obj_inventory']

            self.stoch_continuous = gridrl_params['stoch_continuous']
            self.stoch_discrete = gridrl_params['stoch_discrete']
            self.stoch_value = gridrl_params['stoch_value']
            self.verbose = gridrl_params['env_verbose']

            if gridrl_params['dense_rewards']:
                self.rewards = 'dense'
            self.evaluate = gridrl_params['evaluate']
        else:
            print("gridrl parameters required")
            raise NotImplementedError

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            height = grid_size
            width = grid_size

        # Action space different for Minecraft vs others
        if final_item is None:
            # Original action space
            self.actions = MiniGridEnv.Actions
        else:
            if IDX_TO_ENV[self.env_no] == "craft":
                # Minecraft action space
                self.actions = MiniGridEnv.CraftEnvActions
            elif IDX_TO_ENV[self.env_no] == "treasure":
                # Treasure action space
                self.actions = MiniGridEnv.TreasureEnvActions
            elif IDX_TO_ENV[self.env_no] == "general":
                self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        if not self.fully_obs:
            assert agent_view_size % 2 == 1
            assert agent_view_size >= 3
            self.agent_view_size = agent_view_size
        else:
            self.agent_view_size = 3  # Purely for rendering purpose

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        if self.fully_obs:
            self.image_observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(NO_OBJECTS[self.env_no], height, width),
                dtype='uint8'
            )
        else:
            self.image_observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(NO_OBJECTS[self.env_no], self.agent_view_size,
                       self.agent_view_size),
                dtype='uint8'
            )
        self.inventory_size = len(ITEM_TO_IDX[self.env_no])
        self.inventory = [0 for _ in range(self.inventory_size)]
        self.inventory_observation_space = \
            spaces.Box(low=0, high=np.infty,
                       shape=(self.inventory_size,), dtype='uint8')
        self.observation_space = spaces.Dict({
            'img': self.image_observation_space,
            'vec': self.inventory_observation_space,
        })

        if IDX_TO_ENV[self.env_no] == "craft":
            self.pickaxe_health = {'wood_pickaxe': 0, 'stone_pickaxe': 0,
                                   'iron_pickaxe': 0}

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Keep track of total steps independently
        self.total_steps = 0

        # Normalize rewards
        r_sum = 0
        for k, v in ITEM_TO_REWARD[env_no].items():
            if k in MILESTONE_TO_IDX[env_no][self.dms]:
                r_sum += v
        for k, _ in ITEM_TO_REWARD[env_no].items():
            ITEM_TO_REWARD[env_no][k] = ITEM_TO_REWARD[env_no][k] / r_sum

        # Continuous stochasticity ratios
        self.stoch_continuous_dist = []
        for k, v in ITEM_TO_STOCH[env_no].items():
            self.stoch_continuous_dist.append(v)
        self.stoch_continuous_dist = np.array(self.stoch_continuous_dist)
        self.stoch_continuous_dist /= sum(self.stoch_continuous_dist)

        # Initialize the state
        self.reset()


    def reset(self):
        if not self.fully_obs:
            # Current position and direction of the agent
            self.agent_pos = None
            self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()

        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None

        if not self.fully_obs:
            # Will be set by _gen_grid
            assert self.agent_dir is not None
        else:
            # If fully observable, make default direction "up" (mainly placing)
            self.agent_dir = 3  # up

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Reset inventory
        self.inventory = [0 for _ in range(self.inventory_size)]

        # Reset list of collected items (for dense reward)
        self.collected_items = set()

        # Step count since episode start
        self.step_count = 0

        # Initialize bfs cache
        self.bfs_time = 0
        self.bfs_cache = dict()

        # Return first observation
        obs = self.gen_obs()

        # Coverage statistics
        self.ms_counts = dict.fromkeys(MILESTONE_TO_IDX[self.env_no][self.dms].keys(), 0)

        # Last actions for rendering purposes
        self.last_action = ""

        # Name of last milestone collected (if any) for rendering purposes
        self.last_milestone = ""

        # Time elapsed since last milestone, for rendering
        self.time_since_last_milestone = 0

        # Victim of edge stochasticity at each timestep
        self.removed_item = None

        # Time elapsed since last removal of item, for rendering
        self.time_since_last_removal = 0
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the
        environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall':             'W',
            'floor':            'F',
            'door':             'D',
            'key':              'K',
            'ball':             'A',
            'box':              'B',
            'goal':             'G',
            'lava':             'V',
            'dirt':             'R',
            'tree':             'T',
            'stone':            'S',
            'coal':             'C',
            'iron_ore':         'I',
            'diamond':          'O',
            'crafting_bench':   'E',
            'furnace':          'F',
            'red_door':         '?',
            'yellow_door':      '?',
            'purple_door':      '?',
            'green_door':       '?',
            'red_key':          '?',
            'yellow_key':       '?',
            'purple_key':       '?',
            'green_weight':     '?',
            'green_button':     '?',
            'blue_key':         '?',
            'blue_chest':       '?'
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^',
            4: 'o',  # if no orientation from self.fully_obs
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c is None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _item_reward(self, item):
        """Compute reward for collecting a given item
        """
        if item not in MILESTONE_TO_IDX[self.env_no][self.dms]:
            return 0.

        if self.rewards == "dense":
            if item in ITEM_TO_REWARD[self.env_no] and \
                    item not in self.collected_items:
                reward = ITEM_TO_REWARD[self.env_no][item]
                self.collected_items.add(item)
                return reward
            else:
                return 0.
        elif self.rewards == "sparse" and item == self.final_item:
            r = self._reward()
            return r
        else:
            return 0.

    def _reward(self):
        """Compute the reward to be given upon success
        """
        return 1.0

    def _rand_int(self, low, high):
        """Generate random integer in [low,high[
        """
        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """Generate random float in [low,high[
        """
        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self, obj, top=None, size=None, reject_fn=None,
                  max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        if obj is not None:
            obj.init_pos = (i, j)
            obj.cur_pos = (i, j)

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    @property
    def back_pos(self):
        """
        Get the position of the cell that is right behind the agent
        """

        return self.agent_pos - self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts_old(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set

        This version places the agent at the edge of the viewing window
        """

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - self.agent_view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - self.agent_view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - self.agent_view_size + 1
            topY = self.agent_pos[1] - self.agent_view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - self.agent_view_size // 2
            topY = self.agent_pos[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set

        This version places the agent in the center of the viewing window
        """
        topX = self.agent_pos[0] - self.agent_view_size // 2
        topY = self.agent_pos[1] - self.agent_view_size // 2
        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and
        returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size \
                or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        Check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs['image'], self.env_no)
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def facing(self, item_name):
        """
        Check if agent is "facing" an item.
        If fully observable, "facing" means "adjacent to"
        item_name is str type
        """
        if self.fully_obs:
            for vec in DIR_TO_VEC:
                pos = self.agent_pos + vec
                cell = self.grid.get(*pos)
                if cell is not None and cell.type == item_name:
                    return True
        else:
            if self.fwd_cell is not None and self.fwd_cell.type == item_name:
                return True
        return False

    def near(self, item_name, r=2):
        """
        Check if agent is "near" an item, given a radius `r`
        """
        pos = self.agent_pos
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                p = (pos[0]+i, pos[1]+j)
                if 0 <= p[0] < self.height and 0 <= p[1] < self.width:
                    cell = self.grid.get(p[0], p[1])
                    if cell is not None and cell.type == item_name:
                        return True
        return False

    def set_positional_variables(self):
        """
        Set helpful variables describing area around agent, used to determine
        valid actions
        """
        # Get the position in front of the agent
        self.fwd_pos = self.front_pos
        self.bwd_pos = self.back_pos
        # Get the contents of the cell in front of the agent
        self.fwd_cell = self.grid.get(*self.fwd_pos)
        self.bwd_cell = self.grid.get(*self.bwd_pos)

    def move(self, direction):
        """
        Action: Move in direction
        """
        # Use direction to set new "orientation" for agent
        self.agent_dir = DIR_TO_INT[direction]
        # Reset the positional variables
        self.set_positional_variables()
        # Since agent is oriented correctly, and positional variables are set,
        # we can simply move the agent forward now
        result = self.move_forward()
        return result

    def rotate_left(self):
        """
        Action: Rotate agent to the left
        """
        self.agent_dir = (self.agent_dir - 1) % 4
        return None

    def rotate_right(self):
        """
        Action: Rotate agent to the right
        """
        self.agent_dir = (self.agent_dir + 1) % 4
        return None

    def move_forward(self):
        """
        Action: Move agent forward
        If self.fully_obs, agent will attempt mine item first, then move to
        occupy that tile
        """
        if self.fwd_cell is None or self.fwd_cell.can_overlap():
            # Agent moves forward if empty ahead, or can overlap
            self.agent_pos = self.fwd_pos
            if self.fwd_cell is not None and self.fwd_cell.type == 'goal':
                # Agent rea goal tile; episode ends with reward
                return 'goal'
            if self.fwd_cell is not None and self.fwd_cell.type == 'lava':
                # Agent reaches lava; episode ends with no reward
                return 'lava'
            result = None
        elif self.fwd_cell is not None and self.fully_obs:
            # Agent tries to mine item ahead of it, then move forward
            result = self.mine()
            # Refresh positional variables (tile ahead might change to None)
            self.set_positional_variables()
            # Try to move forward (will succeed if object was mined)
            if self.fwd_cell is None:
                self.agent_pos = self.fwd_pos
        else:
            result = None
        return result

    def move_backward(self):
        """
        Action: Move agent backward
        """
        if self.bwd_cell is None or self.bwd_cell.can_overlap():
            self.agent_pos = self.bwd_pos
            return None
        if self.bwd_cell is not None and self.bwd_cell.type == 'goal':
            return 'goal'
        if self.bwd_cell is not None and self.bwd_cell.type == 'lava':
            return 'lava'
        return None

    def interact(self):
        """
        Actions: Interact with object (all-encompassing action for Treasure)
        """
        en = self.env_no
        obj = self.fwd_cell
        if obj is None:
            return
        item = obj.type
        # Key: add to inventory and remove from grid
        if 'key' in item or item == 'green_weight':
            prev_carrying = self.carrying
            self.inventory[ITEM_TO_IDX[en][item]] += 1
            self.carrying = obj
            self.carrying.cur_pos = np.array([-1, -1])
            if self.single_obj_inventory:
                self.grid.set(*self.fwd_pos, None)
                if prev_carrying is not None:
                    self.grid.set(*prev_carrying.init_pos, prev_carrying)
                    prev_carrying.cur_pos = prev_carrying.init_pos
                    self.inventory[ITEM_TO_IDX[en][prev_carrying.type]] -= 1
            else:
                self.grid.set(*self.fwd_pos, None)
            return item
        # Door: unlock and open
        if 'door' in item:
            for c in ['red', 'blue', 'yellow']:
                # If it's one of these colors, and we have the key, open
                if c in item and \
                        self.inventory[ITEM_TO_IDX[en][c + '_key']] >= 1:
                    change = obj.toggle(self, self.fwd_pos, force_open=True)
                    # remove key from inventory
                    self.inventory[ITEM_TO_IDX[en][c + '_key']] -= 1
                    self.carrying = None
                    if change:
                        return item
        if 'button' in item:
            # NOTE: currently only green button, but we might change this
            for c in ['green']:
                if c in item and \
                        self.inventory[ITEM_TO_IDX[en][c + '_weight']] >= 1:
                    # Press button and remove weight from inventory
                    obj.toggle(self, self.fwd_pos)
                    self.inventory[ITEM_TO_IDX[en][c + '_weight']] -= 1
                    self.carrying = None
                    # Counter to indicate that the button is pressed
                    self.inventory[ITEM_TO_IDX[en]['hurt_back']] += 1
                    # Open corresponding door (we assume it exists)
                    door = self.grid.get(*self.get_object_pos(c + '_door'))
                    change = door.toggle(self, self.fwd_pos, force_open=True)
                    if change:
                        return c + '_door'
        if 'chest' in item:
            # NOTE: currently only have purple chest
            for c in ['purple']:
                if c in item and \
                        self.inventory[ITEM_TO_IDX[en][c + '_key']] >= 1:
                    # Currently, we don't actually "open" the chest,
                    # the episode only ends
                    # NOTE: purple chest designated as "treasure" chest
                    if c == "purple":
                        self.inventory[ITEM_TO_IDX[en]['treasure']] += 1
                    else:
                        NotImplementedError
                    return 'treasure'
        return None

    def pickup(self):
        """
        Action: Pickup item in front of agent
        """
        if self.fwd_cell and self.fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = self.fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(*self.fwd_pos, None)
        return None

    def drop(self):
        """
        Action: Drop item carrying to position in front of agent
        """
        if not self.fwd_cell and self.carrying:
            self.grid.set(*self.fwd_pos, self.carrying)
            self.carrying.cur_pos = self.fwd_pos
            self.carrying = None
        return None

    def toggle(self):
        """
        Action: Toggle item in front of agent
        """
        if self.fwd_cell:
            self.fwd_cell.toggle(self, self.fwd_pos)
        return None

    def mine(self):
        """
        Action: Mine block ahead of agent
        """
        if self.fwd_cell is None:
            return
        # Items that need at minimum a certain pickaxe
        needs_min_wp = ['stone']  # wood
        needs_min_sp = ['iron_ore', 'coal', 'stone']  # stone
        needs_min_ip = ['diamond']  # iron
        # Item to be mined
        item = self.fwd_cell.type
        if item == 'tree':  # in world=tree; in hand=log
            item = 'log'
        # Whether or not we have collected item
        collected = False
        # Try mining by hand first
        if item in ['dirt', 'log']:  # DEBUG: Do not allow crafting_bench
            collected = True
        elif item in needs_min_wp \
                and self.inventory[ITEM_TO_IDX[self.env_no]['wood_pickaxe']] \
                >= 1:
            collected = True
        elif item in needs_min_wp + needs_min_sp \
                and self.inventory[ITEM_TO_IDX[self.env_no]['stone_pickaxe']] \
                >= 1:
            collected = True
        elif item in needs_min_wp + needs_min_sp + needs_min_ip \
                and self.inventory[ITEM_TO_IDX[self.env_no]['iron_pickaxe']] \
                >= 1:
            collected = True
        # If item was mined...
        if collected:
            # Collect item and remove from grid
            self.inventory[ITEM_TO_IDX[self.env_no][item]] += 1
            self.grid.set(*self.fwd_pos, None)
            # Have we achieved the goal of the environment?
            return item
        return None

    def craft_pickaxe(self):
        """
        Action: Craft the strongest pickaxe possible with current inventory
        """
        en = self.env_no
        # How much health to add for each pickaxe
        health_added = {'wood': 5, 'stone': 10, 'iron': 15}
        # Craft if agent has right ingredients and is near crafting_bench
        for material in ['iron', 'stone', 'wood']:
            if self.near('crafting_bench') and \
                    self.inventory[ITEM_TO_IDX[en]['stick']] >= 2 and \
                    self.inventory[ITEM_TO_IDX[en][material]] >= 3:
                self.inventory[ITEM_TO_IDX[en][material + '_pickaxe']] += 1
                self.inventory[ITEM_TO_IDX[en]['stick']] -= 2
                self.inventory[ITEM_TO_IDX[en][material]] -= 3
                self.pickaxe_health[material + '_pickaxe'] += \
                    health_added[material]
                return material + '_pickaxe'
        return None

    def craft_pick(self, material):
        """
        Action: Craft a pickaxe out of specified material
        ('pick' means we can pick the material, unlike default method)
        """
        en = self.env_no
        # How much health to add for each pickaxe
        health_added = {'wood': 5, 'stone': 10, 'iron': 15}
        # Craft if agent has right ingredients and is near crafting_bench
        if self.near('crafting_bench') and \
                self.inventory[ITEM_TO_IDX[en]['stick']] >= 2 and \
                self.inventory[ITEM_TO_IDX[en][material]] >= 3:
            self.inventory[ITEM_TO_IDX[en][material + '_pickaxe']] += 1
            self.inventory[ITEM_TO_IDX[en]['stick']] -= 2
            self.inventory[ITEM_TO_IDX[en][material]] -= 3
            self.pickaxe_health[material + '_pickaxe'] += \
                health_added[material]
            return material + '_pickaxe'
        return None

    def craft_wood(self):
        """
        Action: Craft woods
        """
        en = self.env_no
        if self.inventory[ITEM_TO_IDX[en]['log']] >= 1:
            self.inventory[ITEM_TO_IDX[en]['wood']] += 4
            self.inventory[ITEM_TO_IDX[en]['log']] -= 1
            return "wood"
        return None

    def craft_stick(self):
        """
        Action: Craft sticks
        """
        en = self.env_no
        if self.inventory[ITEM_TO_IDX[en]['wood']] >= 2:
            self.inventory[ITEM_TO_IDX[en]['stick']] += 4
            self.inventory[ITEM_TO_IDX[en]['wood']] -= 2
            return "stick"
        return None

    def craft_crafting_bench(self):
        """
        Action: Craft and place crafting bench
        """
        en = self.env_no
        if self.fwd_cell is None and \
                self.inventory[ITEM_TO_IDX[en]['wood']] >= 4:
            self.inventory[ITEM_TO_IDX[en]['wood']] -= 4
            self.grid.set(*self.fwd_pos, CraftingBench(en))
            return 'crafting_bench'
        return None

    def craft_furnace(self):
        """
        Action: Craft and place furnace
        """
        en = self.env_no
        if self.fwd_cell is None and \
                self.inventory[ITEM_TO_IDX[en]['stone']] >= 4:
            self.inventory[ITEM_TO_IDX[en]['stone']] -= 4
            self.grid.set(*self.fwd_pos, Furnace(en))
            return 'furnace'
        return None

    def craft_iron(self):
        """Action: Craft iron

        Returns:
            result obtained
        """
        en = self.env_no
        if self.near('furnace') and \
                self.inventory[ITEM_TO_IDX[en]['coal']] >= 1 and \
                self.inventory[ITEM_TO_IDX[en]['iron_ore']] >= 1:
            self.inventory[ITEM_TO_IDX[en]['iron']] += 1
            self.inventory[ITEM_TO_IDX[en]['coal']] -= 1
            self.inventory[ITEM_TO_IDX[en]['iron_ore']] -= 1
            return "iron"
        return None

    def get_object(self, item_name):
        """
        Return object if it exists on grid
        """
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == item_name:
                    return cell
        return None

    def get_object_pos(self, item_name):
        """
        Get position of item (returns None if doesn't exist)
        """
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == item_name:
                    return (i, j)
        return None

    def bfs(self, pos, item, n=1, ghost_set={}):
        """
        Search for `item` from starting `pos`, treating anything in
        `ghost_set` as empty space that can be traveled through.
        Return True if found, False if not.
        """
        st = time.time()
        if (item, n) in self.bfs_cache:
            et = time.time()
            self.bfs_time += (et-st)
            return self.bfs_cache[(item, n)]
        count = 0
        fifo = [pos]
        visited = set()
        while len(fifo) > 0:
            # Pop off queue
            node = fifo.pop(0)
            # Check if item is the one(s) we're searching for
            cell = self.grid.get(node[0], node[1])
            if cell is not None and cell.type == item:
                count += 1
            if count >= n:
                et = time.time()
                self.bfs_time += (et-st)
                self.bfs_cache[(item, n)] = 1
                return True
            # Add to visited set
            visited.add(node)
            # Only add neighbors if current node is empty space that can
            # be traveled to/through
            if cell is None or cell.type in ghost_set or \
                    ('door' in cell.type and cell.is_open):
                # Find and add neighbors to queue
                for diff in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new = (node[0] + diff[0], node[1] + diff[1])
                    if 0 <= new[0] < self.height and 0 <= new[1] < self.width \
                            and new not in visited:
                        fifo.append(new)
        et = time.time()
        self.bfs_time += (et-st)
        self.bfs_cache[(item, n)] = 0
        return False

    def object_exists(self, item_name, to_reach=False, n=1, full_search=False):
        """
        Check if object (item_name) exists on grid, and if applicable,
        (if to_reach is True), check if empty square next to object (a crude
        heuristic for whether or not object is reachable, which works under
        our current world generationi procedure)
        Args:
            item_name - (str) name of object
            to_reach - (boolean) can be reached by agent
            n - (int) minimum number of objects that must exist
            full_search - (boolean) search using BFS to see if agent can reach
                without collecting a milestone along the way
        """
        en = self.env_no
        dms = self.dms
        # Create ghost set
        ms_set = set(MILESTONE_TO_IDX[en][dms].keys())
        os = set(OBJECT_TO_IDX[en].keys())
        if 'tree' in os:
            os.remove('tree')
            os.add('log')
        gs = {'dirt'}.union(os).difference(ms_set)\
            .difference({'wall', 'crafting_bench', 'furnace'})

        # To check if object exists and can reach through thorough BFS
        if to_reach and full_search:
            return self.bfs(tuple(self.agent_pos),
                            item_name, n=n, ghost_set=gs)

        # Otherwise, do exhaustive search and/or lazy reach check
        count = 0
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == item_name:
                    if to_reach:
                        # Check if any surrounding square is empty
                        for d in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            c = self.grid.get(i+d[0], j+d[1])
                            # If place next to object is empty or occupied
                            # by agent
                            if c is None or (i+d[0], j+d[1]) == tuple(self.agent_pos):
                                count += 1
                    else:
                        count += 1
                    if count >= n:
                        return True
        return False

    def how_much_input(self, n, inp, outp):
        """
        For craft environment.
        Returns how much of input element is needed to create `n` of ouput
        element, when the crafting rule is `inp` inputs makes `outp` outputs.
        """
        return math.ceil(n / outp) * inp

    def get_affordance(self, obj, n=1, simple=False, enc=set()):
        """
        Args:
        Get affordance value for a particular object
            obj - (str) item name
            n - (int) how many of obj are required
            simple - (boolean) do not recursively call other affordance checks
                (to fix recurrence issue with pickaxese)
            enc: (set) encountered elements in recursion history; if
                current object in the set, we return False (fixes cycles)
        """
        en = self.env_no
        dms = self.dms
        MSI = MILESTONE_TO_IDX[en][dms]
        hmi = self.how_much_input
        # Return False if cycle in recursion
        if obj in enc:
            return False
        # Add current object to enc for future calls
        enc = enc.union({obj})

        if obj == 'log' and self.object_exists('tree', to_reach=True, n=n):
            return True
        if obj == 'wood' and \
                (self.inventory[ITEM_TO_IDX[en]['log']] >= hmi(n, 1, 4) or
                 'log' not in MSI
                 and self.get_affordance('log', n=hmi(n, 1, 4), enc=enc)):
            return True
        if obj == 'stick' and \
                (self.inventory[ITEM_TO_IDX[en]['wood']] >= hmi(n, 2, 4) or
                    'wood' not in MSI
                    and self.get_affordance('wood', n=hmi(n, 2, 4), enc=enc)):
            return True
        if obj == 'crafting_bench' and \
                (self.inventory[ITEM_TO_IDX[en]['wood']] >= hmi(n, 4, 1) or
                    'wood' not in MSI
                    and self.get_affordance('wood', n=hmi(n, 4, 1), enc=enc)):
            return True
        if obj == 'stone' and \
                ((self.inventory[ITEM_TO_IDX[en]['iron_pickaxe']] >= 1 or
                    'iron_pickaxe' not in MSI
                    and self.get_affordance('iron_pickaxe', enc=enc))
                 or (self.inventory[ITEM_TO_IDX[en]['stone_pickaxe']] >= 1 or
                     'stone_pickaxe' not in MSI
                     and self.get_affordance('stone_pickaxe', enc=enc))
                 or (self.inventory[ITEM_TO_IDX[en]['wood_pickaxe']] >= 1 or
                     'wood_pickaxe' not in MSI
                     and self.get_affordance('wood_pickaxe', enc=enc))
                 ) and self.object_exists('stone', to_reach=True,
                                          n=n, full_search=True):
            return True
        if obj == 'furnace' and \
                (self.inventory[ITEM_TO_IDX[en]['stone']] >= hmi(n, 4, 1) or
                    'stone' not in MSI
                    and self.get_affordance('stone', n=hmi(n, 4, 1), enc=enc)):
            return True
        if obj == 'iron_ore' and \
                ((self.inventory[ITEM_TO_IDX[en]['stone_pickaxe']] >= 1 or
                    'stone_pickaxe' not in MSI
                    and self.get_affordance('stone_pickaxe', enc=enc))
                 or (self.inventory[ITEM_TO_IDX[en]['iron_pickaxe']] >= 1 or
                     'iron_pickaxe' not in MSI
                     and self.get_affordance('iron_pickaxe', enc=enc))
                 ) and self.object_exists('iron_ore', to_reach=True,
                                          n=n, full_search=True):
            return True
        if obj == 'coal' and \
                ((self.inventory[ITEM_TO_IDX[en]['stone_pickaxe']] >= 1 or
                    'stone_pickaxe' not in MSI
                    and self.get_affordance('stone_pickaxe', enc=enc))
                 or (self.inventory[ITEM_TO_IDX[en]['iron_pickaxe']] >= 1 or
                     'iron_pickaxe' not in MSI
                     and self.get_affordance('iron_pickaxe', enc=enc))
                 ) and self.object_exists('coal', to_reach=True,
                                          n=n, full_search=True):
            return True
        if obj == 'iron' and \
                (self.inventory[ITEM_TO_IDX[en]['iron_ore']] >= hmi(n, 1, 1) or
                    'iron_ore' not in MSI
                    and self.get_affordance('iron_ore', n=hmi(n, 1, 1), enc=enc)) \
                and (self.inventory[ITEM_TO_IDX[en]['coal']] >= hmi(n, 1, 1) or
                     'coal' not in MSI
                     and self.get_affordance('coal', n=hmi(n, 1, 1), enc=enc)) \
                and (self.object_exists('furnace', to_reach=True) or
                     'furnace' not in MSI
                     and self.get_affordance('furnace', enc=enc)):
            return True
        if obj == 'diamond' and \
                (self.inventory[ITEM_TO_IDX[en]['iron_pickaxe']] >= 1 or
                    'iron_pickaxe' not in MSI
                    and self.get_affordance('iron_pickaxe', enc=enc)) \
                and self.object_exists('diamond', to_reach=True,
                                       n=n, full_search=True):
            return True
        # Pickaxes are dependent on one another, since
        # pickaxe is a single action, and higher level pickaxes will be crafted
        # automatically over lower level ones
        if obj == 'iron_pickaxe' and \
                (self.inventory[ITEM_TO_IDX[en]['stick']] >= hmi(n, 2, 1) or
                    'stick' not in MSI and not simple
                    and self.get_affordance('stick', n=hmi(n, 2, 1), enc=enc)) \
                and (self.inventory[ITEM_TO_IDX[en]['iron']] >= hmi(n, 3, 1) or
                     'iron' not in MSI and not simple
                     and self.get_affordance('iron', n=hmi(n, 3, 1), enc=enc)) \
                and (self.object_exists('crafting_bench', to_reach=True) or
                     'crafting_bench' not in MSI
                     and self.get_affordance('crafting_bench', enc=enc)):
            return True
        if obj == 'stone_pickaxe' \
                and (simple or
                     not self.get_affordance('iron_pickaxe', simple=True, enc=enc)) \
                and (self.inventory[ITEM_TO_IDX[en]['stick']] >= hmi(n, 2, 1)
                     or 'stick' not in MSI and not simple
                     and self.get_affordance('stick', n=hmi(n, 2, 1), enc=enc)) \
                and (self.inventory[ITEM_TO_IDX[en]['stone']] >= hmi(n, 3, 1)
                     or 'stone' not in MSI and not simple
                     and self.get_affordance('stone', n=hmi(n, 3, 1), enc=enc)) \
                and (self.object_exists('crafting_bench', to_reach=True) or
                     'crafting_bench' not in MSI
                     and self.get_affordance('crafting_bench', enc=enc)):
            return True
        if obj == 'wood_pickaxe' \
                and (simple or
                     not self.get_affordance('iron_pickaxe', simple=True, enc=enc)) \
                and (simple or
                     not self.get_affordance('stone_pickaxe', simple=True, enc=enc)) \
                and (self.inventory[ITEM_TO_IDX[en]['stick']] >= hmi(n, 2, 1)
                     or 'stick' not in MSI and not simple
                     and self.get_affordance('stick', n=hmi(n, 2, 1), enc=enc)) \
                and (self.inventory[ITEM_TO_IDX[en]['wood']] >= hmi(n, 3, 1) or
                     'wood' not in MSI and not simple
                     and self.get_affordance('wood', n=hmi(n, 3, 1), enc=enc)) \
                and (self.object_exists('crafting_bench', to_reach=True) or
                     'crafting_bench' not in MSI
                     and self.get_affordance('crafting_bench', enc=enc)):
            return True
        # Get locations of all of the doors/buttons
        red_door = self.get_object("red_door")
        yellow_door = self.get_object("yellow_door")
        green_door = self.get_object("green_door")
        blue_door = self.get_object("blue_door")
        green_button = self.get_object("green_button")
        if obj == 'red_key' and \
                self.inventory[ITEM_TO_IDX[en]['red_key']] == 0:
            if self.object_exists('red_key', to_reach=True, full_search=True):
                return True
            elif self.start_key in ['yellow'] and \
                    red_door is not None and \
                    'red_door' not in MSI \
                    and self.get_affordance('red_door', enc=enc):
                return True
        if obj == 'yellow_key' and \
                self.inventory[ITEM_TO_IDX[en]['yellow_key']] == 0:
            if self.object_exists('yellow_key', to_reach=True, full_search=True):
                return True
            elif self.start_key in ['red'] and \
                    red_door is not None and  \
                    'red_door' not in MSI \
                    and self.get_affordance('red_door', enc=enc):
                return True
        if obj == 'red_door' and \
                (self.inventory[ITEM_TO_IDX[en]['red_key']] >= 1 or
                 'red_key' not in MSI
                  and self.get_affordance('red_key', enc=enc)) \
                and red_door is not None and not red_door.is_open:
            return True
        if obj == 'yellow_door' and \
                (self.inventory[ITEM_TO_IDX[en]['yellow_key']] >= 1 or
                    'yellow_key' not in MSI
                    and self.get_affordance('yellow_key', enc=enc)) \
                and yellow_door is not None and not yellow_door.is_open:
            return True
        if obj == 'green_weight':
            if self.object_exists('green_weight', to_reach=True, full_search=True):
                return True
            elif self.inventory[ITEM_TO_IDX[en]['green_weight']] == 0 \
                    and red_door is not None \
                    and 'red_door' not in MSI \
                    and self.get_affordance('red_door', enc=enc) \
                    and green_button is not None and not green_button.is_pressed:
                return True
        if obj == 'blue_key':
            if self.object_exists('blue_key', to_reach=True, full_search=True):
                return True
            elif self.inventory[ITEM_TO_IDX[en]['blue_key']] == 0 \
                    and yellow_door is not None \
                    and 'yellow_door' not in MSI \
                    and self.get_affordance('yellow_door', enc=enc):
                return True
        if obj == 'blue_door' and \
                (self.inventory[ITEM_TO_IDX[en]['blue_key']] >= 1 or
                    'blue_key' not in MSI
                    and self.get_affordance('blue_key', enc=enc)) \
                and blue_door is not None and not blue_door.is_open:
            return True
        if obj == 'green_door' and \
                (self.inventory[ITEM_TO_IDX[en]['green_weight']] >= 1 or
                    'green_weight' not in MSI
                    and self.get_affordance('green_weight', enc=enc)) \
                and green_door is not None and not green_door.is_open \
                and blue_door is not None \
                and (blue_door.is_open or 'blue_door' not in MSI
                     and self.get_affordance('blue_door', enc=enc)):
            return True
        if obj == 'purple_key':
            if self.object_exists('purple_key', to_reach=True, full_search=True):
                return True
            elif self.inventory[ITEM_TO_IDX[en]['purple_key']] == 0 \
                    and green_door is not None \
                    and 'green_door' not in MSI \
                    and self.get_affordance('green_door', enc=enc):
                return True
        if obj == 'treasure' and \
                red_door is not None \
                and (red_door.is_open or 'red_door' not in MSI
                     and self.get_affordance('red_door', enc=enc)) \
                and (self.inventory[ITEM_TO_IDX[en]['purple_key']] >= 1 or
                     'purple_key' not in MSI
                     and self.get_affordance('purple_key', enc=enc)):
            return True
        return False

    def get_affordances(self):
        """
        Return list of binary values, indicating which items in
        MILESTONE_TO_IDX are currently afforded (i.e. can be completed without
        completing any other milestones in MILESTONE_TO_IDX)
        """
        en = self.env_no
        dms = self.dms
        # set affordances as False by default, and fill in True where app.
        affordances = dict.fromkeys(MILESTONE_TO_IDX[en][dms].keys(), False)

        # Go through all keys and determine affordance status of object
        for obj in affordances.keys():
            aff = self.get_affordance(obj)
            affordances[obj] = aff

        return affordances

    def continuous_stoch(self):
        """
        Provide continuous form of stochasticity in the form of inventory
        disappearances
        """
        # Only perform stochasticity on some steps
        if np.random.random() < self.stoch_value:
            inv = np.array(self.inventory)
            inv[inv>1] = 1.0
            dist = inv * self.stoch_continuous_dist
            if sum(dist) == 0:
                return
            dist /= sum(dist)
            choice = np.random.choice(len(inv), p=dist)
            self.inventory[choice] -= 1  # sorry!
            name = IDX_TO_ITEM[self.env_no][choice]
            if self.verbose:
                print(f"Stoch: Removed {name} from inventory!")
            return name
        return None

    def step(self, action):
        """
        Step through environment; handle actions
        """
        self.bfs_time = 0
        en = self.env_no
        dms = self.dms
        self.step_count += 1
        self.total_steps += 1
        self.time_since_last_milestone += 1
        self.time_since_last_removal += 1

        # if task-agnostic learning is over, switch to target task
        if self.total_steps >= self.task_agnostic_steps:
            self.final_item = self.task_final_item

        reward = 0
        done = False
        timeout = False
        result = None

        # If fully observable, automatically set "direction" to up
        # Later methods will assume this initial orientation (e.g. placing)
        if self.fully_obs:
            self.agent_dir = DIR_TO_INT['up']

        self.set_positional_variables()

        # General Actions
        if hasattr(self.actions, 'left') and action == self.actions.left:
            if self.fully_obs:
                result = self.move('left')
            else:
                result = self.rotate_left()
            self.last_action = 'left'
        elif hasattr(self.actions, 'right') and action == self.actions.right:
            if self.fully_obs:
                result = self.move('right')
            else:
                result = self.rotate_right()
            self.last_action = 'right'
        elif hasattr(self.actions, 'up') and action == self.actions.up:
            if self.fully_obs:
                result = self.move('up')
            else:
                result = self.move_forward()
            self.last_action = 'forward'
        elif hasattr(self.actions, 'down') and action == self.actions.down:
            if self.fully_obs:
                result = self.move('down')
            else:
                result = self.move_backward()
            self.last_action = 'backward'
        elif hasattr(self.actions, "pickup") and action == self.actions.pickup:
            result = self.pickup()
            self.last_action = 'pickup'
        elif hasattr(self.actions, "drop") and action == self.actions.drop:
            result = self.drop()
            self.last_action = 'drop'
        elif hasattr(self.actions, "toggle") and action == self.actions.toggle:
            result = self.toggle()
            self.last_action = 'toggle'
        # Craft Actions
        elif hasattr(self.actions, 'mine') and action == self.actions.mine:
            if not self.fully_obs:
                result = self.mine()
            self.last_action = 'mine'
        elif hasattr(self.actions, 'craft_wood') and \
                action == self.actions.craft_wood:
            result = self.craft_wood()
            self.last_action = 'craft wood'
        elif hasattr(self.actions, 'craft_stick') and \
                action == self.actions.craft_stick:
            result = self.craft_stick()
            self.last_action = 'craft sticks'
        elif hasattr(self.actions, 'craft_crafting_bench') and \
                action == self.actions.craft_crafting_bench:
            result = self.craft_crafting_bench()
            self.last_action = 'craft bench'
        elif hasattr(self.actions, 'craft_pickaxe') and \
                action == self.actions.craft_pickaxe:
            result = self.craft_pickaxe()
            self.last_action = 'craft pickaxe'
        elif hasattr(self.actions, 'craft_furnace') and \
                action == self.actions.craft_furnace:
            result = self.craft_furnace()
            self.last_action = 'craft furnace'
        elif hasattr(self.actions, 'craft_iron') and \
                action == self.actions.craft_iron:
            result = self.craft_iron()
            self.last_action = 'craft iron'
        elif hasattr(self.actions, 'done') and \
                action == self.actions.done:
            self.last_action = 'done'
        # Treasure Actions
        elif hasattr(self.actions, 'interact') and \
                action == self.actions.interact:
            result = self.interact()
            self.last_action = 'interact'
        else:
            assert False, "unknown action"

        # Stochastically remove items from inventory
        if self.stoch_continuous:
            removed_item = self.continuous_stoch()
            if removed_item is not None:
                self.removed_item = removed_item
                self.time_since_last_removal = 0

        # Make sure we only identify the result of an action as a proper
        #   milestone if it is in the MILESTONE_TO_IDX dictionary
        if result in MILESTONE_TO_IDX[en][dms]:
            milestone = result
        else:
            milestone = None

        # Clear bfs cache
        if IDX_TO_ENV[en] == 'treasure' and result is not None:
            # If we open a door or collect a key, completely empty cache
            self.bfs_cache = dict()

        if IDX_TO_ENV[en] == 'craft':
            if result not in [None, 'stick', 'wood', 'wood_pickaxe',
                    'stone_pickaxe', 'iron', 'iron_pickaxe']:
                self.bfs_cache.clear()

        reward = self._item_reward(milestone)
        reward -= self.penalty_scaling
        obs = self.gen_obs()
        if IDX_TO_ENV[en] == 'treasure':
            impossible = False
        else:
            impossible = sum(obs['affordance']) == 0
        # impossible = False  # Ending if impossible messes with returns

        # Add to episodic coverage counts
        if milestone is not None:
            self.ms_counts[milestone] += 1
            self.last_milestone = milestone
            self.time_since_last_milestone = 0

        if self.step_count >= self.max_steps or impossible:
            timeout = True
            done = True
            if self.verbose:
                print("timeout!", self.step_count, os.getpid())

        removal = False
        if self.removed_item is not None and self.time_since_last_removal == 0:
            removal = True

        info = {'ms': milestone, 'timeout': timeout, 'impossible': impossible,
                'success': False, 'steps': self.step_count,
                'steps_remaining': self.max_steps - self.step_count,
                'ms_counts': self.ms_counts,
                'removal': removal}

        if milestone == self.final_item:
            done = True
            info['success'] = True

        if milestone is not None and self.verbose:
            print("Milestone collected:", milestone)

        return obs, reward, done, info

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        If in fully observable mode, return full grid and no mask.
        """

        # If fully obs, return grid and no vis_mask
        if self.fully_obs:
            return self.grid, None

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size,
                               self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2,
                                                   self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution
        encoding)
        """
        en = self.env_no
        dms = self.dms
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        # image = grid.encode(vis_mask)
        # int_image = grid.int_encode(vis_mask)
        sparse_image = grid.sparse_encode(vis_mask, agent_pos=self.agent_pos,
                                          fully_obs=self.fully_obs)

        assert hasattr(self, 'mission'), \
            "environments must define a textual mission string"

        # Make readable inventory for human interactions
        inventory_readable = []
        for i in range(len(self.inventory)):
            name = IDX_TO_ITEM[en][i]
            inventory_readable.append((name, self.inventory[i]))

        self.affordances = self.get_affordances()
        affordance = [None for _ in range(len(MILESTONE_TO_IDX[en][dms]))]
        for k, v in self.affordances.items():
            affordance[MILESTONE_TO_IDX[en][dms][k]] = int(v)

        obs = {
            # Models expect these components
            'img': sparse_image,
            'vec': self.inventory,
            'affordance': affordance,
        }

        if self.print_affordances:
            print("affordances:")
            for k, v in self.affordances.items():
                if v is True:
                    print("a: {}".format(k))

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs, self.env_no)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=True,
               strip_outer_walls=True,
               tile_size=TILE_PIXELS, pred_affordances=None,
               subtask=None):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import minigrid.window
            self.window = minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        highlight_mask = None
        # Only do highlighting if vis_mask and highlighting enabled
        if vis_mask is not None and highlight:
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = self.dir_vec
            r_vec = self.right_vec
            # top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - \
                # r_vec * (self.agent_view_size // 2)
            top_left = self.agent_pos + f_vec * (self.agent_view_size // 2) - \
                r_vec * (self.agent_view_size // 2)

            # Mask of which cells to highlight
            highlight_mask = np.ones(shape=(self.width, self.height),
                                      dtype=np.bool)

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = False

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
            fully_obs=self.fully_obs
         )

        if strip_outer_walls:
            img = img[tile_size:-tile_size, tile_size:-tile_size]

        tiles = []

        en = self.env_no
        dms = self.dms
        affordances = self.get_affordances()
        affordance = [None for _ in range(len(MILESTONE_TO_IDX[en][dms]))]
        for k, v in affordances.items():
            affordance[MILESTONE_TO_IDX[en][dms][k]] = int(v)

        for i, count in enumerate(self.inventory):
            name = IDX_TO_ITEM[self.env_no][i]
            if name == 'hurt_back':
                continue
            render_fn = ITEM_TO_RENDER[self.env_no][name]
            obj_tile = 255 * np.ones((tile_size, tile_size, 3), dtype=np.uint8)
            render_fn(obj_tile)
            count_tile = 255 * np.ones((tile_size, int(tile_size * 1.25), 3), dtype=np.uint8)
            if self.removed_item is not None and self.removed_item == name and self.time_since_last_removal < 3:
                count_tile = cv2.putText(count_tile, f'{count}', (0, tile_size - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                count_tile = cv2.putText(count_tile, f'{count}', (0, tile_size - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            aff_tile = 255 * np.ones((tile_size, tile_size, 3), dtype=np.uint8)
            if name in affordances:
                # Visualize actual affordances
                color = 'green' if affordances[name] else 'red'
                fill_coords(aff_tile, point_in_circle(0.5, 0.5, 0.25), COLORS[color])
                # Visualize predicted affordances
                if pred_affordances is not None:
                    color = 'green' if pred_affordances[name] else 'red'
                    fill_coords(aff_tile, point_in_circle(0.5, 0.5, 0.16), COLORS[color])
            tile = np.concatenate((aff_tile, obj_tile, count_tile), axis=1)
            tiles.append(tile)

        # Add tiles for milestones not in inventory
        for ms in MILESTONE_TO_IDX[en][dms].keys():
            if ms not in ITEM_TO_IDX[en]:
                render_fn = ITEM_TO_RENDER[self.env_no][ms]
                obj_tile = 255 * np.ones((tile_size, tile_size, 3), dtype=np.uint8)
                render_fn(obj_tile)
                count_tile = 255 * np.ones((tile_size, int(tile_size * 1.25), 3), dtype=np.uint8)
                aff_tile = 255 * np.ones((tile_size, tile_size, 3), dtype=np.uint8)
                if ms in affordances:
                    # Visualize actual affordances
                    color = 'green' if affordances[ms] else 'red'
                    fill_coords(aff_tile, point_in_circle(0.5, 0.5, 0.25), COLORS[color])
                    # Visualize predicted affordances
                    if pred_affordances is not None:
                        color = 'green' if pred_affordances[ms] else 'red'
                        fill_coords(aff_tile, point_in_circle(0.5, 0.5, 0.16), COLORS[color])
                tile = np.concatenate((aff_tile, obj_tile, count_tile), axis=1)
                tiles.append(tile)

        columns = []
        n_tiles_h = img.shape[0] // tile_size
        n_cols = math.ceil(len(tiles) / n_tiles_h)
        tiles_per_col = math.ceil(len(tiles) / n_cols)
        large_blank_tile = 255 * np.ones_like(tiles[0])
        blank1 = 255 * np.ones_like(aff_tile)
        blank2 = 255 * np.ones_like(obj_tile)
        blank3 = 255 * np.ones_like(count_tile)
        achieved_ms_tile = 255 * np.ones_like(blank2)
        # Create tile for achieved milestone
        if self.removed_item is not None and self.time_since_last_removal <= 3:
            render_fn = ITEM_TO_RENDER[self.env_no][self.removed_item]
            render_fn(achieved_ms_tile)
        elif len(self.last_milestone) != 0 and self.time_since_last_milestone <= 3:
            render_fn = ITEM_TO_RENDER[self.env_no][self.last_milestone]
            render_fn(achieved_ms_tile)
        for col_i in range(n_cols):
            col = []
            # tiles
            col += tiles[col_i * tiles_per_col:(col_i + 1) * tiles_per_col]
            # top pad
            col = [large_blank_tile for _ in range(n_tiles_h - len(col))] + col
            # Add achieved milestone e
            if col_i == 1 and len(self.last_milestone) != 0:
                mod_large_blank_tile = np.concatenate([achieved_ms_tile, blank1, blank3], axis=1)
                col[2] = mod_large_blank_tile

            columns.append(np.concatenate(col, axis=0))

        # add borders to env
        border_thickness = 2
        vert_bar = np.zeros((img.shape[0], border_thickness, 3), dtype=np.uint8)
        img = np.concatenate((vert_bar, img, vert_bar), axis=1)
        horz_bar = np.zeros((border_thickness, img.shape[1], 3), dtype=np.uint8)
        img = np.concatenate((horz_bar, img, horz_bar), axis=0)

        # glue columns together and add white borders on top and bottome to match size
        glued_cols = np.concatenate(columns, axis=1)
        white_horz_bar = np.ones((border_thickness, glued_cols.shape[1], 3), dtype=np.uint8) * 255
        glued_cols = np.concatenate((white_horz_bar, glued_cols, white_horz_bar), axis=0)

        # Concatenate environment cells with inventory cells
        img = np.concatenate((img, glued_cols), axis=1)

        # Add text indicating latest action
        orgx = len(img[0]) - 6 * tile_size  # text x coord
        orgy = int(tile_size * 0.6)  # text y coord
        org = (orgx, orgy)
        if len(self.last_action) != 0:
            cv2.putText(img, f'Action:  {self.last_action}', org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if subtask is not None:
            # Add text indicating latest subtask
            orgx = len(img[0]) - 6 * tile_size  # text x coord
            orgy = int(tile_size * 1.6)  # text y coord
            org = (orgx, orgy)
            cv2.putText(img, f'Subtask:  {subtask}', org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if len(self.last_milestone) != 0 and self.time_since_last_milestone <= 3 or self.removed_item is not None and self.time_since_last_removal <= 3:
            # Add text indicating collection of milestone
            orgx = len(img[0]) - 6 * tile_size  # text x coord
            orgy = int(tile_size * 2.6)  # text y coord
            org = (orgx, orgy)
            if self.last_milestone == self.final_item and self.time_since_last_milestone <= 3:
                cv2.putText(img, f'Completed', org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            elif self.removed_item is not None and self.time_since_last_removal <= 3:
                cv2.putText(img, f'Removed:', org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, f'Milestone:', org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Add step number
        orgx = len(img[0]) - int(2.1 * tile_size)  # text x coord
        orgy = int(tile_size * 2.6)  # text y coord
        org = (orgx, orgy)
        cv2.putText(img, f'(step {self.step_count})', org, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return
