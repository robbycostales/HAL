import numpy as np
from functools import reduce, partial
import os
import os

from .rendering import point_in_rect, fill_coords, point_in_circle, \
                       point_in_line, \
                       highlight_img, fill_img


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
# Fraction of tile that we fill with object images
FILL_FRAC = 0.99

# Map of environments to indices
ENV_TO_IDX = {
    'general':      0,
    'craft':        1,
    'treasure':     2
}

# Map of state names to integers
STATE_TO_IDX = {
    'open':     0,
    'closed':   1,
    'locked':   2,
    'pressed':  3
}
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

IDX_TO_ENV = dict(zip(ENV_TO_IDX.values(), ENV_TO_IDX.keys()))

# Map of color names to RGB values
COLORS = {
    'red':          np.array([255, 0, 0]),
    'green':        np.array([0, 255, 0]),
    'blue':         np.array([50, 180, 255]),
    'purple':       np.array([112, 39, 195]),
    'yellow':       np.array([175, 175, 0]),
    'grey':         np.array([100, 100, 100]),
    'black':        np.array([0, 0, 0]),
    'white':        np.array([255, 255, 255]),
    'brown':        np.array([120, 100, 50]),
    'light_grey':   np.array([130, 130, 130])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red':          0,
    'green':        1,
    'blue':         2,
    'purple':       3,
    'yellow':       4,
    'grey':         5,
    'black':        6,
    'white':        7,
    'brown':        8,
    'light_grey':   9
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX_G = {     # "General"
    'unseen':           0,
    'empty':            1,
    'agent':            2,
    'wall':             3,
    'floor':            12,
    'door':             13,
    'key':              14,
    'ball':             15,
    'box':              16,
    'goal':             17,
    'lava':             18,
}
OBJECT_TO_IDX_C = {     # "Craft"
    'empty':            0,
    'agent':            1,
    'wall':             2,
    'dirt':             3,
    'tree':             4,
    'stone':            5,
    'coal':             6,
    'iron_ore':         7,
    'diamond':          8,
    'crafting_bench':   9,
    'furnace':          10,
}
OBJECT_TO_IDX_T = {     # "Treasure"
    'empty':                    0,
    'agent':                    1,
    'wall':                     2,
    'red_door':                 3,
    'yellow_door':              4,
    'blue_door':                5,
    'green_door':               6,
    'red_key':                  7,
    'yellow_key':               8,
    'purple_key':               9,
    'green_weight':             10,
    'green_button':             11,
    'blue_key':                 12,
    'purple_chest':             13
}
OBJECT_TO_IDX = [OBJECT_TO_IDX_G, OBJECT_TO_IDX_C, OBJECT_TO_IDX_T]
ALL_OBJECTS = reduce(lambda x, y: x.union(y.keys()), OBJECT_TO_IDX, set())

# Number of objects in each environment (for encoding purposes)
NO_OBJECTS_G = len(OBJECT_TO_IDX_G)
NO_OBJECTS_C = len(OBJECT_TO_IDX_C)
NO_OBJECTS_T = len(OBJECT_TO_IDX_T)
NO_OBJECTS = [NO_OBJECTS_G, NO_OBJECTS_C, NO_OBJECTS_T]

# Number of channels used for sparse encoding (e.g. treasure needs attributes)
NO_CHANNELS_G = NO_OBJECTS_G + len(STATE_TO_IDX)
NO_CHANNELS_C = NO_OBJECTS_C
NO_CHANNELS_T = NO_OBJECTS_T + len(STATE_TO_IDX)
NO_CHANNELS = [NO_CHANNELS_G, NO_CHANNELS_C, NO_CHANNELS_T]

IDX_TO_OBJECT_G = dict(zip(OBJECT_TO_IDX_G.values(), OBJECT_TO_IDX_G.keys()))
IDX_TO_OBJECT_C = dict(zip(OBJECT_TO_IDX_C.values(), OBJECT_TO_IDX_C.keys()))
IDX_TO_OBJECT_T = dict(zip(OBJECT_TO_IDX_T.values(), OBJECT_TO_IDX_T.keys()))
IDX_TO_OBJECT = [IDX_TO_OBJECT_G, IDX_TO_OBJECT_C, IDX_TO_OBJECT_T]

# Inventory for each environment
ITEM_TO_IDX_G = None
ITEM_TO_IDX_C = {
    'log':              0,
    'dirt':             1,
    'wood':             2,
    'stick':            3,
    'stone':            4,
    'wood_pickaxe':     5,
    'stone_pickaxe':    6,
    'iron_pickaxe':     7,
    'iron_ore':         8,
    'iron':             9,
    'coal':             10,
    'diamond':          11,
}
ITEM_TO_IDX_T = {
    'red_key':          0,
    'yellow_key':       1,
    'green_weight':     2,
    'hurt_back':        3,
    'blue_key':         4,
    'purple_key':       5,
    'treasure':         6
}
ITEM_TO_IDX = [ITEM_TO_IDX_G, ITEM_TO_IDX_C, ITEM_TO_IDX_T]

IDX_TO_ITEM_G = None
IDX_TO_ITEM_C = dict(zip(ITEM_TO_IDX_C.values(), ITEM_TO_IDX_C.keys()))
IDX_TO_ITEM_T = dict(zip(ITEM_TO_IDX_T.values(), ITEM_TO_IDX_T.keys()))
IDX_TO_ITEM = [IDX_TO_ITEM_G, IDX_TO_ITEM_C, IDX_TO_ITEM_T]

# Item reward value for flat dense reward setting
ITEM_TO_REWARD_G = None
ITEM_TO_REWARD_C = {
    'log':              1.,
    'wood':             1.,
    'stick':            1.,
    'crafting_bench':   1.,
    'wood_pickaxe':     1.,
    'stone':            1.,
    'furnace':          1.,
    'stone_pickaxe':    1.,
    'iron_ore':         1.,
    'coal':             1.,
    'iron':             1.,
    'iron_pickaxe':     1.,
    'diamond':          1.
}
ITEM_TO_REWARD_T = {
    'red_key':          1.,
    'yellow_key':       1.,
    'green_weight':     1.,
    'blue_key':         1.,
    'green_placed':     1.,
    'purple_key':       1.,
    'treasure':         1.
}
ITEM_TO_REWARD = [ITEM_TO_REWARD_G, ITEM_TO_REWARD_C, ITEM_TO_REWARD_T]

# Ratios at which items will disappear from inventory with continuous stoch
ITEM_TO_STOCH_G = None
ITEM_TO_STOCH_C = {
    'log':              20.,
    'dirt':             0.,
    'wood':             20.,
    'stick':            20.,
    'stone':            10.,
    'wood_pickaxe':     10.,
    'stone_pickaxe':    8.,
    'iron_pickaxe':     2.,
    'iron_ore':         5.,
    'iron':             5.,
    'coal':             5.,
    'diamond':          1.
}
ITEM_TO_STOCH_T = {
    'red_key':          1.,
    'yellow_key':       1.,
    'green_weight':     1.,
    'blue_key':         1.,
    'green_placed':     1.,
    'purple_key':       1.,
    'treasure':         1.
}
ITEM_TO_STOCH = [ITEM_TO_STOCH_G, ITEM_TO_STOCH_C, ITEM_TO_STOCH_T]

MILESTONE_TO_IDX_G = None
MILESTONE_TO_IDX_C = [
    {   # Full set
        'log':              0,
        'wood':             1,
        'stick':            2,
        'crafting_bench':   3,
        'wood_pickaxe':     4,
        'stone':            5,
        'furnace':          6,
        'stone_pickaxe':    7,
        'iron_ore':         8,
        'coal':             9,
        'iron':             10,
        'iron_pickaxe':     11,
        'diamond':          12
    },
    {   # Materials only
        'log':              0,
        'wood':             1,
        'stick':            2,
        'stone':            3,
        'iron_ore':         4,
        'coal':             5,
        'iron':             6,
        'diamond':          7
    },
    {   # Tools only
        'crafting_bench':   0,
        'wood_pickaxe':     1,
        'furnace':          2,
        'stone_pickaxe':    3,
        'iron_pickaxe':     4,
        'diamond':          5
    },
    {   # Robby's reduced set (log, tools, and diamond)
        'log':              0,
        'wood_pickaxe':     1,
        'furnace':          2,
        'stone_pickaxe':    3,
        'iron':             4,
        'iron_pickaxe':     5,
        'diamond':          6
    }
]
MILESTONE_TO_IDX_T = [
    {   # Full set
        'red_key':          0,
        'yellow_key':       1,
        'red_door':         2,
        'yellow_door':      3,
        'green_weight':     4,
        'blue_key':         5,
        'blue_door':        6,
        'green_door':       7,
        'purple_key':       8,
        'treasure':         9
    },
    {   # Keys
        'red_key':          0,
        'yellow_key':       1,
        'green_weight':     2,
        'blue_key':         3,
        'purple_key':       4,
        'treasure':         5
    },
    {   # Doors
        'red_door':         0,
        'yellow_door':      1,
        'blue_door':        2,
        'green_door':       3,
        'treasure':         4
    },
]
MILESTONE_TO_IDX = [MILESTONE_TO_IDX_G, MILESTONE_TO_IDX_C, MILESTONE_TO_IDX_T]

IDX_TO_MILESTONE_G = None
IDX_TO_MILESTONE_C = [dict(zip(M.values(), M.keys())) for M in MILESTONE_TO_IDX_C]
IDX_TO_MILESTONE_T = [dict(zip(M.values(), M.keys())) for M in MILESTONE_TO_IDX_T]
IDX_TO_MILESTONE = [IDX_TO_MILESTONE_G, IDX_TO_MILESTONE_C, IDX_TO_MILESTONE_T]

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

DIR_TO_INT = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
INT_TO_DIR = dict(zip(DIR_TO_INT.values(), DIR_TO_INT.keys()))


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, env_no, obj_type, color):
        assert obj_type in ALL_OBJECTS, obj_type
        assert color in COLOR_TO_IDX, color
        assert env_no in ENV_TO_IDX.values()
        self.type = obj_type
        self.color = color
        self.env_no = env_no
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.env_no][self.type],
                COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state, env_no):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[env_no][type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type in ['empty', 'unseen']:
            return None

        # State, 0: open, 1: closed, 2: locked, 3: pressed
        is_open = state == 0
        is_locked = state == 2
        is_pressed = state == 3

        if obj_type == 'wall':
            v = Wall(env_no, color)
        elif obj_type == 'floor':
            v = Floor(env_no, color)
        elif obj_type == 'ball':
            v = Ball(env_no, color)
        elif obj_type == 'key':
            v = Key(env_no, color)
        elif obj_type == 'box':
            v = Box(env_no, color)
        elif obj_type == 'door':
            v = Door(env_no, color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal(env_no)
        elif obj_type == 'lava':
            v = Lava(env_no)
        elif obj_type == 'dirt':
            v = Dirt(env_no)
        elif obj_type == 'tree':
            v = Tree(env_no)
        elif obj_type == 'stone':
            v = Stone(env_no)
        elif obj_type == 'coal':
            v = Coal(env_no)
        elif obj_type == 'iron_ore':
            v = IronOre(env_no)
        elif obj_type == 'diamond':
            v = Diamond(env_no)
        elif obj_type == 'furnace':
            v = Furnace(env_no)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self, env_no):
        super().__init__(env_no, 'goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, env_no, color='blue', name='floor'):
        super().__init__(env_no, name, color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class GreenButton(Floor):
    def __init__(self, env_no, is_pressed=False):
        super().__init__(env_no, color='green', name='green_button')
        self.is_pressed = is_pressed

    def can_overlap(self):
        return False

    def toggle(self, env, pos):
        # Assume agent has weight (interact handles this)
        self.is_pressed = True
        return True

    def encode(self):
        en = self.env_no

        if self.is_pressed:
            state = 3
        else:
            state = 0

        return (OBJECT_TO_IDX[en][self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img, opacity=1.):
        if self.is_pressed:
            color = np.array(COLORS[self.color]) * 0.45
            fill_coords(img, point_in_rect(0.05, 0.95, 0.8, 0.9), color)
            fill_coords(img, point_in_rect(0.48, 0.52, 0.65, 0.8), color)
            fill_coords(img, point_in_rect(0.25, 0.75, 0.55, 0.65), color)
        else:
            color = np.array(COLORS[self.color]) * 0.45
            fill_coords(img, point_in_rect(0.05, 0.95, 0.8, 0.9), color)
            fill_coords(img, point_in_rect(0.48, 0.52, 0.3, 0.8), color)
            fill_coords(img, point_in_rect(0.25, 0.75, 0.2, 0.3), color)
        highlight_img(img, alpha=(1 - opacity))


class Lava(WorldObj):
    def __init__(self, env_no):
        super().__init__(env_no, 'lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03),
                        (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03),
                        (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03),
                        (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03),
                        (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, env_no, color='grey'):
        super().__init__(env_no, 'wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, env_no, color, is_open=False, is_locked=False,
                 name='door'):
        super().__init__(env_no, name, color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos, force_open=False):
        # If the player has the right key to open the door
        if force_open:
            # Return True if opening door for first time
            if self.is_open:
                return False
            else:
                self.is_locked = False
                self.is_open = True
                return True
        else:
            if self.is_locked:
                if isinstance(env.carrying, Key) and \
                        env.carrying.color == self.color:
                    self.is_locked = False
                    self.is_open = True
                    return True
                return False

            self.is_open = not self.is_open
            return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        en = self.env_no

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[en][self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]
        white = (255, 255, 255)

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), white)
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 *
                        np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), white)
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), white)

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class RedDoor(Door):
    def __init__(self, env_no, is_open=False, is_locked=True):
        super().__init__(env_no, 'red', is_open=is_open, name='red_door',
                         is_locked=is_locked)


class YellowDoor(Door):
    def __init__(self, env_no, is_open=False, is_locked=True):
        super().__init__(env_no, 'yellow', is_open=is_open, name='yellow_door',
                         is_locked=is_locked)


class BlueDoor(Door):
    def __init__(self, env_no, is_open=False, is_locked=True):
        super().__init__(env_no, 'blue', is_open=is_open, name='blue_door',
                         is_locked=is_locked)


class GreenDoor(Door):
    def __init__(self, env_no, is_open=False, is_locked=True):
        super().__init__(env_no, 'green', is_open=is_open, name='green_door',
                         is_locked=is_locked)


class Key(WorldObj):
    def __init__(self, env_no, color='blue', name='key'):
        super(Key, self).__init__(env_no, name, color)

    def can_pickup(self):
        return True

    def render(self, img, opacity=1.):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064),
                    (255, 255, 255))
        highlight_img(img, alpha=(1 - opacity))


class RedKey(Key):
    def __init__(self, env_no):
        super(RedKey, self).__init__(env_no, 'red', name='red_key')


class YellowKey(Key):
    def __init__(self, env_no):
        super(YellowKey, self).__init__(env_no, 'yellow', name='yellow_key')


class PurpleKey(Key):
    def __init__(self, env_no):
        super(PurpleKey, self).__init__(env_no, 'purple', name='purple_key')


class BlueKey(Key):
    def __init__(self, env_no):
        super(BlueKey, self).__init__(env_no, 'blue', name='blue_key')


class Ball(WorldObj):
    def __init__(self, env_no, color='blue', name='ball'):
        super(Ball, self).__init__(env_no, name, color)

    def can_pickup(self):
        return True

    def render(self, img, opacity=1.):
        darker_color = np.array(COLORS[self.color]) * 0.45
        fill_coords(img, point_in_circle(0.5, 0.5, 0.37), darker_color)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
        fill_coords(img, point_in_line(0.19, 0.5, .81, 0.5, 0.04), darker_color)
        highlight_img(img, alpha=(1 - opacity))


class GreenWeight(Ball):
    def __init__(self, env_no):
        super(GreenWeight, self).__init__(env_no, color='green',
                                          name='green_weight')


class Box(WorldObj):
    def __init__(self, env_no, color, contains=None):
        super(Box, self).__init__(env_no, 'box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        white = (255, 255, 255)

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), white)

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class PurpleChest(WorldObj):
    def __init__(self, env_no):
        super(PurpleChest, self).__init__(env_no, 'purple_chest', 'purple')

    def render(self, img, opacity=1.):
        light_purple = (184, 102, 255)
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), light_purple)

        # Wood texture
        fill_coords(img, point_in_rect(0.49, 0.51, 0.12, 1-0.12), c)
        fill_coords(img, point_in_rect(0.28, 0.30, 0.12, 1-0.12), c)
        fill_coords(img, point_in_rect(1-0.30, 1-0.28, 0.12, 1-0.12), c)

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.40, 0.46), c)

        # Key hole
        fill_coords(img, point_in_circle(cx=0.50, cy=0.43, r=0.10), c)
        highlight_img(img, alpha=(1 - opacity))


class Dirt(WorldObj):
    def __init__(self, env_no, color="brown"):
        super(Dirt, self).__init__(env_no, 'dirt', color)

    def render(self, img):
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/dirt_tile.png'),
                 FILL_FRAC)
        # fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Tree(WorldObj):
    def __init__(self, env_no, color="green"):
        super(Tree, self).__init__(env_no, 'tree', color)

    def render(self, img):
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/tree.png'),
                 FILL_FRAC)
        # fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Stone(WorldObj):
    def __init__(self, env_no, color="light_grey"):
        super(Stone, self).__init__(env_no, 'stone', color)

    def render(self, img):
        # fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/stone_tile.png'),
                 FILL_FRAC)


class Coal(WorldObj):
    def __init__(self, env_no, color="black"):
        super(Coal, self).__init__(env_no, 'coal', color)

    def render(self, img):
        # fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0.02, 1., 0.02, 1.), COLORS['light_grey'])
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/coal.png'),
                 FILL_FRAC)


class IronOre(WorldObj):
    def __init__(self, env_no, color="purple"):
        super(IronOre, self).__init__(env_no, 'iron_ore', color)

    def render(self, img):
        # fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0.02, 1., 0.02, 1.), COLORS['light_grey'])
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/iron_ore.png'),
                 FILL_FRAC)


class Diamond(WorldObj):
    def __init__(self, env_no, color="blue"):
        super(Diamond, self).__init__(env_no, 'diamond', color)

    def render(self, img):
        # fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
        fill_coords(img, point_in_rect(0.02, 1., 0.02, 1.), COLORS['light_grey'])
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/diamond.png'),
                 FILL_FRAC)


class CraftingBench(WorldObj):
    def __init__(self, env_no, color="brown"):
        super().__init__(env_no, 'crafting_bench', color)

    def render(self, img):
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/crafting_bench.png'),
                 FILL_FRAC)


class Furnace(WorldObj):
    def __init__(self, env_no, color="grey"):
        super().__init__(env_no, 'furnace', color)

    def render(self, img):
        fill_img(img, os.path.join(os.path.dirname(__file__),
                                   'icons/furnace.png'),
                 FILL_FRAC)

# Need to define these down here so we can use classes defined above
ITEM_TO_RENDER_G = None

ITEM_TO_RENDER_C = {
    'log': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/log.png'),
        frac=FILL_FRAC),
    'dirt': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/dirt_tile.png'),
        frac=FILL_FRAC),
    'wood': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/wood.png'),
        frac=FILL_FRAC),
    'stick': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/stick.png'),
        frac=FILL_FRAC),
    'stone': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/stone_tile.png'),
        frac=FILL_FRAC),
    'wood_pickaxe': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/wood_pickaxe.png'),
        frac=FILL_FRAC),
    'stone_pickaxe': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/stone_pickaxe.png'),
        frac=FILL_FRAC),
    'iron_pickaxe': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/iron_pickaxe.png'),
        frac=FILL_FRAC),
    'iron_ore': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/iron_ore.png'),
        frac=FILL_FRAC),
    'iron': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/iron.png'),
        frac=FILL_FRAC),
    'diamond': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/diamond.png'),
        frac=FILL_FRAC),
    'coal': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/coal.png'),
        frac=FILL_FRAC),
    'crafting_bench': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/crafting_bench.png'),
        frac=FILL_FRAC),
    'furnace': partial(
        fill_img, fill_img_path=os.path.join(os.path.dirname(__file__),
                                             'icons/furnace.png'),
        frac=FILL_FRAC),
}
ITEM_TO_RENDER_T = {
    'red_key': RedKey(2).render,
    'yellow_key': YellowKey(2).render,
    'green_weight': GreenWeight(2).render,
    'hurt_back': GreenButton(2, is_pressed=True).render,
    'blue_key': BlueKey(2).render,
    'purple_key': PurpleKey(2).render,
    'treasure': PurpleChest(2).render,
    'red_door': RedDoor(2).render,
    'yellow_door': YellowDoor(2).render,
    'blue_door': BlueDoor(2).render,
    'green_door': GreenDoor(2).render,

}
ITEM_TO_RENDER = [ITEM_TO_RENDER_G, ITEM_TO_RENDER_C, ITEM_TO_RENDER_T]
