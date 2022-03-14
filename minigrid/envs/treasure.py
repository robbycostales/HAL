import random
import numpy as np

from minigrid.minigrid import MiniGridEnv, Grid, ENV_TO_IDX, YellowDoor,\
                                  RedDoor, BlueDoor, GreenDoor, GreenButton,\
                                  GreenWeight, YellowKey, RedKey, PurpleKey,\
                                  BlueKey, PurpleChest
from minigrid.register import register


class TreasureEnv(MiniGridEnv):
    """Treasure Environment
    """

    def __init__(self, seed, final_item="treasure", rewards="sparse",
                 gridrl_params=None):
        """Initialize TreasureEnv

        Args:
            final_item: The ultimate goal of the environment is to collect this
                item
            rewards: "sparse" to receive one external reward when final_item
                is collected; "dense" to receive exponentially increasing
                rewards as further milestones are collected.
        """

        # NOTE: the following super() call requires final_item to be set, as
        # _gen_grid() is called
        self.final_item = final_item
        self.env_no = ENV_TO_IDX['treasure']
        self.rewards = rewards
        self.size = 11
        super().__init__(
            grid_size=self.size,
            max_steps=30*self.size*self.size,
            final_item=final_item,
            fully_obs=False,
            rewards=rewards,
            seed=seed,
            env_no=self.env_no,
            gridrl_params=gridrl_params,
        )

    def _gen_grid(self, width, height):
        en = self.env_no

        # Create empty grid
        self.grid = Grid(self.env_no, width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate walls that separate rooms
        self.grid.wall_rect(3, 0, 5, height)
        self.grid.wall_rect(0, 3, width, 5)

        # Fill in non-rooms
        self.grid.wall_rect(1, 1, 2, 2)
        self.grid.wall_rect(8, 1, 2, 2)
        self.grid.wall_rect(8, 8, 2, 2)
        self.grid.wall_rect(1, 8, 2, 2)

        def select_new_pos(rs, pos, i, old=[]):
            for j in range(100):
                r = rs[j]
                # if r != no_go[i] and r not in old:
                if r not in old:
                    break
            return (pos[0] + r[0], pos[1] + r[1]), r

        # Get potential coordinates for red and yellow keys in center room
        # ("dumb way, but hey..." ^TM)
        # ris = [(random.randrange(4, 7), random.randrange(4, 7))
        #        for _ in range(100)]  # random initial room coordinates
        ris = [(4, 4), (4, 6), (6, 4), (6, 6)]
        # ris = list(set(ris))
        random.shuffle(ris)
        ris = ris[:2]

        # Decide between three options (both, one, or the other)
        i = np.random.choice(3)
        if i == 0:
            # Both keys
            self.put_obj(YellowKey(en), *ris[0])
            self.put_obj(RedKey(en), *ris[1])
            self.start_key = "both"
        elif i == 1:
            # Red key
            self.put_obj(RedKey(en), *ris[1])
            self.start_key = "red"
        elif i == 2:
            # Yellow key
            self.put_obj(YellowKey(en), *ris[0])
            self.start_key = "yellow"
        else:
            print(i)
            raise NotImplementedError

        # Place doorways and objects in rooms
        rooms = [(4, 1, 3, 2), (8, 4, 2, 3),
                 (4, 8, 3, 2), (1, 4, 2, 3)]  # (x, y, w, h)
        rooms_valid = [(4, 1, 3, 1), (9, 4, 1, 3),
                       (4, 9, 3, 1), (1, 4, 1, 3)]  # (x, y, w, h)
        doorways = [(5, 3), (7, 5), (5, 7), (3, 5)]  # (x, y)
        # Items blocking door, relative to top left for each room
        # no_go = [(1, 1), (0, 1), (1, 0), (1, 1)]
        colors = ['yellow', 'red', 'green', 'blue']
        random.shuffle(colors)  # randomize placement of the doorways
        for i, c in enumerate(colors):
            pos = rooms_valid[i][0:2]
            # Get a few random indices based on room shape
            rs = [(random.randrange(0, rooms_valid[i][2]),
                  random.randrange(0, rooms_valid[i][3]))
                  for _ in range(100)]

            # Yellow: place door and blue key
            if c == 'yellow':
                self.put_obj(YellowDoor(en), *doorways[i])
                p0, r0 = select_new_pos(rs, pos, i)
                self.put_obj(BlueKey(en), *p0)
                if self.start_key == "yellow":
                    p1, _ = select_new_pos(rs, pos, i, old=[r0])
                    self.put_obj(RedKey(en), *p1)

            # Red: place door, green weight, and purple chest
            elif c == 'red':
                self.put_obj(RedDoor(en), *doorways[i])
                p0, r0 = select_new_pos(rs, pos, i)
                self.put_obj(GreenWeight(en), *p0)
                p1, r1 = select_new_pos(rs, pos, i, old=[r0])
                self.put_obj(PurpleChest(en), *p1)
                if self.start_key == "red":
                    p2, _ = select_new_pos(rs, pos, i, old=[r0, r1])
                    self.put_obj(YellowKey(en), *p2)

            # Green: place door and purple key
            elif c == 'green':
                self.put_obj(GreenDoor(en), *doorways[i])
                p, _ = select_new_pos(rs, pos, i)
                self.put_obj(PurpleKey(en), *p)

            # Blue: place door and green button
            elif c == 'blue':
                self.put_obj(BlueDoor(en), *doorways[i])
                p0, _ = select_new_pos(rs, pos, i)
                self.put_obj(GreenButton(en), *p0)
            else:
                raise NotImplementedError

        # Place agent
        self.place_agent(top=(4, 4), size=(3, 3))
        self.mission = f"Obtain the item: {self.final_item}"


class TreasureEnvGreenDoor(TreasureEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="green_door", seed=seed, gridrl_params=gridrl_params)


class TreasureEnvTreasure(TreasureEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="treasure", seed=seed, gridrl_params=gridrl_params)


register(
    id='MiniGrid-TreasureEnv-GreenDoor-v0',
    entry_point='minigrid.envs:TreasureEnvGreenDoor'
)

register(
    id='MiniGrid-TreasureEnv-Treasure-v0',
    entry_point='minigrid.envs:TreasureEnvTreasure'
)
