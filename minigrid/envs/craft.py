import numpy as np
from collections import defaultdict

from minigrid.minigrid import MiniGridEnv, Grid, Tree, Dirt, Stone, Coal,\
                                  IronOre, Diamond, ENV_TO_IDX
from minigrid.register import register


class CraftEnv(MiniGridEnv):
    """Crafting Environment
    """

    def __init__(self, seed, size=12, final_item="diamond", rewards="sparse", gridrl_params=None):
        """Initialize CratEnv

        Args:
            size: Sets both length and width of grid (Note: the outer 2 columns
                and outer 2 rows will be set as `Wall` type, so only
                (size-2, size-2) size play area
            final_item: The ultimate goal of the environment is to collect this
                item
            rewards: "sparse" to receive one external reward when final_item
                is collected; "dense" to receive exponentially increasing
                rewards as further milestones are collected.
        """

        # NOTE: the following super() call requires final_item to be set, as
        # _gen_grid() is called
        self.final_item = final_item
        self.rewards = rewards
        self.env_no = ENV_TO_IDX['craft']
        super().__init__(
            grid_size=size,
            max_steps=30*size*size,
            final_item=final_item,
            fully_obs=False,
            rewards=rewards,
            seed=seed,
            env_no=self.env_no,
            gridrl_params=gridrl_params
        )

    def verify_grid(self, counts):
        """
        Verify environment has enough resources for comfortable success
        """
        # Check resource amounts
        violated = 0
        if counts[str(Tree)] < 5:
            violated += 1
        if counts[str(IronOre)] < 4:
            violated += 1
        if counts[str(Coal)] < 4:
            violated += 1

        # If any violated, return False
        if violated > 0:
            return False
        else:
            return True

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(self.env_no, width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        counts = defaultdict(lambda: 0)

        # Iterate through each block and randomly place objects
        for i in range(1, height-1):
            for j in range(1, width-1):

                if i < height // 2:  # forest region
                    a = [None, Tree, Dirt]
                    p = [36/50, 10/50, 4/50]
                else:  # stone region
                    a = [Stone, Coal, IronOre, Dirt]
                    p = [29/50, 8/50, 8/50, 5/50]

                obj = np.random.choice(a, p=p)
                if obj is not None:
                    self.put_obj(obj(self.env_no), j, i)
                    counts[str(obj)] += 1

        # place one diamond randomly on bottom level
        idxs = (np.random.choice(list(range(1, width-1))), height-2)
        self.put_obj(Diamond(self.env_no), *idxs)
        # Place the agent at a random position and orientation
        self.place_agent(size=(width, height))
        self.mission = f"Obtain the item: {self.final_item}"

        if self.verify_grid(counts):
            return
        else:
            self._gen_grid(width, height)


class CraftEnvLog(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="log", seed=seed, gridrl_params=gridrl_params)


class CraftEnvWood(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="wood", seed=seed, gridrl_params=gridrl_params)


class CraftEnvCraftingBench(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="crafting_bench", seed=seed, gridrl_params=gridrl_params)


class CraftEnvWoodPickaxe(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="wood_pickaxe", seed=seed, gridrl_params=gridrl_params)


class CraftEnvStone(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="stone", seed=seed, gridrl_params=gridrl_params)

class CraftEnvFurnace(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="furnace", seed=seed, gridrl_params=gridrl_params)


class CraftEnvStonePickaxe(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="stone_pickaxe", seed=seed, gridrl_params=gridrl_params)


class CraftEnvIronOre(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="iron_ore", seed=seed, gridrl_params=gridrl_params)


class CraftEnvCoal(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="coal", seed=seed, gridrl_params=gridrl_params)


class CraftEnvIron(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="iron", seed=seed, gridrl_params=gridrl_params)


class CraftEnvIronPickaxe(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="iron_pickaxe", seed=seed, gridrl_params=gridrl_params)


class CraftEnvDiamond(CraftEnv):
    def __init__(self, seed=123, gridrl_params=None):
        super().__init__(final_item="diamond", seed=seed, gridrl_params=gridrl_params)


register(
    id='MiniGrid-CraftEnv-Log-v0',
    entry_point='minigrid.envs:CraftEnvLog'
)

register(
    id='MiniGrid-CraftEnv-Wood-v0',
    entry_point='minigrid.envs:CraftEnvWood'
)

register(
    id='MiniGrid-CraftEnv-CraftingBench-v0',
    entry_point='minigrid.envs:CraftEnvCraftingBench'
)

register(
    id='MiniGrid-CraftEnv-WoodPickaxe-v0',
    entry_point='minigrid.envs:CraftEnvWoodPickaxe'
)

register(
    id='MiniGrid-CraftEnv-Stone-v0',
    entry_point='minigrid.envs:CraftEnvStone'
)

register(
    id='MiniGrid-CraftEnv-Furnace-v0',
    entry_point='minigrid.envs:CraftEnvFurnace'
)

register(
    id='MiniGrid-CraftEnv-StonePickaxe-v0',
    entry_point='minigrid.envs:CraftEnvStonePickaxe'
)

register(
    id='MiniGrid-CraftEnv-IronOre-v0',
    entry_point='minigrid.envs:CraftEnvIronOre'
)

register(
    id='MiniGrid-CraftEnv-Coal-v0',
    entry_point='minigrid.envs:CraftEnvCoal'
)

register(
    id='MiniGrid-CraftEnv-Iron-v0',
    entry_point='minigrid.envs:CraftEnvIron'
)

register(
    id='MiniGrid-CraftEnv-IronPickaxe-v0',
    entry_point='minigrid.envs:CraftEnvIronPickaxe'
)

register(
    id='MiniGrid-CraftEnv-Diamond-v0',
    entry_point='minigrid.envs:CraftEnvDiamond'
)
