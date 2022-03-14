#!/usr/bin/env python3

import argparse
import gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import matplotlib as mpl
from minigrid.window import Window
from minigrid.minigrid import ENV_TO_IDX, MILESTONE_TO_IDX, \
                                  IDX_TO_MILESTONE


def msn_generate(msn, fms, ims):
    """
    Args:
        msn: (int) number of elements to be in milestone set
        fms: (str) name of final milestone
    Returns:
        ITM, MTI (dicts)
    """
    milestones = [(k, v) for k, v in sorted(ims.items())]
    names = [v for _, v in milestones]
    milestone_set = set()
    # First, remove all milestones after final milestone
    idx = names.index(fms)
    milestones = milestones[:idx+1]
    if len(milestones) < msn:
        raise ValueError
    # Add final milestone to milestone set
    ms = milestones.pop(-1)
    milestone_set.add(ms)
    # Choose msn random milestones and add to set
    extras = np.random.choice(len(milestones), size=msn-1, replace=False)
    milestone_set = milestone_set.union([milestones[i] for i in extras])
    # replace IDX_TO_MILESTONE[en][0] and MILESTONE_TO_IDX[en][0]
    # with updated values
    milestones = sorted(list(milestone_set))
    ITM = dict()
    MTI = dict()
    for i in range(len(milestones)):
        ITM[i] = milestones[i][1]
        MTI[milestones[i][1]] = i
    return ITM, MTI


def key_handler(event):
    """
    Actions:

    left (arrow), a     : turn left (or move left)
    right (arrow), d    : turn right (or move right)
    up (arrow), w       : move forward (or move up)
    down (arrow), s     : move backward (or move down)

    t                   : toggle (fwd item)
    p                   : pickup (fwd item)
    o                   : drop (currently held item, or furnace)

    <spacebar>          : mine

    1                   : craft wood
    2                   : craft stick
    3                   : craft crafting bench
    4                   : craft wood pickaxe
    5                   : craft stone pickaxe
    6                   : craft iron pickaxe
    7                   : craft furnace
    8                   : craft iron

    c                   : place crafting bench
    f                   : place furnace

    """

    print('pressed', event.key)

    # Meta
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        window.reset()
        return
    if event.key == 'z':
        # save screenshot
        print("Saving screenshot, enter filename:")
        fname = input()
        window.save_img(fname)

    # Move
    if event.key == 'left' or event.key == 'a':
        window.step(env.actions.left)
        return
    if event.key == 'right' or event.key == 'd':
        window.step(env.actions.right)
        return
    if event.key == 'up' or event.key == 'w':
        window.step(env.actions.up)
        return
    if event.key == 'down' or event.key == 's':
        window.step(env.actions.down)
        return

    # Toggle
    if event.key == 't':
        window.step(env.actions.toggle)
        return

    # Drop / pickup
    if event.key == 'p':
        window.step(env.actions.pickup)
        return
    if event.key == 'o':
        window.step(env.actions.drop)
        return

    # Mine
    if event.key == ' ':  # spacebar
        window.step(env.actions.mine)
        return

    # Craft
    if event.key == '1':
        window.step(env.actions.craft_wood)
    if event.key == '2':
        window.step(env.actions.craft_stick)
    if event.key == '3':
        window.step(env.actions.craft_crafting_bench)
    if event.key == '4':
        window.step(env.actions.craft_wood_pickaxe)
    if event.key == '5':
        window.step(env.actions.craft_stone_pickaxe)
    if event.key == '6':
        window.step(env.actions.craft_iron_pickaxe)
    if event.key == '7':
        window.step(env.actions.craft_furnace)
    if event.key == '8':
        window.step(env.actions.craft_iron)

    # Place
    if event.key == 'c':
        window.step(env.actions.place_crafting_bench)
    if event.key == 'v':
        window.step(env.actions.place_furnace)

    # Interact (treasure)
    if event.key == 'i':
        window.step(env.actions.interact)

    if event.key == 'enter':
        window.step(env.actions.done)
        return


OBJ_TO_ENV = {
        'log': 'MiniGrid-CraftEnv-Log-v0',
        'wood': 'MiniGrid-CraftEnv-Wood-v0',
        'crafting_bench': 'MiniGrid-CraftEnv-CraftingBench-v0',
        'wood_pickaxe': 'MiniGrid-CraftEnv-WoodPickaxe-v0',
        'stone': 'MiniGrid-CraftEnv-Stone-v0',
        'furnace': 'MiniGrid-CraftEnv-Furnace-v0',
        'stone_pickaxe': 'MiniGrid-CraftEnv-StonePickaxe-v0',
        'iron_ore': 'MiniGrid-CraftEnv-IronOre-v0',
        'coal': 'MiniGrid-CraftEnv-Coal-v0',
        'iron': 'MiniGrid-CraftEnv-Iron-v0',
        'iron_pickaxe': 'MiniGrid-CraftEnv-IronPickaxe-v0',
        'diamond': 'MiniGrid-CraftEnv-Diamond-v0',

        'treasure': 'MiniGrid-TreasureEnv-Treasure-v0',
        'green_door': 'MiniGrid-TreasureEnv-GreenDoor-v0',
}

if __name__ == "__main__":
    mpl.use('tkagg')
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="iron",
            type=str, help="Name of item to obtain (defines environment)",
            choices=['log', 'wood', 'crafting_bench', 'wood_pickaxe', 'stone',
                     'furnace', 'stone_pickaxe', 'iron_ore', 'coal', 'iron',
                     'iron_pickaxe', 'diamond', 'green_door', 'treasure'])
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=123
    )
    parser.add_argument(
        "--dms",
        type=int,
        help="default milestone set number",
        default=0
    )
    parser.add_argument(
        "--msn",
        type=int,
        help="number of elements in milestone set",
        default=0
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )
    parser.add_argument(
        '--record',
        default=False,
        help="record and save trajectory",
        action='store_true'
    )
    parser.add_argument(
        "--single-obj-inventory",
        action="store_true",
        help="Only allow agents to hold a single object at a time in treasure\
              environment.")

    parser.add_argument("--stoch-continuous", action="store_true",
                        help="Use continuous form of stochasticity in env")
    parser.add_argument("--stoch-discrete", action="store_true",
                        help="Use discrete form of stochasticity in env")
    parser.add_argument("--stoch-value", type=float, default=1.0,
                        help="The amount of stochasticity to use for each type")

    args = parser.parse_args()

    env_name = OBJ_TO_ENV[args.env]
    if 'Craft' in env_name:
        env_no = ENV_TO_IDX['craft']
    elif 'Treasure' in env_name:
        env_no = ENV_TO_IDX['treasure']
    else:
        env_no = ENV_TO_IDX['general']

    gridrl_params = {
        'dms':                  args.dms,
        'print_affordances':    True,
        'agent_view_size':      7,
        'dense_rewards':        False,
        'task_agnostic_steps':  0,
        'penalty_scaling':      0.01,
        'single_obj_inventory': True,
        'env_verbose':          True,
        'evaluate':             False,

        'stoch_continuous':     args.stoch_continuous,
        'stoch_discrete':       args.stoch_discrete,
        'stoch_value':          args.stoch_value,
    }

    np.random.seed(args.seed)

    msn = args.msn
    if msn > 0:
        en = env_no
        fms = args.env
        ims = IDX_TO_MILESTONE[en][0]
        ITM, MTI = msn_generate(msn, fms, ims)
        IDX_TO_MILESTONE[en][0] = ITM
        MILESTONE_TO_IDX[en][0] = MTI
        print("ITM:", ITM)
        print("MTI:", MTI)

    env = gym.make(env_name, gridrl_params=gridrl_params)

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window(env_no,
                    'minigrid - ' + env_name,
                    craft_args={'env': env, 'args': args},
                    data_path=f'trajectories/{env_name}')
    window.reg_key_handler(key_handler)
    window.reset()

    # Blocking event loop
    window.show(block=True)
