from minigrid.minigrid import ITEM_TO_IDX, ENV_TO_IDX
import time
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
plt.rcParams['keymap.save'].remove('s')  # prevent 's' from saving


class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    Modified to allow use of buttons (instead of key presses) for crafting
    """

    def __init__(self, env_no, title, craft_args=None, data_path=None):
        # Retrieve CraftEnv arguments (MOD)
        if craft_args:
            self.craft_args = craft_args  # entire dictionary
            self.env = craft_args['env']  # environment or wrapper
            self.args = craft_args['args']  # command-line arguments
            self.record = self.args.record
        else:
            self.craft_args = None
            self.env = None
            self.args = None
            self.record = False

        self.env_no = env_no

        # list of dictionaries (to be processed into `replay.py` objects
        # later for flexibility)
        self.trajectory = []
        self.data_path = data_path

        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Buttons (MOD)
        if craft_args:

            e = 0.0125  # offset for centering

            # Left items (collecting; no functionality)
            self.ax_count_diamond = plt.axes([0.0+e, 0.675, 0.1, 0.075])
            self.b_count_diamond = \
                Button(self.ax_count_diamond, 'Diamond\n(0)')
            self.ax_count_coal = plt.axes([0.0+e, 0.575, 0.1, 0.075])
            self.b_count_coal = Button(self.ax_count_coal, 'Coal\n(0)')
            self.ax_count_iron_ore = plt.axes([0.0+e, 0.475, 0.1, 0.075])
            self.b_count_iron_ore = \
                Button(self.ax_count_iron_ore, 'Iron Ore\n(0)')
            self.ax_count_stone = plt.axes([0.0+e, 0.375, 0.1, 0.075])
            self.b_count_stone = Button(self.ax_count_stone, 'Stone\n(0)')
            self.ax_count_log = plt.axes([0.0+e, 0.275, 0.1, 0.075])
            self.b_count_log = Button(self.ax_count_log, 'Log\n(0)')
            self.ax_count_dirt = plt.axes([0.0+e, 0.175, 0.1, 0.075])
            self.b_count_dirt = Button(self.ax_count_dirt, 'Dirt\n(0)')

            # Bottom items (crafting)
            self.ax_craft_wood = plt.axes([0.0+e, 0.02, 0.1, 0.075])
            self.ax_craft_stick = plt.axes([0.125+e, 0.02, 0.1, 0.075])
            self.ax_craft_crafting_bench = plt.axes([0.25+e, 0.02, 0.1, 0.075])
            self.ax_craft_wood_pickaxe = plt.axes([0.375+e, 0.02, 0.1, 0.075])
            self.ax_craft_stone_pickaxe = plt.axes([0.5+e, 0.02, 0.1, 0.075])
            self.ax_craft_iron_pickaxe = plt.axes([0.625+e, 0.02, 0.1, 0.075])
            self.ax_craft_furnace = plt.axes([0.75+e, 0.02, 0.1, 0.075])
            self.ax_craft_iron = plt.axes([0.875+e, 0.02, 0.1, 0.075])

            self.b_craft_wood = \
                Button(self.ax_craft_wood, 'Wood\n(0)')
            self.b_craft_wood.on_clicked(
                lambda x: self.step(self.env.actions.craft_wood))
            self.b_craft_stick = \
                Button(self.ax_craft_stick, 'Stick\n(0)')
            self.b_craft_stick.on_clicked(
                lambda x: self.step(self.env.actions.craft_stick))
            self.b_craft_crafting_bench = \
                Button(self.ax_craft_crafting_bench, 'Craft Bench\n(0)')
            self.b_craft_crafting_bench.on_clicked(
                lambda x: self.step(self.env.actions.craft_crafting_bench))
            self.b_craft_wood_pickaxe = \
                Button(self.ax_craft_wood_pickaxe, 'Wood Pick\n(0)')
            self.b_craft_wood_pickaxe.on_clicked(
                lambda x: self.step(self.env.actions.craft_pickaxe))
            self.b_craft_stone_pickaxe = \
                Button(self.ax_craft_stone_pickaxe, 'Stone Pick\n(0)')
            self.b_craft_stone_pickaxe.on_clicked(
                lambda x: self.step(self.env.actions.craft_pickaxe))
            self.b_craft_iron_pickaxe = \
                Button(self.ax_craft_iron_pickaxe, 'Iron Pick\n(0)')
            self.b_craft_iron_pickaxe.on_clicked(
                lambda x: self.step(self.env.actions.craft_pickaxe))
            self.b_craft_furnace = \
                Button(self.ax_craft_furnace, 'Furnace\n(0)')
            self.b_craft_furnace.on_clicked(
                lambda x: self.step(self.env.actions.craft_furnace))
            self.b_craft_iron = \
                Button(self.ax_craft_iron, 'Iron\n(0)')
            self.b_craft_iron.on_clicked(
                lambda x: self.step(self.env.actions.craft_iron))

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def redraw(self, img):
        ''' based on eponymous function in manual_control.py '''
        if not self.args.agent_view:
            img = self.env.render('rgb_array', tile_size=self.args.tile_size,
                                  highlight=False)

        self.show_img(img)

    def save_img(self, filename):
        img = self.env.render('rgb_array', tile_size=self.args.tile_size,
                              highlight=False)
        plt.imsave(filename + '.png', img)

    def save_trajectory(self):
        # Acquire filename metadata
        timestamp = int(time.time())
        length = len(self.trajectory)

        # Create path
        filename = f"{timestamp}-{length}"
        full_path = self.data_path + "/" + filename

        # Make sure directory exists
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # Save data
        pickle.dump(self.trajectory, open(full_path+".p", "wb+"))
        print("Trajectory saved under {}".format(full_path+".p"))

    def reset(self, done=False):
        ''' based on eponymous function in manual_control.py
            done is passed as True for the purpose of only saving successful
            trajectories upon reset
        '''
        if self.args.seed != -1:
            self.env.seed(self.args.seed)

        if len(self.trajectory) > 0 and done and self.record:
            self.save_trajectory()
        self.trajectory = []

        obs = self.env.reset()
        self.prev_obs = obs

        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.set_caption(self.env.mission)
        self.update_buttons()
        self.redraw(obs)

    def step(self, action):
        ''' based on eponymous function in manual_control.py '''
        # Perform step

        prev_obs = self.prev_obs
        obs, reward, done, info = self.env.step(action)
        print('step=%s, reward=%.2f' % (self.env.step_count, reward))
        print(', '.join(
            f'{item}: {self.env.inventory[index]}'
            for item, index in ITEM_TO_IDX[self.env.env_no].items()))

        # Store trajectory
        if self.trajectory is not None:
            # store newly generated portion of trajectory
            self.trajectory.append({
                'reward': reward,
                'done': done,
                'obs': prev_obs,
                'action': action,
                'next_obs': obs,
                'demo': True,
            })
        self.prev_obs = obs

        # Update matplotlib button inventory figures
        self.update_buttons()

        # Finish or redraw
        if done:
            print('done!')
            self.reset(done=True)
        else:
            self.redraw(obs)
            # print("inventory:", obs['inventory_readable'])

    def update_buttons(self):
        ''' Update buttons based on current inventory status '''
        en = self.env_no
        print("updating buttons")

        if en == ENV_TO_IDX['craft']:
            self.b_count_dirt.label.set_text("Dirt\n({})".format(
                    self.env.inventory[ITEM_TO_IDX[en]['dirt']]))
            self.b_count_log.label.set_text("Log\n({})".format(
                    self.env.inventory[ITEM_TO_IDX[en]['log']]))
            self.b_count_stone.label.set_text("Stone\n({})".format(
                    self.env.inventory[ITEM_TO_IDX[en]['stone']]))
            self.b_count_iron_ore.label.set_text("Iron Ore\n({})".format(
                    self.env.inventory[ITEM_TO_IDX[en]['iron_ore']]))
            self.b_count_coal.label.set_text("Coal\n({})".format(
                    self.env.inventory[ITEM_TO_IDX[en]['coal']]))
            self.b_count_diamond.label.set_text("Diamond\n({})".format(
                    self.env.inventory[ITEM_TO_IDX[en]['diamond']]))

            self.b_craft_wood.label.set_text("Wood\n({})".format(
                self.env.inventory[ITEM_TO_IDX[en]['wood']]))
            self.b_craft_stick.label.set_text("Stick\n({})".format(
                self.env.inventory[ITEM_TO_IDX[en]['stick']]))
            self.b_craft_crafting_bench.label.set_text("Craft Bench")
            self.b_craft_wood_pickaxe.label.set_text("Wood Pick\n({})".format(
                self.env.inventory[ITEM_TO_IDX[en]['wood_pickaxe']]))
            self.b_craft_stone_pickaxe.label.set_text("Stone Pick\n({})".format(
                self.env.inventory[ITEM_TO_IDX[en]['stone_pickaxe']]))
            self.b_craft_iron_pickaxe.label.set_text("Iron Pick\n({})".format(
                self.env.inventory[ITEM_TO_IDX[en]['iron_pickaxe']]))
            self.b_craft_furnace.label.set_text("Furnace")
            self.b_craft_iron.label.set_text("Iron\n({})".format(
                self.env.inventory[ITEM_TO_IDX[en]['iron']]))
        elif en == ENV_TO_IDX['treasure']:
            pass
        else:
            pass

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """
        return
        # self.ax.set_xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
