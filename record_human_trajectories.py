import time
import argparse
import os
import glob
import pickle as pkl

import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from PIL import Image


counter = 0
save_dir = None
test_grid = None


def redraw(img):
    # img = env.get_obs_render(obs['image'], tile_size=args.tile_size)

    window.show_img(img)

def reset():
    global counter, save_dir, test_grid
    counter = 0

    # write save dir parent directory:
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = os.path.join(args.save_dir, f"{len(os.listdir(args.save_dir)):05d}")
    # write save dir
    os.makedirs(save_dir, exist_ok=True)
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    global counter,save_dir
    obs, reward, done, info = env.step(action)

    print("grid", env.grid)

    if save_dir is not None:
        # write image
        Image.fromarray(obs).save(os.path.join(save_dir, f"{counter:05d}.png"))
        # write action
        with open(os.path.join(save_dir, 'actions.txt'), 'a') as f:
            f.write(f"{action}\n")
        # write direction (fed to agent)
        with open(os.path.join(save_dir, 'agent_direction.txt'), 'a') as f:
            f.write(f"{env.agent_dir}\n")
        # write state
        np.save(os.path.join(save_dir, f'{counter:05d}.npy'), env.grid.encode())
        # write state (backup)
        with open(os.path.join(save_dir, f"{counter:05d}.pkl"), 'wb') as f:
            pkl.dump(env.unwrapped, f)

        counter += 1

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)


    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--save_dir',
    type=str,
    default=None,
    help="where to store the image"
)

args = parser.parse_args()

env = gym.make(args.env)

env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)