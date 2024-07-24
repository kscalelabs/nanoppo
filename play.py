"""Play a trained PPO agent in a specified environment."""

import argparse
import logging
import os
from typing import Any, Callable

import jax as j
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
import yaml
from brax.io import model
from brax.mjx.base import State as mjxState
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from tqdm import tqdm
from environment import HumanoidEnv
import wandb

InferenceFn = Callable[[jp.ndarray, jp.ndarray], tuple[jp.ndarray, jp.ndarray]]

def mjx_rollout(env: mujoco.MjModel, n_steps: int = 1000, render_every: int = 2, seed: int = 0,) -> list[mjxState]:
    reset_fn = j.jit(env.reset)
    step_fn = j.jit(env.step)
    rng = j.random.PRNGKey(seed)
    state = reset_fn(rng)
    rollout = [state.pipeline_state]
    for i in tqdm(range(n_steps)):
        # state = step_fn(state, jp.zeros(env.action_size))
        state = step_fn(state, 100*j.random.uniform(rng, (env.action_size,)))
        rollout.append(state.pipeline_state)
        if state.done:
            state = reset_fn(rng)
    return rollout

def render_mjx_rollout(env: mujoco.MjModel, n_steps: int = 1000, render_every: int = 2, seed: int = 0, width: int = 320, height: int = 240) -> np.ndarray:
    rollout = mjx_rollout(env, n_steps, render_every, seed)
    images = env.render(rollout[::render_every], camera="side", width=width, height=height)
    return np.array(images)

def play(n_steps: int, render_every: int, width: int, height: int) -> None:
    env = HumanoidEnv()
    rng = j.random.PRNGKey(0)
    env.reset(rng)
    images_thwc = render_mjx_rollout(env, n_steps, render_every, width=width, height=height)
    fps = int(1/env.dt)
    print(f"Find video at video.mp4 with fps={fps}")
    media.write_video("video.mp4", images_thwc, fps=fps)

if __name__ == "__main__":
    play(n_steps=1000, render_every=2, width=640, height=480)
