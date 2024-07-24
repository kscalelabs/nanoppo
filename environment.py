from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union, Any, NotRequired, TypedDict, Unpack
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

class EnvKwargs(TypedDict):
    sys: base.System
    backend: NotRequired[str]
    n_frames: NotRequired[int]
    debug: NotRequired[bool]

class HumanoidEnv(PipelineEnv):
    def __init__(self, **kwargs: Unpack[EnvKwargs],) -> None:
        mj_model = mujoco.MjModel.from_xml_path("assets/stompylegs.xml")
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model)
        self.initial_qpos = jp.array(mj_model.keyframe("default").qpos)
        self._action_size = mj_model.nu
        sys = mjcf.load_model(mj_model)
        # kwargs stuff
        physics_steps_per_control_step = 4 
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"
        
        super().__init__(sys, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        qpos = (self.initial_qpos)
        qvel = jp.zeros(len(qpos)-1)

        # initialize mjx state
        mjx_state = self.pipeline_init(qpos, qvel)
        obs = self.get_obs(mjx_state, jp.zeros(self._action_size))
        reward, done, zero = jp.zeros(3)

        metrics = {}

        return State(mjx_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        mjx_state = state.pipeline_state
        next_mjx_state = self.pipeline_step(mjx_state, action)
        obs = self.get_obs(mjx_state, action)
        # reward, done = self.reward_fn(mjx_state, action, next_mjx_state)
        reward = jp.zeros(1)
        done = jp.zeros(1)
        return state.replace(pipeline_state=next_mjx_state, obs=obs, reward=reward, done=done)
    
    def get_obs(self, data: MjxState, action: jp.ndarray) -> jp.ndarray:
        """Returns the observation of the environment."""
        return data.qpos
    
def run_env():
    # init env and state
    env = HumanoidEnv()
    rng = jp.zeros(3)
    state = env.reset(rng)
    # sim params
    duration = 2.0
    framerate = 30
    # renderer + scene optinos
    mj_model = mujoco.MjModel.from_xml_path("assets/stompylegs.xml")
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)
    scene_option = mujoco.MjvOption()
    # visualize joints
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    frames = []
    mujoco.mj_resetData(mj_model, mj_data)
    current_time = 0.0
    mark = 0.1
    while current_time < duration:
        print("ALLEN")
        print(current_time)
        # action = jp.random_uniform(rng, shape=(env._action_size,))
        action = np.zeros(env._action_size)
        state = env.step(state, action)

        mj_data.qpos = state.pipeline_state.qpos
        mj_data.qvel = state.pipeline_state.qvel
        mujoco.mj_forward(mj_model, mj_data)

        if len(frames) < current_time * framerate:
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

        # Step the simulation
        mujoco.mj_step(mj_model, mj_data)
        current_time += mj_model.opt.timestep
    media.show_video(frames, fps=framerate)
    print("video done")


if __name__ == "__main__":
    # python environment.py
    # NOTE: its better to use play to test the environment instead
    run_env()