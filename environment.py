from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union, Any, NotRequired, TypedDict, Unpack
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp

import mujoco

from brax import base
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
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

    def rewards(self, mjx_state: MjxState, action: jp.ndarray, next_mjx_state: MjxState) -> Tuple[jp.ndarray, jp.ndarray]:
        min_z = -1.0
        max_z = 2.0
        healthy = jp.where(mjx_state.qpos[2] < min_z, 0.0, 1.0)
        healthy = jp.where(mjx_state.qpos[2] > max_z, 0.0, healthy)
        reward = jp.array(5.0) * healthy

        return reward, healthy
    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        mjx_state = state.pipeline_state
        next_mjx_state = self.pipeline_step(mjx_state, action)
        obs = self.get_obs(mjx_state, action)
        reward, healthy = self.rewards(mjx_state, action, next_mjx_state)
        # reward = jp.array(0.0)
        # done = jp.array(1.0)

        done = 1.0 - healthy
        return state.replace(pipeline_state=next_mjx_state, obs=obs, reward=reward, done=done)

    def get_obs(self, data: MjxState, action: jp.ndarray) -> jp.ndarray:
        """Returns the observation of the environment."""
        return data.qpos
