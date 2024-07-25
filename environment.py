"""Defines the training environment."""

from typing import NotRequired, Tuple, TypedDict, Unpack

import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx.base import State as MjxState
from jax import numpy as jp


class EnvKwargs(TypedDict):
    sys: base.System
    backend: NotRequired[str]
    n_frames: NotRequired[int]
    debug: NotRequired[bool]


class HumanoidEnv(PipelineEnv):
    def __init__(
        self,
        **kwargs: Unpack[EnvKwargs],
    ) -> None:
        mj_model = mujoco.MjModel.from_xml_path("assets/stompylegs.xml")
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
        qpos = self.initial_qpos
        qvel = jp.zeros(len(qpos) - 1)

        # initialize mjx state
        mjx_state = self.pipeline_init(qpos, qvel)
        obs = self.get_obs(mjx_state)
        reward, done, zero = jp.zeros(3)

        metrics = {}

        return State(mjx_state, obs, reward, done, metrics)

    def rewards(
        self, mjx_state: MjxState, action: jp.ndarray, next_mjx_state: MjxState
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        min_z = -1.0
        max_z = 2.0
        healthy = jp.where(mjx_state.qpos[2] < min_z, 0.0, 1.0)
        healthy = jp.where(mjx_state.qpos[2] > max_z, 0.0, healthy)
        reward = jp.array(5.0) * healthy

        return reward, healthy

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        mjx_state = state.pipeline_state

        obs = self.get_obs(mjx_state)

        # action scaling
        action_scale = 2.5
        delta_action = action - mjx_state.qpos[7:]
        delta_action *= action_scale
        action = jp.array(delta_action + mjx_state.qpos[7:])

        next_mjx_state = self.pipeline_step(mjx_state, action)
        # rewards
        reward, healthy = self.rewards(mjx_state, action, next_mjx_state)

        done = 1.0 - healthy
        return state.replace(pipeline_state=next_mjx_state, obs=obs, reward=reward, done=done)

    def get_obs(self, data: MjxState) -> jp.ndarray:
        """Returns the observation of the environment."""
        return data.qpos
