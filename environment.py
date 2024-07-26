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
    def __init__(self, **kwargs: Unpack[EnvKwargs]) -> None:
        # load the model which determines the robot + environment,
        # including the physics and observation/action spaces
        mj_model = mujoco.MjModel.from_xml_path("assets/stompylegs.xml")

        # extracts qpos found in the xml file
        # qpos[:7] is the root position and orientation
        # qpos[7:] is the joint angles
        self.initial_qpos = jp.array(mj_model.keyframe("default").qpos)

        # action size is the number of dof/movable-joints
        self._action_size = mj_model.nu

        # create a brax system from the mj model
        sys = mjcf.load_model(mj_model)

        ## keyword args
        physics_steps_per_control_step = 4
        # n_frames: the number of times to step the physics pipeline for each
        # environment step
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        # backend: string specifying the physics pipeline
        kwargs["backend"] = "mjx"

        # calls super to PipelineEnv, a API to drive a brax system for training and inference
        super().__init__(sys, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        qpos = self.initial_qpos
        qvel = jp.zeros(len(qpos) - 1)

        # initialize mjx state
        mjx_state = self.pipeline_init(qpos, qvel)

        # get current position (qpos)
        obs = self.get_obs(mjx_state)

        # done -> 1.0 if env is terminated, 0.0 otherwise
        reward, done = jp.zeros(2)

        # metrics can be set up to track forward velocity, etc.
        metrics = {}

        return State(mjx_state, obs, reward, done, metrics)

    def rewards(
        self, mjx_state: MjxState, action: jp.ndarray, next_mjx_state: MjxState
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        # if robot falls below min_z or above max_z, reward is 0, otherwise 5
        # healthy is 1 if robot is within min_z and max_z, 0 otherwise
        min_z = -1.0
        max_z = 2.0
        healthy = jp.where(mjx_state.qpos[2] < min_z, 0.0, 1.0)
        healthy = jp.where(mjx_state.qpos[2] > max_z, 0.0, healthy)
        reward = jp.array(5.0) * healthy

        # other rewards and termination conditions can be added here

        return reward, healthy

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        mjx_state = state.pipeline_state

        # get current position (qpos)
        obs = self.get_obs(mjx_state)

        # action scaling
        # NOTE: actions is position based, so we need to scale them to be deltas, then add delta
        action_scale = 2.5
        delta_action = action - mjx_state.qpos[7:]
        delta_action *= action_scale
        action = jp.array(delta_action + mjx_state.qpos[7:])

        # step the physics
        next_mjx_state = self.pipeline_step(mjx_state, action)

        # rewards
        reward, healthy = self.rewards(mjx_state, action, next_mjx_state)
        done = 1.0 - healthy

        # next state becomes current state
        return state.replace(pipeline_state=next_mjx_state, obs=obs, reward=reward, done=done)

    def get_obs(self, data: MjxState) -> jp.ndarray:
        """Returns the observation of the environment."""
        return data.qpos
