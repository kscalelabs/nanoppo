"""Defines the training CLI."""

import argparse
import functools
import os
from datetime import datetime
from typing import Any, NotRequired, TypedDict, Unpack

import jax
import jax.numpy as jp
import mujoco
import yaml
from brax import base, envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, model
from brax.mjx.base import State as mjxState
from etils import epath

import wandb
from mjx.ppo import train as ppo
from mjx.rewards import DEFAULT_REWARD_PARAMS, RewardParams, get_reward_fn


class EnvKwargs(TypedDict):
    sys: base.System
    backend: NotRequired[str]
    n_frames: NotRequired[int]
    debug: NotRequired[bool]


class StompyEnv(PipelineEnv):
    """An environment for humanoid body position, velocities, and angles."""

    def __init__(
        self,
        reward_params: RewardParams = DEFAULT_REWARD_PARAMS,  # TODO: change rewards
        terminate_when_unhealthy: bool = True,
        # reset_noise_scale: float = 1e-2,
        reset_noise_scale: float = 0,
        exclude_current_positions_from_observation: bool = True,
        log_reward_breakdown: bool = True,
        **kwargs: Unpack[EnvKwargs],
    ) -> None:
        path = os.getenv("MODEL_DIR", "") + "assets/stompylegs.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 4  # Should find way to perturb this value in the future
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._reward_params = reward_params
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._log_reward_breakdown = log_reward_breakdown

        print("initial_qpos", mj_model.keyframe("default").qpos)
        self.initial_qpos = mj_model.keyframe("default").qpos
        self.reward_fn = get_reward_fn(self._reward_params, self.dt, include_reward_breakdown=True)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state.

        Args:
            rng: Random number generator seed.

        Returns:
            The initial state of the environment.
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.initial_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        mjx_state = self.pipeline_init(qpos, qvel)
        assert isinstance(mjx_state, mjxState), f"mjx_state is of type {type(mjx_state)}"

        obs = self._get_obs(mjx_state, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        for key in self._reward_params.keys():
            metrics[key] = zero

        return State(mjx_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics.

        Args:
            state: The current state of the environment.
            action: The action to take.

        Returns:
            A tuple of the next state, the reward, whether the episode has ended, and additional information.
        """
        mjx_state = state.pipeline_state
        assert mjx_state, "state.pipeline_state was recorded as None"
        # TODO: determine whether to raise an error or reset the environment

        next_mjx_state = self.pipeline_step(mjx_state, action)

        assert isinstance(next_mjx_state, mjxState), f"next_mjx_state is of type {type(next_mjx_state)}"
        assert isinstance(mjx_state, mjxState), f"mjx_state is of type {type(mjx_state)}"
        # mlutz: from what I've seen, .pipeline_state and .pipeline_step(...)
        # actually return an brax.mjx.base.State object however, the type
        # hinting suggests that it should return a brax.base.State object
        # brax.mjx.base.State inherits from brax.base.State but also inherits
        # from mjx.Data, which is needed for some rewards

        obs = self._get_obs(mjx_state, action)
        reward, is_healthy, reward_breakdown = self.reward_fn(mjx_state, action, next_mjx_state)

        if self._terminate_when_unhealthy:
            done = 1.0 - is_healthy
        else:
            done = jp.array(0)

        state.metrics.update(
            x_position=next_mjx_state.subtree_com[1][0],
            y_position=next_mjx_state.subtree_com[1][1],
            distance_from_origin=jp.linalg.norm(next_mjx_state.subtree_com[1]),
            x_velocity=(next_mjx_state.subtree_com[1][0] - mjx_state.subtree_com[1][0]) / self.dt,
            y_velocity=(next_mjx_state.subtree_com[1][1] - mjx_state.subtree_com[1][1]) / self.dt,
        )

        if self._log_reward_breakdown:
            for key, val in reward_breakdown.items():
                state.metrics[key] = val

        return state.replace(pipeline_state=next_mjx_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjxState, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles.

        Args:
            data: The current state of the environment.
            action: The current action.

        Returns:
            Observations of the environment.
        """
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )


environments = {"stompy": StompyEnv}


def get_env(name: str, **kwargs: Any) -> envs.Env:  # noqa: ANN401
    envs.register_environment(name, environments[name])
    return envs.get_environment(name, **kwargs)


def train(config: dict[str, Any]) -> None:
    wandb.init(
        project=config.get("project_name", "robotic-locomotion-training"),
        name=config.get("experiment_name", "ppo-training"),
    )

    print(f"config: {config}")
    print(f'training on {config["num_envs"]} environments')

    env = get_env(
        # name=config.get("env_name", "default_humanoid"),
        name="stompy",
        reward_params=config.get("reward_params", DEFAULT_REWARD_PARAMS),
        terminate_when_unhealthy=config.get("terminate_when_unhealthy", True),
        reset_noise_scale=config.get("reset_noise_scale", 1e-6),
        exclude_current_positions_from_observation=config.get("exclude_current_positions_from_observation", True),
        log_reward_breakdown=config.get("log_reward_breakdown", True),
    )
    print(f'Env loaded: {config.get("env_name", "could not find environment")}')

    train_fn = functools.partial(
        ppo,
        num_timesteps=config["num_timesteps"],
        num_evals=config["num_evals"],
        reward_scaling=config["reward_scaling"],
        episode_length=config["episode_length"],
        normalize_observations=config["normalize_observations"],
        action_repeat=config["action_repeat"],
        unroll_length=config["unroll_length"],
        num_minibatches=config["num_minibatches"],
        num_updates_per_batch=config["num_updates_per_batch"],
        discounting=config["discounting"],
        learning_rate=config["learning_rate"],
        entropy_cost=config["entropy_cost"],
        num_envs=config["num_envs"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        policy_hidden_layer_sizes=config["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=config["value_hidden_layer_sizes"],
    )

    times = [datetime.now()]

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:  # noqa: ANN401
        times.append(datetime.now())

        wandb.log({"steps": num_steps, "epoch_time": (times[-1] - times[-2]).total_seconds(), **metrics})

    def save_model(current_step: int, make_policy: str, params: dict[str, Any]) -> None:  # noqa: ANN401
        model_path = (
            # "weights/" + config.get("project_name", "model") + config.get("experiment_name", "ppo-training") + ".pkl"
            "weights/model.pkl"
        )
        breakpoint()
        model.save_params(model_path, params)
        print(f"Saved model at step {current_step} to {model_path}")

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, policy_params_fn=save_model)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")


if __name__ == "__main__":
    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Run PPO training with specified config file.")
    # parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    # args = parser.parse_args()

    # Load config from YAML file
    config = "mjx/config.yaml"
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    train(config)