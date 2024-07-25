"""Defines the training CLI."""

from datetime import datetime
from typing import Any, NotRequired, TypedDict, Unpack

import jax.numpy as jp
from brax.io import mjcf, model
from brax.mjx.base import State as mjxState

import wandb
from ppo import train as ppo
from environment import HumanoidEnv
from omegaconf import OmegaConf

config = OmegaConf.create({
    "num_timesteps": 1000000,
    "num_evals": 100,
    "reward_scaling": 0.1,
    "episode_length": 1000,
    "normalize_observations": True,
    "action_repeat": 1,
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.99,
    "learning_rate": 1e-4,
    "entropy_cost": 0.01,
    "num_envs": 8,
    "batch_size": 64,
    "seed": 0,
    "policy_hidden_layer_sizes": [64, 64],
    "value_hidden_layer_sizes": [64, 64]
})

def train(config: dict[str, Any]) -> None:
    env = HumanoidEnv()
    times = [datetime.now()]

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(datetime.now())

    def save_model(current_step: int, make_policy: str, params: dict[str, Any]) -> None:
        model_path = "weights/model.pkl"
        model.save_params(model_path,params)
        print(f"Saved model at {current_step} to {model_path}")

    ppo(num_timesteps=config["num_timesteps"],
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
        environment=env,
        progress_fn=progress,
        policy_params_fn=save_model)

if __name__ == "__main__":
    train(config=config)
