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

import wandb
from scripts.mjx.run_mjx import DEFAULT_REWARD_PARAMS, get_env

InferenceFn = Callable[[jp.ndarray, jp.ndarray], tuple[jp.ndarray, jp.ndarray]]


def mjx_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
) -> list[mjxState]:
    """Rollout a trajectory using MJX.

    It is worth noting that env, a Brax environment, is expected to implement MJX
    in the background. See default_humanoid_env for reference.

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed

    Returns:
        A list of pipeline states of the policy rollout
    """
    # print(f"Rolling out {n_steps} steps with MJX")
    logger.info("Rolling out %d steps with MJX", n_steps)
    reset_fn = j.jit(env.reset)
    step_fn = j.jit(env.step)
    inference_fn = j.jit(inference_fn)
    rng = j.random.PRNGKey(seed)

    state = reset_fn(rng)
    rollout = [state.pipeline_state]
    for i in tqdm(range(n_steps)):
        act_rng, rng = j.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        rollout.append(state.pipeline_state)

        if state.done:
            state = reset_fn(rng)

    return rollout


def render_mjx_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """Rollout a trajectory using MuJoCo and render it.

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
        width: width of rendered frame in pixels
        height: height of rendered frame in pixels

    Returns:
        A list of renderings of the policy rollout with dimensions (T, H, W, C)
    """
    rollout = mjx_rollout(env, inference_fn, n_steps, render_every, seed)
    images = env.render(rollout[::render_every], camera="side", width=width, height=height)

    return np.array(images)


logger = logging.getLogger(__name__)

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"


def play(config: dict[str, Any], n_steps: int, render_every: int, width: int, height: int) -> None:
    wandb.init(
        project=config.get("project_name", "robotic_locomotion_training") + "_test",
        name=config.get("experiment_name", "ppo-training") + "_test",
    )

    # Load environment
    env = get_env(
        name=config.get("env_name", "default_humanoid"),
        reward_params=config.get("reward_params", DEFAULT_REWARD_PARAMS),
        terminate_when_unhealthy=config.get("terminate_when_unhealthy", True),
        reset_noise_scale=config.get("reset_noise_scale", 1e-2),
        exclude_current_positions_from_observation=config.get("exclude_current_positions_from_observation", True),
        log_reward_breakdown=config.get("log_reward_breakdown", True),
    )
    # Reset environment
    rng = j.random.PRNGKey(0)
    env.reset(rng)

    logger.info(
        "Loaded environment %s with env.observation_size: %s and env.action_size: %s",
        config.get("env_name", ""),
        env.observation_size,
        env.action_size,
    )

    # Loading params
    if args.params_path is not None:
        model_path = args.params_path
    else:
        # model_path = "weights/" + config.get("project_name", "model") + config.get("experiment_name", "model") + ".pkl"
        model_path = "weights/model.pkl"
    params = model.load_params(model_path)

    def normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x

    if config.get("normalize_observations", False):
        normalize = (
            running_statistics.normalize
        )  # NOTE: very important to keep training & test normalization consistent

    policy_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=config["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=config["value_hidden_layer_sizes"],
    )
    params = (params[0], params[1].policy)
    # Params are a tuple of (processor_params, PolicyNetwork)
    inference_fn = ppo_networks.make_inference_fn(policy_network)(params)
    print(f"Loaded params from {model_path}")
    print(inference_fn)

    # rolling out a trajectory
    images_thwc = render_mjx_rollout(env, inference_fn, n_steps, render_every, width=width, height=height)
    print(f"Rolled out {len(images_thwc)} steps")

    # render the trajectory
    images_tchw = np.transpose(images_thwc, (0, 3, 1, 2))

    fps = int(1 / env.dt)
    print(f"Writing video to video.mp4 with fps={fps}")
    media.write_video("video.mp4", images_thwc, fps=fps)
    video = wandb.Video(images_tchw, fps=fps, format="mp4")
    wandb.log({"video": video})


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run PPO training with specified config file.")
    # parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    parser.add_argument("--use_mujoco", action="store_true", help="Use mujoco instead of mjx for rendering")
    parser.add_argument("--params_path", type=str, default=None, help="Path to the params file")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of steps to rollout")
    parser.add_argument("--render_every", type=int, default=2, help="Render every nth step")
    parser.add_argument("--width", type=int, default=320, help="width in pixels")
    parser.add_argument("--height", type=int, default=240, help="height in pixels")
    args = parser.parse_args()

    # Load config file
    args.config = "scripts/mjx/config.yaml"
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    play(config, args.n_steps, args.render_every, args.width, args.height)