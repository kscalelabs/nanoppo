"""Play a trained PPO agent in a specified environment."""

from typing import Any, Callable

import jax as j
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
from brax.io import model
from brax.mjx.base import State as mjxState
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from tqdm import tqdm

from environment import HumanoidEnv
from train import config

InferenceFn = Callable[[jp.ndarray, jp.ndarray], tuple[jp.ndarray, jp.ndarray]]


def mjx_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
) -> list[mjxState]:
    reset_fn = j.jit(env.reset)
    step_fn = j.jit(env.step)
    inference_fn = j.jit(inference_fn)
    rng = j.random.PRNGKey(seed)
    state = reset_fn(rng)
    rollout = [state.pipeline_state]
    for i in tqdm(range(n_steps)):
        action_rng, rng = j.random.split(rng)
        # NOTE: actions is position based:
        action, _ = inference_fn(state.obs, action_rng)
        state = step_fn(state, action)
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
    rollout = mjx_rollout(env, inference_fn, n_steps, render_every, seed)
    images = env.render(rollout[::render_every], camera="side", width=width, height=height)
    return np.array(images)


def play(config: dict[str, Any], n_steps: int, render_every: int, width: int, height: int) -> None:
    ## initialize and reset the environment
    env = HumanoidEnv()
    rng = j.random.PRNGKey(0)
    env.reset(rng)

    ## load inference function
    params = model.load_params("model.pkl")
    # params -> (processor_params, PolicyNetwork)
    params = (params[0], params[1].policy)

    normalize = running_statistics.normalize
    policy_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=config["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=config["value_hidden_layer_sizes"],
    )
    inference_fn = ppo_networks.make_inference_fn(policy_network)(params)

    ## render the video's images
    images_thwc = render_mjx_rollout(env, inference_fn, n_steps, render_every, width=width, height=height)
    fps = int(1 / env.dt)

    ## render the trajectory
    print(f"Find video at video.mp4 with fps={fps}")
    media.write_video("video.mp4", images_thwc, fps=fps)


if __name__ == "__main__":
    play(config, n_steps=1000, render_every=2, width=640, height=480)
