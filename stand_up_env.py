"""Defines the environment used for training the agent."""
import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# HUMANOID_STANDUP_XML = "https://raw.githubusercontent.com/openai/gym/master/gym/envs/mujoco/assets/humanoidstandup.xml"
HUMANOID_STANDUP_XML = "assets/stompylegs.xml"
class HumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        observation_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        # xml_path = self._download_mujoco(HUMANOID_STANDUP_XML)
        xml_path = self.get_local_mujoco(HUMANOID_STANDUP_XML)
        super().__init__(str(xml_path), 5, observation_space=observation_space, **kwargs)

    def get_local_mujoco(self, local_path: str) -> Path:
        return Path(local_path).resolve()
    
    # def _download_mujoco(self, url: str, name: str | None = None) -> Path:
    #     if name is None:
    #         name = url.split("/")[-1]

    #     cache_dir = Path(os.environ.get("MUJOCO_ARTIFACTS_DIR", Path.home() / ".cache" / "mujoco"))
    #     cache_dir = cache_dir.expanduser().resolve()
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #     cache_path = cache_dir / name
    #     if cache_path.exists():
    #         return cache_path

    #     # Downloads the file using the provided URL.
    #     response = requests.get(url)
    #     response.raise_for_status()
    #     with cache_path.open("wb") as file:
    #         file.write(response.content)
    #     return cache_path

    def _get_obs(self) -> NDArray[np.float64]:
        data = self.data
        # breakpoint()
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )
        

    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, Any]]:
        self.do_simulation(action, self.frame_skip)

        pos_after = self.data.qpos[2]
        data = self.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            reward,
            False,
            False,
            {
                "reward_linup": uph_cost,
                "reward_quadctrl": -quad_ctrl_cost,
                "reward_impact": -quad_impact_cost,
            },
        )

    def reset_model(self) -> NDArray[np.float64]:
        c = 0.01
        self.init_qpos = self.model.keyframe("default").qpos
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self) -> None:
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20


def run_environment_adhoc() -> None:
    """Runs the humanoid environment with some random actions."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    env = HumanoidEnv(render_mode="human")
    env.reset()

    for _ in range(args.steps):
        action = env.action_space.sample()
        action = (action)*0.0
        _, _, reward, done, info = env.step(action)
        logger.info("Action: %s, Reward: %s, Done: %s, Info: %s", action, reward, done, info)

        if done:
            env.reset()

    env.close()


if __name__ == "__main__":
    # python environment.py
    run_environment_adhoc()
