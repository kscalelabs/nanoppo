#!/usr/bin/env python

"""Runs the training loop for the model."""
from __future__ import annotations
import argparse
import logging
from scripts.walking_env import HumanoidEnv


import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
import random 
import math
from typing import Optional, Union
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from scripts.policy import Reinforce

logger = logging.getLogger(__name__)

# traning with random actions
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
        _, _, reward, done, info = env.step(action)
        logger.info("Action: %s, Reward: %s, Done: %s, Info: %s", action, reward, done, info)

        if done:
            env.reset()

    env.close()
    
# training using Reinforce
def train() -> None: 
    plt.rcParams["figure.figsize"] = (10,5)
    # sets a consistent plot size

    env = HumanoidEnv(render_mode="human")
    # rgb_array

    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

    total_num_episodes = int(5e3) 

    obs_space_dims = 361

    act_space_dims = 12

    rewards_over_seeds = []

    for seed in [1]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = Reinforce(obs_space_dims, act_space_dims)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = wrapped_env.reset(seed=seed)

            done = False
            while not done:
                action = agent.sample_action(obs)
            # print(action)
                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)
            # print(agent)
                # End the episode when either truncated or terminated is true
                #  - truncated: The episode duration reaches max number of timesteps
                #  - terminated: Any of the state space values is no longer finite.
                done = terminated or truncated
                # if (terminated):
                #     print("terminated")
                    
                # if (truncated):
                #     print("truncated")
            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()

            if episode % 100 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)
        rewards_over_seeds.append(reward_over_episodes)

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for Walking"
    )
    plt.savefig('walking_rewards.png')

if __name__ == "__main__":
    # python environment.py
    train()
