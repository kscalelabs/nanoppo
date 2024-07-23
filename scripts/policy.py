from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

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

plt.rcParams["figure.figsize"] = (10, 5)
# sets a consistent plot size


class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 16
        # dimension of first hidden layer
        hidden_space2 = 32
        # dimension of the second hidden layer
        self.policy_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, action_space_dims),
            # nn.Softmax(dim = -1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_net(obs)


class Reinforce:
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.eps = 1e-6

        self.probs = []
        self.rewards = []

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0
        )  # Convert state to tensor and add batch dimension

        prob = self.net(state_tensor)
        probabilities = prob.squeeze(0)  # Get probabilities from network

        # Detach and convert to NumPy array for sampling
        probabilities_np = probabilities.detach().numpy()
        # sample = np.random.choice(elements, p=probabilities_np)
        action = np.random.rand(12)
        tot_prob = 0
        for i in range(12):
            distrib = Normal(probabilities_np[i] + self.eps, 1 + self.eps)
            action[i] = distrib.sample()
            prob = distrib.log_prob(torch.tensor(action[i]))
            tot_prob += prob
        tot_prob /= 12

        self.probs.append(tot_prob)
        # log_prob = torch.log(probabilities[sample])  # Calculate log probability of the selected action
        # self.probs.append(log_prob)  # Store log probability for gradient computation
        return action

    def update(self):
        # take dot product of rewards and log probabilities
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        for logprob, reward in zip(self.probs, self.rewards):
            loss = loss - (logprob * len(self.rewards))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []
