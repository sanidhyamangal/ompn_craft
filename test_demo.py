import gym, os
from itertools import count
import torch
import torch.nn as nn
from gym_psketch.bots.omdec import OMdecBase
from gym_psketch import DictList, Actions
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from gym.wrappers import TimeLimit
from scripts.utils import check_if_buggy_region
import gym_psketch
import numpy as np

env = TimeLimit(gym.make('makebedfull-v0'), max_episode_steps=64)

demo_bot = gym_psketch.DemoBot(env)

demo_obs = env.reset()
done = False
for i in range(64):
    demo_action = demo_bot.get_action(demo_obs)
    print(demo_action)
    next_obs, reward, done, _ = env.step(demo_action)
    demo_obs = next_obs

    if done:
        break
