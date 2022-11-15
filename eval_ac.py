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
from train_ops import log_training_events

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
CSV_FILE = "experiment/eval_ac_1.csv"
TIME_STEPS = 100
env = TimeLimit(gym.make('makebedfull-v0'), max_episode_steps=TIME_STEPS)

state_size = 1075
action_size = 5
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

def eval_ac(actor,  n_iters):
    for iter in range(n_iters):
        visited_path = np.zeros((10,10))
        demo_obs = env.reset()
        obs = DictList(demo_obs)
        state = obs.features.T
        rewards = []

        task_done = 0
        bugs_found = 0
        for i in range(TIME_STEPS):
            # env.render()
            state = torch.FloatTensor(obs.features.T).to(device)
            obs.apply(lambda _t: torch.tensor(_t).float().unsqueeze(0))
            dist = actor(state)

            action = dist.sample().squeeze(0)
            next_obs, reward, done, _ = env.step(action.item())
            if reward:
                task_done += 1
            obs = DictList(next_obs)
            r,visited_path = check_if_buggy_region(obs.pos, visited_path)
            if r:
                bugs_found +=1
            reward += r

            if env.satisfy():
                reward += 5

            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            state = obs.features.T
            if done:
                break
        print('Iteration: {}, Score: {}, Task Done {}, Bugs Detected: {}'.format(iter,sum(rewards).item(),task_done, bugs_found))
        log_training_events([iter, task_done, bugs_found],CSV_FILE)
    env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    eval_ac(actor, 1000)
