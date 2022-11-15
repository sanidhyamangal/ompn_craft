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
CSV_FILE = "experiment/eval_demo.csv"
device = "cpu"
TIME_STEPS = 100
env = TimeLimit(gym.make('makebedfull-v0'), max_episode_steps=TIME_STEPS)

state_size = 1075
action_size = 5
lr = 0.0001



def eval_demo(demo_bot, n_iters):
    planning_errors = 0
    for iter in range(n_iters):
        visited_path = np.zeros((10,10))
        demo_obs = env.reset()
        demo_bot.reset()
        obs = DictList(demo_obs)
        rewards = []
        prev_plan_errors = planning_errors

        task_done = 0
        bugs_found = 0
        for i in range(TIME_STEPS):
            try:
                demo_action = torch.tensor(demo_bot.get_action(demo_obs)).long()
            except:
                if i == 0:
                    planning_errors += 1
                    print('Iteration: {} Planning Error {}'.format(iter, planning_errors))
                else:
                    rewards[-1] += -2
                break

            action = demo_action.squeeze(0)
           
            next_obs, reward, done, _ = env.step(action.item())
            if reward:
                task_done += 1
            demo_obs = next_obs
            obs = DictList(demo_obs)
            r,visited_path = check_if_buggy_region(obs.pos, visited_path)
            if r:
                bugs_found +=1
            reward += r

            #     # print(reward)
            #     reward += 5

            if env.satisfy():
                reward += 5
            
            rewards.append(reward)
            # print(action)
            if done:
                break

        if planning_errors > prev_plan_errors:
            continue
        print('Iteration: {}, Score: {}, Task Done {}, Bugs Detected: {}'.format(iter,
                                                                                                 sum(rewards).item(),
                                                                                                 task_done, bugs_found,
                                                                                                 ))
        

    env.close()


if __name__ == '__main__':
    # if os.path.exists('model/actor.pkl'):
    #     actor = torch.load('model/actor.pkl')
    #     print('Actor Model loaded')
    # else:

    # bot = init_ILModel("bot_best.pkl", device)
    demo_bot = gym_psketch.DemoBot(env)
    eval_demo(demo_bot, 1000)
    # trainIters(actor, critic, bot, demo_bot, n_iters=10000)
