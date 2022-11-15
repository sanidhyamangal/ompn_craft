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

# def init_ILModel(model_ckpt, device):
#     with open(model_ckpt, 'rb') as f:
#         bot = torch.load(f, map_location=device)
#     bot.eval()
#     return bot

# Call reset
# def reset_mem(bot, env):
#     with torch.no_grad():
#         mems = bot.init_memory(torch.tensor([env.env_id]).long())
#     return mems


# def get_action(bot, obs, mems, env):
#     # obs = torch.tensor(obs).float()
#     env_id = torch.tensor([env.env_id]).long()
#     with torch.no_grad():
#         output = bot.get_action(obs, env_id, mems, 'greedy')
#     if bot.is_recurrent:
#         mems = output.mems
#     return output.actions, mems

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
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


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(rewards, masks, gamma=0.95):
    R = 0
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, bot, demo_bot, n_iters):
    print("Entered Training loop")
    planning_errors = 0
    lamb = 1.0
    warmup = 500
    exp_decay_factor = 0.999
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        visited_path = np.zeros((10,10))
        demo_obs = env.reset()
        demo_bot.reset()
        obs = DictList(demo_obs)
        state = obs.features.T
        log_probs = []
        values = []
        rewards = []
        masks = []
        prev_action = None

        prev_plan_errors = planning_errors

        demo_actions = []
        dist_logits = []

        il_actions = 0
        entropy = 0
        # mems = reset_mem(bot, env)



        task_done = 0
        bugs_found = 0
        all_four_bugs = False
        for i in range(TIME_STEPS):
            # env.render()
            # il_action = None
            state = torch.FloatTensor(obs.features.T).to(device)
            #
            obs.apply(lambda _t: torch.tensor(_t).float().unsqueeze(0))
            dist, value = actor(state), critic(state)

            #
            action = dist.sample()
            # print(action)
            # if action == 0:
            # action, mems = get_action(bot, obs, mems, env)
            # print(demo_obs)
            try:
                demo_action = torch.tensor(demo_bot.get_action(demo_obs)).long()
            except:
                if i == 0:
                    planning_errors += 1
                    print('Iteration: {} Planning Error {}'.format(iter, planning_errors))
                else:
                    rewards[-1] += -2
                break

            if iter < warmup:
                action = demo_action
            demo_actions.append(demo_action)
            action = action.squeeze(0)
                # il_action = True
            # il_actions +=1
            # else:action -= 1
            # print(dist)
            next_obs, reward, done, _ = env.step(action.item())
            # if not il_action:action +=1
            if reward:
                task_done += 1
            # reward *= task_done
            demo_obs = next_obs
            obs = DictList(demo_obs)
            r,visited_path = check_if_buggy_region(obs.pos, visited_path)
            if r:
                bugs_found +=1
            reward += r

            # if bugs_found == 4 and not all_four_bugs:
            #     all_four_bugs = True
            #     # print(reward)
            #     reward += 5

            if env.satisfy():
                reward += 5
            # print(action)

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            dist_logits.append(dist.logits)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))


            state = obs.features.T


            if done:
                break

        if planning_errors > prev_plan_errors:
            continue
        print('Iteration: {}, Score: {}, Task Done {}, Bugs Detected: {}, IL Actions: {}'.format(iter,
                                                                                                 sum(rewards).item(),
                                                                                                 task_done, bugs_found,
                                                                                                 il_actions))

        # next_state = torch.FloatTensor(obs.features.T).to(device)
        # next_value = critic(next_state)
        returns = compute_returns(rewards, masks)

        demo_actions = torch.tensor(demo_actions)
        dist_logits = torch.stack(dist_logits)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean() + lamb * F.cross_entropy(input=dist_logits, target=demo_actions,
                                                                     reduction='mean',
                                                                     ignore_index=5)
        critic_loss = advantage.pow(2).mean()


        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

        if iter >= warmup:
            lamb *= exp_decay_factor
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    # if os.path.exists('model/actor.pkl'):
    #     actor = torch.load('model/actor.pkl')
    #     print('Actor Model loaded')
    # else:
    actor = Actor(state_size, action_size).to(device)
    # if os.path.exists('model/critic.pkl'):
    #     critic = torch.load('model/critic.pkl')
    #     print('Critic Model loaded')
    # else:
    critic = Critic(state_size, action_size).to(device)
    # bot = init_ILModel("bot_best.pkl", device)
    demo_bot = gym_psketch.DemoBot(env)
    bot=0
    trainIters(actor, critic, bot, demo_bot, n_iters=10000)
