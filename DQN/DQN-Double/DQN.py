# coding: utf-8

import random

import torch.nn as nn
import torch.autograd as autograd
import torch
import torch.optim as optim

import matplotlib.pyplot as plt
from quanser_robots.common import GentlyTerminating
from collections import deque
import gym
import numpy as np
import torch
import yaml
import os
import copy

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def print_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        config = yaml.load(f)
        print("************************")
        print("*** model configuration ***")
        print(yaml.dump(config["model_config"], default_flow_style=False, default_style=''))
        print("*** train configuration ***")
        print(yaml.dump(config["training_config"], default_flow_style=False, default_style=''))
        print("************************")
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def anylize_env(env, test_episodes = 100,max_episode_step = 500, render = False):
    print("state space shape: ", env.observation_space.shape)
    print("state space lower bound: ", env.observation_space.low)
    print("state space upper bound: ", env.observation_space.high)
    print("action space shape: ", env.action_space.shape)
    print("action space lower bound: ", env.action_space.low)
    print("action space upper bound: ", env.action_space.high)
    print("reward range: ", env.reward_range)
    rewards = []
    steps = []
    for episode in range(test_episodes):
        env.reset()
        step = 0
        episode_reward = 0
        for _ in range(max_episode_step):
            if render:
                env.render()
            step += 1
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
               # print("done with step: %s " % (step))
                break
        steps.append(step)
        rewards.append(episode_reward)
    env.close()
    print("Randomly sample actions for %s episodes, with maximum %s steps per episodes"
          % (test_episodes, max_episode_step))
    print(" average reward per episode: %s, std: %s " % (np.mean(rewards), np.std(rewards) ))
    print(" average steps per episode: ", np.mean(steps))
    print(" average reward per step: ", np.sum(rewards)/np.sum(steps))

class MLP(nn.Module):
    def __init__(self, n_input=4, n_output=3, n_h=1, size_h=256):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()

        assert n_h >= 1, "h must be integer and >= 1"

        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)

        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.relu(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.relu(out)
        out = self.fc_out(out)
        return out

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def plot_fig(episode, all_rewards,avg_rewards, losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('Reward Trend with %s Episodes' % (episode))
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(all_rewards, 'b')
    plt.plot(avg_rewards, 'r')
    plt.subplot(122)
    plt.title('Loss Trend with %s Episodes' % (episode))
    plt.plot(losses)
    plt.show()

def plot(frame_idx, rewards, losses):
    plt.clf()
    plt.close()
    plt.ion()
    plt.figure(figsize=(12 ,5))
    plt.subplot(131)
    plt.title('episode %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.pause(0.0001)

def save_fig(episode, all_rewards, avg_rewards, losses, epsilon, number = 0):
    plt.figure(figsize=(8 ,5))
    plt.title('Reward Trend with %s Episodes' % (episode))
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(all_rewards,'b')
    plt.plot(avg_rewards,'r')
    plt.savefig("storage/reward-"+str(number)+".png")
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('Loss Trend with Latest %s Steps' % (1200))
    plt.plot(losses[-1200:])
    plt.subplot(122)
    plt.title('Epsilon with %s Episodes' % (episode))
    plt.plot(epsilon)
    plt.savefig("storage/loss-"+str(number)+".png")

class Policy(object):
    def __init__(self, env,config):
        # load the configuration file
        model_config = config["model_config"]
        self.n_states = env.observation_space.shape[0]
        self.n_actions = model_config["n_actions"]
        self.use_cuda = model_config["use_cuda"]
        if model_config["load_model"]:
            self.model = torch.load(model_config["model_path"])
        else:
            self.model = MLP(self.n_states, self.n_actions, model_config["n_hidden"],
                             model_config["size_hidden"])
        if self.use_cuda:
            self.model=self.model.cuda()

        training_config = config["training_config"]
        self.current_model = self.model
        self.target_model = copy.deepcopy(self.model)
        self.update_target()

        self.memory_size = training_config["memory_size"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]
        self.gamma = training_config["gamma"]
        self.optimizer = optim.Adam(self.current_model.parameters(),lr= self.lr)
        self.replay_buffer = ReplayBuffer(self.memory_size)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.model.forward(state)
            action  = q_value.max(1)[1].cpu().detach().numpy(  )  # .data[0]
        else:
            action = np.random.randint(size=(1,) ,low=0, high=self.n_actions)
        return action

    # update the target network with current network parameters
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    # change the learning rate during the training process
    def update_lr(self, lr):
        self.lr=lr
        self.optimizer = optim.Adam(self.current_model.parameters(),lr= self.lr)

    def train(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_model(self, model_path = "storage/test.ckpt"):
        torch.save(self.model, model_path)
