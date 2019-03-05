# coding: utf-8

import math
from DQN import *
import numpy as np
import torch
import torch.optim as optim


torch.manual_seed(1234)
np.random.seed(9)
NUM_ACTIONS = 11
env_id ="Qube-v0" # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))
current_model = DQN(env.observation_space.shape[0], NUM_ACTIONS)
target_model  = DQN(env.observation_space.shape[0], NUM_ACTIONS)
current_model = torch.load("storage/qube_test.ckpt")
target_model  = torch.load("storage/qube_test.ckpt")

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()

replay_buffer = ReplayBuffer(100000)

#weight_decay = 3e-5

learning_rate = 3e-3
optimizer = optim.Adam(current_model.parameters() ,lr= learning_rate) #,weight_decay = weight_decay)
epsilon_start = 0.9
epsilon_final = 0.2
epsilon_decay = 3000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

num_frames = 10000
batch_size = 64
gamma      = 0.99

trainer = Trainer(current_model,target_model,replay_buffer,gamma,optimizer)
trainer.update_target()

losses = []
all_rewards = []
for i in range(400):
    if i % 10==0:
        learning_rate = learning_rate * (0.1)
        optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
    episode_reward = 0
    state = env.reset()
    state[4:6]/=20
    episode_count = 0
    for frame_idx in range(1, num_frames + 1):
        env.render()
        epsilon = 0.2 #epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)
        f_action = 5*(action-(NUM_ACTIONS-1)/2)/((NUM_ACTIONS-1)/2)
        next_state, reward, done, _ = env.step(f_action)

        reward = 100*(reward-0.005)
        next_state[4:6]/=20

        replay_buffer.push(state, action[0], reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_count +=1
        if done or episode_count>3000:
            state = env.reset()
            episode_count=0

        if len(replay_buffer) > batch_size:
            loss = trainer.compute_td_loss(batch_size)
            losses.append(loss.item())

        if frame_idx % 200 == 0:
            all_rewards.append(episode_reward)
            episode_reward = 0
            if frame_idx % 9999 == 0:
                plot(frame_idx, all_rewards, losses)
                print(f_action)
                print(state," ",reward)
        if frame_idx % 800 == 0:
            trainer.update_target()

    torch.save(current_model,"storage/qube_test.ckpt")
    print("finish")
env.close()
