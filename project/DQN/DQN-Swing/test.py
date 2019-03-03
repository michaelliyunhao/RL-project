# coding: utf-8

from DQN import *
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument("--path", help="model name in the storage folder", type=str)
args = parser.parse_args()

env_id ="CartpoleSwingShort-v0" # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))

MODEL_PATH = "storage/swing-good.ckpt"

if args.path:
    MODEL_PATH = "storage/"+args.path

NUM_ACTIONS = 11
current_model = torch.load(MODEL_PATH)

if USE_CUDA:
    current_model = current_model.cuda()

num_frames = 10000
while True:
    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    state[4] /= 10
    episode_count = 0
    for frame_idx in range(1, num_frames + 1):
        env.render()
        time.sleep(0.01)
        epsilon = 0
        action = current_model.act(state, epsilon)
        f_action = 16*(action-(NUM_ACTIONS-1)/2)/((NUM_ACTIONS-1)/2)
        next_state, reward, done, _ = env.step(f_action)

        reward = 100*(reward-0.005)
        next_state[4] /= 10

        state = next_state
        episode_reward += reward
        episode_count +=1
        if done :#or episode_count>3000:
            print("done")
            state = env.reset()
            episode_count=0

        if frame_idx % 800 == 0:
            all_rewards.append(episode_reward)
            episode_reward = 0
            plot(frame_idx, all_rewards, losses)
            print(f_action)
            print(state," ",reward)

    print("finish")
env.close()

