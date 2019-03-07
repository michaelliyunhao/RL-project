import torch
import pickle
import os
import yaml
import numpy as np

def load_dataset(f1 = './storage/datasets_hive.pkl', f2 = './storage/labels_hive.pkl'):
    print("Load datas from %s" % f1)
    with open(f1, 'rb') as f:
        train_datasets = pickle.load(f)
    print("Load labels from %s" % f2)
    with open(f2, 'rb') as f:
        train_labels = pickle.load(f)
    print("Datasets shape: %s" % (np.shape(train_datasets)))
    return train_datasets, train_labels

def save_datasets(train_datasets, train_labels,
                  name_train = "datasets1", name_label="labels1" ):
    datasets_path = "./storage/" + name_train + ".pkl"
    labels_path = "./storage/" + name_label + ".pkl"
    with open(datasets_path, 'wb') as f:  # open file with write-mode
        pickle.dump(train_datasets, f, -1)  # serialize and save object

    with open(labels_path, 'wb') as f:  # open file with write-mode
        pickle.dump(train_labels, f, -1)  # serialize and save object

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