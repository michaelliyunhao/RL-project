import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data
import pickle
from Hive import Hive
from Hive import Utilities
import time


# action 逆时针是正
# decode the obs and calculate the reward
# reward = -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)

def calc_reward(obs, action_n, log=False):
    cos_th, sin_th, cos_al, sin_al, th_d, al_d = obs
    al=np.arccos(cos_al)
    th=np.arccos(cos_th)
    al_mod = al % (2 * np.pi) - np.pi
    action = action_n * 5
    cost = al_mod**2 + 5e-3*al_d**2 + 1e-1*th**2 + 2e-2*th_d**2 + 3e-3*action**2
    reward = np.exp(-cost)*0.02
#    if log == True:
#        print("reward: ", reward, " theta: ", th, " theta_dt: ", th_d, " action: ", action)
    
    return reward

class Evaluator(object):
    def __init__(self, obs, model, gamma=0.8):
        self.obs = obs
        self.model = model
        self.gamma = gamma

    def calc_reward(self,obs, action_n, log=False):
    
        
        cos_th, sin_th, cos_al, sin_al, th_d, al_d = obs
        al=np.arccos(cos_al)
        th=np.arccos(cos_th)
        al_mod = al % (2 * np.pi) - np.pi
        action = action_n * 5
        cost = al_mod**2 + 5e-3*al_d**2 + 1e-1*th**2 + 2e-2*th_d**2 + 3e-3*action**2
        reward = np.exp(-cost)*0.02 
#        if log == True:
#            print("reward: ", reward, " theta: ", th, " theta_dt: ", th_d, " action: ", action)
    
        return reward

    def evaluate(self, actions):
        actions = np.array(actions)
        horizon = actions.shape[0]
        rewards = 0
        obs_tmp = self.obs.copy()
        for j in range(horizon):
            inputs = np.zeros([1, 7])
            inputs[0, :6] = obs_tmp.reshape(1, -1)
            inputs[0, 4] = inputs[0, 4] / 30  # scale the theta_dt
            inputs[0, 5] = inputs[0, 5] / 40  # scale the alpha_dt
            inputs[0, 6] = actions[j]
            inputs[inputs > 1] = 1
            inputs[inputs < -1] = -1
            obs_dt_n = self.model.predict(inputs)
            obs_dt_n[0, 4] *= 30  # scale the theta_dt to 30
            obs_dt_n[0, 5] *= 40  # scale the alpha_dt to 40
            obs_tmp = obs_tmp + obs_dt_n[0]
            for i in range(4):
                if obs_tmp[i]>1:
                    obs_tmp[i]=1
                elif obs_tmp[i]<-1:
                    obs_tmp[i]=-1
            rewards -= (self.gamma ** j) * self.calc_reward(obs_tmp, actions[j], log=False)
        return rewards


def select_action_hive(obs, model, horizon, numb_bees=10, max_itrs=10, gamma=0.8):
    evaluator = Evaluator(obs, model, gamma)
    hive_model = Hive.BeeHive(lower=[-1.] * horizon,
                              upper=[1.] * horizon,
                              fun=evaluator.evaluate,
                              numb_bees=numb_bees,
                              max_itrs=max_itrs,
                              verbose=False)
    cost = hive_model.run()
    #  print("Solution: ",hive_model.solution[0])
    # prints out best solution
    #  print("Fitness Value ABC: {0}".format(hive_model.best))
    # plots convergence
    #  Utilities.ConvergencePlot(cost)
    return hive_model.solution[0] * 5


def mpc_dataset_hive(env, model, samples_num, horizon=8, numb_bees=10, max_itrs=10, gamma=0.8):
    model = model.eval()
    datasets = np.zeros([samples_num, 7])
    labels = np.zeros([samples_num, 6])
    obs_old = env.reset()
    for i in range(samples_num):
        env.render()
        action = [select_action_hive(obs_old, model, horizon, numb_bees=numb_bees,
                                     max_itrs=max_itrs, gamma=gamma)]
        action = np.array(action)
        # calc_reward(obs_old,action[0]/2,log=True)
        datasets[i, 0] = obs_old[0]
        datasets[i, 1] = obs_old[1]
        datasets[i, 2] = obs_old[2] 
        datasets[i, 3] = obs_old[3] 
        datasets[i, 4] = obs_old[4] / 30.
        datasets[i, 5] = obs_old[5] / 40.
        datasets[i, 6] = action[0] / 5.
        obs, reward, done, info = env.step(action)
        labels[i, 0] = obs[0] - obs_old[0]
        labels[i, 1] = obs[1] - obs_old[1]
        labels[i, 2] = obs[2] - obs_old[2]
        labels[i, 3] = obs[3] - obs_old[3]
        labels[i, 4] = (obs[4] / 30.) - (obs_old[4] / 30.)
        labels[i, 5] = (obs[5] / 40.) - (obs_old[5] / 40.)
        obs_old = obs
    env.close()
    return datasets, labels



def model_validation(env, model, horizons, samples):
    model = model.eval()
    errors = np.zeros([samples, horizons, 9])  # alpha, cos_th, sin_th,cos_al, sin_al, theta_dt, alpha_dat, reward,theta
    for i in range(samples):
        obs = env.reset()
        actions_n = np.random.uniform(-1, 1, [horizons])
        reward_pred = 0
        reward_real = 0
        obs_pred = obs.copy()
        obs_real = obs.copy()
        for j in range(horizons):  # predicted results
            inputs = np.zeros([1, 7])
            inputs[0, :6] = obs_pred.reshape(1, -1)
            inputs[0, 4] = inputs[0, 4] / 30  # scale the theta_dt
            inputs[0, 5] = inputs[0, 5] / 40  # scale the alpha_dt
            inputs[0, 6] = actions_n[j]
            # inputs = torch.tensor(inputs).to(device).float()
            obs_dt_n = model.predict(inputs)  # model(inputs)
            # obs_dt_n = obs_dt_n.cpu().detach().numpy().reshape(1,3)
            obs_dt_n[0, 4] *= 30  # scale the theta_dt to 30
            obs_dt_n[0, 5] *= 40  # scale the theta_dt to 40
            obs_pred = obs_pred + obs_dt_n[0]
            reward_pred += calc_reward(obs_pred, actions_n[j], log=False)

            obs_real, reward_tmp, done, info = env.step(np.array([actions_n[j]]))
            reward_real += reward_tmp

            error_tmp = obs_real - obs_pred.reshape(6, )
            errors[i, j, 1:7] = abs(error_tmp)
            errors[i, j, 7] = abs(reward_real - reward_pred)
            cos_theta_pred = obs_pred[0]
            if cos_theta_pred > 1:
                cos_theta_pred = 1
            elif cos_theta_pred < -1:
                cos_theta_pred = -1
            sin_theta_pred = obs_pred[1]
            if sin_theta_pred > 1:
                sin_theta_pred = 1
            elif sin_theta_pred < -1:
                sin_theta_pred = -1
            cos_alpha_pred = obs_pred[2]
            if cos_alpha_pred > 1:
                cos_alpha_pred = 1
            elif cos_alpha_pred < -1:
                cos_alpha_pred = -1
            sin_alpha_pred = obs_pred[3]
            if sin_alpha_pred > 1:
                sin_alpha_pred = 1
            elif sin_alpha_pred < -1:
                sin_alpha_pred = -1
            alpha_pred = np.arccos(cos_alpha_pred)
            theta_pred = np.arccos(cos_theta_pred)
            errors[i, j, 0] = abs(np.arccos(obs_real[2]) - alpha_pred)
            errors[i, j, 8] = abs(np.arccos(obs_real[0]) - theta_pred)
    errors_mean = np.mean(errors, axis=0)
    errors_max = np.max(errors, axis=0)
    errors_min = np.min(errors, axis=0)
    errors_std = np.min(errors, axis=0)
    return errors_mean, errors_max, errors_min, errors_std

def plot_model_validation(env, model, horizons, samples, mode="mean"):
    errors = np.zeros([horizons, 9])
    # for i in range(1,horizons+1):
    if mode == "mean":
        errors = model_validation(env, model, horizons, samples)[0]
    if mode == "max":
        errors = model_validation(env, model, horizons, samples)[1]
    if mode == "min":
        errors = model_validation(env, model, horizons, samples)[2]
    if mode == "std":
        errors = model_validation(env, model, horizons, samples)[3]
    plt.ioff()
    plt.figure(figsize=[8, 4])
    plt.plot(np.arange(1, horizons + 1), errors[:, 0] * 180 / 3.1415926)
    plt.title("alpha Angle Error")
    plt.figure(figsize=[8, 4])
    plt.plot(np.arange(1, horizons + 1), errors[:, 8] * 180 / 3.1415926)
    plt.title("theta Angle Error")
    plt.figure(figsize=[8, 4])
    plt.plot(np.arange(1, horizons + 1), errors[:, 1], 'r', label='costh')
    plt.plot(np.arange(1, horizons + 1), errors[:, 2], 'g', label='sinth')
    plt.plot(np.arange(1, horizons + 1), errors[:, 5] / 30, 'b', label='theta_dt')
    plt.plot(np.arange(1, horizons + 1), errors[:, 6] / 40, 'g', label='alpha_dt')
    plt.legend()
    plt.title("State Error")
    plt.figure(figsize=[8, 4])
    plt.plot(np.arange(1, horizons + 1), errors[:, 7])
    plt.title("Reward Error")
    plt.show()
