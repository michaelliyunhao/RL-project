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


class MPC(object):
    def __init__(self, env, config):
        self.env = env
        mpc_config = config["mpc_config"]
        self.horizon = mpc_config["horizon"]
        self.numb_bees = mpc_config["numb_bees"]
        self.max_itrs = mpc_config["max_itrs"]
        self.gamma = mpc_config["gamma"]
        self.action_low = mpc_config["action_low"]
        self.action_high = mpc_config["action_high"]
        self.evaluator = Evaluator(self.gamma)

    def act(self, state, dynamic_model):
        self.evaluator.update(state, dynamic_model)
        optimizer = Hive.BeeHive( lower = [float(self.action_low)] * self.horizon,
                                  upper = [float(self.action_high)] * self.horizon,
                                  fun = self.evaluator.evaluate,
                                  numb_bees = self.numb_bees,
                                  max_itrs = self.max_itrs,
                                  verbose=False)
        cost = optimizer.run()
        #print("Solution: ",optimizer.solution[0])
        #print("Fitness Value ABC: {0}".format(optimizer.best))
        #Utilities.ConvergencePlot(cost)
        return optimizer.solution[0]

class Evaluator(object):
    def __init__(self, gamma=0.8):
        self.gamma = gamma

    def update(self, state, dynamic_model):
        self.state = state
        self.dynamic_model = dynamic_model

    def evaluate(self, actions):
        actions = np.array(actions)
        horizon = actions.shape[0]
        rewards = 0
        state_tmp = self.state.copy()
        for j in range(horizon):
            input_data = np.concatenate( (state_tmp,[actions[j]]) )
            state_dt = self.dynamic_model.predict(input_data)
            state_tmp = state_tmp + state_dt[0]
            rewards -= (self.gamma ** j) * self.get_reward(state_tmp, actions[j])
        return rewards

    def get_reward(self,obs, action_n):
        cos_th, sin_th, cos_al, sin_al, th_d, al_d = obs
        cos_th = min(max(cos_th, -1), 1)
        cos_al = min(max(cos_al, -1), 1)
        al=np.arccos(cos_al)
        th=np.arccos(cos_th)
        al_mod = al % (2 * np.pi) - np.pi
        action = action_n * 5
        cost = al_mod**2 + 5e-3*al_d**2 + 1e-1*th**2 + 2e-2*th_d**2 + 3e-3*action**2
        reward = np.exp(-cost)*0.02
        return reward

def model_validation(env, model, horizons, samples):
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
