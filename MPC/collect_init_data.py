# coding: utf-8
import gym
import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *
from quanser_robots import GentlyTerminating
from quanser_robots.qube import SwingUpCtrl


env = gym.make('Qube-v0')
# env = gym.make('DoublePendulumRR-v0')   
# env = gym.make('CartpoleSwingShort-v0')   


datasets_mode0, labels_mode0 =random_dataset(env, epochs=60, samples_num=200, mode=0, prob=0.3)
datasets_mode1, labels_mode1 =random_dataset(env, epochs=60, samples_num=60, mode=1, prob=0.3)
datasets_mode_1, labels_mode_1 =random_dataset(env, epochs=60, samples_num=60,mode=-1, prob=0.3)

datasets_total = np.concatenate((datasets_mode0,datasets_mode1,datasets_mode_1), axis=0)
labels_total = np.concatenate((labels_mode0,labels_mode1,labels_mode_1), axis=0)

# example of saving data
save_datasets(datasets_total, labels_total,
                  name_train = "qube_datasets6_short", name_label="qube_labels6_short" ) exam
