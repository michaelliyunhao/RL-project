# coding: utf-8

import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C  
# REF is gaussion kenerl function
from mpl_toolkits.mplot3d import Axes3D  
# keep the data visible


# examples of datasets and labels path
datasets_path = './storage/qube_datasets6_short.pkl'
labels_path = './storage/qube_labels6_short.pkl'

train_datasets ,train_labels = load_dataset(datasets_path, labels_path)
train_datasets, train_labels = normlize_datasets(train_datasets,train_labels)
X_train = train_datasets
y_train = train_labels
y_train0 = train_labels[:,0]
y_train1 = train_labels[:,1]
y_train2 = train_labels[:,2]
y_train3 = train_labels[:,3]
y_train4 = train_labels[:,4]
y_train5 = train_labels[:,5]

kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10))
gpr0 = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train0)
gpr1 = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train1)
gpr2 = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train2)
gpr3 = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train3)
gpr4 = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train4)
gpr5 = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train5)
