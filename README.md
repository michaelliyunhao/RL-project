# Reinforcement Learning Course Project
Technische Universität Darmstadt winter semester 2018/2019

Supervisor: Jan Peters, Riad Akrour

This repository contains the PyTorch implementation of Deep Q-Network (DQN) and Model Predictive Control (MPC), 
and the evaluation of them on the [quanser robot platform](https://git.ias.informatik.tu-darmstadt.de/quanser/clients) both in simulation and real world.

## Authors
+ Zuxin Liu
+ Yunhao Li
+ Junfei Xiao

## Algorithms
+ [DQN](https://arxiv.org/abs/1312.5602)
+ [MPC](https://ieeexplore.ieee.org/abstract/document/8463189)

## Platforms
+ Qube
+ Double Pendlum
+ Cartpole Swing-up
+ Cartpole Stab

## Installation
For the installation of the Quanser robot simulation environment, please see [this page](https://git.ias.informatik.tu-darmstadt.de/quanser/clients)

For the implementation of the algorithms, the following packages are required:

+ python = 3.6.2
+ pytorch = 1.0.1
+ numpy = 1.12.1
+ matplotlib = 2.1.1
+ gym

You can simply create the same environment as ours by using [Anaconda](https://www.anaconda.com/).
All the required packages are included in the ```environment.yaml``` file. You can create the environment by the following command

```angular2html
conda env create -f environment.yaml
```
Then, activate your environment by 

```
source activate pytorch
```
