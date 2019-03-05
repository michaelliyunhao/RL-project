# Overview of Experiment on the Quanser-Robot environment

## DQN(Deep Q-learning)

### Cartpole swingup





## MPC(Model Predictive Control)

#### Data:

To collect the initial random data and create enhance data, we define 3 modes to collect data:

+ mode0: the qube is randomly choose actions

+ mode1: the qube trends to move right(actions > 0), which has a probability to randomly choose action which is obove 0

+ mode-1: the qube trends to move left(actions < 0), which has a probability to randomly choose action which is below 0

 probabity = 0.3

 data arichtecture:  
 
 mode0_epochs * mode0_samples_num + mode1_epochs * mode1_samples_num + mode-1_epochs * mode-1_samples_num

### Qube



| data name |  len  | data architecture |  PS
|------|----------|-------------|------|
| qube_datasets1, qube_labels1  |  70000  | 300 * 100 + 200 * 100 + 200 * 100 |     |
| qube_datasets2, qube_labels2  |  30000 | 100 * 100 + 100 * 100 + 100 * 100 |    |
| qube_datasets6_short, qube_labels6_short  |  10800  | 60 * 60 + 60 * 60 + 60 * 60 |   |
| qube_enhance_datasets1, qube_enhance_labels1 |  5000  |    | collect data from the qube100_1 networks  |
| qube_enhance_datasets3, qube_enhance_labels3  |  6000  |     |  qube_enhance_datasets1 and 1000 data trained from qube_80_try2 model  |

Dynamics fitting:

| model name |  method  | optimizer | train data name | train data type  | model architecture |
|------|----------|-------------|-----------|---------|--:|
|  qube_70_2    | neural networks         |   Adam          | qube_enhance_dataset1, qube_enhance_labels1          | enhance data  |  7* 70* 70* 6
|  qube_80_1    | neural networks         |   Adam          | qube_enhance_dataset1, qube_enhance_labels1          | enhance data  |  7 * 80 * 6
| qube_80_2    | neural networks   |  Adam     | qube_datasets2, qube_labels2   | initial random data   | 7 * 80 * 80 * 6
| qube_80_2_new    | neural networks   |  Adam     | qube_datasets1, qube_labels1  | initial random data   | 7 * 80 * 80 * 6
| qube_80_2_new2    | neural networks   |  Adam     | qube_datasets1, qube_labels1 | initial random data   | 7 * 300 * 300 * 6
| qube_80_2_new3    | neural networks   |  Adam     | qube_datasets1, qube_labels1 | initial random data   | 7 * 500 * 500 * 6
| qube_100_1     | neural networks   |  Adam     | qube_datasets6_short, qube_labels6_short |  initial random data  | 7 * 100 * 6
| qube_100_2     | neural networks   |  Adam     | qube_datasets6_short, qube_labels6_short   | initial random data | 7 * 100 * 100 * 6
| GPRmodel     | gaussian process model   |        | qube_datasets6_short, qube_labels6_short   | initial random data |    |


for daynamics fitting, we also try using Gaussian Process Model and auto-sklearn to fit the daynamics function, but the models do not show good performance and the models are too large, so we use neural networks to fit the dynamics function.



Examples:

+ model:  qube_80_1


![2](https://github.com/michaelliyunhao/images/blob/master/qube80_1.png)


+ model:  qube_80_1_try2

![4](https://github.com/michaelliyunhao/images/blob/master/qube_80_1_try2.png)

### Cartpole swingup



### Double cartpole
