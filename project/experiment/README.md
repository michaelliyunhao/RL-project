# Overview of Experiment on the Quanser-Robot environment

## DQN(Deep Q-learning)

### Cartpole swingup





## MPC(Model Predictive Control)

### Qube

Data:

| data name |  collect method  | optimizer |
|------|----------|-------------|

Dynamics fitting:

| model name |  method  | optimizer | train data name | train data type  | model architecture |
|------|----------|-------------|-----------|---------|--:|
|  qube_80_1    | neural networks         |   Adam          | qube_enhance_dataset1, qube_enhance_labels1          | enhance data  |  7*70*70*6
|  h0_100    | neural networks         |   Adam          |           |   | 4+100+3   |
| h1_15     | neural networks        |    Adam        |           |   |
| h1_30     |  neural networks  |  Adam     |      |
| qube_70_2     | neural networks   |  Adam     |      |
| qube_80_2    | neural networks   |  Adam     |      |
| qube_100_1     | neural networks   |  Adam     |      |
| qube_100_2     | neural networks   |  Adam     |      |
| qube_200_2_new     | neural networks   |  Adam     |      |


Examples:

### Cartpole swingup
