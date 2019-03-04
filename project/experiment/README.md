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
|  h0_100    | neural networks         |   Adam          |           |   | mlp(
  (fc_list): ModuleList()
  (fc_in): Linear(in_features=4, out_features=100, bias=True)
  (act): Tanh()
  (fc_out): Linear(in_features=100, out_features=3, bias=True)
)  |
| h1_15     | neural networks        |    Adam        |           |   |
| h1_30     |  neural networks  |  Adam     |      |
| qube_70_2     | neural networks   |  Adam     |      |
| qube_80_2    | neural networks   |  Adam     |      |
| qube_100_1     | neural networks   |  Adam     |      |
| qube_100_2     | neural networks   |  Adam     |      |
| qube_200_2_new     | neural networks   |  Adam     |      |


Examples:

### Cartpole swingup
