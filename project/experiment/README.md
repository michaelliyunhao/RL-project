# Overview of Experiment on the Quanser-Robot environment

## DQN(Deep Q-learning)

### Cartpole swingup





## MPC(Model Predictive Control)

### Qube

Data:

| data name |  collect method  | optimizer |
|------|----------|-------------|
| qube_datasets1, qube_labels1  |  collect method  | optimizer |
| qube_datasets2, qube_labels2  |  collect method  | optimizer |
| qube_datasets3, qube_labels3  |  collect method  | optimizer |
| qube_datasets4, qube_labels4  |  collect method  | optimizer |
| qube_datasets5, qube_labels5  |  collect method  | optimizer |
| qube_datasets6_short, qube_labels6_short  |  collect method  | optimizer |
| qube_enhance_datasets1, qube_enhance_labels1 |  collect method  | optimizer |
| qube_enhance_datasets2, qube_enhance_labels2  |  collect method  | optimizer |
| qube_enhance_datasets3, qube_enhance_labels3  |  collect method  | optimizer |

Dynamics fitting:

| model name |  method  | optimizer | train data name | train data type  | model architecture |
|------|----------|-------------|-----------|---------|--:|
|  qube_80_1    | neural networks         |   Adam          | qube_enhance_dataset1, qube_enhance_labels1          | enhance data  |  7* 70* 70* 6
|  h0_100    | neural networks         |   Adam          |           |   | 4 * 100 * 3   |
| h1_15     | neural networks        |    Adam        |           |   | 4 * 15 *3
| h1_30     |  neural networks  |  Adam     |     | |  4 * 30 * 3
| qube_70_2     | neural networks   |  Adam     |   |   |  7 * 70 * 70 * 6
| qube_80_2    | neural networks   |  Adam     |  |    | 7 * 100 * 6
| qube_80_2_new    | neural networks   |  Adam     |  |    | 7 * 80 * 80 * 6
| qube_80_2_new2    | neural networks   |  Adam     |  |    | 7 * 300 * 300 * 6
| qube_80_2_new3    | neural networks   |  Adam     |  |    | 7 * 500 * 500 * 6
| qube_100_1     | neural networks   |  Adam     |  |    | 7 * 100 * 6
| qube_100_2     | neural networks   |  Adam     |    |  | 7 * 100 * 100 * 6
| qube_200_2_new     | neural networks   |  Adam     |  |    | 7 * 200 * 200 * 6


Examples:

### Cartpole swingup
