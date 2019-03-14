# MPC - Model Predictive Control

This folder contains the implementation of MPC algorithm and the evaluation of it.

The implementation is mainly followed in this paper [here](https://ieeexplore.ieee.org/abstract/document/8463189)

Choose the environment folder and follow the instructions to run everything.

## Overview of the experiment results:

| Environment  | Horizon   |Numb\_bees  |   Max\_itrs  |  Gamma  |
| --------   | -----:  | :----: | :----: | :----: |
| Qube      |  30    |  8  | 30  |  0.98  | 
| CartPole Swingup |  20  | 8   | 20 |  0.99  |
| CartPole Stab   | 12  | 8  |  20 |  0.99  |
| Double CartPole    | 5  | 8 |  20 |  0.99  | 


% For tables use
\begin{table}[h]
% table caption is above the table
\caption{experiment results(Algorithm: MPC)}
\label{tab:1}       % Give a unique label
% For LaTeX tables use
\begin{tabular}{llllll}
\hline\noalign{\smallskip}
Environment & Horizon & Numb\_bees & Max\_itrs & Gamma \\
\noalign{\smallskip}\hline\noalign{\smallskip}
Qube & 30 & 8 & 30  & 0.98  \\
CartPole Swingup & 20 & 8 & 20  & 0.99  \\
CartPole Stab & 12 & 8 & 20 & 0.99 \\
Double CartPole & 5 & 8 & 20  & 0.99 \\
\noalign{\smallskip}\hline
\end{tabular}
\end{table}

### CartpoleStabShort-v0
   episode_rewards:
   
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.98
   
   batch size: 20
   
   weight_decay: 1e-4
   
   num_epochs: 2000
   
### Qube-v0:
   episode_rewards:
   
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.98
   
   batch size: 20
   
   weight_decay: 1e-4
   
   num_epochs: 2000
   
   
### DoublePendulum-v0
   episode_rewards:
   
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.99
   
   batch size: 20

   weight_decay: 1e-4
   
   num_epochs: 2000

### CartpoleSwingShort-v0
   learning_rate: 3e-5
   
   networks architecture:
   
   gamma: 0.98
   
   batch size: 20
   
   weight_decay: 1e-4
   
   num_epochs: 2000

