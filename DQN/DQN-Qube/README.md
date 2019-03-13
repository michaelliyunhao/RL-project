# DQN - CartPoleStab

This folder contains the implementation of DQN algorithm and the evaluation on the CartPoleStab environment

All the hyper-parameters and experiment setting are stored in the ```config.yml``` file

All the results (figure and model) will be stored in the ```./storage``` folder

## How to run

### Test the pre-trained

To try our pre-trained model, simply run

```angularjs
python test.py
```

The script will find the model path specified in the ```config.yml``` file
 
### Train your own model

To train your own model, you can change the hyper-parameters in the ```config.yml``` to whatever you want,
and then run

```angularjs
python train.py
```

The script will load the configurations in the ```config.yml``` file and begin to train

