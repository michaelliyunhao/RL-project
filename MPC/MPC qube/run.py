# coding: utf-8
import gym
import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *
from quanser_robots.common import GentlyTerminating

# datasets:  numpy array, size:[sample number, input dimension]
# labels:  numpy array, size:[sample number, output dimension]

def mpc_iteration_hive(env, model,train_datasets, train_labels, sample_num = 500,
                       horizon = 10, numb_bees = 10, max_itrs = 10, gamma = 0.85,
                       model_name = "emmm"):
    learning_rate = 3e-5
    batch_size = 20
    num_epochs= 50
    weight_decay = 1e-4

    train_datasets_new, train_labels_new = mpc_dataset_hive(env, model, sample_num, horizon, numb_bees, max_itrs, gamma)
    train_datasets = np.concatenate( (train_datasets,train_datasets_new) )
    train_labels = np.concatenate( (train_labels,train_labels_new) )

    train_datasets_norm, train_labels_norm = model.normlize_datasets(train_datasets,train_labels)
    train(model,train_datasets_norm,train_labels_norm,
          learning_rate,batch_size, num_epochs,weight_decay,model_name)
    return train_datasets, train_labels

config_path = "config.yml"
print_config(config_path)
config = load_config(config_path)

env_id ="Qube-v0" # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))

dynamics_model = DynamicModel(config)


#datasets_path = 'storage/datasets_hive.pkl'
#labels_path = 'storage/labels_hive.pkl'
train_datasets ,train_labels = random_dataset(env, epochs=5, samples_num=100, mode=0, prob=0.25) # load_dataset(datasets_path, labels_path)


train_datasets_norm, train_labels_norm = dynamics_model.normlize_datasets(train_datasets,train_labels)
dynamics_model.train(train_datasets_norm,train_labels_norm, config)

print("plot model validation...")
plot_model_validation(env, dynamics_model,horizons=30, samples=300)
#evaluate(model,train_datasets_norm,train_labels_norm)


#sample_num = 600
#train_datasets, train_labels = mpc_iteration_hive(env, \
#                                                  model,train_datasets, \
#                                                  train_labels, sample_num = sample_num, \
#                                                  horizon = 15, numb_bees = 8, \
#                                                  max_itrs = 10, gamma = 0.95, model_name=model_name)



'''
num = 5
for i in range(num):
    prob = i/num
    test_datasets1, test_labels1 = random_dataset(env, epochs = 20, samples_num = 200, mode = 1, prob=prob)
    test_datasets1_norm,test_labels1_norm = normlize_datasets(test_datasets1, test_labels1)
    evaluate(model,test_datasets1_norm,test_labels1_norm)
'''

