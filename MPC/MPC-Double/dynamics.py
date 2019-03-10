import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import pickle
from utils import *

class MLP(nn.Module):
    def __init__(self, n_input=7, n_output=6, n_h=2, size_h=128):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        assert n_h >= 1, "h must be integer and >= 1"
        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)
        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.tanh(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.tanh(out)
        out = self.fc_out(out)
        return out

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)

class DynamicModel(object):
    def __init__(self,config):
        model_config = config["model_config"]
        self.n_states = model_config["n_states"]
        self.n_actions = model_config["n_actions"]
        self.use_cuda = model_config["use_cuda"]
        if model_config["load_model"]:
            self.model = torch.load(model_config["model_path"])
        else:
            self.model = MLP(self.n_states + self.n_actions, self.n_states, model_config["n_hidden"],
                             model_config["size_hidden"])
        if self.use_cuda:
            self.model = self.model.cuda()
            self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
        else:
            self.model = self.model.cpu()
            self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
        training_config = config["training_config"]
        self.n_epochs = training_config["n_epochs"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]
        self.save_model_flag = training_config["save_model_flag"]
        self.save_model_path = training_config["save_model_path"]
        self.exp_number = training_config["exp_number"]
        self.save_loss_fig = training_config["save_loss_fig"]
        self.save_loss_fig_frequency = training_config["save_loss_fig_frequency"]
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # trainset is a dictionary
    def train(self, trainset, testset=0):
        datasets, labels = self.norm_train_data(trainset["data"],trainset["label"])
        if testset != 0:
            test_datasets, test_labels = self.norm_test_data(testset["data"],testset["label"])
        train_dataset = MyDataset(datasets, labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        total_step = len(train_loader)
        print(f"Total training step per epoch [{total_step}]")
        #show_step = int(total_step / 3)
        loss_epochs = []
        for epoch in range(1, self.n_epochs + 1):
            loss_this_epoch = []
            for i, (datas, labels) in enumerate(train_loader):
                datas = self.Variable(torch.FloatTensor(np.float32(datas)))
                labels = self.Variable(torch.FloatTensor(np.float32(labels)))
                self.optimizer.zero_grad()
                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())
                #if (i + 1) % show_step == 0:
                #    print(f"Epoch [{epoch}/{n_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.8f}")
            loss_epochs.append(np.mean(loss_this_epoch))
            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)
            if self.save_loss_fig and epoch % self.save_loss_fig_frequency == 0:
                self.save_figure(epoch, loss_epochs, loss_this_epoch)
                if testset != 0:
                    loss_test = self.validate_model(test_datasets, test_labels)
                print(f"Epoch [{epoch}/{self.n_epochs}], Training Loss: {np.mean(loss_this_epoch):.8f}, "
                      f"Test Loss: {loss_test:.8f}")
        return loss_epochs

    # input a 1d numpy array and return a numpy array
    def predict(self, x):
        x = np.array(x)
        x = self.pre_process(x)
        x_tensor = self.Variable(torch.FloatTensor(x).unsqueeze(0), volatile=True) # not sure here
        out_tensor = self.model(x_tensor)
        out = out_tensor.cpu().detach().numpy()
        out = self.after_process(out)
        return out

    def pre_process(self, x):
        x = (x - self.mean_data) / self.std_data
        return x

    def after_process(self, x):
        x = x * self.std_label + self.mean_label
        return x

    def norm_train_data(self, datas, labels):
        self.mean_data = np.mean(datas, axis=0)
        self.mean_label = np.mean(labels, axis=0)
        self.std_data = np.std(datas, axis=0)
        self.std_label = np.std(labels, axis=0)
        datas = (datas - self.mean_data) / self.std_data
        labels = (labels - self.mean_label) / self.std_label
        return datas, labels

    def norm_test_data(self, datas, labels):
        datas = (datas - self.mean_data) / self.std_data
        labels = (labels - self.mean_label) / self.std_label
        return datas, labels

    def validate_model(self, datasets, labels):
        test_dataset = MyDataset(datasets, labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
        loss_list = []
        for i, (datas, labels) in enumerate(test_loader):
            datas = self.Variable(torch.FloatTensor(np.float32(datas)))
            labels = self.Variable(torch.FloatTensor(np.float32(labels)))
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())
        loss_avr = np.average(loss_list)
        return loss_avr

    def save_figure(self, epoch, loss_epochs,loss_this_epoch):
        plt.clf()
        plt.close("all")
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.title('Loss Trend with %s Epochs' % (epoch))
        plt.plot(loss_epochs)
        plt.subplot(122)
        plt.title('Loss Trend in the latest Epoch')
        plt.plot(loss_this_epoch)
        plt.savefig("storage/loss-" + str(self.exp_number) + ".png")

    def model_validation(self,env, horizon=40, n_sample=200, mpc=[]):
        n_state = env.observation_space.shape[0]
        errors = np.zeros([n_sample, horizon, n_state])
        for i in range(n_sample):
            state = env.reset()
            state_pred = state.copy()
            state_real = state.copy()
            for j in range(horizon):  # predicted results
                if mpc != []:
                    action = mpc.act(state_pred, self)
                    action = np.array([action])
                else:
                    action = env.action_space.sample()
                input_data = np.concatenate((state_pred, action))
                state_dt = self.predict(input_data)
                state_pred = state_pred + state_dt[0]
                state_real, reward, done, info = env.step(action)
                error_tmp = state_real - state_pred
                errors[i, j] = abs(error_tmp)
        errors_mean = np.mean(errors, axis=0)
        errors_max = np.max(errors, axis=0)
        errors_min = np.min(errors, axis=0)
        errors_std = np.min(errors, axis=0)
        return errors_mean, errors_max, errors_min, errors_std

    def plot_model_validation(self, env, horizon=40, n_sample=200, mpc=[], mode="mean"):
        if mode == "mean":
            errors = self.model_validation(env, horizon, n_sample, mpc)[0]
        elif mode == "max":
            errors = self.model_validation(env, horizon, n_sample, mpc)[1]
        elif mode == "min":
            errors = self.model_validation(env, horizon, n_sample, mpc)[2]
        elif mode == "std":
            errors = self.model_validation(env, horizon, n_sample, mpc)[3]
        else:
            return 0
        plt.close("all")
        plt.ioff()
        plt.figure(figsize=[12, 6])
        plt.title(mode + " state error between the predictive model and real world along different horizons")
        plt.xlabel("horizon")
        plt.ylabel("error")
        for i in range(errors.shape[1]):
            plt.plot(errors[:, i], label='state ' + str(i))
            plt.legend()
        plt.savefig("storage/model_error_exp_"+str(self.exp_number)+".png")
        plt.show()

class DatasetFactory(object):
    def __init__(self, env, config):
        self.env = env
        dataset_config = config["dataset_config"]
        self.load_flag = dataset_config["load_flag"]
        self.load_path = dataset_config["load_path"]
        self.n_max_steps = dataset_config["n_max_steps"]
        self.n_random_episodes = dataset_config["n_random_episodes"]
        self.testset_split = dataset_config["testset_split"]
        self.n_mpc_episodes = dataset_config["n_mpc_episodes"]
        self.mpc_dataset_split = dataset_config["mpc_dataset_split"]
        self.n_mpc_itrs = dataset_config["n_mpc_itrs"]
        self.save_flag = dataset_config["save_flag"]
        self.save_path = dataset_config["save_path"]
        self.min_train_samples = dataset_config["min_train_samples"]
        self.random_dataset = []
        self.random_trainset = []
        self.random_testset = []
        self.mpc_dataset = []
        self.mpc_dataset_len = 0
        self.trainset = []
        if self.load_flag:
            self.all_dataset = self.load_dataset()
        else:
            self.all_dataset = []

    # numpy array, collect n_random_episodes data with maximum n_max_steps steps per episode
    def collect_random_dataset(self):
        datasets = []
        labels = []
        for i in range(self.n_random_episodes):
            data_tmp = []
            label_tmp = []
            state_old = self.env.reset()
            for j in range(self.n_max_steps):
                action = self.env.action_space.sample()
                data_tmp.append(np.concatenate((state_old, action)))
                state_new, reward, done, info = self.env.step(action)
                label_tmp.append(state_new - state_old)
                if done:
                    break
                state_old = state_new
            data_tmp = np.array(data_tmp)
            label_tmp = np.array(label_tmp)
            if datasets == []:
                datasets = data_tmp
            else:
                datasets = np.concatenate((datasets, data_tmp))
            if labels == []:
                labels = label_tmp
            else:
                labels = np.concatenate((labels, label_tmp))
        data_and_label = np.concatenate((datasets, labels), axis=1)
        # Merge the data and label into one array and then shuffle
        np.random.shuffle(data_and_label)
        print("Collect random dataset shape: ", datasets.shape)
        testset_len = int(datasets.shape[0] * self.testset_split)
        data_len = datasets.shape[1]
        self.random_testset = {"data": data_and_label[:testset_len, :data_len],
                               "label": data_and_label[:testset_len, data_len:]}
        self.random_trainset = {"data": data_and_label[testset_len:, :data_len],
                                "label": data_and_label[testset_len:, data_len:]}
        self.random_dataset = {"data": datasets, "label": labels}
        self.all_dataset = self.random_dataset

    def collect_mpc_dataset(self, mpc, dynamic_model):
        datasets = []
        labels = []
        reward_episodes = []
        for i in range(self.n_mpc_episodes):
            data_tmp = []
            label_tmp = []
            reward_episode = 0
            state_old = self.env.reset()
            for j in range(self.n_max_steps):
                action = mpc.act(state_old, dynamic_model)
                action = np.array([action])
                data_tmp.append(np.concatenate((state_old, action)))
                state_new, reward, done, info = self.env.step(action)
                reward_episode += reward
                label_tmp.append(state_new - state_old)
                if done:
                    break
                state_old = state_new
            data_tmp = np.array(data_tmp)
            label_tmp = np.array(label_tmp)
            if datasets == []:
                datasets = data_tmp
            else:
                datasets = np.concatenate((datasets, data_tmp))
            if labels == []:
                labels = label_tmp
            else:
                labels = np.concatenate((labels, label_tmp))
            reward_episodes.append(reward_episode)
            print(f"Episode [{i}/{self.n_mpc_episodes}], Reward: {reward_episode:.8f}")
        self.mpc_dataset = {"data": datasets, "label": labels}
        self.mpc_dataset_len = datasets.shape[0]
        print("Totally collect %s data based on MPC" % self.mpc_dataset_len)
        all_datasets = np.concatenate((datasets, self.all_dataset["data"]))
        all_labels = np.concatenate((labels, self.all_dataset["label"]))
        self.all_dataset = {"data": all_datasets, "label": all_labels}
        if self.save_flag:
            self.save_datasets(self.all_dataset)
        return reward_episodes

    def make_dataset(self):
        # calculate how many samples needed from the all datasets
        all_length = max(int(self.mpc_dataset_len / self.mpc_dataset_split), self.min_train_samples)
        sample_length = all_length - self.mpc_dataset_len
        sample_length = min(self.all_dataset["data"].shape[0], sample_length)
        print("Sample %s training data from all previous dataset, total training sample: %s" % (
        sample_length, all_length))
        data_and_label = np.concatenate((self.all_dataset["data"], self.all_dataset["label"]), axis=1)
        # Merge the data and label into one array and then shuffle
        np.random.shuffle(data_and_label)
        testset_len = min(int(all_length * self.testset_split), self.all_dataset["data"].shape[0])
        data_len = self.mpc_dataset["data"].shape[1]

        trainset_data = np.concatenate((self.mpc_dataset["data"], data_and_label[:sample_length, :data_len]))
        trainset_label = np.concatenate((self.mpc_dataset["label"], data_and_label[:sample_length, data_len:]))
        testset_data = data_and_label[testset_len:, :data_len]
        testset_label = data_and_label[testset_len:, data_len:]
        trainset = {"data": trainset_data, "label": trainset_label}
        testset = {"data": testset_data, "label": testset_label}
        return trainset, testset

    # Save dictionary dataset
    def save_datasets(self,data):
        print("Saving all datas to %s" % self.save_path)
        with open(self.save_path, 'wb') as f:  # open file with write-mode
            pickle.dump(data, f, -1)  # serialize and save object

    def load_dataset(self):
        print("Load datas from %s" % self.load_path)
        with open(self.load_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset