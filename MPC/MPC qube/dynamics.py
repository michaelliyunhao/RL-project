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
        self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, datasets, labels):
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
                print(f"Epoch [{epoch}/{self.n_epochs}], Loss: {np.mean(loss_this_epoch):.8f}")
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

    def normlize_datasets(self, datas, labels):
        self.mean_data = np.mean(datas, axis=0)
        self.mean_label = np.mean(labels, axis=0)
        self.std_data = np.std(datas, axis=0)
        self.std_label = np.std(labels, axis=0)
        datas = (datas - self.mean_data) / self.std_data
        labels = (labels - self.mean_label) / self.std_label
        return datas, labels

    #TODO: not sure whether we need this
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
        print(f"Model validation... average loss: {loss_avr:.4f} ")
        #plt.plot(loss_list)
        #plt.show()

    def save_figure(self, epoch, loss_epochs,loss_this_epoch):
        plt.clf()
        plt.close("all")
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.title('Loss Trend with Latest %s Epochs' % (epoch))
        plt.plot(loss_epochs)
        plt.subplot(122)
        plt.title('Loss Trend in the %s Epoch' % (epoch))
        plt.plot(loss_this_epoch)
        plt.savefig("storage/loss-" + str(self.exp_number) + ".png")

class DatasetFactory(object):
    def __init__(self, env, config):
        self.env = env
        dataset_config = config["dataset_config"]
        self.n_max_steps = dataset_config["n_max_steps"]
        self.n_random_episodes = dataset_config["n_random_episodes"]
        self.testset_split = dataset_config["testset_split"]
        self.n_mpc_episodes = dataset_config["n_mpc_episodes"]

        self.all_dataset = None
        self.random_dataset = None
        self.random_trainset = None
        self.random_testset = None
        self.mpc_dataset = None
        self.trainset = None

    # numpy array, collect n_random_episodes data with maximum n_max_steps steps per episode
    def collect_random_dataset(self):
        datasets = None
        labels = None
        for i in range(self.n_random_episodes):
            data_tmp = []
            label_tmp = []
            state_old = self.env.reset()
            for j in range(self.n_max_steps):
                action = self.env.action_space.sample()
                data_tmp.append(np.concatenate((state_old,action)))
                state_new, reward, done, info = self.env.step(action)
                label_tmp.append( state_new-state_old )
                if done:
                    break
                state_old = state_new
            data_tmp = np.array(data_tmp)
            label_tmp = np.array(label_tmp)
            if datasets == None:
                datasets = data_tmp
            else:
                datasets = np.concatenate((datasets,data_tmp))
            if labels == None:
                labels = label_tmp
            else:
                labels = np.concatenate((labels,label_tmp))
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
        self.random_dataset = {"data":datasets, "label":labels}
        self.all_dataset = self.random_dataset

    def collect_mpc_dataset(self,mpc,dynamic_model):
        datasets = None
        labels = None
        for i in range(self.n_mpc_episodes):
            data_tmp = []
            label_tmp = []
            state_old = self.env.reset()
            for j in range(self.n_max_steps):
                action = mpc.act(state_old, dynamic_model)
                data_tmp.append(np.concatenate((state_old, action)))
                state_new, reward, done, info = self.env.step(action)
                label_tmp.append(state_new - state_old)
                if done:
                    break
                state_old = state_new
            data_tmp = np.array(data_tmp)
            label_tmp = np.array(label_tmp)
            if datasets == None:
                datasets = data_tmp
            else:
                datasets = np.concatenate((datasets,data_tmp))
            if labels == None:
                labels = label_tmp
            else:
                labels = np.concatenate((labels,label_tmp))
        self.mpc_dataset = {"data": datasets, "label": labels}
        all_datasets = np.concatenate((datasets, self.all_dataset["data"]))
        all_labels = np.concatenate((labels, self.all_dataset["label"]))
        self.all_dataset = {"data": all_datasets, "label": all_labels}

    def sample_from_dataset(self, dataset, sample_size):
        pass




def random_dataset_old_version(env, epochs=5, samples_num=1000, mode=0, prob=0.25):
    datasets = np.zeros([samples_num * epochs, 7])
    labels = np.zeros([samples_num * epochs, 6])
    for j in range(epochs):
        obs_old = env.reset()
        for i in range(samples_num):
            if mode == 0:
                action = env.action_space.sample()
            elif mode == 1:
                if np.random.uniform() < prob:
                    action = -np.random.uniform() * 3
                    action = np.array([action])
                else:
                    action = np.random.uniform() * 3
                    action = np.array([action])
            elif mode == -1:
                if np.random.uniform() < prob:
                    action = np.random.uniform() * 3
                    action = np.array([action])
                else:
                    action = -np.random.uniform() * 3
                    action = np.array([action])

            datasets[j * samples_num + i, 0] = obs_old[0]
            datasets[j * samples_num + i, 1] = obs_old[1]
            datasets[j * samples_num + i, 2] = obs_old[2]
            datasets[j * samples_num + i, 3] = obs_old[3]
            datasets[j * samples_num + i, 4] = obs_old[4] / 30.
            datasets[j * samples_num + i, 5] = obs_old[5] / 40.
            datasets[j * samples_num + i, 6] = action / 5.

            # env.render()
            action = np.array(action)
            obs, reward, done, info = env.step(action)
            labels[j * samples_num + i, 0] = obs[0] - obs_old[0]
            labels[j * samples_num + i, 1] = obs[1] - obs_old[1]
            labels[j * samples_num + i, 2] = obs[2] - obs_old[2]
            labels[j * samples_num + i, 3] = obs[3] - obs_old[3]
            labels[j * samples_num + i, 4] = (obs[4] / 30.) - (obs_old[4] / 30.)
            labels[j * samples_num + i, 5] = (obs[5] / 40.) - (obs_old[5] / 40.)
            obs_old = obs
    return datasets, labels