import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import pickle

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

    def train(self, datasets, labels, config):
        training_config = config["training_config"]
        n_epochs = training_config["n_epochs"]
        lr = training_config["learning_rate"]
        batch_size = training_config["batch_size"]
        save_model_flag = training_config["save_model_flag"]
        save_model_path = training_config["save_model_path"]
        exp_number = training_config["exp_number"]
        save_loss_fig = training_config["save_loss_fig"]
        criterion = nn.MSELoss(reduction='elementwise_mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_dataset = MyDataset(datasets, labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        total_step = len(train_loader)
        print(f"Total training step per epoch [{total_step}]")
        #show_step = int(total_step / 3)
        loss_epochs = []
        for epoch in range(1, n_epochs + 1):
            loss_this_epoch = []
            for i, (datas, labels) in enumerate(train_loader):
                datas = self.Variable(torch.FloatTensor(np.float32(datas)))
                labels = self.Variable(torch.FloatTensor(np.float32(labels)))
                optimizer.zero_grad()
                outputs = self.model(datas)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_this_epoch.append(loss.item())
                #if (i + 1) % show_step == 0:
                #    print(f"Epoch [{epoch}/{n_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.8f}")
            loss_epochs.append(np.mean(loss_this_epoch))
            if save_model_flag:
                torch.save(self.model, save_model_path)
            if save_loss_fig and epoch % 20 == 0:
                plt.clf()
                plt.close()
                plt.figure(figsize=(12, 5))
                plt.subplot(121)
                plt.title('Loss Trend with Latest %s Epochs' % (epoch))
                plt.plot(loss_epochs)
                plt.subplot(122)
                plt.title('Loss Trend in the %s Epoch' % (epoch))
                plt.plot(loss_this_epoch)
                plt.savefig("storage/loss-" + str(exp_number) + ".png")
                print(f"Epoch [{epoch}/{n_epochs}], Loss: {np.mean(loss_this_epoch):.8f}")
        return loss_epochs

    # input a 1d numpy array and return a numpy array
    def predict(self, x):
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

class DatasetFactory(object):
    def __init__(self, env, config):
        self.env = env
        self.random_dataset = []

    # numpy array
    def collect_random_dataset(self, episodes=5, n_max_steps=1000):
        datasets = []
        labels = []
        for i in range(episodes):
            data_tmp = []
            label_tmp = []
            state_old = self.env.reset()
            for j in range(n_max_steps):
                action = self.env.action_space.sample()
                data_tmp.append(np.concatenate((state_old,action)))
                state_new, reward, done, info = self.env.step(action)
                label_tmp.append( state_new-state_old )
                if done:
                    break
                state_old = state_new
            datasets.append(data_tmp)
            labels.append(label_tmp)
        datasets = np.array(datasets)
        labels = np.array(labels)
        self.random_dataset = {"data":datasets, "label":labels}
        return datasets, labels

class MyDataset(data.Dataset):
    def __init__(self, datas, labels):
        self.datas = torch.tensor(datas)
        self.labels = torch.tensor(labels)

    def __getitem__(self, index):  # return tensor
        datas, target = self.datas[index], self.labels[index]
        return datas, target

    def __len__(self):
        return len(self.datas)

def min_max_scaler(d_in):  # scale the data to the range [0,1]
    d_max = np.max(d_in)
    d_min = np.min(d_in)
    d_out = (d_in - d_min) / (d_max - d_min)
    return d_out, d_min, d_max







def model_evaluation(model, test_loader, device=torch.device('cuda'),
                     criterion = nn.MSELoss(reduction='elementwise_mean')):
    model = model.eval()
    total_step = len(test_loader)
    loss_list = []
    for i, (images, labels) in enumerate(test_loader):
        # Move tensors to the configured device
        images = images.to(device).float()
        labels = labels.to(device).float()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
    loss_avr = np.average(loss_list)
    loss_max = np.max(loss_list)
    loss_min = np.min(loss_list)
    loss_std = np.std(loss_list)
    print(f"Average: {loss_avr:.4f}, max: {loss_max:.4f}, min: {loss_min:.4f}, std: {loss_std:.4f}, ")
    plt.plot(np.arange(total_step), loss_list)
    plt.show()


def evaluate(model, datasets,labels,device=torch.device('cuda'),
             criterion = nn.MSELoss(reduction='elementwise_mean')):
    model = model.eval()
    test_dataset = MyDataset(datasets, labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1)
    total_step=len(test_loader)
    loss_test_list = []
    for i, (data, labels) in enumerate(test_loader):
        # Move tensors to the configured device
        data = data.to(device).float()
        labels = labels.to(device).float()
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss_test_list.append(loss.item())
    loss_avr=np.average(loss_test_list)
    loss_max=np.max(loss_test_list)
    loss_min=np.min(loss_test_list)
    loss_std=np.std(loss_test_list)
    print(f"Test dataset average: {loss_avr:.8f}, max: {loss_max:.8f}, min: {loss_min:.8f}, std: {loss_std:.8f}, ")
    fig = plt.figure(figsize=(16,3))
    plt.plot(np.arange(total_step),loss_test_list)
    plt.title("Test Loss")
    plt.show()


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