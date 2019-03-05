# coding: utf-8

from DQN import *
import argparse
from quanser_robots import GentlyTerminating

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="model name in the storage folder", type=str)
args = parser.parse_args()

plt.style.use('seaborn')
env = GentlyTerminating(gym.make('QubeRR-v0'))

MODEL_PATH = "storage/qube.ckpt"

if args.path:
    MODEL_PATH = "storage/"+args.path

NUM_ACTIONS = 11
current_model = torch.load(MODEL_PATH)

if USE_CUDA:
    current_model = current_model.cuda()

s_all, a_all = [], []

# data_rr=[]
data_rr = torch.load("storage/data_rr.pkl")
n_epoch = 1
epsilon = 0.3

for i in range(n_epoch):
    print(i)
    obs_old = env.reset()
    obs_old[4:6] /= 20
    done = False
    epsilon -=  0.03
    while not done:

        env.render()
        action = current_model.act(obs_old, 0.0)
        f_action = 5 * (action - (NUM_ACTIONS - 1) / 2) / ((NUM_ACTIONS - 1) / 2)
        obs_new, reward, done, info = env.step(f_action)
        reward = 100*(reward-0.005)
        obs_new[4:6] /= 20
        data_rr.append([obs_old, action[0], reward, obs_new, done])
        obs_old = obs_new
        s_all.append(info['s'])
        a_all.append(info['a'])

env.close()

print("save collected data")
torch.save(data_rr,"storage/data_rr.pkl")

fig, axes = plt.subplots(5, 1, figsize=(5, 8), tight_layout=True)

s_all = np.stack(s_all)
a_all = np.stack(a_all)

n_points = s_all.shape[0]
t = np.linspace(0, n_points * env.unwrapped.timing.dt_ctrl, n_points)
for i in range(4):
    state_labels = env.unwrapped.state_space.labels[i]
    axes[i].plot(t, s_all.T[i], label=state_labels, c='C{}'.format(i))
    axes[i].legend(loc='lower right')
action_labels = env.unwrapped.action_space.labels[0]
axes[4].plot(t, a_all.T[0], label=action_labels, c='C{}'.format(4))
axes[4].legend(loc='lower right')

axes[0].set_ylabel('ang pos [rad]')
axes[1].set_ylabel('ang pos [rad]')
axes[2].set_ylabel('ang vel [rad/s]')
axes[3].set_ylabel('ang vel [rad/s]')
axes[4].set_ylabel('voltage [V]')
axes[4].set_xlabel('time [seconds]')
plt.show()


