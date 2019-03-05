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


obs = env.reset()
s_all, a_all = [], []
for i in range(3):
    done = False
	print(i)
    while not done:
        env.render()
		obs[4:6] /= 20
	    action = current_model.act(obs, 0)
	    f_action = 5 * (action - (NUM_ACTIONS - 1) / 2) / ((NUM_ACTIONS - 1) / 2)
	    obs, rwd, done, info = env.step(f_action)
	    s_all.append(info['s'])
	    a_all.append(info['a'])

env.close()


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


