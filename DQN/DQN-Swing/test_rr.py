# coding: utf-8

from DQN import *
import argparse
from swingup import *

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="model name in the storage folder", type=str)
args = parser.parse_args()

MODEL_PATH = "storage/swing-good.ckpt"

if args.path:
    MODEL_PATH = "storage/"+args.path

NUM_ACTIONS = 11
current_model = torch.load(MODEL_PATH)

if USE_CUDA:
    current_model = current_model.cuda()

use_plot = True
render = True

window = 500
collect_fr = 10
plot_fr = 10
render_fr = 10

if use_plot:
    plt.ion()
    plot = PlotSignal(window=window)

# Initialize Controller & Environment:
env, ctrl = get_env_and_controller(long_pendulum=False, simulation=False, swinging=True, mouse_control=False)

n_episodes = 10
time_steps = 10000000

for i in range(n_episodes):
    print("\n\n###############################")
    print("Episode {0}".format(0))

    # Reset the environment:
    env.reset()
    obs, _, _, _ = env.step(np.zeros(1))
    # Start the Control Loop:
    print("\nStart Controller:\t\t\t", end="")
    for n in range(time_steps):
        obs[4] /= 10
        action = current_model.act(obs, 0)
        f_action = 16 * (action - (NUM_ACTIONS - 1) / 2) / ((NUM_ACTIONS - 1) / 2)
        obs, _, done, _ = env.step(f_action)
        if done:
            print("Physical Limits or End of Time reached")
            break

        if render and np.mod(n, render_fr) == 0:
            env.render()

        if use_plot and np.mod(n, collect_fr) == 0:
            alpha, theta = get_angles(obs[1], obs[2])
            plot.update(theta=theta, alpha=alpha, theta_dt=obs[4], volt=act[0], u=act[1], x=obs[0])
            env.render()

        if use_plot and np.mod(n, plot_fr) == 0:
            plot.plot_signal()

    # Stop the cart:
    env.step(np.zeros(1))

env.close()



