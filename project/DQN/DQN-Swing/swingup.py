import numpy as np
import time
import gym
import matplotlib.pyplot as plt

from quanser_robots.cartpole.ctrl import SwingUpCtrl, MouseCtrl
from quanser_robots.common import GentlyTerminating, Logger


def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta


class PlotSignal:
    def __init__(self, window=10000):
        self.window = window
        self.values = {}

    def update(self, **argv):
        for k in argv:
            if k not in self.values:
                self.values[k] = [argv[k]]
            else:
                self.values[k].append(argv[k])
            self.values[k] = self.values[k][-self.window:]

    def plot_signal(self):
        N = len(self.values)
        plt.clf()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.pause(0.0000001)

    def last_plot(self):
        N = len(self.values)
        plt.clf()
        plt.ioff()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.show()


def do_trajectory(env, ctrl, plot, time_steps=10000, use_plot=True,
                  collect_fr=10, plot_fr=10, render=True, render_fr=10):

    obs = env.reset()
    for n in range(time_steps):
        act = ctrl(obs)
        obs, _, done, _ = env.step(np.array(act[0]))

        if done:
            print("Physical Limits or End of Time reached")
            break

        if render:
            if n % render_fr == 0:
                env.render()

        if use_plot:

            if n % collect_fr == 0:
                alpha, theta = get_angles(obs[1], obs[2])
                plot.update(theta=theta, alpha=alpha, theta_dt=obs[4], volt=act[0], u=act[1], x=obs[0])
                env.render()

            if n % plot_fr == 0:
                plot.plot_signal()


def get_env_and_controller(long_pendulum=True, simulation=True, swinging=True, mouse_control=False):
    pendulum_str = {True:"Long", False:"Short"}
    simulation_str = {True:"", False:"RR"}
    task_str = {True:"Swing", False:"Stab"}

    if not simulation:
        pendulum_str = {True: "", False: ""}

    mu = 7.5 if long_pendulum else 19.
    env_name = "Cartpole%s%s%s-v0" % (task_str[swinging], pendulum_str[long_pendulum], simulation_str[simulation])
    if not mouse_control:
        return Logger(GentlyTerminating(gym.make(env_name))), SwingUpCtrl(long=long_pendulum, mu=mu)
    else:
        return Logger(GentlyTerminating(gym.make(env_name))), MouseCtrl()


def main():
    use_plot = False
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
            act = ctrl(obs)
            obs, _, done, _ = env.step(act[0] * np.ones(1))

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
