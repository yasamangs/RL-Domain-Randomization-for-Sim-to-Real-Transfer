import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))

import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
import stable_baselines3
import numpy as np
import statistics
import scipy.stats
import gym
from PIL import Image
from cycler import cycler
from env.hopper import *
from stable_baselines3.common.evaluation import evaluate_policy


class Callback():
    def __init__(self, agent, env, args):
        """ initializes a callback object to access 
        the internal state of the RL agent over training iterations

        args:
            agent: RL agent
            env: target environment
            args: argument parser

        evaluation metrics:
            episode rewards
            episode lengths
        """
        self.test_episodes = args.test_episodes
        self.episode_rewards = list()
        self.episode_lengths = list()
        self.agent = agent
        self.env = env

    def _on_step(self, num_episodes, args, verbose=1) -> bool:
        """ monitors performance """
        if num_episodes % args.eval_frequency == 0:
            episode_rewards, episode_lengths = evaluate_policy(self.agent,
                                                               self.env, self.test_episodes,
                                                               return_episode_rewards=True)
            er, el = np.array(episode_rewards), np.array(episode_lengths)
            self.episode_rewards.append(er.mean())
            self.episode_lengths.append(el.mean())
            if verbose > 0 and num_episodes % int(args.train_episodes * 0.25) == 0:
                print(
                    f'training episode: {num_episodes} | test episodes: {self.test_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}')

        return True


def display(frame, step, episode, reward):
    """ renders the (state -> action) step in the target environment
    as a graphics interchange format frame
    """
    image = Image.fromarray(frame)
    drawer = ImageDraw.Draw(image)
    color = (255, 255, 255) if np.mean(image) < 128 else (0, 0, 0)
    drawer.text((image.size[0] / 20, image.size[1] / 18),
                f'episode: {episode} | step: {step} | reward: {reward:.2f}', fill=color)

    return image


def multiprocess(args, train_env, test_env, train, seeds):
    """ processes multiple sequential training and testing sessions 
    with different seeds (to counteract variance)

    args:
        train_env: source environment
        test_env: target environment
        train: training function
        seeds: fibonacci seeds
    """
    model = None
    if args.input_model is not None:
        model = args.input_model

    print(f'\nmodel to train: {model}\n')

    pool = {'rewards': list(), 'lengths': list(),
            'times': list(), 'weights': list()}
    for iter, seed in enumerate(seeds):
        print(f'\ntraining session: {iter + 1}')
        print("----------------")
        for key, value in zip(pool.keys(), train(args, seed, train_env, test_env, model)):
            pool[key].append(value)
            if key == 'rewards':
                path = os.path.join(args.directory, f'iter-{iter}-rewards.npy')
                np.save(path, value)

    return pool


def stack(args, metric, records):
    """ stacks training sessions outputs 

    args:
        metric: evaluation metric
        records: set of evaluation records
    """
    stacks = [(statistics.mean(elements), statistics.stdev(elements))
              for elements in list(zip(*records))]
    xs = np.array(
        [(index + 1) * args.eval_frequency for index in range(len(stacks))])
    ys = np.array([stack[0] for stack in stacks])
    sigmas = np.array([stack[1] for stack in stacks])

    return metric, xs, ys, sigmas


def track(metric, xs, ys, sigmas, args, label, filename):
    """ plots the RL agent's performance in the target environment 
    (according to the evaluation metric) over training episodes

    args:
        metric: evaluation metric
        xs: training episodes (x-axis values)
        ys: set of evaluation records (y-axis values)
        sigmas: set of evaluation records variances
    """
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
              '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    plt.plot(xs, ys, alpha=1, label=label)
    plt.fill_between(xs, ys - sigmas, ys + sigmas, alpha=0.5)

    plt.xlabel('episodes')
    plt.ylabel(f'episode {metric}')
    plt.grid(True)
    plt.legend()

    plt.savefig(f'{args.directory}/{filename}.png', dpi=300)
    plt.close()


def collect(env, seed, maxit=10):
    """ collects trajectories in the environment """
    data = list()
    env.seed(seed)
    num_episodes = 0
    while num_episodes < maxit:
        done = False
        episode = list()
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            episode.append((obs, action, reward, next_state, done))
            obs = next_state
        num_episodes += 1
        data.append(episode)
    return data


def wasserstein_distance(real_data, sim_data):
    """ computes the Wasserstein distance (Earth Mover's Distance) 
    between the state-action pairs from real-world data and simulation data 
    """
    real_state_actions = []
    sim_state_actions = []

    for episode in real_data:
        for step in episode:
            obs, action, _, _, _ = step
            real_state_actions.append(np.hstack((obs, action)))

    for episode in sim_data:
        for step in episode:
            obs, action, _, _, _ = step
            sim_state_actions.append(np.hstack((obs, action)))

    real_state_actions = np.array(real_state_actions)
    sim_state_actions = np.array(sim_state_actions)

    distance = 0
    for i in range(real_state_actions.shape[1]):
        distance += scipy.stats.wasserstein_distance(
            real_state_actions[:, i], sim_state_actions[:, i])

    return distance


def optimize_params(real_data, sim_data, seed,
                    maxit: int = 100, dist: str = 'uniform',
                    learning_rate: float = 2.5e-5, verbose: bool = False):
    """ optimizes the physical parameters (masses) of a simulation environment 
    to minimize the Wasserstein distance between the real and simulation trajectory distributions
    """
    parts = ['torso', 'thigh', 'leg', 'foot']
    # initial guess for link masses
    masses = np.array([2.53429174, 3.92699082, 2.71433605, 5.0893801])
    print(f'initial physical parameters:')
    print(f'-------------------------------------')
    for part, mass in zip(parts, masses):
        print(f'{part}: {mass}')

    for iter in range(maxit):
        base = wasserstein_distance(real_data, sim_data)
        losses = np.zeros(len(masses))
        for m in range(len(masses)):
            per_masses = masses.copy()
            per_masses[m] += learning_rate

            sim_env = gym.make('CustomHopper-source-UDR-v0', params=per_masses)
            sim_env.set_randomness(dist)

            per_sim_data = collect(sim_env, seed)
            loss = wasserstein_distance(real_data, per_sim_data)
            losses[m] = loss

        # compute gradient by finite difference approximation
        gradients = (losses - base) / learning_rate
        masses -= learning_rate * gradients

        # ensure masses within valid bounds
        masses = np.clip(masses, 0.01, 10.0)

        # update sim_data for the next iteration
        sim_env = gym.make('CustomHopper-source-UDR-v0', params=masses)
        sim_env.set_randomness(dist)

        sim_data = collect(sim_env, seed)

        # debugging prints
        if verbose:
            print(f'iteration: {iter + 1}/{maxit}')
            print(f'masses: {masses}')
            print(f'loss: {loss}')
            print(f'gradients: {gradients}')
            print('-' * 30)

    print(f'optimal physical parameters:')
    print(f'---------------------------')
    for part, mass in zip(parts, masses):
        print(f'{part}: {mass}')
    return masses
