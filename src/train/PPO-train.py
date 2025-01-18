""" PPO (Proximal Policy Optimization) algorithm
Custom Hopper 
MuJoCo environment
"""

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import stable_baselines3
import numpy as np
import warnings
import argparse
import imageio
import torch
import time
import gym
from PIL import Image
from cycler import cycler
from env.hopper import *
from stable_baselines3 import PPO
from collections import OrderedDict
from utils import display, multiprocess, stack, track
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--test', action='store_true',
                        help='test the model')
    parser.add_argument('--render', action='store_true',
                        help='render the simulator')
    parser.add_argument('--device', default='cpu', type=str,
                        help='network device [cpu, cuda]')
    parser.add_argument('--train-env', default='source', type=str,
                        help='training environment')
    parser.add_argument('--test-env', default='target', type=str,
                        help='testing environment')
    parser.add_argument('--train-episodes', default=10000, type=int,
                        help='number of training episodes')
    parser.add_argument('--test-episodes', default=50, type=int,
                        help='number of testing episodes')
    parser.add_argument('--eval-frequency', default=100, type=int,
                        help='evaluation frequency over training iterations')
    parser.add_argument('--learning-rate', default=2.5e-4, type=float,
                        help='learning rate')
    parser.add_argument('--input-model', default=None, type=str,
                        help='pre-trained input model (in .mdl format)')
    parser.add_argument('--directory', default='results', type=str,
                        help='path to the output location for checkpoint storage (model and rendering)')
    return parser.parse_args()


class Callback(BaseCallback):
    def __init__(self, agent, env, args, verbose=1):
        super(Callback, self).__init__(verbose)
        """ initializes a callback object to access 
        the internal state of the RL agent over training iterations

        args:
            agent: PPO agent
            env: target environment
            args: argument parser

        evaluation metrics:
            episode rewards
            episode lengths
        """
        self.train_episodes = args.train_episodes
        self.eval_frequency = args.eval_frequency
        self.test_episodes = args.test_episodes
        self.episode_rewards = list()
        self.episode_lengths = list()
        self.num_episodes = 0
        self.agent = agent
        self.flag = False
        self.env = env

    def _on_step(self) -> bool:
        """ monitors performance """
        self.num_episodes += np.sum(self.locals['dones'])
        if self.num_episodes % self.eval_frequency == 0:
            if not self.flag:
                episode_rewards, episode_lengths = evaluate_policy(self.agent,
                                                                   self.env, self.test_episodes,
                                                                   return_episode_rewards=True)
                er, el = np.array(episode_rewards), np.array(episode_lengths)
                self.episode_rewards.append(er.mean())
                self.episode_lengths.append(el.mean())
                if self.verbose > 0 and self.num_episodes % int(self.train_episodes * 0.25) == 0:
                    print(
                        f'training episode: {self.num_episodes} | test episodes: {self.test_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}')
                self.flag = True  # mark evaluation as due
        if self.num_episodes % self.eval_frequency != 0:
            self.flag = False  # reset evaluation flag

        if self.num_episodes >= self.train_episodes:
            return False  # stop training

        return True


def train(args, seed, train_env, test_env, model):
    """ trains the agent in the source environment

    args:
        seed: seed of the training session
        model: model to train
    """
    env = gym.make(train_env)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy = 'MlpPolicy'

    if model is not None:
        agent = PPO.load(model,
                         env=env,
                         seed=seed,
                         device=args.device)
    else:
        agent = PPO(policy,
                    env=env,
                    seed=seed,
                    device=args.device,
                    learning_rate=args.learning_rate,
                    batch_size=128,
                    ent_coef=0.0,
                    n_steps=1024,
                    gamma=0.99)

    test_env = gym.make(test_env)
    test_env.seed(seed)

    callback = Callback(agent, test_env, args)

    total_timesteps = args.train_episodes * 500
    start_time = time.time()
    agent.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - start_time

    return callback.episode_rewards, callback.episode_lengths, train_time, agent.policy.state_dict()


def test(args, test_env, seed):
    """ tests the agent in the target environment """
    env = gym.make(test_env)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy = 'MlpPolicy'
    model = None

    if args.train:
        model = f'{args.directory}/PPO-({args.train_env} to {args.test_env}).mdl'
        agent = PPO.load(model,
                         env=env,
                         device=args.device)
    else:
        if args.input_model is not None:
            model = args.input_model
            agent = PPO.load(model,
                             env=env,
                             device=args.device)

    print(f'\nmodel to test: {model}\n')

    frames = list()
    num_episodes = 0
    episode_rewards = []
    while num_episodes < args.test_episodes:
        done = False
        obs = env.reset()
        rewards, steps = (0, 0)
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            obs = next_state

            steps += 1
            if args.render and num_episodes < 5:
                frame = env.render(mode='rgb_array')
                frames.append(display(frame,
                                      step=steps,
                                      reward=rewards,
                                      episode=num_episodes + 1))
        num_episodes += 1
        episode_rewards.append(rewards)
    er = np.array(episode_rewards)
    print(
        f'\ntest episodes: {num_episodes} | reward: {er.mean():.2f} +/- {er.std():.2f}\n')

    if args.render:
        imageio.mimwrite(
            f'{args.directory}/PPO-({args.train_env} to {args.test_env})-test.gif', frames, fps=30)

    env.close()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    if not os.path.exists(args.directory):
        os.mkdir(args.directory)

    train_env, test_env = tuple(f'CustomHopper-{x}-v0'
                                for x in [args.train_env,
                                          args.test_env])

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('\nWARNING: GPU not available, switch to CPU\n')
        args.device = 'cpu'

    # validate environment registration
    try:
        env = gym.make(train_env)
    except gym.error.UnregisteredEnv:
        raise ValueError(f'ERROR: environment {train_env} not found')

    try:
        env = gym.make(test_env)
    except gym.error.UnregisteredEnv:
        raise ValueError(f'ERROR: environment {test_env} not found')

    # validate model loading
    if args.input_model is not None and not os.path.isfile(args.input_model):
        raise FileNotFoundError(
            f'ERROR: model file {args.input_model} not found')

    if args.train:
        pool = multiprocess(args, train_env, test_env, train, seeds=[1, 2, 3])
        for metric, records in zip(('reward', 'length'), (pool['rewards'], pool['lengths'])):
            metric, xs, ys, sigmas = stack(args, metric, records)
            if metric == 'reward':
                path = os.path.join(
                    args.directory, f'PPO-({args.train_env} to {args.test_env})-rewards.npy')
                np.save(path, ys)
            track(metric, xs, ys, sigmas, args,
                  label='PPO',
                  filename=f'PPO-({args.train_env} to {args.test_env})-{metric}')
        print(
            f'\ntraining time: {np.mean(pool["times"]):.2f} +/- {np.std(pool["times"]):.2f}')
        print("-------------")

        policy = 'MlpPolicy'
        agent = PPO(policy, env=env, device=args.device)

        total = [rewards[-1] for rewards in pool['rewards']]
        agent.policy.load_state_dict(pool['weights'][
                                     (np.abs(
                                         np.array(total) - np.mean(total))
                                      ).argmin()])

        agent.save(
            f'{args.directory}/PPO-({args.train_env} to {args.test_env}).mdl')
        print(
            f'\nmodel checkpoint storage: {args.directory}/PPO-({args.train_env} to {args.test_env}).mdl\n')

    if args.test:
        test(args, test_env, seed=1)


if __name__ == '__main__':
    main()
