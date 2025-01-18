import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))

import gym
import torch
import argparse
import warnings
import itertools
import numpy as np
from agents.ac import A2C, A2CPolicy
from env.hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-episodes', default=10000, type=int,
                        help='number of training episodes')
    parser.add_argument('--test-episodes', default=100, type=int,
                        help='number of testing episodes')
    parser.add_argument('--device', default='cpu', type=str,
                        help='network device [cpu, cuda]')
    return parser.parse_args()


def train(seed: int,
          device: str = 'cpu',
          train_episodes: int = 10000,
          train_env: str = 'CustomHopper-source-v0', **kwargs) -> A2C:
    """ trains the agent in the source environment """
    env = gym.make(train_env)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy = A2CPolicy(
        env.observation_space.shape[-1], env.action_space.shape[-1], **kwargs)
    agent = A2C(policy,
                device=device, **kwargs)

    num_episodes = 0
    num_timesteps = 0
    while num_episodes < train_episodes:
        env.seed(seed)

        done = False
        obs = env.reset()
        while not done:
            action, action_log_prob, state_value = agent.predict(obs)
            next_state, reward, done, _ = env.step(action)
            agent.store_outcome(obs, action_log_prob,
                                reward, done, state_value)
            obs = next_state
            num_timesteps += 1
            if num_timesteps % agent.batch_size == 0:
                agent.update_policy()
        num_episodes += 1

    return agent


def test(seed: int, agent: A2C,
         test_episodes: int = 100,
         test_env: str = 'CustomHopper-target-v0') -> float:
    """ tests the agent in the target environment """
    env = gym.make(test_env)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_episodes = 0
    episode_rewards = []
    while num_episodes < test_episodes:
        env.seed(seed)

        done = False
        obs = env.reset()
        rewards, steps = (0, 0)
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            obs = next_state
        num_episodes += 1
        episode_rewards.append(rewards)
    er = np.array(episode_rewards)

    return er


def pooling(kwargs: dict, seed, device, train_episodes, test_episodes):

    agent = train(seed=seed,
                  device=device,
                  train_episodes=train_episodes, **kwargs)

    return test(seed, agent,
                test_episodes=test_episodes), kwargs


def gridsearch(args, params, seeds=[1, 2, 3]):
    results = []
    keys = list(params.keys())
    for param in list(itertools.product(*params.values())):
        kwargs = dict(zip(keys, param))
        pool = list()
        for iter, seed in enumerate(seeds):
            er, _ = pooling(kwargs,
                            seed=seed,
                            device=args.device,
                            train_episodes=args.train_episodes,
                            test_episodes=args.test_episodes)
            pool.append(er)
        res = np.array(pool)
        cov = res.std() / res.mean()  # coefficient of variation
        score = res.mean() * (1 - cov)
        print(
            f'score: {score:.2f} | reward: {res.mean():.2f} +/- {res.std():.2f} | params: {kwargs}')
        results.append([score, res.mean(), res.std(), kwargs])

    results.sort(key=lambda x: x[0], reverse=True)
    print(f'\ngrid search - ranking scores:')
    print("----------------------------")
    for rank, candidate in enumerate(results):
        print(
            f'{rank + 1} | score: {candidate[0]:.2f} | reward: {candidate[1]:.2f} +/- {candidate[2]:.2f} | params: {candidate[3]}')
        if rank + 1 == 3:
            break

    return max(results, key=lambda x: x[0])


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    params = {
        'learning_rate': [1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4]
    }

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('\nWARNING: GPU not available, switch to CPU\n')
        args.device = 'cpu'

    prime = gridsearch(args, params)
    print(
        f'\nmaximum score: {prime[0]:.2f} | reward: {prime[1]:.2f} +/- {prime[2]:.2f} | optimal params: {prime[3]}')
    print("-------------")


if __name__ == '__main__':
    main()
