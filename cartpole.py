# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:12:12 2021

@author: hp
"""

from __future__ import print_function, division
from builtins import range

import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler, Nystroem

ALPHA = 0.1
GAMMA = 0.99

def epsilon_greedy(model, s, eps = 0.1):
    p = np.random.random()
    if p < (1- eps):
        values = model.predict_all_actions(s)
        return np.argmax(values)
    else:
        return model.env.action_space.sample()
    
def gather_samples(env, n_eps = 10000):
    samples = []
    for _ in range(n_eps):
        s = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))
            samples.append(sa)
            
            s, r, done, info = env.step(a)
        
    return samples

class Model:
    def __init__(self, env):
        self.env = env
        samples = gather_samples(env)
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.w = np.zeros(dims)
        
    def predict(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x @ self.w
    
    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in range(self.env.action_space.n)]
    
    def grad(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x
    
def test_agent(model, env, n_eps = 20):
    reward_per_episode = np.zeros(n_eps)
    for it in range(n_eps):
        done = False
        episode_reward = 0
        s = env.reset()
        while not done:
            a = epsilon_greedy(model, s, eps = 0)
            s, r, done, info = env.step(a)
            episode_reward += r
        reward_per_episode[it] = episode_reward
    
    return np.mean(reward_per_episode)

def watch_agent(model, env, eps):
    done = False
    episode_reward = 0
    s = env.reset()
    while not done:
        a = epsilon_greedy(model, s, eps = eps)
        s, r, done, info = env.step(a)
        env.render()
        episode_reward += r
    env.close()
    print(f"Episode reward: {episode_reward}")
    
if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    
    model = Model(env)
    reward_per_episode = []
    
    watch_agent(model, env, eps = 0)
    n_episodes = 1500
    
    for it in range(n_episodes):
        s = env.reset()
        episode_reward = 0
        done = False
        while not done:
            a = epsilon_greedy(model, s)
            s2, r, done, info = env.step(a)
            
            if done:
                target = r
            else:
                values = model.predict_all_actions(s2)
                target = r + GAMMA * np.max(values)
            
            g = model.grad(s, a)
            err = target - model.predict(s, a)
            model.w += g * ALPHA * err
            
            episode_reward += r
            
            s = s2
        
        if (it + 1) % 50 == 0:
            print(f"Episode: {it + 1}, Reward: {episode_reward}")
            
        if it > 20 and np.mean(reward_per_episode[-20:]) == 200:
            print("Early exit")
            break
        
        reward_per_episode.append(episode_reward)
        
    test_reward = test_agent(model, env)
    print(f"Average test reward: {test_reward}")
    
    plt.plot(reward_per_episode)
    plt.title("Reward per episode")
    plt.show()
    
    watch_agent(model, env, eps = 0)
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    