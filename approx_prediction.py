# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:29:38 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_policy, print_values
from sklearn.kernel_approximation import RBFSampler, Nystroem

ALPHA = 0.01
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'L', 'D', 'R')

def epsilon_greedy(greedy, s, eps = 0.1):
    p = np.random.random()
    if p < (1 - eps):
        return greedy[s]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def gather_samples(grid, n_episodes = 10000):
    samples = []
    
    for _ in range(n_episodes):
        s = grid.reset()
        samples.append(s)
        while not grid.game_over():
            a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            r = grid.move(a)
            s = grid.current_state()
            samples.append(s)
        
    return samples
    
class Model:
    def __init__(self, grid):
        samples = gather_samples(grid)
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.w = np.zeros(dims)
    
    def predict(self, s):
        x = self.featurizer.transform([s])[0]
        return x @ self.w
    
    def grad(self, s):
        x = self.featurizer.transform([s])[0]
        return x
    
if __name__ == '__main__':
    grid = standard_grid()
    
    print("rewards: ")
    print_values(grid.rewards, grid)
    
    greedy_policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U'
        }
    
    model = Model(grid)
    mse_per_episode = []
    
    episodes = 10000
    for it in range(episodes):
        #if (it + 1) % 100 == 0:
         #   print(it + 1)
        
        s = grid.reset()
        Vs = model.predict(s)
        episode_err = 0
        n_steps = 0
        
        while not grid.game_over():
            a = epsilon_greedy(greedy_policy, s)
            r = grid.move(a)
            s2 = grid.current_state()
            
            if grid.is_terminal(s2):
                target = r
            else:
                Vs2 = model.predict(s2)
                target = r + GAMMA * Vs2
                
            err = target - Vs
            g = model.grad(s)
            model.w += err * ALPHA * g
            
            n_steps += 1
            episode_err += err * err
            s = s2
            Vs = Vs2
    
        mse = episode_err / n_steps
        mse_per_episode.append(mse)
        
    plt.plot(mse_per_episode)
    plt.title("MSE per episode")
    plt.show()
    
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0
        
    print("values: ")
    print_values(V, grid)
    
    print("policy: ")
    print_policy(greedy_policy, grid)
        