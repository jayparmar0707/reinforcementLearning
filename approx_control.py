# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:25:28 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gridworld import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_policy, print_values
from sklearn.kernel_approximation import Nystroem, RBFSampler

ALPHA = 0.1
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ACTION2INT = {a: i for i, a in enumerate(ALL_POSSIBLE_ACTIONS)}
INT2ONEHOT = np.eye(len(ACTION2INT))

def epsilon_greedy(model, s, eps = 0.1):
    p = np.random.random()
    if p < (1 - eps):
        values = model.predict_all_actions(s)
        return ALL_POSSIBLE_ACTIONS[np.argmax(values)]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    
def one_hot(k):
    return INT2ONEHOT[k]

def merge_state_actions(s, a):
    ai = one_hot(ACTION2INT[a])
    return np.concatenate((s, ai))

def gather_samples(grid, n_eps = 1000):
    samples = []
    for _ in range(n_eps):
        s = grid.reset()
        while not grid.game_over():
            a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            sa = merge_state_actions(s, a)
            samples.append(sa)
            
            r = grid.move(a)
            s = grid.current_state()
        
    return samples
            
        
class Model:
    def __init__(self, grid):
        samples = gather_samples(grid)
        
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.w = np.zeros(dims)
        
    def predict(self, s, a):
        sa = merge_state_actions(s, a)
        x = self.featurizer.transform([sa])[0]
        return x @ self.w
    
    def grad(self, s, a):
        sa = merge_state_actions(s, a)
        x = self.featurizer.transform([sa])[0]
        return x
    
    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in ALL_POSSIBLE_ACTIONS]
    
if __name__ == '__main__':
    grid = negative_grid(step_cost = -0.1)
    
    print("rewards: ")
    print_values(grid.rewards, grid)
    
    model = Model(grid)
    reward_per_episode = []
    state_visit_count = {}
    
    n_eps = 20000
    
    for it in range(n_eps):
        if (it + 1) % 1000 == 0:
            print(it + 1)
        
        s = grid.reset()
        state_visit_count[s] = state_visit_count.get(s, 0) + 1
        episode_reward = 0
        while not grid.game_over():
            a = epsilon_greedy(model, s)
            r = grid.move(a)
            s2 = grid.current_state()
            
            state_visit_count[s2] = state_visit_count.get(s2, 0) + 1
            
            if grid.is_terminal(s2):
                target = r
            else:
                values = model.predict_all_actions(s2)
                target = r + GAMMA * np.max(values)
                
            g = model.grad(s, a)
            err = target - model.predict(s, a)
            model.w += err * g * ALPHA
            
            s = s2
            
            episode_reward += r
        
        reward_per_episode.append(episode_reward)
        
        
    plt.plot(reward_per_episode)
    plt.title('Reward per episode')
    plt.show()
        
    V = {}
    greedy_policy = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            values = model.predict_all_actions(s)
            V[s] = np.max(values)
            greedy_policy[s] = ALL_POSSIBLE_ACTIONS[np.argmax(values)]
        else:
            V[s] = 0
            
    print("values: ")
    print_values(V, grid)
    print("policy: ")
    print_policy(greedy_policy, grid)
        
    print("state visit count: ")
    state_visit_count_arr = np.zeros((grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            if (i, j) in state_visit_count:
                state_visit_count_arr[i, j] = state_visit_count[(i, j)]
                
    df = pd.DataFrame(state_visit_count_arr)
    print(df)
                
        