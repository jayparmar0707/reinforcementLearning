# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:43:12 2020

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt

EPS = 0.1
NUM_TRIALS = 10000
BANDIT_PROBS = [0.2, 0.5, 0.75]

class bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0
        
    def pull(self):
        return np.random.random() < self.p
    
    def update(self, x):
        self.N = self.N + 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        
def experiment():
    bandits = [bandit(p) for p in BANDIT_PROBS]
    
    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_times_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print("optimal_j: ", optimal_j)
    
    for i in range(NUM_TRIALS):
        if np.random.uniform(0, 1) < EPS:
            j = np.random.randint(len(bandits))
            num_times_explored += 1
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])
            
        if j == optimal_j:
            num_times_optimal += 1
        
        x = bandits[j].pull()
        
        rewards[i] = x
        
        bandits[j].update(x) 
        
    for b in bandits:
        print("mean estimate: ", b.p_estimate)
        
    print("Total reward earned: ", rewards.sum())
    print("overall win rate: ", rewards.sum() / NUM_TRIALS)
    print("num_times_exploited: ", num_times_exploited)
    print("num_times_explored: ", num_times_explored)
    print("num_times_optimal: ", num_times_optimal)
    
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBS))
    plt.show()
    
if __name__ == "__main__":
    experiment()