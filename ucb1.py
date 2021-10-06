# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:46:08 2021

@author: hp
"""

from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

BANDIT_PROBS = [0.2, 0.5, 0.75]
NUM_TRIALS = 100000

class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0.
        self.N = 0.
        
    def pull(self):
        return np.random.uniform(0, 1) < self.p
    
    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) *self.p_estimate + x)/ self.N
            
def ucb(mean, n, nj):
    return mean + np.sqrt(2 * np.log(n) / nj)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBS]
    rewards = np.empty(NUM_TRIALS)
    total_plays = 0
    
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        
    for i in range(NUM_TRIALS):
        j = np.argmax([ucb(b.p_estimate, NUM_TRIALS, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        rewards[i] = x
        
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)
    
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBS))
    plt.xscale('log')
    plt.show()
    
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBS))
    plt.show()
    
    for b in bandits:
        print(b.p_estimate)
        
    print("Total rewards: ", rewards.sum())
    print("Win rate: ", rewards.sum() / NUM_TRIALS)
    print("Number of times each bandit was selected:", [b.N for b in bandits])
    
    return cumulative_average

if __name__ == "__main__":
    experiment()