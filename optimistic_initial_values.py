# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:01:57 2020

@author: hp
"""

from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

BANDIT_PROBS = [0.2, 0.5, 0.8]
EPS = 0.01
NUM_TRIALS = 100000

class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 5.
        self.N = 1.
    def pull(self):
        return np.random.random() < self.p
    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        
def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBS]
    
    rewards = np.zeros(NUM_TRIALS)
    
    for i in range(NUM_TRIALS):
        j = np.argmax([b.p_estimate for b in bandits])
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
        
    for b in bandits:
        print("mean estimate:", b.p_estimate)
        
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandits:", [b.N for b in bandits])
    
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.ylim([0, 1])
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBS))
    plt.show()
    
if __name__ == "__main__":
    experiment()