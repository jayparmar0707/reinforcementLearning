# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:15:49 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

np.random.seed(1)
NUM_TRIALS = 10000
BANDIT_PROBS = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1.
        self.b = 1.
        self.N = 0.
        
    def pull(self):
        return np.random.uniform(0, 1) < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1
        
def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label = f"real p: {b.p: .4f}, win rate: {b.a - 1} / {b.N}")
    
    plt.title(f"Bandit dist. after {trial} trials")
    plt.legend()
    plt.show()
    
def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBS]
    rewards = np.empty(NUM_TRIALS)
    sample_points = [50, 100, 500, 1000, 2500, 5000, 7500, 9999]
    
    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])
        
        if i in sample_points:
            plot(bandits, i)
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
        
    print("Total rewards: ", rewards.sum())
    print("Win rate: ", rewards.sum() / NUM_TRIALS)
    print("Each bandit run: ", [b.N for b in bandits])

if __name__ == "__main__":
    experiment()    