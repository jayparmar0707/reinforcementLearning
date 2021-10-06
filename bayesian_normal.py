# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:28:04 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

BANDIT_MEANS = [1, 2, 3]
NUM_TRIALS = 2000

class Bandit:
    def __init__(self,true_mean):
        self.true_mean = true_mean
        self.m = 0
        self.tau = 1
        self.lambda_ = 1
        self.sum_x = 0
        self.N = 0
        
    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        self.sum_x += x
        self.lambda_ += self.tau
        self.N += 1
        self.m = self.tau * self.sum_x / self.lambda_
        
def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    
    for b in bandits:
        y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
        plt.plot(x, y, label = f"true mean: {b.true_mean: .4f} number of plays: {b.N}")
            
    plt.title(f"Bandit distribution after {trial} trials")
    plt.legend()
    plt.show()
    
def run_experiment():
    bandits = [Bandit(m) for m in BANDIT_MEANS]
    rewards = np.empty(NUM_TRIALS)
    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    
    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])
        
        if i in sample_points:
            plot(bandits, i)
            
        x = bandits[j].pull()
        bandits[j].update(x)
        rewards[i] = x
        
        
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)
    
    plt.plot(cumulative_average)
    for m in BANDIT_MEANS:
        plt.plot(np.ones(NUM_TRIALS) * m)
        
    plt.show()
    
    return cumulative_average
    
if __name__ == "__main__":
    run_experiment()
        
        
        
        
        
        
