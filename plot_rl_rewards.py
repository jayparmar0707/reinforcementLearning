# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:08:33 2021

@author: hp
"""
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type = str, required = True,
                  help = 'either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_trader_rewards/{args.mode}.npy')

print(f"mean: {a.mean(): .2f}, min: {a.min(): .2f}, max: {a.max(): .2f}")

if args.mode == 'train':
    plt.plot(a)
else:
    plt.hist(a, bins = 20)

plt.title(args.mode)
plt.show()