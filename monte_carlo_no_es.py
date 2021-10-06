# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:27:09 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from iterative_policy_evaluation_deterministic import print_policy, print_values
from gridworld import negative_grid, standard_grid
from monte_carlo_es import max_dict

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps = 0.1):
    p = np.random.random()
    
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    
def play_game(grid, policy):
    s = (2, 0)
    grid.set_state(s)
    a = random_action(policy[s])
    
    states_actions_rewards = [(s, a, 0)]
    while True:
        r = grid.move(a)
        s = grid.current_state()
        
        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_action(policy[s])
            states_actions_rewards.append((s, a, r))
            
    states_actions_returns = []
    first = True
    G = 0
    
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    
    states_actions_returns.reverse()
    return states_actions_returns

if __name__ == '__main__':
    grid = negative_grid(step_cost = -0.9)
    
    print("rewards:")
    print_values(grid.rewards, grid)
    
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    Q = {}
    returns = {}
    states = grid.all_states()
    
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s, a)] = []
        else:
            pass
        
    deltas = []
    
    for t in range(10000):
        if t % 1000 == 0:
            print(t)
            
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()

        for s, a, G in states_actions_returns:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
                
        deltas.append(biggest_change)
    
        for s in policy.keys():
            a = max_dict(Q[s])[0]
            policy[s] = a
            
    plt.plot(deltas)
    plt.show()

    V = {}
    
    for s in policy.keys():
        V[s] = max_dict(Q[s])[1]
    
    print("Values:")
    print_values(V, grid)
    
    print("Policy: ")
    print_policy(policy, grid)








    
    
    
    
    
    
    
    
    
    