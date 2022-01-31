# -*- coding: utf-8 -*-
# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy

# Environment
env = gym.make("Taxi-v3")
env.render()
# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 2000
num_of_steps = 100 # per each episode

#Parameters for decreasing the exploration rate.
epsilon = 1
max_epsilon = 1               
min_epsilon = 0.01
decay_rate = 0.01        

tr_rewards = []
tr_actions = []

#Starting with all zeros.
Q_table = numpy.zeros((500, 6))
# Training
for i in range(num_of_episodes):
    state = env.reset()
    tot_reward = 0
    tot_actions = 0
    for j in range(num_of_steps):
        random_val = random.uniform(0, 1)
        if (random_val < epsilon):
            action = random.randint(0, 5)
        else:
            action = numpy.argmax(Q_table[state])
        
        tot_actions += 1
        next_state, reward, done, info = env.step(action)
        
        Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * numpy.max(Q_table[next_state]) - Q_table[state, action]) 
        tot_reward += reward
        state = next_state
        
        if done == True:
            print(tot_reward)
            break
    
    #Decreasing the exploration rate.
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*numpy.exp(-decay_rate*num_of_episodes)
    
    tr_actions.append(tot_actions)
    tr_rewards.append(tot_reward)
            
print(f'Average of total reward: {numpy.mean(tr_rewards)}\nAverage of the total actions: {numpy.mean(tr_actions)}')

#Testing
env.reset()
rewards = []
actions = []
for i in range(10):
    state = env.reset()
    done = False
    total_rewards = 0
    total_actions = 0
    for j in range(25):
        env.render()
        
        action = numpy.argmax(Q_table[state])
        total_actions += 1
        
        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            actions.append(total_actions)
            break
        state = new_state
print(f'Average of total reward: {numpy.mean(rewards)}\nAverage of the total actions: {numpy.mean(actions)}')
env.close()