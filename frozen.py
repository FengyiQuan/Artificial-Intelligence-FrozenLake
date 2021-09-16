import gym
from QLearningAgent import QLearningAgent
from DynaQAgent import DynaQAgent
import numpy as np
import time, pickle, os
from DeepQLearningAgent import DeepQLearningAgent
env = gym.make('FrozenLake8x8-v0')
env.reset()

# env.reset()
# env.render()
epsilon = 0.5

epsilon_min = 0.001
lr_rate = 0.1
gamma = 0.9
step = 5
max_steps = 100




qLearning = QLearningAgent(env, epsilon, gamma, lr_rate)
dynaq = DynaQAgent(env, epsilon, gamma, lr_rate, step)
dynaq0 = DynaQAgent(env, epsilon, gamma, lr_rate, 0)

deepQ1 = DeepQLearningAgent(env,epsilon,gamma,lr_rate)
#deepQ2 = DQNAgent(state_size=1,action_size=4,epsilon=epsilon,gamma=gamma,lr_rate=lr_rate)

def train(agent, total_episodes):
    if agent == qLearning:
        print("this is Q learning")
    if agent == dynaq:
        print("this is dynaQ with",dynaq.steps)
    win = 0
    for ep in range(1,total_episodes + 1):
        state = env.reset()
        t = 0
        s = 0
        while t < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, next_state, reward, action)
            state = next_state
            t += 1
            s += reward
            if done:
                # when decays epsilon starts from 0.9 else set epsilon = 0.5
                # if agent.epsilon > epsilon_min:  # Epsilon Update
                #     agent.epsilon -= 1/total_episodes
                break
        if s >= 1 : win += 1
        if ep % 5000 == 0:
            #time.sleep(5)
            print('Episode',ep,'win:', win)


    #print ('Train Win: ',win)




train(dynaq,110000)

# def play(agent, numberEpisode=5000):
#     win = 0
#     for episode in range(numberEpisode):
#         state = env.reset()
#         t = 0
#         score = 0
#         while t < 100:
#             #env.render()
#             action = agent.choose_play_action(state)
#             next_state, reward, done, info = env.step(action)
#             state = next_state
#             t+=1
#             score +=reward
#             if done:
#                 break
#         if score >= 1:
#             win += 1
#     print('Episodes: ',total_episodes,'Play Win: ',win)
#



# train()
#
# with open("frozenLake_qTable.pkl", 'wb') as f:
#     pickle.dump(QLearningAgent.Q, f)
#
# with open("frozenLake_qTable.pkl", 'rb') as f:
# 	QLearningAgent.Q = pickle.load(f)

#print(QLearningAgent.Q)

