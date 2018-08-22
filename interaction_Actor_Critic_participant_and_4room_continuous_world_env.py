#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:27:31 2018

@author: jack.lingheng.meng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:35:51 2018

@author: jack.lingheng.meng
"""



import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import os

from Environment.FourRoomContinuousWorld import FourRoomContinuousWorld
from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic



if __name__ == '__main__':
    
    with tf.Session() as sess:
        # Instantiate LAS environment object
        env = FourRoomContinuousWorld('127.0.0.1', 19997)
        observation = env.reset()
        
        #######################################################################
        #                          Instatiate Agent                       #
        #######################################################################
        agent_name = 'Single_Agent'
        agent = LASAgent_Actor_Critic(sess, 'Single_Agent',
                                      env.observation_space,
                                      env.action_space,
                                      actor_lr = 0.0001, actor_tau = 0.001,
                                      critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                      minibatch_size = 64,
                                      max_episodes = 50000, max_episode_len = 1000,
                                      # Exploration Strategies
                                      exploration_action_noise_type = 'ou_0.2',
                                      exploration_epsilon_greedy_type = 'epsilon-greedy-max_1_min_0.05_decay_0.999',
                                      # Save Summaries
                                      save_dir = os.path.join(os.path.abspath('..'),'Four_Room_World_Experiment_results',agent_name),
                                      experiment_runs = '20180822-102426', #datetime.now().strftime("%Y%m%d-%H%M%S"),
                                      # Save and Restore Actor-Critic Model
                                      restore_actor_model_flag = True,
                                      restore_critic_model_flag = True)
        #######################################################################
        
        # Step counter
        
       
        episode_num = 1000
        for episode in range(episode_num):
            observation = env.reset()
            done = False
            reward = 0
            i = 1
            while True:
                # LAS interacts with environment.
                action = agent.perceive_and_act(observation, reward, done)
                if done:
                    break
                # delay the observing of consequence of LASAgent's action
                observation, reward, done, info = env.step(action[0])
                
                i += 1
            print('Episode: {}, Steps: {}'.format(episode, i))