#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:15:42 2018

@author: jack.lingheng.meng
"""
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
def plot_cumulative_reward(cumulativeReward):
    line, = plt.plot(cumulativeReward, '-+')
    plt.ion()
    #plt.ylim([0,10])
    plt.show()
    plt.pause(0.0001)

class RandomLASAgent():
    """
    Single LAS agent contorl all actuators i.e. non-distributed
    
    """
    def __init__(self, observation_space, action_space):
        """
        Parameters
        ----------
        observation_space: gym.spaces.Box
            
        action_space: gym.spaces.Box
            
        """
        self.action_space = action_space             # gym.spaces.Box object
        self.observation_space = observation_space   # gym.spaces.Box object
        
        # ========================================================================= #
        #                 Initialize Temprary Memory                                #
        # ========================================================================= # 
        # Temporary hard memory: storing every experience
        self._memory = deque(maxlen = 10000)
        # Temporary memory: variables about last single experience
        self._firstExperience = True
        self._observationOld = []   # observation at time t
        self._observationNew = []   # observation at time t+1
        self._actionOld = []        # action at time t
        self._actionNew = []        # action at time t+1
        # Cumulative reward
        self._cumulativeReward = 0
        self._cumulativeRewardMemory = deque(maxlen = 10000)
        self._rewardMemory = deque(maxlen = 10000)
        
    def interact(self, observation, reward, done):
        self._observation = observation
        self._reward = reward
        self._done = done
        
        self._rewardMemory.append([self._reward])
        self._cumulativeReward += reward
        self._cumulativeRewardMemory.append([self._cumulativeReward])
        # plot in real time
#        if len(self._memory) %200 == 0:
#            #plot_cumulative_reward(self._cumulativeRewardMemory)
#            plot_cumulative_reward(self._rewardMemory)
        
        self._actionNew = self._act()
        return self._actionNew
    
    def _act(self):
        # Sample function provided by gym.spaces.Box is much slower, so use our
        # own sample action is a better choice.
        action = self.action_space.sample()
#        smas = np.random.randn(self.env.smas_num)
#        lights_color = np.random.uniform(0,1,self.env.lights_num*3)
#        action = np.concatenate((smas, lights_color))

        return action
