#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 21:46:33 2018

@author: jack.lingheng.meng
"""
from Environment.FourRoomGridWorld import FourRoomGridWorld
import time

if __name__ == '__main__':
    four_room_grid_world_env = FourRoomGridWorld()
    observation, reward, done, info = four_room_grid_world_env.reset()
    
    print('Interacting ...')
    
    while not done:
        # Random Action
        action = four_room_grid_world_env.action_space.sample()
        # Interact with environment
        observation, reward, done, info = four_room_grid_world_env.step(action)
        
        time.sleep(0.1)
    
    print('Interaction done.')