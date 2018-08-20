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
    
    while not done:
        action = four_room_grid_world_env.action_space.sample()
        observation, reward, done, info = four_room_grid_world_env.step(action)
        if done:
            print('done:'.format(done))
        time.sleep(0.1)