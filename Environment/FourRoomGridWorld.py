#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:43:02 2018

@author: jack.lingheng.meng
"""

try:
    from .VrepRemoteApiBindings import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means it is very likely')
    print ('that either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import gym
from gym import spaces
import numpy as np
import numpy.matlib as matlib
import random
import math
import warnings

import time

from collections import deque
import re

from .UtilitiesForEnv import get_all_object_name_and_handle, get_object_position

from IPython.core.debugger import Tracer

class FourRoomGridWorld(gym.Env):
    def __init__(self, IP = '127.0.0.1', Port = 19997):
        """
        Instantiate FourRoomGridWorld. 
        
        Parameters
        ----------
        IP: string default = '127.0.0.1'
            IP address to connect V-REP server.
         
        Port: int default = 19997
            Port to communicate with V-REP server.
                
        """
        print ('Initialize FourRoomGridWorld ...')
        # ========================================================================= #
        #                      Initialize V-REP related work                        #
        # ========================================================================= # 
        # Connect to V-REP server
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart(IP,Port,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            print ('FourRoomGridWorld connected to remote V-REP API server')
        else:
            print ('FourRoomGridWorld failed connecting to remote V-REP API server')
        
        # Initialize operation mode of communicated command in V-REP
        #   To get sensor data
        #     vrep.simx_opmode_buffer:   does not work, don't know why?
        #     vrep.simx_opmode_blocking: too slow
        #     vrep.simx_opmode_oneshot:  works pretty good
        self._def_op_mode = vrep.simx_opmode_blocking
        
        self._set_joint_op_mode = vrep.simx_opmode_oneshot
        self._set_light_op_mode = vrep.simx_opmode_oneshot
        self._set_visitor_op_mode = vrep.simx_opmode_oneshot

        self._get_prox_op_mode = vrep.simx_opmode_oneshot 
        self._get_light_op_mode = vrep.simx_opmode_oneshot
        
        
        # Start simulating in V-REP
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        # ========================================================================= #
        #     Call utility function to get object names, handles and positions      #
        # ========================================================================= #        
        self.wallBrickHandles, self.wallBrickNames, \
        self.floorTileHandles, self.floorTileNames, \
        self.hallwayHandles, self.hallwayNames, \
        self.goalHandles, self.goalNames,\
        self.standingParticipantHandles, self.standingParticipantNames = get_all_object_name_and_handle(self.clientID, self._def_op_mode, vrep)
        
        self.wallBrickPositions = get_object_position(self.wallBrickHandles, self.clientID, self._def_op_mode, vrep)
        self.floorTilePositions = get_object_position(self.floorTileHandles, self.clientID, self._def_op_mode, vrep)
        self.hallwayPositions = get_object_position(self.hallwayHandles, self.clientID, self._def_op_mode, vrep)
        self.goalPositions = get_object_position(self.goalHandles, self.clientID, self._def_op_mode, vrep)
        self.standingParticipantPositions = get_object_position(self.standingParticipantHandles, self.clientID, self._def_op_mode, vrep)
        # Save initial position of participant for reset
        self.initial_standingParticipantPositions = self.standingParticipantPositions
        # ========================================================================= #
        #               Initialize action and observation space                 #
        # ========================================================================= # 
        print("Initialize action and observation space...")
        # Discrete Observation:
        #   valide states are in Floor Tile and Hallyway
        self.observation_handles = np.concatenate((self.floorTileHandles, self.hallwayHandles))
        self.observation_positions = np.concatenate((self.floorTilePositions,self.hallwayPositions))
        self.observation_dim = len(self.observation_handles)
        self.observation_space = spaces.Discrete(self.observation_dim)
        # Discrete Action:
        #   0. up 
        #   1. down
        #   2. left
        #   3. right
        self.action_dim = 4
        self.action_space = spaces.Discrete(self.action_dim)
        
        print("Initialization of FourRoomGridWorld done!")
       
    def step(self, action):
        """
        Take one step of interaction.
        
        Parameters
        ----------
        action: int
        
        Returns
        -------
        observation: int
            obervation of environment after taking an action
        reward: float
            reward after taking an action
        done: bool
            whether simulation is done or not.
        info:
            some information for debugging
        """
        # To imporove system's robustness, we need to just ignore this action
        # rather than throw an exception.
        if np.sum(np.isnan(action)) != 0:
            raise ValueError("Find nan value in action!")
        # Check action to avoid improer action command
        if not self.action_space.contains(int(action)):
            raise ValueError('action value: {} is not in action_space'.format(action))
        # Taking action
        self._act(action)

        # Set a small sleep time to avoid getting nan sensor data
        time.sleep(0.01)

        # Observe current state
        self.observation = self._self_observe()

        # Calculate reward
        self.reward, done = self._reward_function(self.observation)
        
        info = []
        return self.observation, self.reward, done, info 

    def _self_observe(self):
        """
        return state of participant by mapping position to index in:
            self.floorTileHandles and self.hallwayHandles
        
        Returns
        -------
        new_state: int
        """
        # Get participant position
        standingParticipantPositions = get_object_position(self.standingParticipantHandles, 
                                                           self.clientID, 
                                                           self._def_op_mode, vrep)
        new_state = -1
        for index, position in enumerate(self.observation_positions):
            if standingParticipantPositions[0][0] == position[0] and standingParticipantPositions[0][1] == position[1]:
                new_state = index
                break
        if new_state == -1:
            raise Exception('Did not find new_state.')
            
        return new_state
    
    def _reward_function(self, observation):
        """
        If current state is in goal state, reward = 1.0 and done = True. 
        Otherwise, reward = 0.0 and done = False.
        
        Parameters
        ----------
        observation: int
            state of participant which is also the index in 
            self.observation_handles and self.observation_positions.
        
        Returns
        -------
        reward: float
        
        done: bool
            if current state is in goal state, done = True. Otherwise, done = False
        """
        # Check if observation is goal
        reward = 0.0
        done = False
        for goal_position in self.goalPositions:
            x = self.observation_positions[observation][0]
            y = self.observation_positions[observation][1]
            if x == goal_position[0] and y == goal_position[1]:
                reward = 1.0
                done = True
                break
        return reward, done
        
    def _act(self, action):
        """
        Take the action in V-REP
        """
        targetPosition, targetOrientation = self._transition_model(action)
        # Move to target position
        vrep.simxSetObjectPosition(self.clientID, 
                                   self.standingParticipantHandles[0], 
                                   -1, 
                                   targetPosition, 
                                   vrep.simx_opmode_blocking)
        # Move to target orientation
        vrep.simxSetObjectOrientation(self.clientID,
                                      self.standingParticipantHandles[0],
                                      -1, 
                                      targetOrientation, 
                                      vrep.simx_opmode_blocking)

    def _stochastic_primitive_action(self, action):
        """
        Get the stochatic action given action chosen by agent.
        
        Parameters
        ----------
        action: int
        
        Returns
        -------
        stochastic_action: int
        """
        random.seed(1)
        rand_number = random.random()
        if rand_number < 2/3:
            stochastic_action = action
        else:
            rest_actions = []
            for i in range(self.action_space.n):
                if i != action:
                    rest_actions.append(i)
            act_index = random.randint(0, len(rest_actions))
            stochastic_action = rest_actions[act_index]
        return stochastic_action

    def _transition_model(self, action):
        """
        Return the target position of participant after taking action.
        
        Parameters
        ----------
        action: int
        
        Returns
        -------
        targetPosition: [x, y, z]
        targetOrientation: 
            0. up: [0,0,90]
            1. down: [0,0,-90]
            2. left: [0,0,180]
            3. right: [0,0,0]
        """
        # Get participant position
        standingParticipantPositions = get_object_position(self.standingParticipantHandles, 
                                                           self.clientID, 
                                                           self._def_op_mode, vrep)
        # standingParticipantPositions only has one participant
        x = standingParticipantPositions[0][0]
        y = standingParticipantPositions[0][1]
        z = standingParticipantPositions[0][2]
        # Get stochastic action
        stochastic_action = self._stochastic_primitive_action(action)
        
        # Calculate new state
        if stochastic_action == 0:
            # up
            targetPosition = [x, y+1, z]
            targetOrientation = [0, 0, math.pi/2]
        elif stochastic_action == 1:
            # down
            targetPosition = [x, y-1, z]
            targetOrientation = [0, 0, -math.pi/2]
        elif stochastic_action == 2:
            # left
            targetPosition = [x-1, y, z]
            targetOrientation = [0, 0, math.pi]
        elif stochastic_action == 3:
            # right
            targetPosition = [x+1, y, z]
            targetOrientation = [0, 0, 0]
        else:
            raise ValueError('Wrong stochastic_action value.')
        # Check if targetPosition is Wall Brick
        for wall_brick in self.wallBrickPositions:
            if targetPosition[0] == wall_brick[0] and targetPosition[1] == wall_brick[1]:
                #print('Jump to wall. Do not move.')
                targetPosition = standingParticipantPositions[0]
                break
        return targetPosition, targetOrientation
                
    def reset(self):
        """
        Returns
        -------
        obseravtion:
        rward:
        done:
        info:
        """
        # Reset participant position
        vrep.simxSetObjectPosition(self.clientID, 
                                   self.standingParticipantHandles[0], 
                                   -1, 
                                   self.initial_standingParticipantPositions[0], 
                                   vrep.simx_opmode_blocking)
        
        # Get state
        self.observation = self._self_observe()

        # Calculate reward
        self.reward, done = self._reward_function(self.observation)
        
        info = []
        return self.observation, self.reward, done, info 