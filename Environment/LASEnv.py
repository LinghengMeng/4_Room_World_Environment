#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:55:34 2018

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
import warnings

import time

from collections import deque
import re

from .UtilitiesForEnv import get_all_object_name_and_handle, deprecated

from IPython.core.debugger import Tracer

class LASEnv(gym.Env):
    def __init__(self, IP = '127.0.0.1', Port = 19997,
                 reward_function_type = 'occupancy'):
        """
        Instantiate LASEnv. LASEnv is the interface between LAS and Environment. Thus, LASEnv is the internal environment of LAS.
        
        Parameters
        ----------
        IP: string default = '127.0.0.1'
            IP address to connect V-REP server.
         
        Port: int default = 19997
            Port to communicate with V-REP server.
        
        reward_function_type: str default = 'occupancy'
            Choose reward function type.
            Options:
                1. 'occupancy'
                
        """
        print ('Initialize LASEnv ...')
        # ========================================================================= #
        #                      Initialize V-REP related work                        #
        # ========================================================================= # 
        # Connect to V-REP server
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart(IP,Port,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            print ('LASEnv connected to remote V-REP API server')
        else:
            print ('LASEnv failed connecting to remote V-REP API server')
        
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
        
#        self._get_prox_op_mode = vrep.simx_opmode_streaming 
#        self._get_light_op_mode = vrep.simx_opmode_streaming
        
        # Start simulating in V-REP
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        # ========================================================================= #
        #           Call utility function to get object names and handles           #
        # ========================================================================= #        
        self.proxSensorHandles, self.proxSensorNames, \
        self.lightHandles, self.lightNames, \
        self.jointHandles, self.jointNames, \
        self.visitorNames, self.visitorHandles = get_all_object_name_and_handle(self.clientID, self._def_op_mode, vrep)
        # ========================================================================= #
        #               Initialize LAS action and observation space                 #
        # ========================================================================= # 
        print("Initialize LAS action and observation space...")
        self.prox_sensor_num = len(self.proxSensorHandles)
        self.smas_num = len(self.jointHandles)
        self.lights_num = len(self.lightHandles)
        # Sensor range:
        #   prox sensor: 0 or 1 (Only Proximity Sensor, no light data)
        self.sensors_dim = self.prox_sensor_num
        self.obs_max = np.array([1.]*self.sensors_dim)      
        self.obs_min = np.array([0.]*self.sensors_dim)
        # Actuator range:
        #   sma: not sure ??
        #   light intensity(only use B-color-channel): [0, 1]
        self.actuators_dim = self.smas_num + self.lights_num 
        # Set action range to [-1, 1], when send command to V-REP useing 
        # (action + 1) / 2 to transform action to range [0,1]
        self.act_max = np.array([1]*self.actuators_dim)
        self.act_min = np.array([-1]*self.actuators_dim)
        # Agent should be informed about observation_space and action_space to initialize
        # agent's observation and action dimension and value limitation.
        self.observation_space = spaces.Box(self.obs_min, self.obs_max, dtype = np.float32)
        self.action_space = spaces.Box(self.act_min, self.act_max, dtype = np.float32)
        print("Initialization of LAS done!")
        # save the name of each entry in observation and action
        light_name = matlib.repmat(np.reshape(self.lightNames,[len(self.lightNames),1]), 1,1).flatten()
        self.observation_space_name = self.proxSensorNames # only proxSensor
        self.action_space_name = np.concatenate((self.jointNames, light_name))
        # ========================================================================= #
        #                    Initialize Reward Function Type                        #
        # ========================================================================= #        
        self.reward_function_type = reward_function_type
    
    def step(self, action):
        """
        Take one step of interaction.
        
        Parameters
        ----------
        action: ndarray
            action[0: smas_num] are action corresponding to sma
            action[smas_num: end] are action corresponding to light color
        
        Returns
        -------
        observation: ndarray
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
            Tracer()()
            raise ValueError("Find nan value in action!")
        # Clip action to avoid improer action command
        action = np.clip(action, self.act_min, self.act_max)
        action = (action + 1) / 2.0
        # Split action for light and sma
        action = np.ndarray.flatten(action)  # flatten array
        action_smas = action[:self.smas_num]
        action_lights_intensity = action[self.smas_num:]
        # Taking action

        #vrep.simxPauseCommunication(self.clientID,True)     #temporarily halting the communication thread 
        self._set_all_joint_position(action_smas)
        # Actually only set light color
        self._set_all_light_intensity(action_lights_intensity)
        #vrep.simxPauseCommunication(self.clientID,False)    #and evaluated at the same time

        # Set a small sleep time to avoid getting nan sensor data
        time.sleep(0.01)

        # Observe current state
        try:
            self.observation = self._self_observe()
        except ValueError:
            self._nanObervationFlag = 1
            print("Observation has NAN value.")
        
        # This reward is non-interactive reward i.e. it's not affected by visitor.
        # Therefore, it's only used for tunning hyper-parameters of LASAgent
        if self.reward_function_type == 'occupancy':
            #   1. 'IR_distance': based on IR distance from detected object to IR
            #   2. 'IR_state_ratio': the ratio of # of detected objects and all # 
            #                        of IR sensors 
            #   3. 'IR_state_number': the number of detected objects
            self.reward = self._reward_occupancy(self.observation, 'IR_distance')
        else:
            raise ValueError('No reward function: {}'.format(self.reward_function_type))
        
        
        done = False
        info = []
        return self.observation, self.reward, done, info

    def _self_observe(self):
        """
        This observe function is for LAS:
            distance from the detected object to proximity sensors (if no 
            detected object, the value is 0)
            
        Returns
        -------
        observation: ndarray (proxDistances)
        """
        # For IRs only proxDistances is used in ROM Exhibit simulation
        _, proxDistances, _ = self._get_all_prox_data()
        # Combine observation (No light data)
        observation = proxDistances
        return observation

    def _reward_occupancy(self, observation, reward_type = 'IR_distance'):
        """
        Calculate reward based on occupancy i.e. the # of IRs been activated
        
        Parameters
        ----------
        observation: array
            observation array
            
        reward_type: string default='IR_distance'
            1. 'IR_distance': based on IR distance from detected object to IR
            2. 'IR_state_ratio': the ratio of # of detected objects and all # 
                                 of IR sensors 
            3. 'IR_state_number': the number of detected objects
        
        Returns
        -------
        reward: float
            the value of reward
        """
        prox_distances = observation[:self.prox_sensor_num]
        # Make here insistent with IR data
        #   1. 'IR_distance': based on IR distance from detected object to IR
        #   2. 'IR_state_ratio': the ratio of # of detected objects and all # 
        #                        of IR sensors 
        #   3. 'IR_state_number': the number of detected objects
        reward_temp = 0.0
        if reward_type == 'IR_distance':
            for distance in prox_distances:
                if distance != 0:
                    reward_temp += 1/distance
        elif reward_type == 'IR_state_ratio':
            for distance in prox_distances:
                if distance != 0:
                    reward_temp += 1
            reward_temp = reward_temp / len(prox_distances)
        elif reward_type == 'IR_state_number':
            for distance in prox_distances:
                if distance != 0:
                    reward_temp += 1
        else:
            raise Exception('Please choose a proper reward type!')
        self.reward = reward_temp
        return self.reward

    def reset(self):
        """
        Reset environment.
        
        Returns
        -------
        observation:
            
        reward:
            
        done:
            
        info:
            
        """
        #vrep.simxStopSimulation(self.clientID, self._def_op_mode)
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        
        self.observation = self._self_observe()
        self.reward = self._reward_occupancy(self.observation)
        
        done = False
        return self.observation#, self.reward, done
        
    def destroy(self):
        """
        Stop simulation on server and release connection.
        """
        vrep.simxStopSimulation(self.clientID, self._def_op_mode)
        vrep.simxFinish(self.clientID)
        
    def close_connection(self):
        """
        Release connnection, but not stop simulation on server.
        """
        vrep.simxFinish(self.clientID)
    # ========================================================================= #
    #                                                                           # 
    #                           Set Sma and Light Color                         #
    #                                                                           # 
    # ========================================================================= #     
    def _set_all_joint_position(self, targetPosition):
        """
        Set all joint position.
        
        Parameters
        ----------
        targetPosition: ndarray
        target position of each joint
        """
        jointNum = self.smas_num
        if jointNum != len(targetPosition):
            raise ValueError('Joint targetPosition is error!!!')

        for i in range(jointNum):
            res = vrep.simxSetJointTargetPosition(self.clientID, self.jointHandles[i], targetPosition[i], self._set_joint_op_mode)
    
    def _set_all_light_intensity(self, action_lights_intensity):
        """
        Set all light intensity which actually only set the B-color-channel of
        light.
        
        Parameters
        ----------
        action_lights_intensity: array
            the value of light intensity
        """
        # R = 1 & G = 1
        action_lights_color = np.ones((len(action_lights_intensity), 3))
        for i, B_color in enumerate(action_lights_intensity):
            action_lights_color[i,2] = B_color
        self._set_all_light_state_and_color(action_lights_color)
    
    def _set_all_light_state_and_color(self, targetColor):
        
        """
        Set all light sate and color
        
        Parameters
        ----------
        targetColor: (n,3) ndarray
            target color of each light
        """
        lightNum = self.lights_num
        if len(targetColor) != (lightNum):
            raise ValueError('len(targetColor) != lightNum*3')
        
        ######## Inner function: remote function call to set light state #######
        def _set_light_state_and_color(clientID, name, handle, targetState, targetColor, opMode):
            emptyBuff = bytearray()
            res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID,
                                                                           name,
                                                                           vrep.sim_scripttype_childscript,
                                                                           'setLightStateAndColor',
                                                                           [handle, targetState],targetColor,[],emptyBuff,
                                                                           opMode)
            return res
        ########################       inner function end       ###############
        # light state is always on.
        targetState = np.ones(self.lights_num, dtype = int) 
        for i, lightColor in enumerate(targetColor):
           res = _set_light_state_and_color(self.clientID, str(self.lightNames[i]), self.lightHandles[i], targetState[i], targetColor[i,:], self._set_light_op_mode)
    # ========================================================================= #
    #                                                                           #         
    #                    Get Proximity Sensor and Light Data                    #
    #                                                                           #
    # ========================================================================= # 
    def _get_all_prox_data(self):
        """
        Get all proximity sensory data.
        
        Returns
        -------
        proxStates: array (Not used in ROM System)
            state of each IR sensor
        
        proxDistances: array
            distance from IR to detected object i.e. the less the distance, the
            closer the obeject is to IR, but the exception is when no object being
            dected the distance is 0.
            
        proxPosition: (n,3) ndarray (Not used in ROM System)
            position of detected object relative to the sensor reference frame
        """
        proxSensorNum = len(self.proxSensorHandles)
        proxStates = np.zeros(proxSensorNum)
        proxDistances = np.zeros(proxSensorNum)
        proxPosition = np.zeros([proxSensorNum, 3])
        for i in range(proxSensorNum):
            code, proxStates[i], proxPosition[i,:], handle, snv = vrep.simxReadProximitySensor(self.clientID, self.proxSensorHandles[i], self._get_prox_op_mode)
            proxPosition[i,:] = np.multiply(proxPosition[i,:], proxStates[i])
            proxDistances[i] = np.sqrt(np.sum(np.square(proxPosition[i,:])))
            #print('code:{}, proxState:{}, proxPosition:{}'.format(code,proxStates[i],np.multiply(proxPosition[i,:], proxStates[i])))
            if np.sum(np.isnan(proxStates[i])) != 0:
                raise ValueError("Find nan value in proximity sensor data!")
        return proxStates, proxDistances, proxPosition
    
    def _get_all_light_intensity(self):
        """
        This function is used to get all lights intensity which is indicated by
        B-color channel from original RGB clolor.
        
        Returns
        -------
        lightIntensity: array
            each entry corresponds to one intensity of illumination of one light
        """
        lightStates, lightDiffsePart, lightSpecularPart = self._get_all_light_data()
        lightNum = len(self.lightHandles)
        lightIntensity = np.zeros(lightNum)
        for i, lightColor in enumerate(lightDiffsePart):
            lightIntensity[i] = lightColor[2] # 0:R 1:G 2:B
        return lightIntensity
    
    def _get_all_light_data(self):
        """
        Get all light data.
        
        Returns
        -------
        lightStates: ndarray
            light states
        lightDiffsePart: ndarray
            light color
        lightSpecularPart: ndarray
            also light color, currently not used
        """
        lightNum = len(self.lightHandles)
        #print("lightNum:{}".format(lightNum))
        lightStates = np.zeros(lightNum)
        lightDiffsePart = np.zeros([lightNum,3])
        lightSpecularPart = np.zeros([lightNum,3])
        
        # inner function to get light state and color
        def _get_light_state_and_color(clientID, name , handle, op_mode):
            emptyBuff = bytearray()
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,
                                                                                   name,
                                                                                   vrep.sim_scripttype_childscript,
                                                                                   'getLightStateAndColor',
                                                                                   [handle],[],[],emptyBuff,
                                                                                   op_mode)
            if res==vrep.simx_return_ok:
                #print ('getLightStateAndColor works! ',retStrings[0]) # display the reply from V-REP (in this case, just a string)
                lightState = retInts[0]
                diffusePart = [retFloats[0],retFloats[1],retFloats[2]]
                specularPart = retFloats[3],retFloats[4],retFloats[5]
                return lightState, diffusePart, specularPart
            else:
                warnings.warn("Remote function call: getLightStateAndColor fail in Class AnyLight.")
                return 0, [0,0,0], [0,0,0]
        # inner function end
        
        for i in range(lightNum):
            lightStates[i], lightDiffsePart[i,:], lightSpecularPart[i,:] = _get_light_state_and_color(self.clientID, str(self.lightNames[i]), self.lightHandles[i], self._get_light_op_mode)            
            if np.sum(np.isnan(lightDiffsePart[i,:])) != 0:
                raise ValueError("Find nan value in light color!")
        return lightStates, lightDiffsePart, lightSpecularPart    
    
