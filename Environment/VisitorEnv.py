#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:56:08 2018

@author: jack.lingheng.meng
"""

try:
    from .VrepRemoteApiBindings import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import gym
from gym import spaces
import time
import numpy as np
import warnings

from .UtilitiesForEnv import get_all_object_name_and_handle, deprecated

class VisitorEnv(gym.Env):
    def __init__(self, IP = '127.0.0.1',Port = 19999):
        """
        Instantiate LASEnv. LASEnv is the interface between LAS and Environment. Thus, LASEnv is the internal environment of LAS.
        
        Parameters
        ----------
        IP: string default = '127.0.0.1'
        IP address to connect V-REP server.
        
        
        Port: int default = 19999
        Port to communicate with V-REP server.
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
        #             Initialize single Visitor action and observation space        #
        # ========================================================================= # 
        print("Initialize single Visitor action and observation space...")
        self.lights_num = len(self.lightHandles)
        self.visitor_num = 1 # single visitor
        # Observation space for single visitor
        #   lightStates
        #   lightColors
        #   lightPositions
        #   visitorBodyPosition
        self.sensors_dim = self.lights_num * (1+3+3) + self.visitor_num * 3
        # Action space for single visitor (later we might need more action like wave hand...)
        #   move: 0 or 1
        #   target position: (x, y, z)
        self.actuators_dim = 1 + self.visitor_num * 3
        # Sensor range:
        #   light state: 0 or 1
        #   light color: [0, 1] * 3 
        #   light position: [width, length, height] depends on the size of floor
        #   visitorBodyPosition: [width, length, height] depends on the size of floor
        obs_light_max = np.array([1.] * self.lights_num * (1+3))
        obs_light_min = np.array([0.] * self.lights_num * (1+3))
        obs_light_pos_max = np.array([9., 9., 9.] * self.lights_num)
        obs_light_pos_min = -np.array([9., 9., 9.] * self.lights_num)
        obs_visitor_tar_pos_max = np.array([9., 9., 9.] * self.visitor_num)
        obs_visitor_tar_pos_min = -np.array([9., 9., 9.] * self.visitor_num)
        
        self.obs_max = np.concatenate([obs_light_max.flatten(), obs_light_pos_max.flatten(),obs_visitor_tar_pos_max.flatten()])
        self.obs_min = np.concatenate([obs_light_min.flatten(), obs_light_pos_min.flatten(),obs_visitor_tar_pos_min.flatten()])
        
        # Actuator range:
        #   visitor's target position: [width, length, height] depends on the size of floor
        self.act_max = np.array([1., 9., 9., 9.] * self.visitor_num)
        self.act_min = -np.array([0., 9., 9., 9.] * self.visitor_num)
        
        self.observation_space = spaces.Box(self.obs_min, self.obs_max)
        self.action_space = spaces.Box(self.act_min, self.act_max)
        print("Initialization of single Visitor environment done!")

        # ========================================================================= #
        #                       Initialize other variables                          #
        # ========================================================================= #         
        self.reward = 0
        self.done = False
        self.info = []
        self.observation = []
        
    def step(self, visitorName, action):
        """
        A specific interface for red excited visitor:
            return observation:
                light state: observation[:lightNum]
                light color: observation[lightNum:lightNum * 4]
                light position: observation[lightNum * 4:lightNum * 5]
                visitor position: observation[lightNum*5:]
        """
        action = np.clip(action,self.act_min, self.act_max)
        move = action[0]
        position = action[1:3] # we can leave z coordinate

        # if move == 1, move; otherwise don't move.
        if move == 1:
            #vrep.simxPauseCommunication(self.clientID,True)
            #print("Set Position in Vrep: {}".format(position))
            self._set_single_visitor_position(visitorName, position)
            #vrep.simxPauseCommunication(self.clientID,False)
        
        self.observation = self._self_observe(visitorName)
        #print("len(observation):{}".format(len(observation)))
        self.reward = 0
        self.done = False
        self.info = []
        return self.observation, self.reward, self.done, self.info
    
    def _set_single_visitor_position(self, visitorName, position):
        visitorIndex = np.where(self.visitorNames == visitorName)
        if len(visitorIndex[0]) == 0:
            print("Not found visitor: {}".format(visitorName))
        else:
            vrep.simxSetObjectPosition(self.clientID, self.visitorHandles[visitorIndex], -1, [position[0],position[1],0], self._set_visitor_op_mode)
    
    def _get_single_visitor_body_position(self, visitorName):
        """
        Give visitorName, return bodyPosition
        """
        bodyPosition = np.zeros(3)
        visitorBodyIndex = np.where(self.visitorNames == visitorName)
        if len(visitorBodyIndex[0]) == 0:
            print("Not found visitor: {}".format(visitorName))
        else:
            res, bodyPosition = vrep.simxGetObjectPosition(self.clientID, self.visitorHandles[visitorBodyIndex], -1, self._get_light_op_mode)
        #print("Visitor position: {}".format(position))
        return np.array(bodyPosition)
        
    def _self_observe(self,visitorName):
        """
        This obervave function is for visitors:
            light state: observation[:lightNum]
            light color: observation[lightNum:lightNum * 4]
            light position: observation[lightNum * 4:lightNum * 5]
            visitor position: observation[lightNum*5:]
        """
        lightStates, lightDiffsePart, lightSpecularPart = self._get_all_light_data()
        lightPositions = self._get_all_light_position()
        visitorBodyPosition = self._get_single_visitor_body_position(visitorName)
        obser_for_red_light_excited_visitor = np.concatenate((lightStates,
                                                                   lightDiffsePart.flatten(),
                                                                   lightPositions.flatten(),
                                                                   visitorBodyPosition.flatten()))
        
        return obser_for_red_light_excited_visitor        
        
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
                return -1, [0,0,0], [0,0,0]
        # inner function end
        
        for i in range(lightNum):
           lightStates[i], lightDiffsePart[i,:], lightSpecularPart[i,:] = _get_light_state_and_color(self.clientID, str(self.lightNames[i]), self.lightHandles[i], self._get_light_op_mode)
           if lightStates[i] == -1:
               lightStates[i] = 0
               lightDiffsePart[i,:] = [0, 0, 0]
               lightSpecularPart[i,:] = [0, 0, 0]
           elif lightStates[i] == 0:
               lightDiffsePart[i,:] = [0, 0, 0]
               lightSpecularPart[i,:] = [0, 0, 0]
        return lightStates, lightDiffsePart, lightSpecularPart

    def _get_all_light_position(self):
        """
        Get all lights position.
        
        Returns
        -------
        lightPositions: ndarray
        """
        lightNum = self.lights_num
        #print("_get_all_light_position lightNum:{}".format(lightNum))
        lightPositions = np.zeros([lightNum, 3]) # 3: (x, y, z)
        for i in range(lightNum):
            res, lightPositions[i,:] = vrep.simxGetObjectPosition(self.clientID, self.lightHandles[i], -1, self._get_light_op_mode)
        return lightPositions         
        
    def _reward(self):
        """ calculate reward for visitor, currently it's always 0."""
        reward = 0
        return reward
    
    @deprecated('Please use method "VisitorEnv._self_observe(bodyName)" to get initial obervation for visitor.')
    def reset(self, bodyName):
        """
        Reset environment for a single visitor.
        
        Parameters
        ----------
        bodyName: string
            the name of visitor's body.
        
        Returns
        -------
        observation:
        reward:
        done:
        info:
        """
        vrep.simxStartSimulation(self.clientID, self._def_op_mode)
        
        self.observation = self._self_observe(bodyName)
        
        self.reward = self._reward()
        
        self.done = False
        self.info = []
        return self.observation, self.reward, self.done, self.info       

    @deprecated('You should not call destory() in visitor environment, if you want to keep LAS working. \
                Please call "VisitorEnv.close_connection() to release connection."')
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
        
        