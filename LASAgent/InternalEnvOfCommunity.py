#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:47:50 2018

@author: jack.lingheng.meng
"""
import os
from datetime import datetime
import numpy as np
import csv
from collections import deque
import tensorflow as tf
from gym import spaces
from LASAgent.InternalEnvOfAgent import InternalEnvOfAgent
from LASAgent.RandomLASAgent import RandomLASAgent

class InternalEnvOfCommunity(object):
    """
    This class provides an internal environment for a community of agents to 
    interact with external environment.
    """
    def __init__(self, sess, community_name, community_size,
                 observation_space, action_space, 
                 observation_space_name, action_space_name,
                 x_order_MDP = 1,
                 x_order_MDP_observation_type = 'concatenate_observation',
                 occupancy_reward_type = 'IR_distance',
                 interaction_mode = 'real_interaction',
                 load_pretrained_agent_flag = False):
        """
        Initialize internal environment for an agent
        Parameters
        ----------
        community_name: string
            the name of the community this internal environment serves for
        
        community_size: int
            the # of agents living in the community 
        
        observation_space: gym.spaces.Box datatype
            firt-order observation space of "agent_name", if we use x_order_MDP
            the actual_observation_space should be:
                (observation_space * x_order_MDP)
        
        action_space: gym.spaces.Box datatype
            this is actually actuator space
        
        observation_space_name: list of string
            gives the name of each entry in observation space
        
        action_space_name: list of strings
            gives the name of each entry in action space
        
        x_order_MDP: int default=1
            define the order of MDP. If x_order_MDP != 1, we combine multiple
            observations as one single observation.
        
        x_order_MDP_observation_type: default = 'concatenate_observation'
            ways to generate observation for x_order_MDP:
                1. 'concatenate_observation'
                2. 'average_observation'
        
        occupancy_reward_type: string default = 'IR_distance'
            1. 'IR_distance': based on IR distance from detected object to IR
            2. 'IR_state_ratio': the ratio of # of detected objects and all # 
                                 of IR sensors 
            3. 'IR_state_number': the number of detected objects
        
        interaction_mode: string default = 'real_interaction'
            indicate interaction mode: 
                1) 'real_interaction': interact with real robot
                2) 'virtual_interaction': interact with virtual environment
        
        load_pretrained_agent_flag: boolean default = False
            if == True: load pretrained agent, otherwise randomly initialize.
        """
        self.tf_session = sess
        
        self.x_order_MDP = x_order_MDP
        self.x_order_MDP_observation_sequence = deque(maxlen = self.x_order_MDP)
        self.x_order_MDP_observation_type = x_order_MDP_observation_type
        
        self.occupancy_reward_type = occupancy_reward_type
        self.interaction_mode = interaction_mode
        
        self.load_pretrained_agent_flag = load_pretrained_agent_flag
        
        # Initialize community
        self.community_name = community_name
        self.community_size = community_size
        ####################################################################
        #                          Configuration
        ####################################################################
        # Config 1: with shared sensor
        self.community_config_obs = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15'],
                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
                                     'agent_3':['node#9','node#8','node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
                                     }
        self.community_config_act = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
                                     }
#        # Config 2: no shared sensor
#        self.community_config_obs = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
#                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
#                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
#        self.community_config_act = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
#                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
#                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
        # Config 3: share all sensor
#        self.community_config_obs = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15',\
#                                                'node#14','node#13','node#12','node#11','node#10','node#9','node#8','node#7','node#6',\
#                                                'node#5','node#4','node#3','node#2','node#1','node#0'],
#                                     'agent_2':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15',\
#                                                'node#14','node#13','node#12','node#11','node#10','node#9','node#8','node#7','node#6',\
#                                                'node#5','node#4','node#3','node#2','node#1','node#0'],
#                                     'agent_3':['node','node#22','node#21','node#20','node#19','node#18','node#17','node#16','node#15',\
#                                                'node#14','node#13','node#12','node#11','node#10','node#9','node#8','node#7','node#6',\
#                                                'node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
#        self.community_config_act = {'agent_1':['node','node#22','node#21','node#20','node#19','node#18','node#17'],
#                                     'agent_2':['node#16','node#15','node#14','node#13','node#12','node#11','node#10','node#9','node#8'],
#                                     'agent_3':['node#7','node#6','node#5','node#4','node#3','node#2','node#1','node#0']
#                                     }
        ####################################################################
        
        self.observation_space = observation_space
        self.observation_space_name = observation_space_name
        self.action_space = action_space
        self.action_space_name = action_space_name
        # Information on Partition Config of observation and action space
        self.agent_community_partition_config = \
                self._create_community_partition_from_config(self.community_name,
                                                             self.community_size,
                                                             self.observation_space,
                                                             self.observation_space_name,
                                                             self.action_space,
                                                             self.action_space_name,
                                                             self.community_config_obs,
                                                             self.community_config_act,
                                                             self.x_order_MDP,
                                                             self.x_order_MDP_observation_type)
        # Creat a community of agents
        #   1. 'random_agent'
        #   2. 'actor_critic_agent'
        self.agent_community = self._create_agent_community(self.agent_community_partition_config,
                                                            self.load_pretrained_agent_flag,
                                                            agent_config = 'actor_critic_agent')  
        ####################################################################
        #                          Initialize Summary
        ####################################################################
        self.total_step_counter = 0
        
        #####################################################################
        #                 Interaction data saving directory                 #
        #                                                                   #
        # Note: each agent living in this community has its own Interaction #
        #       data saving directory. Here is a saving of interaction from #
        #       the perspective of Agent-Community.
        #####################################################################
        self.interaction_data_dir = os.path.join(os.path.abspath('..'),
                                                      'ROM_Experiment_results',
                                                      self.community_name,
                                                      'interaction_data')
        if not os.path.exists(self.interaction_data_dir):
            os.makedirs(self.interaction_data_dir)
        self.interaction_data_file = os.path.join(self.interaction_data_dir,
                                                  datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
        with open(self.interaction_data_file, 'a') as csv_datafile:
            fieldnames = ['Time', 'Observation_queue', 'Observation_partition',
                          'Reward_partition', 
                          'Action_partition', 'Action']
            writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
            writer.writeheader()
        
        
    def interact(self, observation, external_reward = 0, done = False):
        """
        The interface function interacts with external environment.
        
        Parameters
        ----------
        observation: ndarray
            the observation received from external environment
        external_reward: float (optional)
            this parameter is used only when external reward is provided by 
            simulating environment
            
        Returns
        -------
        action: ndarray
            the action chosen by intelligent agent
        """
        # Add the (x_order_MDP)th observation
        self.x_order_MDP_observation_sequence.append(observation)
        if len(self.x_order_MDP_observation_sequence) != self.x_order_MDP:
            raise Exception('Feeded observation size is not equal to x_order_MDP!')
        # Generate partitioned observation for x_order_MDP with multiple-agents
        #   Note: use shallow copy:
        #             self.x_order_MDP_observation_sequence.copy(),
        #         otherwise the self.x_order_MDP_observation_sequence is reset to empty,
        #         after call this function.
        observation_partition = self._generate_observation_for_x_order_MDP(self.x_order_MDP_observation_sequence.copy(),
                                                                           self.x_order_MDP_observation_type,
                                                                           self.agent_community_partition_config)
        # Partition reward
        #   1. 'IR_distance': based on IR distance from detected object to IR
        #   2. 'IR_state_ratio': the ratio of # of detected objects and all # 
        #                        of IR sensors 
        #   3. 'IR_state_number': the number of detected objects
        reward_partition = self._partition_reward(observation_partition,
                                                  self.agent_community_partition_config,
                                                  self.x_order_MDP,
                                                  self.occupancy_reward_type)
        for agent_name in reward_partition.keys():
            print('Reward of {} is: {}'.format(agent_name,reward_partition[agent_name]))
        # Collect actions from each agent
        action_partition = self._collect_action(observation_partition,
                                                reward_partition,
                                                self.agent_community)
        # Combine actions from agents
        action = self._combine_action(action_partition, self.agent_community_partition_config)
        
        self.total_step_counter += 1
        # Logging interaction data
        self._logging_interaction_data(self.x_order_MDP_observation_sequence.copy(),
                                       observation_partition,
                                       reward_partition,
                                       action_partition,
                                       action)
        
        return action
    
    def _create_community_partition_from_config(self, community_name,
                                                community_size, 
                                                observation_space,
                                                observation_space_name,
                                                action_space,
                                                action_space_name,
                                                community_config_obs,
                                                community_config_act,
                                                x_order_MDP,
                                                x_order_MDP_observation_type):
        """
        Partition a community consisting of #community_size agents according to
        configuration.
        
        Parameters
        ----------
        community_name: string
            the name of the community this internal environment serves for
        
        community_size: int
            size of community i.e. # of agents in the community
        
        observation_space: gym.spaces.Box datatype
            observation space of "agent_name"
        
        observation_space_name: list of string
            gives the name of each entry in observation space
        
        action_space: gym.spaces.Box datatype
            action space of "agent_name"
        
        action_space_name: list of strings
            gives the name of each entry in action space
        
        community_config_obs: 
            give the information on how to configurate observation for the community
        
        community_config_act:
            give the information on how to configurate action for the community
        
        x_order_MDP: int 
            define the order of MDP. If x_order_MDP != 1, we combine multiple
            observations as one single observation.
        
        x_order_MDP_observation_type: string
            ways to generate observation for x_order_MDP:
                1. 'concatenate_observation'
                2. 'average_observation'
        
        Returns
        -------
        agent_community_partition: dictionary
            a dictionary of community partition configuration in where:
                agent_community_partition = {'agent_name_1': {'obs_mask':agent_obs_mask,
                                                              'act_mask':agent_act_mask,
                                                              'observation_space':observation_space,
                                                              'action_space':action_space
                                                              }}
        """
        agent_community_partition = {}
        # Find partition mask for observation and action
        for agent_conf_index in self.community_config_obs.keys():
            agent_name = community_name + '_' + agent_conf_index
            agent_obs_mask = np.zeros(len(observation_space_name))
            agent_act_mask = np.zeros(len(action_space_name))
            for obs_node in self.community_config_obs[agent_conf_index]:
                for i in range(len(observation_space_name)):
                    if observation_space_name[i].endswith(obs_node):
                        agent_obs_mask[i] = 1
            for act_node in self.community_config_act[agent_conf_index]:
                for j in range(len(action_space_name)):
                    if action_space_name[j].endswith(act_node):
                        agent_act_mask[j] = 1
            agent_community_partition[agent_name] = {}
            agent_community_partition[agent_name]['obs_mask'] = agent_obs_mask
            agent_community_partition[agent_name]['act_mask'] = agent_act_mask
        # Create observation and action space and their corresponding name 
        # for each agent
        for agent_name in agent_community_partition.keys():
            # observation
            obs_dim = int(np.sum(agent_community_partition[agent_name]['obs_mask']))
            obs_low = np.zeros(obs_dim)
            obs_high = np.zeros(obs_dim)
            obs_name = [] # name for observation entry
            obs_temp_i = 0
            for obs_i in range(len(agent_community_partition[agent_name]['obs_mask'])):
                if agent_community_partition[agent_name]['obs_mask'][obs_i] == 1:
                    obs_low[obs_temp_i] = observation_space.low[obs_i]
                    obs_high[obs_temp_i] = observation_space.high[obs_i]
                    obs_name.append(observation_space_name[obs_i])
                    obs_temp_i += 1
            # action
            act_dim = int(np.sum(agent_community_partition[agent_name]['act_mask']))
            act_low = np.zeros(act_dim)
            act_high = np.zeros(act_dim)
            act_name = [] # name for action entry
            act_temp_i = 0
            for act_i in range(len(agent_community_partition[agent_name]['act_mask'])):
                if agent_community_partition[agent_name]['act_mask'][act_i] == 1:
                    act_low[act_temp_i] = action_space.low[act_i]
                    act_high[act_temp_i] = action_space.high[act_i]
                    act_name.append(action_space_name[act_i])
                    act_temp_i += 1
            # Generate observation_space accroding to:
            #       x_order_MDP and x_order_MDP_observation_type
            if x_order_MDP_observation_type == 'concatenate_observation':
                obs_low = np.tile(obs_low, x_order_MDP)
                obs_high = np.tile(obs_high, x_order_MDP)
                agent_community_partition[agent_name]['observation_space'] = spaces.Box(low=obs_low,high=obs_high, dtype = np.float32)
                agent_community_partition[agent_name]['observation_space_name'] = np.tile(obs_name, x_order_MDP)
            elif x_order_MDP_observation_type == 'average_observation':
                agent_community_partition[agent_name]['observation_space'] = spaces.Box(low=obs_low,high=obs_high, dtype = np.float32)
                agent_community_partition[agent_name]['observation_space_name'] = obs_name
            else:
                raise Exception()
            agent_community_partition[agent_name]['action_space'] = spaces.Box(low=act_low, high=act_high, dtype = np.float32)
            agent_community_partition[agent_name]['action_space_name'] = act_name
        return agent_community_partition
        
    def _create_agent_community(self, agent_community_partition_config,
                                load_pretrained_agent_flag,
                                agent_config = 'random_agent'):
        """
        Create agent community according to community partition configuration
        and agent configuration.
        
        Parameters
        ----------
        agent_community_partition_config: dict of dict
            contains information on how to partition observation and action space
        
        load_pretrained_agent_flag: boolean default = False
            if == True: load pretrained agent, otherwise randomly initialize.
        
        agent_config: (not determined)
            contains information on how to configurate each agent:
                1. 'random_agent': random agent
                2. 'actor_critic_agent': actor_critic agent
        
        Returns
        -------
        agent_community: dict
            a dict of agent living in the community:
                agent_community = {'agent_name': agent}
        """
        agent_community = {}
        if agent_config == 'random_agent':
            for agent_name in agent_community_partition_config.keys():
                observation_space = agent_community_partition_config[agent_name]['observation_space']
                action_space = agent_community_partition_config[agent_name]['action_space']
                # Instantiate learning agent
                agent_community[agent_name] = RandomLASAgent(observation_space, action_space)
                print('Create random_agent community done!')
        elif agent_config == 'actor_critic_agent':
            for agent_name in agent_community_partition_config.keys():
                observation_space = agent_community_partition_config[agent_name]['observation_space']
                action_space = agent_community_partition_config[agent_name]['action_space']
                observation_space_name = agent_community_partition_config[agent_name]['observation_space_name']
                action_space_name = agent_community_partition_config[agent_name]['action_space_name']
                # Instantiate LAS-agent
                #   x_order_MDP = 1: 
                #       observations have been combined, so use x_order_MDP=1
                #   x_order_MDP_observation_type = 'concatenate_observation':
                #       doesn't matter, since x_order_MDP = 1
                #   occupancy_reward_type = 'IR_distance':
                #       doesn't matter, since interaction_mode = 'virtual_interaction'
                #   interaction_mode = 'virtual_interaction':
                #       because rward is provided by InternalEnvOfCommunity
                x_order_MDP = 1
                x_order_MDP_observation_type = 'concatenate_observation' # doesn't matter
                occupancy_reward_type = 'IR_distance'                    # doesn't matter
                interaction_mode = 'virtual_interaction'
                agent_community[agent_name] = InternalEnvOfAgent(self.tf_session,\
                                                                 agent_name, 
                                                                 observation_space, 
                                                                 action_space,
                                                                 observation_space_name, 
                                                                 action_space_name,
                                                                 x_order_MDP,
                                                                 x_order_MDP_observation_type,
                                                                 occupancy_reward_type,
                                                                 interaction_mode,
                                                                 load_pretrained_agent_flag)
            print('Create actor_critic_agent community done!')
        else:
            raise Exception('Please choose a right agent type!')
        return agent_community
        
    def _partition_observation(self, observation, agent_community_partition_config):
        """
        Partition whole observation into each agent's observation field according
        to community partition configuration.
        
        Parameters
        ----------
        observation: ndarray
            observation of whole external environment.
            
        agent_community_partition_config: dict of dict
            contains info on how to partition whole observation and action.
        
        Returns
        -------
        observation_partition: dict
            a dist of observation where each value corresponds to the observation
            of one agent:
                observation_partition = {'agent_name': observation}
        
        """
        observation_partition = {}
        for agent_name in agent_community_partition_config.keys():
            obs_index = []
            obs_index = np.where(agent_community_partition_config[agent_name]['obs_mask'] == 1)
            observation_temp = []
            observation_temp = observation[obs_index]
            observation_partition[agent_name] = observation_temp
        return observation_partition
    
    def _generate_observation_for_x_order_MDP(self, observation_sequence,
                                              x_order_MDP_observation_type,
                                              agent_community_partition_config):
        """
        
        Parameters
        ----------
        observation_sequence: deque object
            with x_order_MDP observations
            
        x_order_MDP_observation_type: string
            ways to generate observation for x_order_MDP:
                1. 'concatenate_observation'
                2. 'average_observation'
        
        agent_community_partition_config: dict of dict
            contains info on how to partition whole observation and action.
        
        Returns
        -------
        observation
        """
        observation_partition = {}
        if x_order_MDP_observation_type == 'concatenate_observation':
            # Initialize observation_partition
            for agent_name in agent_community_partition_config.keys():
                observation_partition[agent_name] = []
            # Extract observation for each agent and concatenate obs-sequence
            while observation_sequence:
                obs_temp = observation_sequence.popleft()
                obs_partition_temp = self._partition_observation(obs_temp, agent_community_partition_config)
                for agent_name_temp in obs_partition_temp.keys():
                    observation_partition[agent_name_temp] = np.append(observation_partition[agent_name_temp],obs_partition_temp[agent_name_temp])
        elif x_order_MDP_observation_type == 'average_observation':
            observation_partition = 0
        else:
            raise Exception('Please choose a proper x_order_MDP_observation_type!')
        
        return observation_partition
    
    def _partition_reward(self, observation_partition, 
                          agent_community_partition_config,
                          x_order_MDP,
                          reward_type = 'IR_distance'):
        """
        Partition reward based on observation_partition and agent_community_partition_config
        
        Parameters
        ----------
        observation_partition: dict
            observation_partition = {'agent_name': observation}
            
        agent_community_partition_config: dict of dict
            agent_community_partition_config = {'agent_name_1': {'obs_mask':agent_obs_mask,
                                                                 'act_mask':agent_act_mask,
                                                                 'observation_space':observation_space,
                                                                 'action_space':action_space
                                                                 }}
        x_order_MDP: int 
            define the order of MDP. If x_order_MDP != 1, we combine multiple
            observations as one single observation.
            
        reward_type: string
            1. 'IR_distance': based on IR distance from detected object to IR
            2. 'IR_state_ratio': the ratio of # of detected objects and all # 
                                 of IR sensors 
            3. 'IR_state_number': the number of detected objects
        
        Returns
        -------
        reward_partition: dict
            a dict of reward:
                reward_partition = {'agent_name': reward}
        
        """
        reward_partition = {}
        for agent_name in observation_partition.keys():
            IR_data = []
            for i, name in enumerate(agent_community_partition_config[agent_name]['observation_space_name']):
                # Get IR sensor info
                if 'ir_node' in name:
                    IR_data.append(observation_partition[agent_name][i])
            # Make here insistent with IR data
            # 1. 'IR_distance': sum of reciprocal of distance from detected 
            #                   object to IR.
            # 2. 'IR_state': ratio of # of detected objects and # of IR
            reward_temp = 0.0
            if reward_type == 'IR_distance':
                for distance in IR_data:
                    if distance != 0:
                        reward_temp += 1/distance
            elif reward_type == 'IR_state_ratio':
                for distance in IR_data:
                    if distance != 0:
                        reward_temp += 1
                reward_temp = reward_temp / len(IR_data)
            elif reward_type == 'IR_state_number':
                for distance in IR_data:
                    if distance != 0:
                        reward_temp += 1
            else:
                raise Exception('Please choose a proper reward type!')
            # Average occupancy
            reward_partition[agent_name] = reward_temp / x_order_MDP
        return reward_partition
    
    def _collect_action(self, observation_partition, reward_partition, agent_community):
        """
        Collect actions from each agent into a dict.
        
        Parameters
        ----------
        observation_partition: dict
            a dict of observation partitions:
                observation_partition = {'agent_name': observation}
        
        reward_partition: dict
            a dict of reward partitions:
                reward_partition = {'agent_name': reward}
        
        agent_community: dict
            a dict of agents:
                agent_community = {'agent_name': agent_object}
        
        Returns
        -------
        action_partition: dict
            a dict of actions:
                action_partition = {'agent_name': action}
        
        """
        done = False
        action_partition = {}
        for agent_name in agent_community.keys():
            action_partition[agent_name] = agent_community[agent_name].interact(observation_partition[agent_name],\
                            reward_partition[agent_name],done)
        return action_partition
    
    def _combine_action(self, action_partition, agent_community_partition_config):
        """
        Combine each agent's action into a whole action.
        
        Parameters
        ----------
        action_partition: dict
            a dict of actions:
                action_partition = {'agent_name': action}
        
        agent_community_partition_config: dict of dict
            contains info on how to partition whole observation and action.
        
        Returns
        -------
        action: ndarray
            an array of action on the whole action space
        """
        action = np.zeros(self.action_space.shape)
        for agent_name in agent_community_partition_config.keys():
            act_index = []
            act_index = np.where(agent_community_partition_config[agent_name]['act_mask']==1)
            action[act_index] = action_partition[agent_name]
        return action
    
    def _extrinsic_reward_func(self, observation):
        """
        This function is used to provide extrinsic reward.
        """
        reward = 1
        return reward
    
    def start(self):
        """
        This interface function is to load pretrained models.
        """
        
        
    def stop(self):
        """
        This interface function is to save trained models.
        """
        
    def feed_observation(self, observation):
        """
        This interface function only receives observation from environment, but
        not return action.
        
        (Training could also be done when feeding observation.)
        
        Parameters
        observation: ndarray
            the observation received from external environment
        """
        self.x_order_MDP_observation_sequence.append(observation)
        
        # Train all agent when feeding observation
        for agent_name in self.agent_community.keys():
            self.agent_community[agent_name].agent._train()
    
    def _logging_interaction_data(self, x_order_MDP_observation_sequence,
                                  observation_partition,
                                  reward_partition,
                                  action_partition,
                                  action):
        """
        Saving interaction data
        
        Parameters
        ----------
        observation_for_x_order_MDP: array
        
        reward: float
        
        action: array
        
        """
        with open(self.interaction_data_file, 'a') as csv_datafile:
            fieldnames = ['Time', 'Observation_queue', 'Observation_partition',
                          'Reward_partition', 
                          'Action_partition', 'Action']
            writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
            writer.writerow({'Time':datetime.now().strftime("%Y%m%d-%H%M%S"),
                             'Observation_queue': x_order_MDP_observation_sequence,
                             'Observation_partition': observation_partition,
                             'Reward_partition': reward_partition,
                             'Action_partition': action_partition,
                             'Action':action})
# =================================================================== #
#                   Initialization Summary Functions                  #
# =================================================================== #     










    