#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:12:06 2018

@author: jack.lingheng.meng
"""
from gym import spaces
from datetime import datetime
import os
import numpy as np
from collections import deque
import csv

from LASAgent.LASAgent_Actor_Critic import LASAgent_Actor_Critic


class InternalEnvOfAgent(object):
    """
    This class provides Internal Environment for an agent.
    """
    def __init__(self, sess, agent_name, 
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
        agent_name: string
            the name of the agent this internal environment serves for
        
        observation_space: gym.spaces.Box datatype
            observation space of "agent_name". if we use x_order_MDP
            the actual_observation_space should be:
                (observation_space * x_order_MDP)
        
        action_space: gym.spaces.Box datatype
            action space of "agent_name"
        
        observation_space_name: string array
            name of each entry of observation space
            (hold for complex reward function definition.)
        
        action_space_name: string array
            name of each entry of action space 
            (hold for complex reward function definition.)
        
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
                        i.e. reward is provided
        
        load_pretrained_agent_flag: boolean default = False
            if == True: load pretrained agent, otherwise randomly initialize.
        """
        self.tf_session = sess
        
        self.x_order_MDP = x_order_MDP
        self.x_order_MDP_observation_sequence = deque(maxlen = self.x_order_MDP)
        self.x_order_MDP_observation_type = x_order_MDP_observation_type
        
        self.occupancy_reward_type = occupancy_reward_type
        self.interaction_mode = interaction_mode
        #####################################################################
        #                       Initialize agent                            #
        #####################################################################
        self.agent_name = agent_name
        
        self.observation_space = observation_space
        self.actual_observation_space = spaces.Box(low = np.tile(self.observation_space.low,self.x_order_MDP),
                                                   high = np.tile(self.observation_space.high,self.x_order_MDP), 
                                                   dtype = np.float32)
        self.action_space = action_space
        
        self.observation_space_name = observation_space_name
        self.action_space_name = action_space_name
        
        # Model saving directory
        self.agent_model_save_dir = os.path.join(os.path.abspath('..'),'ROM_Experiment_results',self.agent_name)
        
        if load_pretrained_agent_flag == False:
            self.agent = LASAgent_Actor_Critic(self.tf_session,
                                               self.agent_name,
                                               self.actual_observation_space,
                                               self.action_space,
                                               actor_lr = 0.0001, actor_tau = 0.001,
                                               critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                               minibatch_size = 64,
                                               max_episodes = 50000, max_episode_len = 1000,
                                               # Exploration Strategies
                                               exploration_action_noise_type = 'ou_0.2',
                                               exploration_epsilon_greedy_type = 'none',
                                               # Save Summaries
                                               save_dir = self.agent_model_save_dir,
                                               experiment_runs = datetime.now().strftime("%Y%m%d-%H%M%S"),
                                               # Save and Restore Actor-Critic Model
                                               restore_actor_model_flag = False,
                                               restore_critic_model_flag = False)
        elif load_pretrained_agent_flag == True:
            self._initialize_pretrained_agent()
        else:
            raise Exception('Please set load_pretrained_agent parameter!')
        #####################################################################
        #                 Interaction data saving directory                 #
        #####################################################################
        self.interaction_data_dir = os.path.join(os.path.abspath('..'),
                                                      'ROM_Experiment_results',
                                                      self.agent_name,
                                                      'interaction_data')
        if not os.path.exists(self.interaction_data_dir):
            os.makedirs(self.interaction_data_dir)
        self.interaction_data_file = os.path.join(self.interaction_data_dir,
                                                  datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')
        with open(self.interaction_data_file, 'a') as csv_datafile:
            fieldnames = ['Time', 'Observation', 'Reward', 'Action']
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
        observation_for_x_order_MDP = self._generate_observation_for_x_order_MDP(self.x_order_MDP_observation_sequence.copy(),
                                                                                 self.x_order_MDP_observation_type)
        if self.interaction_mode == 'real_interaction':
            reward = self._reward_occupancy(observation_for_x_order_MDP)
        elif self.interaction_mode == 'virtual_interaction':
            reward = external_reward
        else:
            raise Exception('Please choose right interaction mode!')
        print('Reward of {} is: {}'.format(self.agent_name, reward))
        done = False
        action = self.agent.perceive_and_act(observation_for_x_order_MDP, reward, done)
        # Logging interaction data
        self._logging_interaction_data(observation_for_x_order_MDP,
                                       reward,
                                       action)
        return action
    
    def _generate_observation_for_x_order_MDP(self, observation_sequence,
                                              x_order_MDP_observation_type):
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
        observation_for_x_order_MDP
        """
        observation_for_x_order_MDP = []
        if x_order_MDP_observation_type == 'concatenate_observation':
            # Extract observation for each agent and concatenate obs-sequence
            while observation_sequence:
                obs_temp = observation_sequence.popleft()
                observation_for_x_order_MDP = np.append(observation_for_x_order_MDP, obs_temp)
        elif x_order_MDP_observation_type == 'average_observation':
            observation_for_x_order_MDP = 0
        else:
            raise Exception('Please choose a proper x_order_MDP_observation_type!')
        
        return observation_for_x_order_MDP
    
    def _reward_occupancy(self, observation,
                          x_order_MDP,
                          reward_type = 'IR_distance'):
        """
        Calculate reward based on occupancy i.e. the IRs data
        
        Parameters
        ----------
        observation: array
            observation array
        
        x_order_MDP: int default=1
            define the order of MDP. If x_order_MDP != 1, we combine multiple
            observations as one single observation.
        
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
        # 
        self.reward = reward_temp / x_order_MDP
        return self.reward
    
    def _initialize_pretrained_agent(self):
        """
        This function is to load pretrained models. This function 
        is actually to reinitialize an agent with pretrained models.
        
        (Call this function only if you want to start learning with most recently 
        trained agent.)
        """
        # Search for most recent model in model directory
        model_directory = os.path.join(self.agent_model_save_dir,'models')
        model_created_date = []
        for directory_temp in os.listdir(model_directory):
            # Only compare date
            if '2018' in directory_temp:
                model_created_date.append(directory_temp)
        directory_of_most_recent_models = max(model_created_date)
        # Instantiate Agent With the Pretrained Model
        self.agent = LASAgent_Actor_Critic(self.tf_session,
                                           self.agent_name,
                                           self.actual_observation_space,
                                           self.action_space,
                                           actor_lr = 0.0001, actor_tau = 0.001,
                                           critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                                           minibatch_size = 64,
                                           max_episodes = 50000, max_episode_len = 1000,
                                           # Exploration Strategies
                                           exploration_action_noise_type = 'ou_0.2',
                                           exploration_epsilon_greedy_type = 'none',
                                           # Save Summaries
                                           save_dir = self.agent_model_save_dir,
                                           experiment_runs = directory_of_most_recent_models,
                                           # Save and Restore Actor-Critic Model
                                           restore_actor_model_flag = True,
                                           restore_critic_model_flag = True)
        
        
    def stop(self):
        """
        This interface function is to save trained models for the agent:
            1. actor-critic model
            2. environment model
        
        (Try to call this function before shut down learning to maintain most
        recently trained agent, although there is a periodic saving which cannot
        ensure saving the most recent trained agent.)
        """
        # Save Actor-Critic model
        self.agent.extrinsic_actor_model.save_actor_network(self.agent.episode_counter)
        self.agent.extrinsic_critic_model.save_critic_network(self.agent.episode_counter)
        # Save Environment Model
        self.agent.saved_env_model_counter += 1
        self.agent.environment_model.save_environment_model_network(self.agent.saved_env_model_counter)
        # Save Replay Buffer ?? (not really necessary)
        
        # Save
        
        
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
        self.agent._train()

    def _logging_interaction_data(self, observation_for_x_order_MDP,
                                  reward,
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
            fieldnames = ['Time', 'Observation', 'Reward', 'Action']
            writer = csv.DictWriter(csv_datafile, fieldnames = fieldnames)
            writer.writerow({'Time':datetime.now().strftime("%Y%m%d-%H%M%S"),
                             'Observation': observation_for_x_order_MDP, 
                             'Reward': reward, 
                             'Action':action})
        
        
        
       