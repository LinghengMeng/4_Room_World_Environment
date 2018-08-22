#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:41:09 2018

@author: jack.lingheng.meng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow import keras
import numpy as np
import pandas as pd
import gym
from gym import wrappers
import tflearn
from tflearn.models.dnn import DNN
import argparse
import pprint as pp
from collections import deque
import time


from LASAgent.replay_buffer import ReplayBuffer
from LASAgent.noise import AdaptiveParamNoiseSpec,NormalActionNoise,OrnsteinUhlenbeckActionNoise

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, name, sess, observation_space,  action_space,
                 learning_rate, tau, batch_size,
                 actor_model_save_path = '',
                 target_actor_model_save_path = '',
                 restore_model_flag=False,
                 restore_model_version = 0):
        """
        Parameters
        ----------
        name: str
            
        sess: tf.Session
            
        observation_space: gym.spaces.Box
            
        action_space: gym.spaces.Box
            
        learning_rate: float
            
        tau: float
            
        batch_size: int
            
        restore_model_flag: bool default=False
        
        actor_model_save_path_and_name: str default = 'results/models/actor_model.ckpt'
        
        target_actor_model_save_path_and_name: str default = 'results/models/target_actor_model.ckpt'
        """
        
        self.name = name
        self.sess = sess
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.action_bound_high = action_space.high
        self.action_bound_low = action_space.low
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        
        # Info for load pre-trained actor models
        self.actor_model_save_path = actor_model_save_path
        self.target_actor_model_save_path = target_actor_model_save_path
        
        self.restore_model_flag = restore_model_flag
        self.restore_model_version = self._find_the_most_recent_model_version() 
        if self.restore_model_flag and self.restore_model_version == -1:
            raise Exception('You do not have pretrained models.\nPlease set "load_pretrained_agent_flag = False".')
        
        with tf.variable_scope(self.name) as self.scope:
            # Initialize or Restore Actor Network
            self.inputs, self.out, self.scaled_out, self._actor_model = self.create_actor_network()
            if self.restore_model_flag == True:
                temp_name = os.path.join(self.actor_model_save_path,
                                         self.name + '_' + str(self.restore_model_version)+'.ckpt')
                self._actor_model.load(temp_name)
                print('Restored actor model: {}'.format(temp_name))
            
            self.network_params = tf.trainable_variables(scope=self.name)
    
            # Initialize or Restore Target Network
            self.target_inputs, self.target_out, self.target_scaled_out, self._target_actor_model = self.create_actor_network()
            if self.restore_model_flag == True:
                temp_name = os.path.join(self.actor_model_save_path,
                                         self.name + '_target_' + str(self.restore_model_version)+'.ckpt')
                self._target_actor_model.load(temp_name)
                print('Restored target actor model: {}'.format(temp_name))
                
            self.target_network_params = tf.trainable_variables(scope=self.name)[len(self.network_params):]
    
            # Op for periodically updating target network with online network
            # weights
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                    for i in range(len(self.target_network_params))]
    
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
    
            # Combine the gradients here
            # The reason of negative self.action_gradient here is we want to do 
            # gradient ascent, and AdamOptimizer will do gradient ascent when applying
            # a gradient.
            self.unnormalized_actor_gradients = tf.gradients(
                    self.scaled_out, self.network_params, -self.action_gradient) 
            #Tracer()()
            self.actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size), self.unnormalized_actor_gradients))
    
            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.actor_gradients, self.network_params))
            # This is useless, since we use tf.variable_scope
            self.num_trainable_vars = len(
                self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        """
        
        """
        inputs = tflearn.input_data(shape=[None, self.s_dim],name = 'ActorInput')
        h1 = tflearn.fully_connected(inputs, 20)
        h1 = tflearn.activations.relu(h1)
        h1 = tflearn.layers.normalization.batch_normalization(h1)
        h2 = tflearn.fully_connected(h1, 10)
        h2 = tflearn.activations.relu(h2)
        h2 = tflearn.layers.normalization.batch_normalization(h2)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            h2, self.a_dim, activation='tanh', weights_init=w_init, name = 'ActorOutput') # action space is shifted to [-1,1]
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound_high, name = 'ActorScaledOutput')
        model = DNN(scaled_out, tensorboard_verbose = 3)

        return inputs, out, scaled_out, model
        
    def save_actor_network(self, actor_save_time = 0):
        """save actor and target actor model"""
        self._actor_model.save(os.path.join(self.actor_model_save_path,
                                            self.name + '_' + str(actor_save_time)+'.ckpt'))
        self._target_actor_model.save(os.path.join(self.target_actor_model_save_path,
                                                   self.name +'_target_' + str(actor_save_time)+'.ckpt'))
        print('Save actor networks.')

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
#        return self._actor_model.predict(inputs)

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })
#        return self._target_actor_model.predict(inputs)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def _find_the_most_recent_model_version(self):
        """
        Returns
        -------
        the_most_recent_model_version: int
            the most recent model version. If no saved model, return -1.
        """
        # Find the most recent version
        model_version = []
        for file_name_temp in os.listdir(self.actor_model_save_path):
            if self.name+'_target_' in file_name_temp:
                _, version_temp = file_name_temp.split('.')[0].split(self.name+'_target_')
                model_version.append(version_temp)
        if len(model_version) != 0:
            the_most_recent_model_version = max([int(i) for i in model_version])
        else:
            the_most_recent_model_version = -1
        return the_most_recent_model_version

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, name, sess, observation_space, action_space,
                 learning_rate, tau, gamma,
                 critic_model_save_path = 'results/models',
                 target_critic_model_save_path = 'results/models',
                 restore_model_flag=False,
                 restore_model_version = 0):
        """
        Parameters
        ----------
        name: str
            The name of this cirtic network. Giving a name to object of CriticNetwork
            is necessary to avoid messing up trainable variables together.
        sess: tf.Session
            tf.Session to run computational graph
        observation_space: gym.spaces.Box
            observation space of environment
        action_space: gym.spaces.Box
            action space of environment
        learning_rate: float
            learning rate to train CriticNetwork
        tau: float
            hyper-parameter weighting the update of target network
        gamma: float
            discount rate
        critic_model_save_path: str default = 'results/models'
            path of critic model we are going to save
        target_critic_model_save: str default = 'results/models/target_critic_model.ckpt'
            path of target critic model we are going to save
        restore_model_flag: bool default=False:
            indicator of whether to restore a pre-trained critic network
        restore_model_version: int default = 0
            if restore model, this parameter gives the number of specific version
            of models we are going to restore
        """
        # name is necessary, since we will reuse this graph multiple times.
        self.name = name
        self.sess = sess
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        
        # Info for save and load pre-trained critic models
        self.critic_model_save_path = critic_model_save_path
        self.target_critic_model_save_path = target_critic_model_save_path
        
        self.restore_model_flag = restore_model_flag
        self.restore_model_version = self._find_the_most_recent_model_version()
        if self.restore_model_flag and self.restore_model_version == -1:
            raise Exception('You do not have pretrained models.\nPlease set "load_pretrained_agent_flag = False".')
        
        with tf.variable_scope(self.name) as self.scope:
            # Create the critic network
            self.inputs, self.action, self.out, self._critic_model = self.create_critic_network()
            if self.restore_model_flag == True:
                temp_name = os.path.join(self.critic_model_save_path,
                                         self.name + '_' + str(self.restore_model_version)+'.ckpt')
                self._critic_model.load(temp_name)
                print('Restored critic model: {}'.format(temp_name))
            # We should avoid using tf.trainable_variables(), since it will
            # mix all trainable variables together.
            self.network_params = tf.trainable_variables(scope=self.name)
    
            # Target Network
            self.target_inputs, self.target_action, self.target_out, self._target_critic_model = self.create_critic_network()
            if self.restore_model_flag == True:
                temp_name = os.path.join(self.critic_model_save_path,
                                         self.name + '_target_' + str(self.restore_model_version)+'.ckpt')
                self._target_critic_model.load(temp_name)
                print('Restore target critic model: {}'.format(temp_name))
            self.target_network_params = tf.trainable_variables(scope=self.name)[len(self.network_params):]
            #Tracer()()
            # Op for periodically updating target network with online network
            # weights with regularization
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                + tf.multiply(self.target_network_params[i], 1. - self.tau))
                    for i in range(len(self.target_network_params))]
    
            # Network target (y_i)
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
    
            # Define loss and optimization Op
            self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
            # Get the gradient of the net w.r.t. the action.
            # For each action in the minibatch (i.e., for each x in xs),
            # this will sum up the gradients of each critic output in the minibatch
            # w.r.t. that action. Each output is independent of all
            # actions except for one.
            self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim], name = 'CriticInputState')
        action = tflearn.input_data(shape=[None, self.a_dim], name = 'CriticInputAction')
        
        h1_inputs = tflearn.fully_connected(inputs, 20)
        h1_inputs = tflearn.activations.relu(h1_inputs)
        h1_inputs = tflearn.layers.normalization.batch_normalization(h1_inputs)
        
        h1_action = tflearn.fully_connected(action, 20)
        h1_action = tflearn.activations.relu(h1_action)
        h1_action = tflearn.layers.normalization.batch_normalization(h1_action)
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        merged1 = tflearn.layers.merge([h1_inputs, h1_action], mode='concat')
        h2 = tflearn.fully_connected(merged1, 20)
        h2 = tflearn.activations.relu(h2)
        h2 = tflearn.layers.normalization.batch_normalization(h2)
        
        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(h2, 1, weights_init=w_init, name = 'CriticOutput')
        model = DNN(out, tensorboard_verbose = 3)
        return inputs, action, out, model

    def save_critic_network(self, critic_save_time = 0):
        """
        Function used to save critic and target critic model
        Parameters
        ----------
        critic_save_time: int default = 0
            the time when save this ciritic and its target critic models.
        """
        self._critic_model.save(model_file = os.path.join(self.critic_model_save_path,
                                                          self.name + '_' + str(critic_save_time)+'.ckpt'))
        self._target_critic_model.save(model_file = os.path.join(self.target_critic_model_save_path,
                                                                 self.name + '_target_' + str(critic_save_time)+'.ckpt'))
        print('Save critic networks.')

    def train(self, inputs, action, predicted_q_value):
        """
        Returns
        -------
        loss: mean square error
            
        out: output of Critic_Network
            
        optimize: tf.operation
        """
        return self.sess.run([self.loss, self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def _find_the_most_recent_model_version(self):
        """
        Returns
        -------
        the_most_recent_model_version: int
            the most recent model version
        """
        # Find the most recent version
        model_version = []
        for file_name_temp in os.listdir(self.critic_model_save_path):
            if self.name+'_target_' in file_name_temp:
                _, version_temp = file_name_temp.split('.')[0].split(self.name+'_target_')
                model_version.append(version_temp)
        if len(model_version) != 0:
            the_most_recent_model_version = max([int(i) for i in model_version])
        else:
            the_most_recent_model_version = -1
        return the_most_recent_model_version
# ===========================
#   Environment Model DNNs
# ===========================
class EnvironmentModelNetwork(object):
    """
    Environment Model is to learn the state trasition dynamics. Specifically,
    it learns the maping:
        from (observation_t, action_t) to (observation_t+1, reward_t)
    """
    def __init__(self, name, sess, observation_space, action_space,
                 learning_rate = 0.0001,
                 # Save Environment Model
                 env_model_save_path = 'results/models/',
                 # Restore Environment Model
                 env_restore_flag = False,
                 env_model_restore_path_and_name = 'results/models/env_model.ckpt'):
        """
        Initialize environment model.
        
        Parameters
        ----------
        name: str
        
        sess: tf.Session
        
        observation_space: gym.spaces.Box
            
        action_space: gym.spaces.Box
            
        learning_rate: float default = 0.0001
            learning rate
        env_model_save_path: str default = 'results/models/'
            the path to save environment model
                 # Restore Environment Model
        env_restore_flag: bool default = False
            idicate whether restore a previously trained environment model
        env_model_restore_path_and_name: str default = 'results/models/env_model.ckpt'
            the path and name of previously trained environment model that is 
            going to restore
        """
        self.name = name
        self.sess = sess
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.learning_rate = learning_rate
        # Parameters for save environment model
        self.env_model_save_path = env_model_save_path
        # Parameters for restoring environment model
        self.env_restore_flag = env_restore_flag
        self.env_model_restore_path_and_name = env_model_restore_path_and_name
        
        # Create Environment Model
        with tf.variable_scope(self.name) as self.scope:
            self.obs_input, self.act_input, self.obs_output, self.reward_output, self.env_model = self.create_environment_model_network()
            if self.env_restore_flag == True:
                print('restore environment model')
                self.env_model = keras.models.load_model(self.env_model_restore_path_and_name)
       
    def create_environment_model_network(self):
        """
        Create environment model network.
        
        Returns
        -------
        observation_inputs: tf.Tensor
            tensor of observation input
        action_inputs: tf.Tensor
            tensor of action input
        observation_output: tf.Tensor
            tensor of observation output
        reward_output: tf.Tensor
            tensor of reward output
        model: Keras model
        """
        observation_inputs = keras.Input(shape=(self.observation_dim,), name='EnvModel_observation_input')
        obs_h1 = keras.layers.Dense(units = 20, activation='relu')(observation_inputs)
        obs_h1 = keras.layers.BatchNormalization()(obs_h1)
        action_inputs = keras.Input(shape=(self.action_dim,), name='EnvModel_action_input')
        act_h1 = keras.layers.Dense(units = 20, activation='relu')(action_inputs)
        act_h1 = keras.layers.BatchNormalization()(act_h1)
        
        merged_inputs = keras.layers.add([obs_h1, act_h1], name = 'EnvModel_merged_input')
        merged_h1 = keras.layers.Dense(units=20, activation = 'relu')(merged_inputs)
        merged_h1 = keras.layers.BatchNormalization()(merged_h1)
        
        observation_output = keras.layers.Dense(units = (self.observation_dim), activation = 'relu')(merged_h1)
        reward_output = keras.layers.Dense(units = (1), activation = 'relu')(merged_h1)
        
        model = keras.Model(inputs = [observation_inputs, action_inputs],
                            outputs = [observation_output, reward_output])
        adam = keras.optimizers.Adam(lr = 0.001)
        model.compile(optimizer = adam, loss = 'mse')
        # Plot Environment Model
        #keras.utils.plot_model(model, to_file = 'actor_model.png',show_shapes=True, show_layer_names=True)
        
        return observation_inputs, action_inputs, observation_output, reward_output, model 
    
    def train(self, observation_inputs, action_inputs, observation_output, reward_output):
        """
        Train environment model.
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
        action_inputs: ndarray
            action at time step t
        observation_output: ndarray
            array combined of (observation,reward): 
                observation at time step t+1
                reward at time step t
        """
        self.env_model.fit(x=[observation_inputs,action_inputs],
                           y=[observation_output, reward_output],
                           verbose = 0)
    
    def evaluate(self, observation_inputs, action_inputs, observation_output, reward_output):
        """
        Evaluate environment model. The testing samples should be different from
        training samples.
        
        Parameters
        ----------
        observation_inputs: ndarray
            observation at time step t
        action_inputs: ndarray
            action at time step t
        observation_output: ndarray
            array combined of (observation,reward): 
                observation at time step t+1
                reward at time step t
        
        Returns
        -------
        np.sqrt(np.mean(loss)):
            square root of mean squared error on the test set.
        """
        loss_all, obs_loss, r_loss = self.env_model.evaluate(x=[observation_inputs,action_inputs],
                                                             y=[observation_output, reward_output],
                                                             verbose = 0)
        return np.sqrt(loss_all)
    
    def save_environment_model_network(self, env_save_time):
        """
        save environment model
        Parameters
        ----------
        env_save_time: int
            the time step when saving the environment model
        """
        keras.models.save_model(self.env_model,
                                filepath = self.env_model_save_path + '/env_model_' + str(env_save_time) + '.ckpt')
        
    def predict_target(self, observation, action):
        """
        Predict next step observation and reward based on current observation
        and action.
        
        Parameters
        ----------
        observation: ndarray
            observation at time step t
        action: ndarray
            action taken at time step t
        Returns
        -------
        prediction: list [predicted_observation, predicted_reward]
            prediction of observation and reward resulted from taken action in 
            observation.
        """
        return self.env_model.predict([observation, action])
    def get_environment_model_weights(self):
        """
        
        """
        return self.env_model.get_weights()
# ===========================
#  Intrinsic Motivation Components
# ===========================
class KnowledgeBasedIntrinsicMotivationComponent():
    def __init__(self, env_models_dir, sliding_window_size):
        """
        Parameters
        ----------
        env_models_dir: str
            the directory where environment models are saved in.
        sliding_window_size: int
            the size of sliding window
        """
        self.env_models_dir = env_models_dir
        
        if int(sliding_window_size) < 2:
            raise ValueError("sliding_window_size should be an int >= 2")
        self.sliding_window_size = sliding_window_size
        self.env_models = deque(maxlen=self.sliding_window_size)
        
    def load_env_models_from_disk(self):
        """
        Call when new environment model is saved, rather than calculate 
        knowledge-based intrinsic motivation, to:
            1. load newest environment model
            2. reduce cost of reading disk
        """
        env_model_names = glob.glob(self.env_models_dir+"/env_model_*.ckpt")
        nums_env_models = []
        for i in range(len(env_model_names)):
            model_num, _ = env_model_names[i].split('env_model_')[1].split('.ckpt')
            nums_env_models.append(int(model_num))
        
        nums_env_models.sort()
        #total_num_env_models = len(nums_env_models)
        num_newest_env_model = nums_env_models[-1]
        
        self.env_models.append(keras.models.load_model(os.path.join(self.env_models_dir,
                                                                    "env_model_"+ str(num_newest_env_model)+".ckpt")))
        
    def set_weights_of_oldest_env_model_with_newest_env_model(self, newest_weights):
        """
        Set the weights of oldest env model in self.env_models with the weights
        of newest environment model.
        
        This function is used after self.env_models is fully filled with env 
        models.
        
        Parameters
        ----------
        newest_weights: a list of all weight tensors in the model, as Numpy arrays
            env_model.get_weights()
        """
        if len(self.env_models) == 0:
            print('self.env_models is empty. Loading from disk ...')
            self.load_env_models_from_disk()
        elif len(self.env_models) != self.sliding_window_size:
            print('Adding new env model ...')
            temp_env_model = self.env_models[0]
            temp_env_model.set_weights(newest_weights)
            self.env_models.append(temp_env_model)
        else:
            print('Updating oldest env model to newest ...')
            oldest_env_model = self.env_models.popleft()
            oldest_env_model.set_weights(newest_weights)
            self.env_models.append(oldest_env_model)
        
    def knowledge_based_intrinsic_reward(self, obs_old, act_old,
                                         r_new, obs_new):
        """
        Calculate knowledge-based intrinsic motivation.
        Called every step.
        
        Parameters
        ----------
        obs_old: ndarray
            observation at time step t
        act_old:
            action at time step t
        r_new:
            reward at time stem t
        obs_new:
            observation at time step t+1
            
        Returns
        -------
        knowledge_based_intrinsic_reward: float
            knowledge based intrinsic rewward
        """
        if len(self.env_models) != self.sliding_window_size:
            raise ValueError('self.env_models) != self.sliding_window_size')
        
        # Calculate Learning Progress
        first_half_window = int(self.sliding_window_size/2)
        first_half_norm_error = 0
        second_half_norm_error = 0
        for i in range(self.sliding_window_size):
            prediction = self.env_models[i].predict([obs_old, act_old])
            obs_error = obs_new - prediction[0]
            r_error = r_new - prediction[1]
            pred_error = np.concatenate((obs_error, r_error), axis = 1)
#            sr_mse = np.sqrt(np.mean(pred_error ** 2))
            norm_error = np.linalg.norm(pred_error, ord = 2)
            if i <= first_half_window:
                first_half_norm_error += norm_error
            else:
                second_half_norm_error += norm_error
        learning_progress = abs(first_half_norm_error - second_half_norm_error) / (self.sliding_window_size/2)
        
        return learning_progress
        
    def visualize_knowledge_based_intrinsic_motivation(self, intrinsically_motivated_actor,
                                                       intrinsically_motivated_critic):
        """
        visualize intrinsic motivation using T-SNE.
        Our data has the form:
            action = Actor(observation)
            interest = Critic(observation, action)
        Parameters
        ----------
        intrinsically_motivated_actor:
            
        intrinsically_motivated_critic:
            
        """

# ===========================
#   Living Architecture System Agent
# ===========================

class LASAgent_Actor_Critic():
    """
    LASAgent is the learning agent of Living Architecture System. Basically, it
    consists of three main components coding policies and value-functions:
        1. Extrinsically motivated actor-critic model
        2. Knowledge-based Intrinsically motivated actor-critic model
        3. Competence-based Intrinsically motivated actor-critic model
    And two main components producing intrinsic motivation
        1. Knowledge-based intrinsic motivation
        2. Competence-based intrinsic motivation
    """
    def __init__(self, sess, agent_name,
                 observation_space, action_space,
                 actor_lr = 0.0001, actor_tau = 0.001,
                 critic_lr = 0.0001, critic_tau = 0.001, gamma = 0.99,
                 minibatch_size = 64,
                 max_episodes = 50000, max_episode_len = 1000,
                 # Exploration Strategies
                 exploration_action_noise_type = 'ou_0.2',
                 exploration_epsilon_greedy_type = 'none',
                 # Save Summaries
                 save_dir = '',
                 experiment_runs = '',
                 # Save and Restore Actor-Critic Model
                 restore_actor_model_flag = False,
                 restore_critic_model_flag = False,
                 restore_model_version = 0):
        """
        Intialize LASAgent.
        
        Parameters
        ----------
        actor_lr: float default = 0.0001
            actor model learning rate
        
        actor_tau: float default = 0.001
            target actor model updating weight
        
        critic_lr: float default = 0.0001
            critic model learning rate
        
        critic_tau: float default = 0.001
            target critic model updating weight
        
        gamma default:int = 0.99
            future reward discounting paramter
        
        minibatch_size:int default = 64
            size of minibabtch
        
        max_episodes:int default = 50000
            maximum number of episodes
        
        max_episode_len: int default = 1000
            maximum lenght of each episode
        
        exploration_action_noise_type: str default = 'ou_0.2',
            set up action noise. Options:
                1. 'none' (no action noise)
                2. 'adaptive-param_0.2'
                3. 'normal_0.2'
                4. 'ou_0.2' 
        
        exploration_epsilon_greedy_type: str default = 'none',
            set up epsilon-greedy.
            1. If exploration_epsilon_greedy_type == 'none', no epsilon-greedy.
            2. 'epsilon-greedy-max_1_min_0.05_decay_0.999'
        
        save_dir: string default='')
            directory to save tensorflow summaries and pre-trained models
        
        experiment_runs: str default = ''
            directory to save summaries of a specific run 
        
        restore_actor_model_flag: bool default = False
            indicate whether load pre-trained actor model
        
        restore_critic_model_flag: bool default = False
            indicate whetther load pre-trained critic model
        """
        # Produce a string describes experiment setting
        self.experiment_setting = ['LAS Environment:' + '<br />' +\
                                   '1. action_space: ' + str(action_space.shape) + '<br />' +\
                                   '2. observation_space: ' + str(observation_space.shape) + '<br /><br />' +\
                                   'LASAgent Hyper-parameters: ' + '<br />' +\
                                   '1. actor_lr: ' + str(actor_lr) + '<br />' +\
                                   '2. actor_tau: ' + str(actor_tau) + '<br />' +\
                                   '3. critic_lr: ' + str(critic_lr) + '<br />' +\
                                   '4. critic_tau: ' + str(critic_tau) + '<br />' +\
                                   '5. gamma: ' + str(gamma) + '<br />' +\
                                   '6. minibatch_size: ' + str(minibatch_size) + '<br />' +\
                                   '7. max_episodes: ' + str(max_episodes) + '<br />' +\
                                   '8. max_episode_len: ' + str(max_episode_len) + '<br />' +\
                                   '9. action_noise_type: ' + str(exploration_action_noise_type) + '<br />' +\
                                   '10.epsilon_greedy_type: ' + str(exploration_epsilon_greedy_type) + '<br />' +\
                                   '11.restore_actor_model_flag: ' + str(restore_actor_model_flag) + '<br />' +\
                                   '12.restore_critic_model_flag: ' + str(restore_critic_model_flag)][0]
        # Init Environment Related Parameters
        self.sess = sess
        self.agent_name = agent_name
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.save_dir = save_dir
        self.experiment_runs = experiment_runs
        # Temporary Memory
        self.first_experience = True
        self.observation_old = []
        self.action_old = []
        self.reward_new = []
        self.observation_new = []
        # =================================================================== #
        #                Initialize Global Hyper-parameters                   #
        # =================================================================== #        
        self.max_episodes = max_episodes
        self.max_episode_len = max_episode_len
        self.episode_counter = 1
        self.steps_counter = 1      # Steps elapsed in one episode
        self.total_step_counter = 1 # Steps elapsed in whole life
        self.render_env = False
        # Save trained models every 5 episodes (saving to disk is very 
        # time-consuming, so we shouldn't save models so frequently.)
        # If physical system run in 2Hz and each episode = 1000 steps,
        # each episode takes 9 mins, we save models every 45 mins is enough
        # if we need to shut down every one hour.
        self.save_actor_critic_models_every_xxx_episodes = 5
        # =================================================================== #
        #                 Initialize Replay Buffers for                       #
        #         Extrinsic and Intrinsic Policy, and Environment Model        #
        # =================================================================== #         
        # ********************************************* #
        #         Replay Buffer for Extrinsic Policy    #
        # ********************************************* #
        self.buffer_size = 1000000
        self.random_seed = 1234
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.random_seed)
        # ********************************************* #
        #        Replay Buffer for Environment Model    #
        # ********************************************* #
        # 5% experience will be saved in test buffer
        self.env_model_buffer_test_ratio = 0.2
        # 1. Training Buffer
        self.env_model_train_buffer_size = 100000
        self.env_model_train_buffer = ReplayBuffer(self.env_model_train_buffer_size, self.random_seed)
        # 2. Test Buffer:
        #    For examing whether our environemnt model is converged, we can 
        #    save a small set of testing samples that will not be used to 
        #    training environment. Note that this test set should not have too
        #    much past experiences either too much recent experiences.
        self.env_model_test_buffer_size = 10000
        self.env_model_test_samples_size = 1000
        self.env_model_test_buffer = ReplayBuffer(self.env_model_test_buffer_size, self.random_seed)
        # ****************************************************** #
        #   Replay Buffer for Knowledge-based Intrinsic Policy   #
        # ****************************************************** #
        self.knowledge_based_intrinsic_policy_buffer_size = 1000000
        self.knowledge_based_intrinsic_policy_replay_buffer = ReplayBuffer(self.knowledge_based_intrinsic_policy_buffer_size,
                                                                           self.random_seed)
        # ****************************************************** #
        #   Replay Buffer for Competence-based Intrinsic Policy  #
        # ****************************************************** #
        self.competence_based_intrinsic_policy_buffer_size = 1000000
        self.competence_based_intrinsic_policy_replay_buffer = ReplayBuffer(self.competence_based_intrinsic_policy_buffer_size,
                                                                            self.random_seed)
        # =================================================================== #
        #      Initialize Parameters for Both Actor and Critic Model          #
        # =================================================================== #        
        self.minibatch_size = 64
        # Common Saving Directory (we should use os.path.join(), change to it later)
        self.models_dir = os.path.join(self.save_dir,'models',self.experiment_runs)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # =================================================================== #
        #       Initialize Extrinsically Motivated Actor-Critic Model         #
        # =================================================================== #
        # Extrinsically Motivated Actor
        self.extrinsically_motivated_actor_name = self.agent_name+'_extrinsically_motivated_actor_name'
        self.extrinsic_actor_lr = actor_lr
        self.extrinsic_actor_tau = actor_tau
        # Restore Pre-trained Actor Modles
        self.extrinsic_actor_model_save_path = self.models_dir
        self.target_extrinsic_actor_model_save_path = self.models_dir
        
        self.restore_extrinsic_actor_model_flag = restore_actor_model_flag
        self.restore_extrinsic_actor_model_version = restore_model_version
        
        self.extrinsic_actor_model = ActorNetwork(self.extrinsically_motivated_actor_name,
                                        self.sess, 
                                        self.observation_space, 
                                        self.action_space,
                                        self.extrinsic_actor_lr, 
                                        self.extrinsic_actor_tau,
                                        self.minibatch_size,
                                        self.extrinsic_actor_model_save_path,
                                        self.target_extrinsic_actor_model_save_path,
                                        self.restore_extrinsic_actor_model_flag,
                                        self.restore_extrinsic_actor_model_version)
        # Extrinsically Motivated Critic
        self.extrinsically_motivated_critic_name = self.agent_name+'_extrinsically_motivated_critic_name'
        self.extrinsic_critic_lr = critic_lr
        self.extrinsic_critic_tau = critic_tau
        self.extrinsic_gamma = gamma
        # Restore Pre-trained Critic Model
        self.extrinsic_critic_model_save_path = self.models_dir
        self.target_extrinsic_critic_model_save_path = self.models_dir
        
        self.restore_extrinsic_critic_model_flag = restore_critic_model_flag
        self.restore_extrinsic_critic_model_version = restore_model_version
        
        self.extrinsic_critic_model = CriticNetwork(self.extrinsically_motivated_critic_name,
                                          self.sess,
                                          self.observation_space,
                                          self.action_space,
                                          self.extrinsic_critic_lr,
                                          self.extrinsic_critic_tau,
                                          self.extrinsic_gamma,
                                          self.extrinsic_critic_model_save_path,
                                          self.target_extrinsic_critic_model_save_path,
                                          self.restore_extrinsic_critic_model_flag,
                                          self.restore_extrinsic_critic_model_version)
        # =================================================================== #
        #                     Initialize Environment Model                    #
        # =================================================================== #
        self.environment_model_name = self.agent_name+'current_environment_model_name'
        self.env_model_lr = 0.0001
        self.env_model_minibatch_size = 200
        self.env_model_save_path = self.models_dir
        self.save_env_model_every_xxx_episodes = 5
        self.saved_env_model_counter = 0
        self.env_restore_flag = False
        self.env_model_restore_path_and_name = 'results/models/env_model.ckpt'
        self.environment_model = EnvironmentModelNetwork(self.environment_model_name,
                                                         self.sess,
                                                         self.observation_space,
                                                         self.action_space,
                                                         self.env_model_lr,
                                                         self.env_model_save_path,
                                                         self.env_restore_flag,
                                                         self.env_model_restore_path_and_name)
        # Immediately save one env model to allow knowledge-based intrinsic
        # motivation to know the structure of environment model.
        self.saved_env_model_counter += 1
        self.environment_model.save_environment_model_network(self.saved_env_model_counter)
        # =================================================================== #
        #                     Initialize Knowledge-based                      #
        #               Intrinsically Motivated Actor-Critic Model            #
        # =================================================================== #
        # Initialize Knowledge-based Intrinsic Motivation Component
        self.knowledge_based_intrinsic_reward = 0
        # Note; actual window size = sliding window size * save_env_model_every_xxx_steps
        self.knowledge_based_intrinsic_reward_sliding_window_size = 4 
        self.update_newest_env_model_every_xxx_steps = 200
        self.knowledge_based_intrinsic_motivation_model = KnowledgeBasedIntrinsicMotivationComponent(self.models_dir,
                                                                                                     self.knowledge_based_intrinsic_reward_sliding_window_size)
        # Intrinsically Motivated Actor
        self.knowledge_based_intrinsic_actor_name = self.agent_name+'_knowledge_based_intrinsic_actor_name'
        self.knowledge_based_intrinsic_actor_lr = actor_lr
        self.knowledge_based_intrinsic_actor_tau = actor_tau
        # Restore Pre-trained Actor Motivated by Knowledge-based Intrinsic Motivation
        
        self.knowledge_based_intrinsic_actor_model_save_path = self.models_dir
        self.target_knowledge_based_intrinsic_actor_model_save_path = self.models_dir
        
        self.restore_knowledge_based_intrinsic_actor_model_flag = False
        self.restore_knowledge_based_intrinsic_actor_model_version = restore_model_version
        
        self.knowledge_based_intrinsic_actor_model = ActorNetwork(self.knowledge_based_intrinsic_actor_name,
                                                                  self.sess,
                                                                  self.observation_space,
                                                                  self.action_space,
                                                                  self.knowledge_based_intrinsic_actor_lr,
                                                                  self.knowledge_based_intrinsic_actor_tau,
                                                                  self.minibatch_size,
                                                                  self.knowledge_based_intrinsic_actor_model_save_path,
                                                                  self.target_knowledge_based_intrinsic_actor_model_save_path,
                                                                  self.restore_knowledge_based_intrinsic_actor_model_flag,
                                                                  self.restore_knowledge_based_intrinsic_actor_model_version)
        # Intrinsically Motivated Critic
        self.knowledge_based_intrinsic_critic_name = self.agent_name+'_knowledge_based_intrinsic_critic_name'
        self.knowledge_based_intrinsic_critic_lr = critic_lr
        self.knowledge_based_intrinsic_critic_tau = critic_tau
        self.knowledge_based_intrinsic_critic_gamma = gamma
        
        # Restore Pre-trained Critic Model
        
        self.knowledge_based_intrinsic_critic_model_save_path = self.models_dir
        self.target_knowledge_based_intrinsic_critic_model_save_path = self.models_dir
        
        self.restore_knowledge_based_intrinsic_critic_model_flag = False
        self.restore_knowledge_based_intrinsic_critic_model_version = restore_model_version
        
        self.knowledge_based_intrinsic_critic_model = CriticNetwork(self.knowledge_based_intrinsic_critic_name,
                                                                    self.sess,
                                                                    self.observation_space,
                                                                    self.action_space,
                                                                    self.knowledge_based_intrinsic_critic_lr,
                                                                    self.knowledge_based_intrinsic_critic_tau,
                                                                    self.knowledge_based_intrinsic_critic_gamma,
                                                                    self.knowledge_based_intrinsic_critic_model_save_path,
                                                                    self.target_knowledge_based_intrinsic_critic_model_save_path,
                                                                    self.restore_knowledge_based_intrinsic_critic_model_flag,
                                                                    self.restore_knowledge_based_intrinsic_critic_model_version)
        # =================================================================== #
        #                    Initialize Competence-based                      #
        #               Intrinsically Motivated Actor-Critic Model            #
        # =================================================================== #
        # Competence-based Intrinsic Motivation
        self.competence_based_intrinsic_reward = 0
        
        
        # =================================================================== #
        #                  Initialize Exploration Strategies                  #
        # =================================================================== #        
        # 1. Action Noise to Maintain Exploration
        self.exploration_action_noise_type = exploration_action_noise_type
        self.actor_noise = self._init_action_noise(self.exploration_action_noise_type, self.action_space.shape[0])
        # 2. Epsilon-Greedy
        self.exploration_epsilon_greedy_type = exploration_epsilon_greedy_type # 'epsilon-greedy-max_1_min_0.05_decay_0.999'
        self.epsilon_max, self.epsilon_min, self.epsilon_decay = self._init_epsilon_greedy(self.exploration_epsilon_greedy_type)
        self.epsilon = self.epsilon_max
        # 3. Knowledge-based Intrinsic Motivation (for future implementation)
        
        # 4. Competence-based Intrinsic Motivation
        
        
        # =================================================================== #
        #                       Initialize Summary Ops                        #
        # =================================================================== #        
        self.summary_dir = os.path.join(self.save_dir,'summary',self.experiment_runs)
        if not os.path.isdir(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.episode_rewards = 0
        
        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        
        # Summarize Extrinsically Motivated Actor-Critic Training
        self.summary_ops_accu_rewards, self.summary_vars_accu_rewards = self._init_summarize_accumulated_rewards()
        self.summary_ops_action_reward, self.summary_action, self.summary_reward = self._init_summarize_action_and_reward()
        self.summary_ops_critic_loss, self.summary_critic_loss = self._init_summarize_actor_critic()
        # Summarize Knowledge-based Intrinsic Motivation Component
        self.summary_ops_kb_reward, self.sum_kb_reward = self._init_summarize_knowledge_based_intrinsic_reward()
        # Summarize Environment Model Training
        self.summary_ops_env_loss, self.summary_env_loss = self._init_summarize_environment_model()
        # Summarize Experiment Setting
        self.summary_ops_experiment_setting, self.summary_experiment_setting = self._init_summarize_experiment_setting()
        summary_str_experiment_setting = self.sess.run(self.summary_ops_experiment_setting,
                                                       feed_dict = {self.summary_experiment_setting: self.experiment_setting})
        self.writer.add_summary(summary_str_experiment_setting)
        # Initialize hyper-parameters for visualize extrinsic state action value
        self._init_visualize_extrinsic_actor_critic()
#        self._init_visualize_extrinsic_state_action_value_function()
#        self._init_visualize_extrinsic_action_value_given_a_specific_state()
        # =================================================================== #
        #                    Initialize Tranable Variables                    #
        # =================================================================== #        
        self.sess.run(tf.global_variables_initializer()) # Make sure to initialze tensors before use
        self.extrinsic_actor_model.update_target_network()
        self.extrinsic_critic_model.update_target_network()

# =================================================================== #
#                       Main Interaction Functions                    #
# =================================================================== #
    def perceive_and_act(self, observation, reward, done):
        """
        Perceive observation and reward, then return action based on current
        observation.
        
        Parameters
        ----------
        observation: ndarray
            observation
        reward: float
            reward of previous action
        done: bool
            whether current simulation is done
            
        Returns
        -------
        action: ndarray
            action generated by agent
        """
        self.observation_new = observation
        self.reward_new = reward
        self.done = done
        #Tracer()()
        
        # *********************************** # 
        #            Produce Action           #
        # *********************************** #
        # If this is the first action, no one single complete experience to remember
        if self.first_experience:
            action = self._act()
            
            self.action_old = action
            self.observation_old = self.observation_new
            self.first_experience = False
            self.total_step_counter += 1
            return action
        # Action, added exploration noise
        action = self._act()
        
        # *********************************** # 
        #    Write Summaries for Analysis     #
        # *********************************** #
        # 1. Save Step Summaries
        summary_str_action_rewards = self.sess.run(self.summary_ops_action_reward,
                                                   feed_dict = {self.summary_action: self.action_old,
                                                                self.summary_reward: self.reward_new})
        self.writer.add_summary(summary_str_action_rewards, self.total_step_counter)
        # 2. Save Episode Summaries
        self.episode_rewards += self.reward_new
        if self.steps_counter == self.max_episode_len or done == True:
            # *********************************** # 
            #  Save Trained Models episodically   #
            # *********************************** #
            # Save trained models every xxx_episodes 
            if self.episode_counter % self.save_actor_critic_models_every_xxx_episodes == 0:
                self.extrinsic_actor_model.save_actor_network(self.episode_counter)
                self.extrinsic_critic_model.save_critic_network(self.episode_counter)
            # Save Environment Model to disk every "xxx_episodes"
            if (self.episode_counter % self.save_env_model_every_xxx_episodes) == 0:
                self.saved_env_model_counter += 1
                self.environment_model.save_environment_model_network(self.saved_env_model_counter)
                print('Saved {}th environment model.'.format(self.saved_env_model_counter))
            # Save data for visualize extrinsic state action value
            if self.episode_counter % self.embedding_extrinsic_state_action_value_episodic_frequency == 0:
                self.peoriodically_save_extrinsic_state_action_value_embedding(self.episode_counter)
#            # Save data for visualize extrinsic action values given a state
#            if self.episode_counter % self.embedding_extrinsic_action_value_given_a_state_episodic_frequency == 0:
#                self.peoriodically_save_extrinsic_action_value_given_a_state(self.observation_new,
#                                                                         self.episode_counter)
            # Episodic Summary
            summary_str = self.sess.run(self.summary_ops_accu_rewards,
                                        feed_dict = {self.summary_vars_accu_rewards: self.episode_rewards})
            self.writer.add_summary(summary_str,self.episode_counter)
            self.writer.flush()
            
            # Reset Summary Data
            self.steps_counter = 1
            self.episode_rewards = 0
            self.episode_counter += 1
            
        else:
            self.steps_counter += 1
        
        # *********************************** # 
        #        Remember Experiences         #
        # *********************************** #
        # 1. Extrinsic Policy Replay Buffer
        self.replay_buffer.add(np.reshape(self.observation_old, (self.extrinsic_actor_model.s_dim,)),
                               np.reshape(self.action_old, (self.extrinsic_actor_model.a_dim,)),
                               self.reward_new,
                               self.done,
                               np.reshape(self.observation_new, (self.extrinsic_actor_model.s_dim,)))
        # 2. Environment Model Replay Buffer
        if np.random.rand(1) <= self.env_model_buffer_test_ratio:
            self.env_model_test_buffer.add(np.reshape(self.observation_old, (self.extrinsic_actor_model.s_dim,)),
                                           np.reshape(self.action_old, (self.extrinsic_actor_model.a_dim,)),
                                           self.reward_new,
                                           self.done,
                                           np.reshape(self.observation_new, (self.extrinsic_actor_model.s_dim,)))
        else:
            self.env_model_train_buffer.add(np.reshape(self.observation_old, (self.extrinsic_actor_model.s_dim,)),
                                            np.reshape(self.action_old, (self.extrinsic_actor_model.a_dim,)),
                                            self.reward_new,
                                            self.done,
                                            np.reshape(self.observation_new, (self.extrinsic_actor_model.s_dim,)))
        # 3. Intrinsc Policy Replay Buffer
        #    The Learning Progress plays the role of intrinsic reward
        #    a. knowledge-based intirnsic motivation
        # number of env models in knowledge_based_intrinsic_motivation_model >= sliding window size
        if len(self.knowledge_based_intrinsic_motivation_model.env_models) == self.knowledge_based_intrinsic_reward_sliding_window_size:
            self.k_based_intrinsic_r = self.knowledge_based_intrinsic_motivation_model.knowledge_based_intrinsic_reward(np.reshape(self.observation_old, (1,self.extrinsic_actor_model.s_dim)),
                                                                                                                        np.reshape(self.action_old, (1,self.extrinsic_actor_model.a_dim)),
                                                                                                                        np.reshape(self.reward_new, (1,1)),
                                                                                                                        np.reshape(self.observation_new, (1,self.extrinsic_actor_model.s_dim)))
            self.knowledge_based_intrinsic_policy_replay_buffer.add(np.reshape(self.observation_old, (self.extrinsic_actor_model.s_dim,)),
                                                                np.reshape(self.action_old, (self.extrinsic_actor_model.a_dim,)),
                                                                self.k_based_intrinsic_r,
                                                                self.done,
                                                                np.reshape(self.observation_new, (self.extrinsic_actor_model.s_dim,)))
            # Summarize Knowledge-based Intrinsic Reward
            sum_str = self.sess.run(self.summary_ops_kb_reward,
                                    feed_dict={self.sum_kb_reward:self.k_based_intrinsic_r})
            self.writer.add_summary(sum_str, self.total_step_counter)
        #    b. competence-based intrinsic motivation
        
        
        # *********************************** # 
        #             Train Models            #
        # *********************************** #
        self._train()
        
        
        # *********************************** # 
        #       Reset Temporary Variables     #
        # *********************************** #
        # Before return, set observation and action as old.
        self.observation_old = self.observation_new
        self.action_old = action
        self.total_step_counter += 1
        return action
    
    def _act(self):
        """
        Produce action based on current observation.
        """
        # Epsilon-Greedy
        if self.exploration_epsilon_greedy_type != 'none':
            if np.random.rand(1) <= self.epsilon:
                action = np.reshape(self.action_space.sample(), [1,self.action_space.shape[0]])
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                if self.total_step_counter % 2000 == 0:
                    print("epsilon:{}".format(self.epsilon))
                return action
        # Action Noise
        if self.exploration_action_noise_type != 'none':
            action = self.extrinsic_actor_model.predict(np.reshape(self.observation_new, (1, self.extrinsic_actor_model.s_dim))) + self.actor_noise() #The noise is too huge.
        else:
            action = self.extrinsic_actor_model.predict(np.reshape(self.observation_new, (1, self.extrinsic_actor_model.s_dim)))
        
        return action
    
    def _train(self):
        """
        Train Actor-Critic Model
        """
        # Keep adding experience to the memory until
        # there are at least minibatch size samples
        # ************************************************** # 
        #  Train Extrinsically Motivated Actor-Critic Model  #
        # ************************************************** #
        if self.replay_buffer.size() > self.minibatch_size:
            # Random Samples
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                self.replay_buffer.sample_batch(int(self.minibatch_size))
            # Prioritized Samples
            
            # Calculate targets
            target_q = self.extrinsic_critic_model.predict_target(
                s2_batch, self.extrinsic_actor_model.predict_target(s2_batch))

            y_i = []
            for k in range(int(self.minibatch_size)):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.extrinsic_critic_model.gamma * target_q[k])

            # Update the critic given the targets
            critic_loss, predicted_q_value, _ = self.extrinsic_critic_model.train(\
                                                                                  s_batch,
                                                                                  a_batch,
                                                                                  np.reshape(y_i, (int(self.minibatch_size), 1)))
            # Summarize critic training loss
            summary_critic_loss_str = self.sess.run(self.summary_ops_critic_loss,
                                                    feed_dict = {self.summary_critic_loss: critic_loss})
            self.writer.add_summary(summary_critic_loss_str, self.total_step_counter)
            
            # Update the actor policy using the sampled gradient
            a_outs = self.extrinsic_actor_model.predict(s_batch)
            grads = self.extrinsic_critic_model.action_gradients(s_batch, a_outs)
            self.extrinsic_actor_model.train(s_batch, grads[0])
            
            # Update target networks
            self.extrinsic_actor_model.update_target_network()
            self.extrinsic_critic_model.update_target_network()
        # ************************************************** # 
        #               Train Environment Model              #
        # ************************************************** #
        if self.env_model_train_buffer.size() > self.env_model_minibatch_size:
            # Train env model every step
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.env_model_train_buffer.sample_batch(int(self.env_model_minibatch_size))
            self.environment_model.train(s_batch,
                                         a_batch,
                                         s2_batch,
                                         np.reshape(r_batch, (int(self.env_model_minibatch_size), 1)))
            # Every "update_newest_env_model_every_xxx_steps" steps, replace the
            # oldeest env model in knowledge-based intrinsic motiavtion compoent
            # with the newest env model.
            if (self.total_step_counter % self.update_newest_env_model_every_xxx_steps) == 0: 
                newest_env_weights = self.environment_model.get_environment_model_weights()
                self.knowledge_based_intrinsic_motivation_model.set_weights_of_oldest_env_model_with_newest_env_model(newest_env_weights)
                
            # Evaluate on test buffer every step
            if self.env_model_test_buffer.size() > self.env_model_test_samples_size:
                s_batch_test, a_batch_test, r_batch_test, t_batch_test, s2_batch_test =\
                        self.env_model_test_buffer.sample_batch(int(self.env_model_test_samples_size))
                env_loss = self.environment_model.evaluate(s_batch_test,
                                                           a_batch_test,
                                                           s2_batch_test,
                                                           np.reshape(r_batch_test, (int(self.env_model_test_samples_size), 1)))
                # Summaries of Training Environment Model
                summary_env_loss_str = self.sess.run(self.summary_ops_env_loss,
                                                     feed_dict = {self.summary_env_loss: env_loss})
                self.writer.add_summary(summary_env_loss_str, self.total_step_counter)
        # ********************************************************************* # 
        #  Train Knowledge-based Intrinsically Motivated Actor-Critic Model     #
        # ********************************************************************* #
        
# =================================================================== #
#                 Intrinsic Motivation Components                     #
# =================================================================== #

        
    def competence_based_intrinsic_motivation_component(self):
        """
        Returns
        -------
        competence_based_intrinsic_reward: float
        """
        


# =================================================================== #
#                    Initialization Helper Functions                  #
# =================================================================== # 
    def _init_epsilon_greedy(self, exploration_epsilon_greedy_type):
        """
        Initialize hyper-parameters for epsilon-greedy.
        Parameters
        ----------
        exploration_epsilon_greedy_type: str default = 'epsilon-greedy-max_1_min_0.05_decay_0.999'
            str for setting epsilon greedy. Please keep the format and just change float numbers.
            For default 'epsilon-greedy-max_1_min_0.05_decay_0.999', it means:
                maximum epsilon = 1
                minimum spsilom = 0.05
                epsilon decay = 0.999
            If exploration_epsilon_greedy_type == 'none', no epsilon-greedy.
        
        Returns
        -------
        epsilon_max: float
            maximum epsilon
        epsilon_min: float
            minimum spsilom
        epsilon_decay: float
            epsilon decay
        """
        if exploration_epsilon_greedy_type == 'none':
            epsilon_max=0
            epsilon_min=0
            epsilon_decay=0
        else:
            _, epsilon_max, _, epsilon_min, _, epsilon_decay = exploration_epsilon_greedy_type.split('_')
        
        return float(epsilon_max), float(epsilon_min), float(epsilon_decay)
    
    def _init_action_noise(self, action_noise_type='ou_0.2', nb_actions=1):
        """
        Initialize action noise object.
        
        Parameters
        ----------
        action_noise_type: str default = 'ou_0.2'
            type of action noise:
                1. 'none' (no action noise)
                2. 'adaptive-param_0.2'
                3. 'normal_0.2'
                4. 'ou_0.2'
        nb_actions: int default = 1
            dimension of action space
        
        Returns
        -------
            action_noise: object of ActionNoise class.
        """
        if action_noise_type == 'none':
            pass
        elif 'adaptive-param' in action_noise_type:
            _, stddev = action_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            return param_noise
        elif 'normal' in action_noise_type:
            _, stddev = action_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            return action_noise
        elif 'ou' in action_noise_type:
            _, stddev = action_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            return action_noise
        else:
            raise RuntimeError('unknown noise type "{}"'.format(action_noise_type))
# =================================================================== #
#                   Initialization Summary Functions                  #
# =================================================================== # 
    def _init_summarize_accumulated_rewards(self):
        """
        Function used for building summaries.
        """
        episode_rewards = tf.Variable(0.)
        episode_rewards_sum = tf.summary.scalar("Accumulated_Rewards", episode_rewards)
        
        summary_ops = tf.summary.merge([episode_rewards_sum])
    
        return summary_ops, episode_rewards
    
    def _init_summarize_action_and_reward(self):
        """
        Histogram summaries of action and reward
        
        Returns
        -------
        summary_ops: tf.ops
            ops to summarize action and reward
        action: tf.placeholder
            placeholder for feeding action
        reward: tf.placeholder
            placeholder for feeding reward
        """
        action = tf.placeholder(tf.float32, shape=[1,self.action_space.shape[0]])
        reward = tf.placeholder(tf.float32)
        
        action_sum = tf.summary.histogram("action", action)
        reward_sum = tf.summary.scalar("reward", reward)
        
        summary_ops = tf.summary.merge([action_sum, reward_sum])
        return summary_ops, action, reward
    
    def _init_summarize_actor_critic(self):
        """
        Summarize data from actor-critic model.
        """
        loss_critic = tf.placeholder(dtype = tf.float32)
        loss_critic_sum = tf.summary.scalar('loss_critic', loss_critic)
        loss_critic_sum_op = tf.summary.merge([loss_critic_sum])
        return loss_critic_sum_op, loss_critic
    
    def _init_summarize_experiment_setting(self):
        """
        Summarize experiment setting
        """
        experiemnt_setting = tf.placeholder(tf.string)
        experiemnt_setting_sum = tf.summary.text('Actor_Critic_Agent_Experiment_Setting', experiemnt_setting)
        summary_ops = tf.summary.merge([experiemnt_setting_sum])
        return summary_ops, experiemnt_setting
       
    def _init_summarize_environment_model(self):
        """
        Summarize environment model training
        """
        loss = tf.placeholder(dtype = tf.float32)
        loss_sum = tf.summary.scalar('loss_env_model', loss)
        loss_sum_op = tf.summary.merge([loss_sum])
        return loss_sum_op, loss
    
    def _init_summarize_knowledge_based_intrinsic_reward(self):
        """ """
        knowledge_based_intrinsic_reward = tf.placeholder(tf.float32)
        summary = tf.summary.scalar('knowledge_based_intrinsic_reward', knowledge_based_intrinsic_reward)
        sum_op = tf.summary.merge([summary])
        return sum_op, knowledge_based_intrinsic_reward
# =================================================================== #
#             Visualize Extrinsic State-Action Value                  #
# =================================================================== #    
    def _init_visualize_extrinsic_actor_critic(self):
        self.embeded_vars_of_extrinsic_actor_critic_file_name = 'embeded_vars_of_extrinsic_actor_critic.ckpt'
        self.embeded_vars_of_extrinsic_actor_critic_config = projector.ProjectorConfig()
        """
        Visualize state-action value of sampled (state,action) pair.
        """
        self.embedding_extrinsic_state_action_value_sample_size = 10000
        # save every 2 episode
        self.embedding_extrinsic_state_action_value_episodic_frequency = 2 
        
        # Generate embeded data: (sample_size, state_dim + action_dim)
        # Note: embedded data only need to save once, while metadata need to save
        # several times.
        act_dim = self.action_space.shape[0]
        obs_dim = self.observation_space.shape[0]
        embeded_data = np.zeros((self.embedding_extrinsic_state_action_value_sample_size, obs_dim+act_dim))
        self.embeded_extrinsic_action_samples = np.zeros((self.embedding_extrinsic_state_action_value_sample_size, act_dim))
        self.embeded_extrinsic_state_samples = np.zeros((self.embedding_extrinsic_state_action_value_sample_size, obs_dim))
        for i in range(self.embedding_extrinsic_state_action_value_sample_size):
            act_sample = self.action_space.sample()
            obs_sample = self.observation_space.sample()
            embeded_data[i,:] = np.concatenate((obs_sample,act_sample))
            self.embeded_extrinsic_action_samples[i,:] = act_sample
            self.embeded_extrinsic_state_samples[i,:] = obs_sample
        # Initialize embedding variable
        self.embedding_extrinsic_state_action_value_var = tf.Variable(embeded_data,
                                                                      dtype=tf.float32,
                                                                      name = 'extrinsic_state_action_value')
        self.sess.run(self.embedding_extrinsic_state_action_value_var.initializer)
        """ 
        Visualize state-action value of sampled action given a specific state.
        """
        self.embedding_extrinsic_action_value_given_a_state_sample_size = 10000
        self.embedding_extrinsic_action_value_given_a_state_episodic_frequency = 2
        
        # Generate embedding data: (sample_size, action_dim)
        act_dim = self.action_space.shape[0]
        self.embeded_extrinsic_action_samples = np.zeros((self.embedding_extrinsic_action_value_given_a_state_sample_size,act_dim))
        for i in range(self.embedding_extrinsic_action_value_given_a_state_sample_size):
            self.embeded_extrinsic_action_samples[i,:] = self.action_space.sample()
        # Initialize embedding variable
        self.embeded_extrinsic_action_value_given_a_state_var = tf.Variable(self.embeded_extrinsic_action_samples,
                                                                            dtype = tf.float32,
                                                                            name = 'extrinsic_action_value_given_a_state')
        self.sess.run(self.embeded_extrinsic_action_value_given_a_state_var.initializer)
        
        
        # Save embedding vars
        saver_embed = tf.train.Saver([self.embedding_extrinsic_state_action_value_var,
                                      self.embeded_extrinsic_action_value_given_a_state_var])
        saver_embed.save(self.sess,
                         os.path.join(self.summary_dir, self.embeded_vars_of_extrinsic_actor_critic_file_name))
    def _init_visualize_extrinsic_state_action_value_function(self):
        """
        Visualize state-action value of sampled (state,action) pair.
        """
        self.embedding_extrinsic_state_action_value_sample_size = 10000
        # save every 2 episode
        self.embedding_extrinsic_state_action_value_episodic_frequency = 2 
        
        # Generate embeded data: (sample_size, state_dim + action_dim)
        # Note: embedded data only need to save once, while metadata need to save
        # several times.
        act_dim = self.action_space.shape[0]
        obs_dim = self.observation_space.shape[0]
        embeded_data = np.zeros((self.embedding_extrinsic_state_action_value_sample_size, obs_dim+act_dim))
        self.embeded_extrinsic_action_samples = np.zeros((self.embedding_extrinsic_state_action_value_sample_size, act_dim))
        self.embeded_extrinsic_state_samples = np.zeros((self.embedding_extrinsic_state_action_value_sample_size, obs_dim))
        for i in range(self.embedding_extrinsic_state_action_value_sample_size):
            act_sample = self.action_space.sample()
            obs_sample = self.observation_space.sample()
            embeded_data[i,:] = np.concatenate((obs_sample,act_sample))
            self.embeded_extrinsic_action_samples[i,:] = act_sample
            self.embeded_extrinsic_state_samples[i,:] = obs_sample
        # Initialize embedding variable
        self.embedding_extrinsic_state_action_value_var = tf.Variable(embeded_data,
                                                                      dtype=tf.float32,
                                                                      name = 'extrinsic_state_action_value')
        self.sess.run(self.embedding_extrinsic_state_action_value_var.initializer)
        # Save embedding
        saver_embed = tf.train.Saver([self.embedding_extrinsic_state_action_value_var])
        saver_embed.save(self.sess,
                         os.path.join(self.summary_dir,self.embeded_vars_of_extrinsic_actor_critic_file_name))

    def peoriodically_save_extrinsic_state_action_value_embedding(self, version_num):
        """
        Preparing embeded data and metadata, then call function peoriodically
        to write these data into tensorflow summary.
        """
        # metadata should be saved peoriodically and separatively to associate with differetn embedding
        metadata_file_name = 'embedding_extrinsic_state_action_value_meta_' + str(version_num) +'.tsv'
        
        # Preparing metadate which will be associated with embedding when write
        # these data to summary. Thus, the file name of metadata is:
        #   (embeded_data_name+'_metadata.tsv')
        action_value_batch = self.extrinsic_critic_model.predict(self.embeded_extrinsic_state_samples,
                                                                 self.embeded_extrinsic_action_samples)
        metadata = pd.DataFrame(action_value_batch)
        metadata.columns = ['embedding_extrinsic_state_action_value_'+str(version_num)]
        metadata.to_csv(os.path.join(self.summary_dir, metadata_file_name),
                        sep = '\t')
        
        # Associate metadata with embedding:
        # embeddings {
        #   tensor_name: 'word_embedding'
        #   metadata_path: '$LOG_DIR/metadata.tsv'}
        embedding = self.embeded_vars_of_extrinsic_actor_critic_config.embeddings.add()
        embedding.tensor_name = self.embedding_extrinsic_state_action_value_var.name
        embedding.metadata_path = metadata_file_name
        projector.visualize_embeddings(self.writer,
                                       self.embeded_vars_of_extrinsic_actor_critic_config)
        
    def _init_visualize_extrinsic_action_value_given_a_specific_state(self):
        """ 
        Visualize state-action value of sampled action given a specific state.
        """
        self.embedding_extrinsic_action_value_given_a_state_sample_size = 10000
        self.embedding_extrinsic_action_value_given_a_state_episodic_frequency = 2
        
        # Generate embedding data: (sample_size, action_dim)
        act_dim = self.action_space.shape[0]
        self.embeded_extrinsic_action_samples = np.zeros((self.embedding_extrinsic_action_value_given_a_state_sample_size,act_dim))
        for i in range(self.embedding_extrinsic_action_value_given_a_state_sample_size):
            self.embeded_extrinsic_action_samples[i,:] = self.action_space.sample()
        # Initialize embedding variable
        self.embeded_extrinsic_action_value_given_a_state_var = tf.Variable(self.embeded_extrinsic_action_samples,
                                                                            dtype = tf.float32,
                                                                            name = 'extrinsic_action_value_given_a_state')
        self.sess.run(self.embeded_extrinsic_action_value_given_a_state_var.initializer)
        # Save embedding var
        saver_embed = tf.train.Saver([self.embeded_extrinsic_action_value_given_a_state_var])
        saver_embed.save(self.sess,
                         os.path.join(self.summary_dir, self.embeded_vars_of_extrinsic_actor_critic_file_name))
        
    def peoriodically_save_extrinsic_action_value_given_a_state(self,
                                                                state,
                                                                version_num):
        """
        Peoriodically called to visualize action value of a given state
        
        Parameters
        ----------
        state:
        
        version_num:
            
        """
        metadata_file_name = 'embedding_extrinsic_action_value_given_a_state_meta_' + str(version_num) + '.tsv'
        # Generate metadata
        state_samples = np.tile(state, [self.embedding_extrinsic_action_value_given_a_state_sample_size,1])
        action_value_batch = self.extrinsic_critic_model.predict(state_samples,
                                                                 self.embeded_extrinsic_action_samples)
        # Save metadata
        metadata = pd.DataFrame(action_value_batch)
        metadata.columns = ['embedding_extrinsic_action_value_given_a_state'+str(version_num)]
        metadata.to_csv(os.path.join(self.summary_dir, metadata_file_name),
                        sep = '\t')
        # Associate metadata with embedding
        embedding = self.embeded_vars_of_extrinsic_actor_critic_config.embeddings.add()
        embedding.tensor_name = self.embeded_extrinsic_action_value_given_a_state_var.name
        embedding.metadata_path = metadata_file_name
        projector.visualize_embeddings(self.writer,
                                       self.embeded_vars_of_extrinsic_actor_critic_config)
        
