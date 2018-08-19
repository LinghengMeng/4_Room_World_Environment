#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 12:06:07 2018

@author: jack.lingheng.meng
"""
from VrepRemoteApiBindings import vrep
import csv
import os


def load_model(clientID, model_name = 'wall_brick', location_file = 'wall_brick_location.csv'):
    """
    Parameters
    ----------
    clientID: int
        the V-REP clientID 
    
    model_name: string
        the name of model you are going to load:
            1. wall_brick
            2. floor_tile
            3. hallway
            4. goal
            5. action_down
            6. action_up
            7. action_left
            8. action_right
            9. 
    
    location_file: string
        the file given by this string contains the location of model. 
        Each entry in the file has the format:
            name, x, y, z
    
    Returns
    -------
    object_handles: list
        a list of object handles
    
    object_positions: list
        a list of positions of objects
    """
    object_handles = []
    object_positions = []
    with open(location_file, newline = '') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader) # remove first row
        for i, row in enumerate(csv_reader):
            object_name = row[0]
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            #print('{}th {} location: ({}, {}, {})'.format(i, model_name, x, y, z))
            model_path = os.path.join(os.path.abspath('.'),'4room_world_models',model_name+'.ttm')
            res, handle = vrep.simxLoadModel(clientID, model_path, 0, vrep.simx_opmode_blocking)
            if res != vrep.simx_return_ok:
                raise Exception('vrep.simxLoadModel does not work!')
            else:
                object_handles.append(handle)
                #print('handle: {}'.format(handle))
            
            vrep.simxSetObjectPosition(clientID, handle, -1, [x, y, z], vrep.simx_opmode_oneshot)
            res, objPosition = vrep.simxGetObjectPosition(clientID, handle, -1, vrep.simx_opmode_blocking)
            if res != vrep.simx_return_ok:
                raise Exception('vrep.simxGetObjectPosition does not work.')
            else:
                object_positions.append(objPosition)
                #print("{} handle: {}, position: {}".format(object_name, handle, objPosition))
    return object_handles, object_positions

def creat_4room_world_scene(clientID):
    """
    Parameters
    ----------
    clientID: int
        the V-REP clientID 
    """
    # load wall brick
    wall_brick_handles, wall_brick_positions = load_model(clientID, 'wall_brick', location_file = 'wall_brick_location.csv')
    # load floor tile
    floor_tile_handles, floor_tile_positions = load_model(clientID, 'floor_tile', location_file = 'floor_tile_location.csv')
    # load hallway
    hallway_handles, hallway_positions = load_model(clientID, 'hallway', location_file = 'hallway_location.csv')
    # load goal
    goal_handles, goal_positions = load_model(clientID, 'goal', location_file = 'goal_location.csv')
    # load participant
    standing_participant_handles, standing_participant_positions = load_model(clientID, 'standing_participant', location_file = 'standing_participant_location.csv')

if __name__ == '__main__':
    """
    Load models into the scene
    """
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    if clientID != -1:
        print('Connected to remote API server')
    else:
        print('Failed connecting to remote API server')
    # create 4room_world scene
    creat_4room_world_scene(clientID)

    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)