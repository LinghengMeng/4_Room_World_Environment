#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:31:18 2018

@author: jack.lingheng.meng
"""

import numpy as np
import functools
import warnings

def get_all_object_name_and_handle(clientID, opMode, vrep):
    """
    This function will abstract objects' name and handle by distinguishing their 
    corresponding object type, as long as these object naming with substring "_node#". 
    
    Parameters
    ----------
    clientID:
        The clientID return from vrep.simxStart().
    opMode: 
        The operation mode to call vrep.simxGetObjectGroupData().
    vrep:
        vrep
        
    # When call vrep.simxGetObjectGroupData to abstract object name and handle
    # choose appropriate objectType parameter:
        #                   joint:  vrep.sim_object_joint_type
        #        proximity sensor:  vrep.sim_object_proximitysensor_type
        #                   light:  vrep.sim_object_light_type
        #        visitor position:  vrep.sim_object_dummy_type
    """
    dataType = 0    # 0: retrieves the object names (in stringData.)
    print("Getting objects' names and handles ...")
    # Wall Brick, Floor Tile, Hallway and Goal object are all shpae in V-REP scene.
    wallBrickIndex = []
    floorTileIndex = []
    hallwayIndex = []
    goalIndex = []
    
    rc = vrep.simx_return_initialize_error_flag
    while rc != vrep.simx_return_ok:
        rc, shapeObjectHandles, intData, floatData, shapeObjectNames = vrep.simxGetObjectGroupData(clientID,vrep.sim_object_shape_type, dataType, opMode)
        if rc==vrep.simx_return_ok:
            print ('Get Wall Brick Success!!!!!') # display the reply from V-REP (in this case, just a string)
            for i, name in enumerate(shapeObjectNames):
                if 'Wall' in name:
                    print("Wall Brick: {}, and handle: {}".format(name, shapeObjectHandles[i]))
                    wallBrickIndex.append(i)
                if 'Floor' in name:
                    print("Floor Tile: {}, and handle: {}".format(name, shapeObjectHandles[i]))
                    floorTileIndex.append(i)
                if 'Hallway' in name:
                    print("Hallway: {}, and handle: {}".format(name, shapeObjectHandles[i]))
                    hallwayIndex.append(i)
                if 'Goal' in name:
                    print("Goal: {}, and handle: {}".format(name, shapeObjectHandles[i]))
                    goalIndex.append(i)
            break
        else:
            print ('Fail to get Wall Brick!!!')
    
    # Standing Participants are dummy in V-REP scene.
    standingParticipantIndex = []
    rc = vrep.simx_return_initialize_error_flag
    while rc != vrep.simx_return_ok:
        rc, standingParticipantHandles, intData, floatData, standingParticipantNames = vrep.simxGetObjectGroupData(clientID,vrep.sim_object_dummy_type, dataType, opMode)
        if rc==vrep.simx_return_ok:
            print ('Get StandingParticipant Objest Success!!!!!') # display the reply from V-REP (in this case, just a string)
            for i, name in enumerate(standingParticipantNames):
                if "Participant" in name:
                    print("Standing Participant: {}, and handle: {}".format(name, standingParticipantHandles[i]))
                    standingParticipantIndex.append(i)
            break
        else:
            print ('Fail to get Standing ParticipantNames!!!')

    
    shapeObjectHandles = np.array(shapeObjectHandles)
    shapeObjectNames = np.array(shapeObjectNames)
    
    standingParticipantHandles = np.array(standingParticipantHandles)
    standingParticipantNames = np.array(standingParticipantNames)
    
    # All objects handels and names
    wallBrickHandles = shapeObjectHandles[wallBrickIndex]
    wallBrickNames = shapeObjectNames[wallBrickIndex]
    
    floorTileHandles = shapeObjectHandles[floorTileIndex]
    floorTileNames = shapeObjectNames[floorTileIndex]
    
    hallwayHandles = shapeObjectHandles[hallwayIndex]
    hallwayNames = shapeObjectNames[hallwayIndex]
    
    goalHandles = shapeObjectHandles[goalIndex]
    goalNames = shapeObjectNames[goalIndex]
    
    standingParticipantHandles= standingParticipantHandles[standingParticipantIndex]
    standingParticipantNames = standingParticipantNames[standingParticipantIndex]
    
    
    return wallBrickHandles, wallBrickNames, \
           floorTileHandles, floorTileNames, \
           hallwayHandles, hallwayNames, \
           goalHandles, goalNames,\
           standingParticipantHandles, standingParticipantNames

def get_object_position(objectHandle, clientID, opMode, vrep):
    """
    Get the position of objects listed in objectHandle.
    
    Parameters
    ----------
    objectHandle: list
        contains a list of object handles
    clientID:
        The clientID return from vrep.simxStart().
    
    opMode: 
        The operation mode to call vrep.simxGetObjectGroupData().
    vrep:
        vrep
    
    Returns
    -------
    position: ndarray
        position of objects in objectHandle.
        Each entry has the format:
            [x, y, z]
            
    """
    position = np.zeros([len(objectHandle), 3])
    for i, handle in enumerate(objectHandle):
        res, position[i,:] = vrep.simxGetObjectPosition(clientID, handle, -1, vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            raise Exception('vrep.simxGetObjectPosition does not work.')
        #print("{} handle: {}, position: {}".format(object_name, handle, objPosition))
    return position
    
def deprecated(msg=''):
    def dep(func):
        '''This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.'''

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn_explicit(
                "Call to deprecated function {}. {}".format(func.__name__, msg),
                category=DeprecationWarning,
                filename=func.func_code.co_filename,
                lineno=func.func_code.co_firstlineno + 1
            )
            return func(*args, **kwargs)

        return new_func

    return deprecated






















