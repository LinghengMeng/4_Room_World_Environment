# 4_Room_World_Environment
This repository provides a simulation of 4-Room-World environment based on [V-REP](http://www.coppeliarobotics.com).

There are in totall **4 Types of 4-Room-World Environment** with various complexities implemented in this repository. For more detail information on these virtual environments, please refer to [4 Types of 4-Room-World Environment](https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Environment/README.md).


## Create 4-Room-World Scene (Optional)
FourRoomScene folder contains files to create a scene using V-REP api(If you do not have V-REP, please download from [V-REP](http://www.coppeliarobotics.com/downloads.html)).
   
  * 4room_world_models: V-REP models used to create a scene
      
  * VrepRemoteApiBindings: V-REP dynamic lib and api-script
      
  * CSV files: specify position of loaded objects
  
         * "wall_brick_location.csv": position of wall brick
         * "floor_tile_location.csv": position of wall floor tile
         * "hallway_location.csv": position of hallway
         * "goal_location.csv": position of goal
         * "standing_participant_location.csv": initial position of standing participant which is static.
         
  * Script to create a scene:
    `create_4room_world.py`
    
  * V-REP scene created by script: `4_room_world.ttt`
  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_scene2.png" width="250" height="250" />     &nbsp;  <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_Scene.png" width="250" height="250" /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Figure 1.** 4Room World Screenshots

## Load Scene to V-REP

## Choose 4-Room-World Virtual Environment

  1. FourRoomGridWorld
  2. FourRoomContinuousWorld
  3. FourRoomCameraVisualWorld
  4. FourRoomFirstPersonVisualWorld


Implemented in `Environment/FourRoomGridWorld.py`
