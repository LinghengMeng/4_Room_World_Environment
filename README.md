# 4_Room_World_Environment
This repository provides a simulation of 4-Room-World environment.

## Create 4-Room-World Scene
  FourRoomScene folder contains files to create a scene using V-REP api.
   
  * 4room_world_models: V-REP models used to create a scene
      
  * VrepRemoteApiBindings: V-REP dynamic lib and api-script
      
  * CSV files: specify position of loaded objects
  
         * wall_brick_location.csv: position of wall brick
         * floor_tile_location.csv: position of wall floor tile
         * hallway_location.csv: position of hallway
         * goal_location.csv: position of goal
         * standing_participant_location.csv: initial position of standing participant which is static.
         
  * Script to create a scene:
    `create_4room_world.py`
    
  * V-REP scene created by script: `4_room_world.ttt`
  
<img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_Scene.png" width="400" height="400" />       <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_scene2.png" width="400" height="400" /> 

## Virtual Environment
