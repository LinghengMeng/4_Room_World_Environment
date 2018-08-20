# 4_Room_World_Environment
This repository provides a simulation of 4-Room-World environment based on [V-REP](http://www.coppeliarobotics.com).



## Create 4-Room-World Scene
  FourRoomScene folder contains files to create a scene using V-REP api.
   
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
  
&nbsp; &nbsp; &nbsp; <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_scene2.png" width="250" height="250" />     &nbsp;  <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_axis_Legend.png" width="250" height="250" />  &nbsp; <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_Legend.png"  height="250" /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (a) 4Room World Screenshot  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (b) Axis Legend &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (c) Action Space and Goal Legend

## Virtual Environment
### FourRoomGridWorld
>The cells of the grid corresponds to the states of the environment of four rooms as shown in **4Room World Screenshot**. From any state the agent can perform one of four actions, **up**, **down**, **left** or **right**, which have a stochastic effect. With probability 2/3, the actions cause the agent to move one cell in the corresponding direction, and with probability 1/3, the agent moves instead in one of the other three directions, each with probability 1/9. In either case, if the movement would take the agent into a wall then the agent remians in the same cell. Rewards are **zero** on all state transitions except transiting into goal state which has reward **one**. 
