# 4 Types of 4-Room-World Environment
This folder contains in totall **4 Types of 4-Room-World Environment** which are extenstions of discrete 4-Room-Grid World proposed in [Sutton et al., AI1991].

## Virtual Environment
### 1. FourRoomGridWorld
For 4-Room Grid World, the state and action are discrete value.
>The cells of the grid corresponds to the states of the environment of four rooms as shown in **4Room World Screenshot**. From any state the agent can perform one of four actions, **up**, **down**, **left** or **right**, which have a stochastic effect. With probability 2/3, the actions cause the agent to move one cell in the corresponding direction, and with probability 1/3, the agent moves instead in one of the other three directions, each with probability 1/9. In either case, if the movement would take the agent into a wall then the agent remians in the same cell. Rewards are **zero** on all state transitions except transiting into goal state which has reward **one**. 

* Implemented in `Environment/FourRoomGridWorld.py`
* Observation Space: gym.spaces.Discrete(104)
   * 100 ordinary floor states
   * 4 hallways
* Action Space: gym.spaces.Discrete(4)
   * up: 0
   * down: 1
   * left: 2
   * right: 3

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_axis_Legend.png" width="250" height="250" /> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_Legend.png"  height="250" /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; (a) State &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (b) Action and Goal

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Figure 1. 4-Room Grid World**

### 2. FourRoomContinuousWorld
For 4-Room continuous World, the state and action are continuous value.
* Implemented in `Environment/FourRoomGridWorld.py`
* Observation Space: gym.spaces.Box()
   * x-position: [-1, 1] corresponding to axis [-6, 6] in V-REP scene.
   * y-position: [-1, 1] corresponding to axis [-6, 6] in V-REP scene.
* Action Space: gym.spaces.BOx()
   * orientation: [-1, 1] corresponding to [-pi, pi] in V-REP scene.
   * stride: [-1, 1] corresponding to [-1, 1] in V-REP scene.

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_Continuous_World_State_Legend.png" width="250" height="250" /> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://github.com/LinghengMeng/4_Room_World_Environment/blob/master/Images/4Room_Continuous_World_Action_Legend.png"  height="250" /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; (a) State &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (b) Action and Goal

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Figure 2. 4-Room Continuous World**

### 3. FourRoomCameraVisualWorld
For 4-Room Camera Visual World, the states are sequence of images coming from a camera overlooking the whole world.

### 4. FourRoomFirstPersonVisualWorld
For 4-Room First Person Visual Worl, the states are sequence of image coming from eyes of participant which can only partially observe the state of the world.









## Reference
[Sutton, Richard S., Doina Precup, and Satinder Singh. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial intelligence 112, no. 1-2 (1999): 181-211.](https://ac.els-cdn.com/S0004370299000521/1-s2.0-S0004370299000521-main.pdf?_tid=5e385c67-79e7-4e07-af80-d4bb0abbbb93&acdnat=1534771730_5258b8e295835695ebde7f6976d1291d)
