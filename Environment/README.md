# 4 Types of 4-Room-World Environment
This folder contains in totall **3 Types of 4-Room-World Environment** which are extenstions of discrete 4-Room-Grid World proposed in [Sutton et al., AI1991].

## Virtual Environment
### 1. FourRoomGridWorld
FOr 4-Room Grid World, the state and action are discrete value.
>The cells of the grid corresponds to the states of the environment of four rooms as shown in **4Room World Screenshot**. From any state the agent can perform one of four actions, **up**, **down**, **left** or **right**, which have a stochastic effect. With probability 2/3, the actions cause the agent to move one cell in the corresponding direction, and with probability 1/3, the agent moves instead in one of the other three directions, each with probability 1/9. In either case, if the movement would take the agent into a wall then the agent remians in the same cell. Rewards are **zero** on all state transitions except transiting into goal state which has reward **one**. 

Implemented in `Environment/FourRoomGridWorld.py`

### 2. FourRoomContinuousWorld
For 4-Room continuous World, the state and action are continuous value.

### 3. FourRoomCameraVisualWorld
For 4-Room Camera Visual World, the states are sequence of images coming from a camera overlooking the whole world.

### 4. FourRoomFirstPersonVisualWorld
For 4-Room First Person Visual Worl, the states are sequence of image coming from eyes of participant which can only partially observe the state of the world.










[Sutton, Richard S., Doina Precup, and Satinder Singh. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial intelligence 112, no. 1-2 (1999): 181-211.](https://ac.els-cdn.com/S0004370299000521/1-s2.0-S0004370299000521-main.pdf?_tid=5e385c67-79e7-4e07-af80-d4bb0abbbb93&acdnat=1534771730_5258b8e295835695ebde7f6976d1291d)
