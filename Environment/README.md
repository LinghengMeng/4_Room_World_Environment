# 3 Types of 4-Room-World Environment
This folder contains in totall **3 Types of 4-Room-World Environment** which are extenstions of discrete 4-Room-Grid World in [Sutton et al., AI1991].

## Virtual Environment
### FourRoomGridWorld
>The cells of the grid corresponds to the states of the environment of four rooms as shown in **4Room World Screenshot**. From any state the agent can perform one of four actions, **up**, **down**, **left** or **right**, which have a stochastic effect. With probability 2/3, the actions cause the agent to move one cell in the corresponding direction, and with probability 1/3, the agent moves instead in one of the other three directions, each with probability 1/9. In either case, if the movement would take the agent into a wall then the agent remians in the same cell. Rewards are **zero** on all state transitions except transiting into goal state which has reward **one**. 

Implemented in `Environment/FourRoomGridWorld.py`















[Sutton, Richard S., Doina Precup, and Satinder Singh. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial intelligence 112, no. 1-2 (1999): 181-211.](https://ac.els-cdn.com/S0004370299000521/1-s2.0-S0004370299000521-main.pdf?_tid=5e385c67-79e7-4e07-af80-d4bb0abbbb93&acdnat=1534771730_5258b8e295835695ebde7f6976d1291d)
