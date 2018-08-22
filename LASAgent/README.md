# LASAgent classes
This folder contains implemented Learning Algorithms for Living Architecture System. In theory, **Living Architecture Sytem** and **Visitor** could share the same set of Learning Algorithms to realize control of intelligent agent. However, in practice, using learning algorithm to control visitor is very hard and complex. Therefore, in our implementation, we separatively maintain control algorithms for [Living Architecture System](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/LASAgent) and [Visitor](https://github.com/UWaterloo-ASL/LAS_Gym/tree/ROM_Agent_Community_LM/VisitorAgent).

## Intermediate Internal Environment Classes
To ensure reusability, we provides two intermediate classes for realistic interaction in which reward signal is not provided by environment, at the same time seamlessly working with virtual environment with interfaces as in [OpenAI Gym](https://gym.openai.com/docs/). 
1. **Internal Environment for Single Agent** is mainly used to receive observation from, calculate reward and deliver action chosen by agent to real or virtual external environment.
   * InternalEnvOfAgent.py
2. **Internal Environment for Agent Community** is used to decompose the whole system into several sub-systems, at the same time calculate reward signal for each sub-system.
   * InternalEnvOfCommunity.py

## Learning Agent Classes

1. **Actor-Critic LASAgent** [Lillicrap et al. ICLR2016]
   * Implemented in `LASAgent_Actor_Critic.py`
2. **Random action LASAgent**
   * Implememted in `RandomLASAgent.py`


## Reference:

[Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).](https://arxiv.org/pdf/1509.02971.pdf)
